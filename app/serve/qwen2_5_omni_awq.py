#!/usr/bin/env python3
"""
Qwen2.5-Omni AWQ Quantized Audio Transcription HTTP Service
Based on low-VRAM mode and AWQ quantization for reduced GPU memory usage
"""

import io
import os
import sys
import tempfile
import importlib.util
import threading
import uuid
import time
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

import numpy as np
import torch
import librosa
from flask import Flask, request, jsonify, send_file
from huggingface_hub import hf_hub_download
from opencc import OpenCC
from concurrent.futures import ThreadPoolExecutor, as_completed

# AWQ quantization imports
from awq.models.base import BaseAWQForCausalLM
from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# ==================== Global Configuration ====================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024  # 4GB

# Global variables
model = None
processor = None
opencc_converter = None  # OpenCC converter for Traditional Chinese
model_lock = threading.Lock()
processing_lock = threading.Lock()
job_status = {}
job_lock = threading.Lock()

# Parallel processing
model_inference_semaphore = None  # Will be initialized based on GPU memory
max_parallel_workers = 3  # Default parallel workers

# Idle tracking
last_activity_time = None
is_processing = False
idle_check_interval = 30  # Check every 30 seconds
idle_timeout = 300  # 5 minutes
model_config = {}  # Store model configuration for auto-reload

# Directories - use absolute paths to ensure correct location
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS_DIR = os.path.join(WORK_DIR, "inputs")
TEMP_DIR = os.path.join(WORK_DIR, "temp")
OUTPUTS_DIR = os.path.join(WORK_DIR, "outputs")

os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ==================== Low-VRAM Module Replacement ====================

def replace_transformers_module():
    """Replace transformers module with low-VRAM implementation"""
    original_mod_name = 'transformers.models.qwen2_5_omni.modeling_qwen2_5_omni'

    # Find the low-VRAM mode file
    possible_paths = [
        'low-VRAM-mode/modeling_qwen2_5_omni_low_VRAM_mode.py',
        'serve/low-VRAM-mode/modeling_qwen2_5_omni_low_VRAM_mode.py',
        '../low-VRAM-mode/modeling_qwen2_5_omni_low_VRAM_mode.py',
    ]

    new_mod_path = None
    for path in possible_paths:
        if os.path.exists(path):
            new_mod_path = path
            break

    if new_mod_path is None:
        raise FileNotFoundError("Could not find modeling_qwen2_5_omni_low_VRAM_mode.py")

    print(f"[INFO] Loading low-VRAM mode from: {new_mod_path}")

    # Remove original module if already loaded
    if original_mod_name in sys.modules:
        del sys.modules[original_mod_name]

    # Load custom module
    spec = importlib.util.spec_from_file_location(original_mod_name, new_mod_path)
    new_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(new_mod)

    # Inject into sys.modules
    sys.modules[original_mod_name] = new_mod

    return new_mod

# Replace module before importing model classes
low_vram_module = replace_transformers_module()
Qwen2_5OmniDecoderLayer = low_vram_module.Qwen2_5OmniDecoderLayer
Qwen2_5OmniForConditionalGeneration = low_vram_module.Qwen2_5OmniForConditionalGeneration

# ==================== AWQ Model Wrapper ====================

class Qwen2_5_OmniAWQForConditionalGeneration(BaseAWQForCausalLM):
    """AWQ quantization wrapper for Qwen2.5-Omni"""
    layer_type = "Qwen2_5OmniDecoderLayer"
    max_seq_len_key = "max_position_embeddings"
    modules_to_not_convert = ["visual", "audio_tower"]  # Don't quantize visual/audio modules

    @staticmethod
    def get_model_layers(model):
        return model.thinker.model.layers

    @staticmethod
    def get_act_for_scaling(module):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model, device: str):
        """Move embedding and special modules to specified device"""
        model.thinker.model.embed_tokens = model.thinker.model.embed_tokens.to(device)
        model.thinker.visual = model.thinker.visual.to(device)
        model.thinker.audio_tower = model.thinker.audio_tower.to(device)

        # Move rotary embeddings
        model.thinker.visual.rotary_pos_emb = model.thinker.visual.rotary_pos_emb.to(device)
        model.thinker.model.rotary_emb = model.thinker.model.rotary_emb.to(device)

        # Move rotary embeddings for all layers
        for layer in model.thinker.model.layers:
            layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)

    @staticmethod
    def get_layers_for_scaling(module, input_feat, module_kwargs):
        """Configure layers for AWQ scaling"""
        layers = []

        # Attention input
        layers.append(dict(
            prev_op=module.input_layernorm,
            layers=[
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ],
            inp=input_feat["self_attn.q_proj"],
            module2inspect=module.self_attn,
            kwargs=module_kwargs,
        ))

        # Attention output
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(dict(
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat["self_attn.o_proj"],
            ))

        # MLP layers
        layers.append(dict(
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.gate_proj, module.mlp.up_proj],
            inp=input_feat["mlp.gate_proj"],
            module2inspect=module.mlp,
        ))

        layers.append(dict(
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
            inp=input_feat["mlp.down_proj"],
        ))

        return layers

# ==================== Parallel Processing Initialization ====================

def init_parallel_processing(max_workers=None):
    """
    Initialize parallel processing with semaphore

    Args:
        max_workers: Maximum parallel workers (None = auto-detect based on GPU memory)

    Returns:
        Actual max_workers value used
    """
    global model_inference_semaphore, max_parallel_workers

    if max_workers is None:
        # Auto-detect based on GPU memory
        if torch.cuda.is_available():
            total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if total_mem_gb >= 24:
                max_workers = 4
            elif total_mem_gb >= 16:
                max_workers = 3
            elif total_mem_gb >= 12:
                max_workers = 2
            else:
                max_workers = 1
        else:
            max_workers = 1

    max_parallel_workers = max_workers
    model_inference_semaphore = threading.Semaphore(max_workers)

    print(f"[INFO] Parallel processing initialized with max_workers={max_workers}")

    if torch.cuda.is_available():
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[INFO] GPU Memory: {total_mem_gb:.1f} GB total")

    return max_workers

# ==================== Model Loading ====================

def load_model_processor(checkpoint_path, flash_attn2=False, local_model=False):
    """Load AWQ quantized model and processor"""
    global model, processor, opencc_converter, last_activity_time, model_config

    with model_lock:
        if model is not None:
            print("[INFO] Model already loaded")
            last_activity_time = time.time()
            return

        print(f"[INFO] Loading AWQ model from: {checkpoint_path}")
        print(f"[INFO] Flash Attention 2: {flash_attn2}")
        print(f"[INFO] Local model: {local_model}")

        # Store model configuration for auto-reload
        model_config = {
            'checkpoint_path': checkpoint_path,
            'flash_attn2': flash_attn2,
            'local_model': local_model
        }

        # Initialize OpenCC converter for Simplified to Traditional Chinese (Taiwan standard)
        if opencc_converter is None:
            try:
                opencc_converter = OpenCC('s2tw')
                print("[INFO] OpenCC Simplified to Traditional Chinese (Taiwan) converter initialized")
            except Exception as e:
                print(f"[WARNING] Failed to initialize OpenCC: {e}")
                opencc_converter = None

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load AWQ quantized model
        attn_impl = "flash_attention_2" if flash_attn2 else "eager"
        model = Qwen2_5_OmniAWQForConditionalGeneration.from_quantized(
            checkpoint_path,
            model_type="qwen2_5_omni",
            torch_dtype=torch.float16,
            attn_implementation=attn_impl
        )

        # Load speaker dictionary
        if local_model:
            spk_path = os.path.join(checkpoint_path, 'spk_dict.pt')
        else:
            spk_path = hf_hub_download(repo_id=checkpoint_path, filename='spk_dict.pt')

        print(f"[INFO] Loading speakers from: {spk_path}")
        model.model.load_speakers(spk_path)

        # Move components to CUDA
        print("[INFO] Moving model components to CUDA...")
        model.model.thinker.model.embed_tokens = model.model.thinker.model.embed_tokens.to(device)
        model.model.thinker.visual = model.model.thinker.visual.to(device)
        model.model.thinker.audio_tower = model.model.thinker.audio_tower.to(device)
        model.model.thinker.visual.rotary_pos_emb = model.model.thinker.visual.rotary_pos_emb.to(device)
        model.model.thinker.model.rotary_emb = model.model.thinker.model.rotary_emb.to(device)

        for layer in model.model.thinker.model.layers:
            layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)

        # Load processor
        print("[INFO] Loading processor...")
        processor = Qwen2_5OmniProcessor.from_pretrained(checkpoint_path)

        # Print GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[INFO] GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        print("[INFO] Model loaded successfully!")

        # Update last activity time
        last_activity_time = time.time()

def unload_model():
    """Unload model and clear CUDA cache - INTERNAL USE ONLY (assumes lock is held)"""
    global model, processor

    if model is None:
        return

    print("[INFO] Unloading model due to idle timeout...")

    try:
        # Move model components to CPU before deletion to free GPU memory
        if torch.cuda.is_available():
            print("[INFO] Moving model components to CPU...")
            try:
                # Move main model components to CPU
                if hasattr(model, 'model') and hasattr(model.model, 'thinker'):
                    model.model.thinker = model.model.thinker.cpu()
                elif hasattr(model, 'cpu'):
                    model.cpu()
            except Exception as e:
                print(f"[WARNING] Error moving model to CPU: {e}")

        # Delete model and processor
        del model
        del processor
        model = None
        processor = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Print memory stats
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[INFO] GPU Memory after unload: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        print("[INFO] Model unloaded and CUDA cache cleared")

    except Exception as e:
        print(f"[ERROR] Error during model unload: {e}")
        import traceback
        traceback.print_exc()

def update_activity():
    """Update last activity timestamp"""
    global last_activity_time
    last_activity_time = time.time()

def idle_monitor():
    """Monitor idle time and unload model if idle for too long"""
    global model, is_processing, last_activity_time, idle_timeout

    while True:
        time.sleep(idle_check_interval)

        # Skip if currently processing
        if is_processing:
            continue

        # Skip if no activity recorded yet
        if last_activity_time is None:
            continue

        # Check idle time
        idle_time = time.time() - last_activity_time

        if idle_time >= idle_timeout:
            # Acquire lock and unload model
            with model_lock:
                # Double check conditions after acquiring lock
                if model is not None and not is_processing:
                    print(f"[INFO] Model idle for {idle_time:.0f} seconds, unloading...")
                    # Call unload_model (which doesn't acquire lock itself)
                    unload_model()
                    last_activity_time = None

# ==================== Audio Processing ====================

def sanitize_filename(filename):
    """Sanitize filename for security"""
    import re
    # Remove path separators and dangerous characters
    filename = os.path.basename(filename)
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    return filename[:255]  # Limit length

def convert_to_wav(input_path, request_id):
    """
    Convert any audio/video file to WAV format using FFmpeg

    Args:
        input_path: Path to input file (any audio/video format)
        request_id: Unique request identifier for logging

    Returns:
        Path to converted WAV file
    """
    import subprocess

    # Generate output path
    output_path = os.path.join(TEMP_DIR, f"{request_id}_converted.wav")

    print(f"[{request_id}] Converting to WAV format using FFmpeg...")
    print(f"[{request_id}] Input: {input_path}")
    print(f"[{request_id}] Output: {output_path}")

    try:
        # Check file magic bytes to detect actual format
        with open(input_path, 'rb') as f:
            header = f.read(12)

        # Detect MP4/MOV by checking for ftyp box
        is_mp4 = (len(header) >= 12 and
                  header[4:8] == b'ftyp')

        print(f"[{request_id}] File magic bytes suggest MP4: {is_mp4}")

        # FFmpeg command: convert to 16kHz mono WAV
        cmd = ['ffmpeg']

        # If it looks like MP4 but has wrong extension, specify format
        if is_mp4:
            cmd.extend(['-f', 'mov,mp4,m4a,3gp,3g2,mj2'])  # MP4 demuxer

        cmd.extend([
            '-i', input_path,
            '-vn',               # No video
            '-ar', '16000',      # Sample rate 16kHz
            '-ac', '1',          # Mono
            '-f', 'wav',         # WAV format output
            '-y',                # Overwrite output file
            output_path
        ])

        # Run FFmpeg
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300  # 5 minutes timeout
        )

        if result.returncode != 0:
            error_msg = result.stderr.decode('utf-8', errors='ignore')
            print(f"[{request_id}] FFmpeg stderr: {error_msg}")
            raise RuntimeError(f"FFmpeg conversion failed (returncode {result.returncode})")

        print(f"[{request_id}] Conversion successful: {output_path}")
        return output_path

    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg conversion timeout (>5 minutes)")
    except Exception as e:
        raise RuntimeError(f"FFmpeg conversion error: {str(e)}")

def transcribe_audio_file(audio_path, request_id, max_new_tokens=8192, temperature=0.1,
                          repetition_penalty=1.1, enable_s2t=True, use_semaphore=True):
    """
    Transcribe a single audio file using AWQ model (thread-safe)

    Args:
        audio_path: Path to audio file
        request_id: Unique request identifier
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        repetition_penalty: Repetition penalty
        enable_s2t: Enable simplified to traditional Chinese conversion (default: True)
        use_semaphore: Whether to use semaphore for concurrency control (default: True)

    Returns:
        Transcribed text
    """
    global model_inference_semaphore

    if use_semaphore and model_inference_semaphore is not None:
        with model_inference_semaphore:
            return _transcribe_audio_file_impl(
                audio_path, request_id, max_new_tokens,
                temperature, repetition_penalty, enable_s2t
            )
    else:
        return _transcribe_audio_file_impl(
            audio_path, request_id, max_new_tokens,
            temperature, repetition_penalty, enable_s2t
        )

def _transcribe_audio_file_impl(audio_path, request_id, max_new_tokens, temperature, repetition_penalty, enable_s2t):
    """Internal implementation of transcribe_audio_file"""
    global model, processor, opencc_converter, model_config

    # Auto-reload model if it was unloaded
    if model is None and model_config:
        print(f"[{request_id}] Model not loaded, reloading...")
        load_model_processor(**model_config)
    elif model is None:
        raise RuntimeError("Model not loaded and no configuration available")

    # Update activity timestamp
    update_activity()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Messages with transcription prompt
    system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
    user_prompt = "請將音訊內容精確轉錄為繁體中文文字。格式要求：1) 標點符號：每句話以句號(。)、問號(?)或驚嘆號(!)結尾,語意停頓處加入逗號(,)、頓號(、)或分號(;) 2) 分句與換行：每個完整句子獨立成行,話題轉換時另起一段(空行),引用經文或列舉項目時每項獨立成行 3) 段落結構：根據語意邏輯分段,段落間用空行分隔,相同主題保持在同一段 4) 聖經引用格式：使用《書卷名章:節》格式,例如《約翰福音3:16》神愛世人,甚至將他的獨生子賜給他們,叫一切信他的,不致滅亡,反得永生。聖經書卷包含：舊約(創世記、出埃及記、利未記、民數記、申命記、約書亞記、士師記、路得記、撒母耳記上、撒母耳記下、列王紀上、列王紀下、歷代志上、歷代志下、以斯拉記、尼希米記、以斯帖記、約伯記、詩篇、箴言、傳道書、雅歌、以賽亞書、耶利米書、耶利米哀歌、以西結書、但以理書、何西阿書、約珥書、阿摩司書、俄巴底亞書、約拿書、彌迦書、那鴻書、哈巴谷書、西番雅書、哈該書、撒迦利亞書、瑪拉基書)、新約(馬太福音、馬可福音、路加福音、約翰福音、使徒行傳、羅馬書、哥林多前書、哥林多後書、加拉太書、以弗所書、腓立比書、歌羅西書、帖撒羅尼迦前書、帖撒羅尼迦後書、提摩太前書、提摩太後書、提多書、腓利門書、希伯來書、雅各書、彼得前書、彼得後書、約翰一書、約翰二書、約翰三書、猶大書、啟示錄) 5) 直接輸出轉錄文字,不包含任何解釋、評論、標記或元資料。"

    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": system_prompt},
        ]},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": user_prompt}
        ]},
    ]

    print(f"[{request_id}] Processing audio file...")

    try:
        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Use process_mm_info like official demo
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

        # Process inputs
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True
        ).to(device)

        print(f"[{request_id}] Generating transcription...")

        # Generate
        with torch.no_grad():
            output = model.generate(
                **inputs,
                use_audio_in_video=True,
                return_audio=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
            )

        # Decode - extract text_ids from output[0]
        text_output = processor.batch_decode(
            output[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # Extract response from list
        response = text_output[0] if isinstance(text_output, list) else text_output

        # Simplified to Traditional Chinese conversion
        if enable_s2t and opencc_converter is not None:
            try:
                response = opencc_converter.convert(response)
                print(f"[{request_id}] Applied OpenCC Simplified to Traditional Chinese conversion")
            except Exception as e:
                print(f"[{request_id}] Warning: OpenCC conversion failed: {e}")

        # Clean up CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[{request_id}] Transcription complete: {len(response)} chars")
        print(f"[{request_id}] ====== TRANSCRIPTION RESULT ======")
        print(response)
        print(f"[{request_id}] ====== END OF TRANSCRIPTION ======")

        return response

    except Exception as e:
        print(f"[{request_id}] ERROR during transcription: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def process_audio_segments(audio_path, request_id, segment_duration=600, overlap_duration=10,
                          enable_parallel=True, parallel_workers=None, **kwargs):
    """
    Process long audio by splitting into overlapping segments

    Args:
        audio_path: Path to audio file
        request_id: Unique request identifier
        segment_duration: Duration of each segment in seconds
        overlap_duration: Overlap duration at segment boundaries in seconds (default: 10s)
        enable_parallel: Enable parallel processing (default: True)
        parallel_workers: Number of parallel workers (None = use global default)
        **kwargs: Additional arguments for transcription

    Returns:
        Combined transcription text
    """
    print(f"[{request_id}] Processing audio file: {audio_path}")

    # Convert to WAV if needed
    converted_path = None

    # Try to load directly first
    try:
        audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        # If direct loading fails, convert with FFmpeg
        print(f"[{request_id}] Direct loading failed, converting with FFmpeg...")
        converted_path = convert_to_wav(audio_path, request_id)
        audio_array, sr = librosa.load(converted_path, sr=16000, mono=True)

    duration = len(audio_array) / sr
    duration_mins = duration / 60
    print(f"[{request_id}] Audio duration: {duration:.2f}s ({duration_mins:.2f} mins), sr: {sr}")

    # If short enough, process directly
    if duration <= segment_duration:
        print(f"[{request_id}] Audio short enough, processing directly")
        return transcribe_audio_file(audio_path, request_id, **kwargs)

    # Split into overlapping segments
    segment_samples = int(segment_duration * sr)
    overlap_samples = int(overlap_duration * sr)
    step_samples = segment_samples - overlap_samples

    segments = []
    segment_positions = []  # Track start/end positions for overlap removal

    for i in range(0, len(audio_array), step_samples):
        start_idx = i
        end_idx = min(i + segment_samples, len(audio_array))
        segment = audio_array[start_idx:end_idx]

        # Only add if segment is long enough (at least 5 seconds)
        if len(segment) >= sr * 5:
            segments.append(segment)
            segment_positions.append({
                'start': start_idx / sr,  # Start time in seconds
                'end': end_idx / sr,      # End time in seconds
                'has_overlap_start': i > 0,  # Has overlap with previous segment
                'has_overlap_end': end_idx < len(audio_array)  # Has overlap with next segment
            })

        if end_idx >= len(audio_array):
            break

    segment_duration_mins = segment_duration / 60
    overlap_mins = overlap_duration / 60
    print(f"[{request_id}] Split into {len(segments)} overlapping segments ({segment_duration_mins:.1f} mins each, {overlap_mins:.1f} mins overlap)")

    # Determine parallel processing
    if parallel_workers is None:
        parallel_workers = max_parallel_workers

    use_parallel = enable_parallel and len(segments) > 1

    if use_parallel:
        print(f"[{request_id}] Using parallel processing with {parallel_workers} workers")
    else:
        print(f"[{request_id}] Using serial processing")

    # Process segments
    results = []
    temp_files = []

    try:
        # Save all segments to temp files first
        import soundfile as sf
        for idx, segment in enumerate(segments):
            temp_file = os.path.join(TEMP_DIR, f"{request_id}_segment_{idx}.wav")
            sf.write(temp_file, segment, sr)
            temp_files.append(temp_file)

        if use_parallel:
            # Parallel processing
            results = _process_segments_parallel(
                segments, segment_positions, temp_files,
                request_id, parallel_workers, **kwargs
            )
        else:
            # Serial processing
            results = _process_segments_serial(
                segments, segment_positions, temp_files,
                request_id, **kwargs
            )

        # Intelligent merging: remove overlapping content
        merged_result = merge_overlapping_transcriptions(results, segment_positions, request_id)

        # Log final combined result
        print(f"[{request_id}] ====== FINAL COMBINED TRANSCRIPTION ======")
        print(merged_result)
        print(f"[{request_id}] ====== END OF COMBINED TRANSCRIPTION ======")
        print(f"[{request_id}] Total length: {len(merged_result)} chars")

        return merged_result

    finally:
        # Clean up any remaining temp files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

        # Clean up converted WAV file
        if converted_path and os.path.exists(converted_path):
            try:
                os.remove(converted_path)
                print(f"[{request_id}] Cleaned up converted file: {converted_path}")
            except:
                pass

def _process_segments_parallel(segments, segment_positions, temp_files, request_id, parallel_workers, **kwargs):
    """Process audio segments in parallel using ThreadPoolExecutor"""
    results = [None] * len(segments)

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {}

        # Submit all tasks
        for idx, temp_file in enumerate(temp_files):
            segment_start_time = segment_positions[idx]['start']
            segment_start_mins = segment_start_time / 60

            future = executor.submit(
                transcribe_audio_file,
                temp_file,
                f"{request_id}_seg{idx}",
                use_semaphore=True,  # Use semaphore for GPU concurrency control
                **kwargs
            )
            futures[future] = (idx, temp_file, segment_start_mins)

        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            idx, temp_file, segment_start_mins = futures[future]
            completed += 1

            try:
                result = future.result()
                results[idx] = result
                print(f"[{request_id}] Segment {idx+1}/{len(segments)} completed "
                      f"({completed}/{len(segments)} done, {len(result)} chars, "
                      f"starts at {segment_start_mins:.1f} mins)")
            except Exception as e:
                print(f"[{request_id}] Segment {idx+1} failed: {e}")
                results[idx] = f"[Error in segment {idx}]"

            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass

    return results

def _process_segments_serial(segments, segment_positions, temp_files, request_id, **kwargs):
    """Process audio segments serially (original implementation)"""
    results = []

    for idx, temp_file in enumerate(temp_files):
        segment_start_time = segment_positions[idx]['start']
        segment_start_mins = segment_start_time / 60
        print(f"[{request_id}] Processing segment {idx+1}/{len(segments)} (starts at {segment_start_mins:.1f} mins)...")

        try:
            result = transcribe_audio_file(temp_file, f"{request_id}_seg{idx}", use_semaphore=False, **kwargs)
            results.append(result)
            print(f"[{request_id}] Segment {idx+1}/{len(segments)} completed: {len(result)} chars")
        except Exception as e:
            print(f"[{request_id}] Segment {idx} failed: {e}")
            results.append(f"[Error in segment {idx}]")

        # Clean up temp file
        try:
            os.remove(temp_file)
        except:
            pass

    return results

def merge_overlapping_transcriptions(results, segment_positions, request_id):
    """
    Intelligently merge overlapping transcription segments

    Args:
        results: List of transcription results
        segment_positions: List of segment position info
        request_id: Request identifier for logging

    Returns:
        Merged transcription text
    """
    if len(results) == 0:
        return ""

    if len(results) == 1:
        return results[0]

    print(f"[{request_id}] Merging {len(results)} overlapping segments...")

    merged_parts = []

    for idx, result in enumerate(results):
        if idx == 0:
            # First segment: keep everything
            merged_parts.append(result)
            print(f"[{request_id}] Segment 0: kept full content ({len(result)} chars)")
        else:
            # Subsequent segments: try to find overlap with previous segment
            prev_result = results[idx - 1]
            overlap_removed = remove_overlap_content(prev_result, result, request_id, idx)

            if overlap_removed:
                merged_parts.append(overlap_removed)
                print(f"[{request_id}] Segment {idx}: removed overlap, kept {len(overlap_removed)} chars (removed ~{len(result) - len(overlap_removed)} chars)")
            else:
                # If overlap detection fails, keep full content with separator
                merged_parts.append(result)
                print(f"[{request_id}] Segment {idx}: overlap detection failed, kept full content ({len(result)} chars)")

    # Join with double newline to maintain paragraph structure
    final_result = "\n\n".join(merged_parts)

    # Clean up excessive newlines
    while "\n\n\n" in final_result:
        final_result = final_result.replace("\n\n\n", "\n\n")

    return final_result.strip()

def remove_overlap_content(prev_text, current_text, request_id, segment_idx):
    """
    Remove overlapping content from current segment by comparing with previous segment

    Strategy:
    1. Split both texts into sentences
    2. Find the last few sentences of prev_text in the beginning of current_text
    3. Remove the overlapping sentences from current_text

    Args:
        prev_text: Previous segment transcription
        current_text: Current segment transcription
        request_id: Request identifier for logging
        segment_idx: Current segment index

    Returns:
        Current text with overlap removed, or None if no overlap found
    """
    # Split into sentences (consider Chinese punctuation)
    import re

    # Split by sentence-ending punctuation
    def split_sentences(text):
        # Split by 。 ! ? but keep the punctuation
        sentences = re.split('([。!?]+)', text)
        # Recombine sentences with their punctuation
        result = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                result.append(sentences[i] + sentences[i+1])
        # Add last part if exists
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1])
        return [s.strip() for s in result if s.strip()]

    prev_sentences = split_sentences(prev_text)
    current_sentences = split_sentences(current_text)

    if not prev_sentences or not current_sentences:
        return None

    # Try to find overlap: check last N sentences of prev against first M sentences of current
    max_check = min(10, len(prev_sentences), len(current_sentences))  # Check up to 10 sentences

    best_overlap_count = 0

    # Check for sentence-level matches
    for n in range(1, max_check + 1):
        # Get last n sentences from previous
        prev_tail = prev_sentences[-n:]

        # Check if they appear at the start of current
        if len(current_sentences) >= n:
            current_head = current_sentences[:n]

            # Check similarity (allowing for minor differences due to transcription variations)
            matches = 0
            for prev_sent, curr_sent in zip(prev_tail, current_head):
                # Calculate similarity (simple character overlap)
                similarity = calculate_similarity(prev_sent, curr_sent)
                if similarity > 0.7:  # 70% similarity threshold
                    matches += 1

            if matches >= n * 0.7:  # At least 70% of sentences match
                best_overlap_count = n

    if best_overlap_count > 0:
        # Remove overlapping sentences from current text
        remaining_sentences = current_sentences[best_overlap_count:]
        result = "".join(remaining_sentences)
        print(f"[{request_id}] Segment {segment_idx}: detected {best_overlap_count} overlapping sentences")
        return result

    # If no sentence-level overlap, try character-level overlap for last sentence
    # This handles cases where a sentence was cut mid-way
    prev_tail = prev_text[-200:].strip()  # Last 200 chars

    for length in range(min(100, len(prev_tail)), 10, -5):  # Try different lengths
        tail_fragment = prev_tail[-length:]
        if tail_fragment in current_text[:300]:  # Check in first 300 chars
            pos = current_text.find(tail_fragment)
            if pos != -1 and pos < 100:  # Found near the beginning
                result = current_text[pos + len(tail_fragment):].strip()
                print(f"[{request_id}] Segment {segment_idx}: detected character-level overlap ({length} chars)")
                return result

    return None

def calculate_similarity(text1, text2):
    """Calculate similarity between two texts using character overlap"""
    if not text1 or not text2:
        return 0.0

    # Remove whitespace for comparison
    t1 = text1.replace(" ", "").replace("\n", "")
    t2 = text2.replace(" ", "").replace("\n", "")

    if not t1 or not t2:
        return 0.0

    # Count matching characters
    matches = sum(c1 == c2 for c1, c2 in zip(t1, t2))
    max_len = max(len(t1), len(t2))

    return matches / max_len if max_len > 0 else 0.0

# ==================== Flask Routes ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global model

    status = {
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

    if torch.cuda.is_available():
        status["gpu_available"] = True
        status["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
        status["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
    else:
        status["gpu_available"] = False

    return jsonify(status)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Transcribe audio and return text file"""
    return _transcribe_impl(return_format='file')

@app.route('/transcribe/json', methods=['POST'])
def transcribe_json():
    """Transcribe audio and return JSON"""
    return _transcribe_impl(return_format='json')

@app.route('/transcribe/srt', methods=['POST'])
def transcribe_srt():
    """Transcribe audio and return SRT subtitle file"""
    return _transcribe_impl(return_format='srt')

def _transcribe_impl(return_format='file'):
    """
    Common transcription implementation

    Args:
        return_format: 'file', 'json', or 'srt'
    """
    global model, processor, is_processing, model_config

    # Auto-reload model if needed
    if model is None and model_config:
        print("[INFO] Model not loaded, reloading...")
        load_model_processor(**model_config)
    elif model is None:
        return jsonify({"error": "Model not loaded and no configuration available"}), 503

    # Validate request
    if 'file' not in request.files and not request.data:
        return jsonify({"error": "No audio file provided"}), 400

    # Get audio data
    if 'file' in request.files:
        file = request.files['file']
        filename = sanitize_filename(file.filename)
        audio_data = file.read()
    else:
        audio_data = request.data
        filename = "audio.wav"

    # Validate file size
    if len(audio_data) == 0:
        return jsonify({"error": "Empty audio file"}), 400

    if len(audio_data) > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({"error": "File too large"}), 413

    # Generate request ID
    request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Save uploaded file
    input_path = os.path.join(INPUTS_DIR, f"{request_id}_{filename}")
    with open(input_path, 'wb') as f:
        f.write(audio_data)

    print(f"[{request_id}] Received audio file: {filename} ({len(audio_data)} bytes)")

    try:
        # Process with lock to serialize requests
        with processing_lock:
            # Mark as processing to prevent idle unload
            is_processing = True
            update_activity()

            print(f"[{request_id}] Starting transcription...")

            # Get parameters
            segment_duration = int(request.form.get('segment_duration', 600))
            max_new_tokens = int(request.form.get('max_new_tokens', 8192))
            temperature = float(request.form.get('temperature', 0.1))
            repetition_penalty = float(request.form.get('repetition_penalty', 1.1))
            enable_s2t = request.form.get('enable_s2t', 'true').lower() == 'true'

            # Transcribe
            transcription = process_audio_segments(
                input_path,
                request_id,
                segment_duration=segment_duration,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                enable_s2t=enable_s2t
            )

            # Save output
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            output_filename = f"{timestamp}.txt"
            output_path = os.path.join(OUTPUTS_DIR, output_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcription)

            print(f"[{request_id}] Transcription saved to: {output_path}")
            print(f"[{request_id}] ====== SAVED TRANSCRIPTION ======")
            print(transcription)
            print(f"[{request_id}] ====== END OF SAVED TRANSCRIPTION ======")

            # Return based on format
            if return_format == 'json':
                return jsonify({
                    "status": "success",
                    "transcription": transcription,
                    "output_file": output_filename,
                    "timestamp": datetime.now().isoformat()
                })

            elif return_format == 'srt':
                # Generate SRT format
                srt_content = generate_srt_from_text(transcription, segment_duration)

                srt_buffer = io.BytesIO()
                srt_buffer.write(srt_content.encode('utf-8'))
                srt_buffer.seek(0)

                return send_file(
                    srt_buffer,
                    as_attachment=True,
                    download_name='audio.srt',
                    mimetype='text/plain'
                )

            else:  # return_format == 'file'
                return send_file(
                    output_path,
                    as_attachment=True,
                    download_name='audio.txt',
                    mimetype='application/octet-stream'
                )

    except Exception as e:
        print(f"[{request_id}] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        # Mark processing complete
        is_processing = False
        update_activity()

        # Clean up input file
        try:
            if os.path.exists(input_path):
                os.remove(input_path)
        except:
            pass

def generate_srt_from_text(text, segment_duration):
    """Generate SRT subtitle format from text"""
    lines = text.split('\n\n')

    srt_lines = []
    for idx, line in enumerate(lines, 1):
        if not line.strip():
            continue

        start_time = (idx - 1) * segment_duration
        end_time = idx * segment_duration

        start_h = int(start_time // 3600)
        start_m = int((start_time % 3600) // 60)
        start_s = int(start_time % 60)
        start_ms = int((start_time % 1) * 1000)

        end_h = int(end_time // 3600)
        end_m = int((end_time % 3600) // 60)
        end_s = int(end_time % 60)
        end_ms = int((end_time % 1) * 1000)

        srt_lines.append(f"{idx}")
        srt_lines.append(f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms:03d} --> {end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms:03d}")
        srt_lines.append(line.strip())
        srt_lines.append("")

    return '\n'.join(srt_lines)

@app.route('/transcribe/async', methods=['POST'])
def transcribe_async():
    """Start async transcription job"""
    global model, model_config

    # Auto-reload model if needed
    if model is None and model_config:
        print("[INFO] Model not loaded, reloading...")
        load_model_processor(**model_config)
    elif model is None:
        return jsonify({"error": "Model not loaded and no configuration available"}), 503

    # Validate request
    if 'file' not in request.files and not request.data:
        return jsonify({"error": "No audio file provided"}), 400

    # Get audio data
    if 'file' in request.files:
        file = request.files['file']
        filename = sanitize_filename(file.filename)
        audio_data = file.read()
    else:
        audio_data = request.data
        filename = "audio.wav"

    # Generate job ID
    job_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Save file
    input_path = os.path.join(INPUTS_DIR, f"{job_id}_{filename}")
    with open(input_path, 'wb') as f:
        f.write(audio_data)

    # Get parameters
    params = {
        'segment_duration': int(request.form.get('segment_duration', 600)),
        'max_new_tokens': int(request.form.get('max_new_tokens', 8192)),
        'temperature': float(request.form.get('temperature', 0.1)),
        'repetition_penalty': float(request.form.get('repetition_penalty', 1.1)),
        'enable_s2t': request.form.get('enable_s2t', 'true').lower() == 'true'
    }

    # Initialize job status
    with job_lock:
        job_status[job_id] = {
            "status": "processing",
            "created_at": datetime.now().isoformat(),
            "progress": 0
        }

    # Start background thread
    def process_job():
        global is_processing

        try:
            # Mark as processing
            is_processing = True
            update_activity()

            transcription = process_audio_segments(input_path, job_id, **params)

            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            output_filename = f"{timestamp}.txt"
            output_path = os.path.join(OUTPUTS_DIR, output_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcription)

            with job_lock:
                job_status[job_id] = {
                    "status": "completed",
                    "transcription": transcription,
                    "output_file": output_filename,
                    "completed_at": datetime.now().isoformat()
                }

        except Exception as e:
            with job_lock:
                job_status[job_id] = {
                    "status": "failed",
                    "error": str(e),
                    "failed_at": datetime.now().isoformat()
                }

        finally:
            # Mark processing complete
            is_processing = False
            update_activity()

            # Clean up input file
            try:
                if os.path.exists(input_path):
                    os.remove(input_path)
            except:
                pass

    thread = threading.Thread(target=process_job, daemon=True)
    thread.start()

    return jsonify({
        "status": "accepted",
        "job_id": job_id,
        "message": f"Use /status/{job_id} to check progress"
    }), 202

@app.route('/status/<job_id>', methods=['GET'])
def check_status(job_id):
    """Check async job status"""
    with job_lock:
        if job_id not in job_status:
            return jsonify({"error": "Job not found"}), 404

        return jsonify(job_status[job_id])

@app.route('/transcribe/file', methods=['POST'])
def transcribe_file():
    """
    Transcribe a file that already exists in /app/inputs directory

    Usage:
        POST /transcribe/file
        Content-Type: application/json
        {
            "filename": "20250912 聖經學校 16 罪與悔改 翻譯語音.mp4",
            "segment_duration": 600,
            "max_new_tokens": 8192,
            "temperature": 0.1,
            "repetition_penalty": 1.1,
            "enable_s2t": true
        }
    """
    global model, model_config, is_processing

    # Auto-reload model if needed
    if model is None and model_config:
        print("[INFO] Model not loaded, reloading...")
        load_model_processor(**model_config)
    elif model is None:
        return jsonify({"error": "Model not loaded and no configuration available"}), 503

    # Get JSON data
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400

    # Get filename
    filename = data.get('filename')
    if not filename:
        return jsonify({"error": "No filename provided"}), 400

    # Sanitize filename for security
    filename = sanitize_filename(filename)

    # Check if file exists
    input_path = os.path.join(INPUTS_DIR, filename)
    if not os.path.exists(input_path):
        return jsonify({"error": f"File not found: {filename}"}), 404

    # Generate request ID
    request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

    print(f"[{request_id}] Processing file from inputs: {filename}")

    try:
        # Process with lock to serialize requests
        with processing_lock:
            # Mark as processing to prevent idle unload
            is_processing = True
            update_activity()

            print(f"[{request_id}] Starting transcription...")

            # Get parameters from JSON
            segment_duration = int(data.get('segment_duration', 600))
            max_new_tokens = int(data.get('max_new_tokens', 8192))
            temperature = float(data.get('temperature', 0.1))
            repetition_penalty = float(data.get('repetition_penalty', 1.1))
            enable_s2t = data.get('enable_s2t', True)

            # Transcribe
            transcription = process_audio_segments(
                input_path,
                request_id,
                segment_duration=segment_duration,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                enable_s2t=enable_s2t
            )

            # Save output
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            output_filename = f"{timestamp}.txt"
            output_path = os.path.join(OUTPUTS_DIR, output_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcription)

            print(f"[{request_id}] Transcription saved to: {output_path}")
            print(f"[{request_id}] ====== SAVED TRANSCRIPTION ======")
            print(transcription)
            print(f"[{request_id}] ====== END OF SAVED TRANSCRIPTION ======")

            return jsonify({
                "status": "success",
                "transcription": transcription,
                "output_file": output_filename,
                "timestamp": datetime.now().isoformat()
            })

    except Exception as e:
        print(f"[{request_id}] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        # Mark processing complete
        is_processing = False
        update_activity()

# ==================== Main ====================

def main():
    parser = ArgumentParser(description="Qwen2.5-Omni AWQ Audio Transcription Service")

    # Model arguments
    parser.add_argument("--checkpoint-path", type=str, default="Qwen/Qwen2.5-Omni-7B-AWQ",
                        help="Model checkpoint path or HuggingFace repo")
    parser.add_argument("--local-model", action="store_true",
                        help="Use local model instead of downloading from HF")
    parser.add_argument("--flash-attn2", action="store_true",
                        help="Enable Flash Attention 2")

    # Processing arguments
    parser.add_argument("--segment-duration", type=int, default=600,
                        help="Audio segment duration in seconds (default: 600 = 10 minutes)")
    parser.add_argument("--max-new-tokens", type=int, default=8192,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="Repetition penalty")

    # Parallel processing arguments
    parser.add_argument("--max-workers", type=int, default=3,
                        help="Maximum parallel workers for segment processing (default: 3)")
    parser.add_argument("--disable-parallel", action="store_true",
                        help="Disable parallel processing")

    # Server arguments
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to bind to")
    parser.add_argument("--idle-timeout", type=int, default=300,
                        help="Idle timeout in seconds before unloading model")

    args = parser.parse_args()

    # Store args globally for routes to access
    app.config['ARGS'] = args

    # Load model
    print("=" * 60)
    print("Qwen2.5-Omni AWQ Audio Transcription Service")
    print("=" * 60)

    try:
        load_model_processor(
            args.checkpoint_path,
            flash_attn2=args.flash_attn2,
            local_model=args.local_model
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Initialize parallel processing
    if not args.disable_parallel:
        init_parallel_processing(max_workers=args.max_workers)
    else:
        print("[INFO] Parallel processing disabled")

    # Start idle monitor thread
    monitor_thread = threading.Thread(target=idle_monitor, daemon=True)
    monitor_thread.start()
    print(f"[INFO] Idle monitor started (timeout: {idle_timeout}s)")

    # Start server
    print(f"\n[INFO] Starting server on {args.host}:{args.port}")
    print(f"[INFO] Endpoints:")
    print(f"  POST /transcribe         - Transcribe audio (returns text file)")
    print(f"  POST /transcribe/json    - Transcribe audio (returns JSON)")
    print(f"  POST /transcribe/srt     - Transcribe audio (returns SRT)")
    print(f"  POST /transcribe/async   - Start async transcription")
    print(f"  POST /transcribe/file    - Transcribe file from /app/inputs directory")
    print(f"  GET  /status/<job_id>    - Check async job status")
    print(f"  GET  /health             - Health check")
    print("=" * 60)

    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        threaded=True
    )

if __name__ == "__main__":
    main()