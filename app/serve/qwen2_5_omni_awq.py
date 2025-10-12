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

        # Initialize OpenCC converter for Simplified to Traditional Chinese
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
    """Sanitize filename for security - preserves Unicode characters"""
    import re
    # Remove path separators and dangerous characters, but keep Unicode chars
    filename = os.path.basename(filename)
    # Remove only dangerous characters: / \ : * ? " < > |
    filename = re.sub(r'[/\\:*?"<>|]', '', filename)
    # Replace multiple spaces with single space
    filename = re.sub(r'\s+', ' ', filename)
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
        # Use simple, robust settings that work for all formats
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-vn',               # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
            '-ar', '16000',      # Sample rate 16kHz
            '-ac', '1',          # Mono
            '-y',                # Overwrite output file
            output_path
        ]

        print(f"[{request_id}] FFmpeg command: {' '.join(cmd)}")

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
            raise RuntimeError(f"FFmpeg conversion failed (returncode {result.returncode}): {error_msg}")

        # Verify output file was created and has content
        if not os.path.exists(output_path):
            raise RuntimeError("FFmpeg did not create output file")

        output_size = os.path.getsize(output_path)
        if output_size == 0:
            raise RuntimeError("FFmpeg created empty output file")

        print(f"[{request_id}] Conversion successful: {output_path} ({output_size} bytes)")
        return output_path

    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg conversion timeout (>5 minutes)")
    except Exception as e:
        raise RuntimeError(f"FFmpeg conversion error: {str(e)}")

def transcribe_audio_file(audio_path, request_id, max_new_tokens=8192, temperature=0.1,
                          repetition_penalty=1.1, enable_s2t=True):
    """
    Transcribe a single audio file using AWQ model

    Args:
        audio_path: Path to audio file
        request_id: Unique request identifier
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        repetition_penalty: Repetition penalty
        enable_s2t: Enable simplified to traditional Chinese conversion (default: True)

    Returns:
        Transcribed text
    """
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
    user_prompt = "請將音訊內容精確轉錄為中文文字。格式要求：1) 標點符號：每句話以句號(。)、問號(?)或驚嘆號(!)結尾,語意停頓處加入逗號(,)、頓號(、)或分號(;) 2) 聖經引用格式：使用《書卷名章:節》格式,例如《約翰福音3:16》神愛世人,甚至將他的獨生子賜給他們,叫一切信他的,不致滅亡,反得永生。聖經書卷包含：舊約(創世記、出埃及記、利未記、民數記、申命記、約書亞記、士師記、路得記、撒母耳記上、撒母耳記下、列王紀上、列王紀下、歷代志上、歷代志下、以斯拉記、尼希米記、以斯帖記、約伯記、詩篇、箴言、傳道書、雅歌、以賽亞書、耶利米書、耶利米哀歌、以西結書、但以理書、何西阿書、約珥書、阿摩司書、俄巴底亞書、約拿書、彌迦書、那鴻書、哈巴谷書、西番雅書、哈該書、撒迦利亞書、瑪拉基書)、新約(馬太福音、馬可福音、路加福音、約翰福音、使徒行傳、羅馬書、哥林多前書、哥林多後書、加拉太書、以弗所書、腓立比書、歌羅西書、帖撒羅尼迦前書、帖撒羅尼迦後書、提摩太前書、提摩太後書、提多書、腓利門書、希伯來書、雅各書、彼得前書、彼得後書、約翰一書、約翰二書、約翰三書、猶大書、啟示錄) 3) 直接輸出轉錄文字,不包含任何解釋、評論、標記或元資料。"

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
        # Handle different output formats - the output[0] might be a 2D tensor or tuple
        generated_ids = output[0]

        # If it's a tuple, extract the first element
        if isinstance(generated_ids, tuple):
            generated_ids = generated_ids[0]

        if isinstance(generated_ids, torch.Tensor):
            if generated_ids.dim() > 1 and generated_ids.size(0) == 1:
                # Single batch: extract first element [1, seq_len] -> [seq_len]
                generated_ids = generated_ids[0]

        text_output = processor.batch_decode(
            [generated_ids] if isinstance(generated_ids, torch.Tensor) and generated_ids.dim() == 1 else generated_ids,
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

def process_audio_segments(audio_path, request_id, segment_duration=600, **kwargs):
    """
    Process long audio by splitting into segments

    Args:
        audio_path: Path to audio file
        request_id: Unique request identifier
        segment_duration: Duration of each segment in seconds
        **kwargs: Additional arguments for transcription

    Returns:
        Combined transcription text
    """
    print(f"[{request_id}] Processing audio file: {audio_path}")

    # Create output file for accumulating results
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_filename = f"{request_id}_{timestamp}.txt"
    output_path = os.path.join(OUTPUTS_DIR, output_filename)
    print(f"[{request_id}] Output file: {output_path}")

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

    # Split into non-overlapping segments
    segment_samples = int(segment_duration * sr)
    segments = []

    for i in range(0, len(audio_array), segment_samples):
        start_idx = i
        end_idx = min(i + segment_samples, len(audio_array))
        segment = audio_array[start_idx:end_idx]

        # Only add if segment is long enough (at least 5 seconds)
        if len(segment) >= sr * 5:
            segments.append(segment)

        if end_idx >= len(audio_array):
            break

    segment_duration_mins = segment_duration / 60
    print(f"[{request_id}] Split into {len(segments)} segments ({segment_duration_mins:.1f} mins each)")

    # Process each segment
    results = []
    temp_files = []

    try:
        for idx, segment in enumerate(segments):
            # Save segment to temp file
            temp_file = os.path.join(TEMP_DIR, f"{request_id}_segment_{idx}.wav")
            import soundfile as sf
            sf.write(temp_file, segment, sr)
            temp_files.append(temp_file)

            segment_start_time = idx * segment_duration
            segment_start_mins = segment_start_time / 60
            print(f"[{request_id}] Processing segment {idx+1}/{len(segments)} (starts at {segment_start_mins:.1f} mins)...")

            try:
                result = transcribe_audio_file(temp_file, f"{request_id}_seg{idx}", **kwargs)
                results.append(result)

                # Output segment result to log immediately
                print(f"[{request_id}] ====== SEGMENT {idx+1}/{len(segments)} RESULT (starts at {segment_start_mins:.1f} mins) ======")
                print(result)
                print(f"[{request_id}] ====== END OF SEGMENT {idx+1} ({len(result)} chars) ======")
                print()  # Empty line for readability

                # Append segment result to output file immediately
                with open(output_path, 'a', encoding='utf-8') as f:
                    if idx > 0:
                        f.write("\n\n")  # Add separator between segments
                    f.write(result)
                print(f"[{request_id}] Segment {idx+1} appended to: {output_path}")

            except Exception as e:
                print(f"[{request_id}] Segment {idx} failed: {e}")
                results.append(f"[Error in segment {idx}]")
                # Append error to output file
                with open(output_path, 'a', encoding='utf-8') as f:
                    if idx > 0:
                        f.write("\n\n")
                    f.write(f"[Error in segment {idx}]")

            # Clean up segment file
            try:
                os.remove(temp_file)
            except:
                pass

        # Simple merging: join all segments
        merged_result = "\n\n".join(results)

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
            segment_duration = int(request.form.get('segment_duration', 300))
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
        'segment_duration': int(request.form.get('segment_duration', 300)),
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
            segment_duration = int(data.get('segment_duration', 300))
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
    parser.add_argument("--segment-duration", type=int, default=300,
                        help="Audio segment duration in seconds (default: 300 = 5 minutes)")
    parser.add_argument("--max-new-tokens", type=int, default=8192,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="Repetition penalty")

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