#!/usr/bin/env python3
"""
Qwen2.5-Omni Enhanced Version with REST API, Advanced Audio Processing, and Memory Management
Combines the Gradio UI interface with AWQ version features
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

# Original imports
import ffmpeg
import gradio as gr
import soundfile as sf
import modelscope_studio.components.base as ms
import modelscope_studio.components.antd as antd
import gradio.processing_utils as processing_utils
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from gradio_client import utils as client_utils
from qwen_omni_utils import process_mm_info

# ==================== Flask API Configuration ====================

# Create Flask app for REST API
flask_app = Flask(__name__)
flask_app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024  # 4GB

# ==================== Global Configuration ====================

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

# ==================== Model Loading with Memory Optimization ====================

def _load_model_processor(args):
    global model, processor, opencc_converter, last_activity_time, model_config
    
    with model_lock:
        if model is not None:
            print("Model already loaded")
            last_activity_time = time.time()
            return model, processor
        
        # Initialize OpenCC converter for Simplified to Traditional Chinese
        if opencc_converter is None:
            try:
                opencc_converter = OpenCC('s2tw')
                print("OpenCC Simplified to Traditional Chinese (Taiwan) converter initialized")
            except Exception as e:
                print(f"Warning: Failed to initialize OpenCC: {e}")
                opencc_converter = None

        # Store model configuration for auto-reload
        model_config = {
            'checkpoint_path': args.checkpoint_path,
            'flash_attn2': args.flash_attn2,
            'cpu_only': args.cpu_only
        }

        if args.cpu_only:
            device_map = 'cpu'
        else:
            device_map = 'cuda'

        # Check if flash-attn2 flag is enabled and load model accordingly
        if args.flash_attn2:
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                args.checkpoint_path,
                torch_dtype='auto',
                attn_implementation='flash_attention_2',
                device_map=device_map
            )
        else:
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                args.checkpoint_path, 
                device_map=device_map, 
                torch_dtype='auto'
            )

        processor = Qwen2_5OmniProcessor.from_pretrained(args.checkpoint_path)
        
        # In audio-only mode, we might want to optimize memory usage
        if args.audio_only:
            print("Running in audio-only mode - model loaded for audio processing only")
        
        # Print GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        print("Model loaded successfully!")
        last_activity_time = time.time()
        return model, processor

def unload_model():
    """Unload model and clear CUDA cache - INTERNAL USE ONLY (assumes lock is held)"""
    global model, processor

    if model is None:
        return

    print("Unloading model due to idle timeout...")

    try:
        # Move model components to CPU before deletion to free GPU memory
        if torch.cuda.is_available():
            print("Moving model components to CPU...")
            try:
                model.cpu()
            except Exception as e:
                print(f"Warning: Error moving model to CPU: {e}")

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
            print(f"GPU Memory after unload: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        print("Model unloaded and CUDA cache cleared")

    except Exception as e:
        print(f"Error during model unload: {e}")
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
                    print(f"Model idle for {idle_time:.0f} seconds, unloading...")
                    # Call unload_model (which doesn't acquire lock itself)
                    unload_model()
                    last_activity_time = None

# ==================== Audio Processing Functions ====================

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
                          repetition_penalty=1.1, enable_s2t=True, custom_user_prompt=None):
    """
    Transcribe a single audio file using the model

    Args:
        audio_path: Path to audio file
        request_id: Unique request identifier
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        repetition_penalty: Repetition penalty
        enable_s2t: Enable simplified to traditional Chinese conversion (default: True)
        custom_user_prompt: Custom user prompt for transcription (default: None, uses default prompt)

    Returns:
        Transcribed text
    """
    global model, processor, opencc_converter, model_config

    # Auto-reload model if it was unloaded
    if model is None and model_config:
        print(f"[{request_id}] Model not loaded, reloading...")
        args = type('Args', (), model_config)()  # Mock args object
        args.cpu_only = model_config.get('cpu_only', False)
        load_model_processor_with_args(args)
    elif model is None:
        raise RuntimeError("Model not loaded and no configuration available")

    # Update activity timestamp
    update_activity()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Messages with transcription prompt
    system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
    
    # Use custom user prompt if provided, otherwise use default
    user_prompt = custom_user_prompt or DEFAULT_USER_PROMPT

    print(f"[{request_id}] Processing audio file...")

    try:
        # Load audio using librosa
        audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Create messages with audio array
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": system_prompt},
            ]},
            {"role": "user", "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": user_prompt}
            ]},
        ]

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

        # Extract only the actual transcription content, removing system/user/assistant tags
        # The transcription should be the part after the last "assistant\n" tag
        if "assistant\n" in response:
            response = response.split("assistant\n")[-1].strip()
        elif "assistant\n\n" in response:
            response = response.split("assistant\n\n")[-1].strip()

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

def load_model_processor_with_args(args):
    """Wrapper to load model with args object (for compatibility with idle monitoring)"""
    global model, processor
    _load_model_processor(args)

# ==================== Flask API Routes ====================

@flask_app.route('/health', methods=['GET'])
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

@flask_app.route('/transcribe', methods=['POST'])
def transcribe():
    """Transcribe audio and return text file"""
    return _transcribe_impl(return_format='file')

@flask_app.route('/transcribe/json', methods=['POST'])
def transcribe_json():
    """Transcribe audio and return JSON"""
    return _transcribe_impl(return_format='json')

def _transcribe_impl(return_format='file'):
    """
    Common transcription implementation

    Args:
        return_format: 'file', 'json', or 'srt'
    """
    global model, processor, is_processing, model_config

    # Auto-reload model if needed
    if model is None and model_config:
        print("Model not loaded, reloading...")
        args = type('Args', (), model_config)()  # Mock args object
        args.cpu_only = model_config.get('cpu_only', False)
        load_model_processor_with_args(args)
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

    if len(audio_data) > flask_app.config['MAX_CONTENT_LENGTH']:
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

# ==================== Default Prompts ====================

# Default transcription prompt
DEFAULT_USER_PROMPT = "你是一個音訊轉錄助手，請完全按照原始音訊內容轉錄，不要說任何與音訊無關的話。格式要求：1) 直接輸出轉錄文字,不包含任何解釋、評論、標記或元資料。 2) 標點符號：每句話以句號(。)、問號(?)或驚嘆號(!)結尾,語意停頓處加入逗號(,)、頓號(、)或分號(;) 3) 聖經引用格式：使用《書卷名章:節》格式,例如《約翰福音3:16》神愛世人,甚至將他的獨生子賜給他們,叫一切信他的,不致滅亡,反得永生。聖經書卷包含：舊約(創世記、出埃及記、利未記、民數記、申命記、約書亞記、士師記、路得記、撒母耳記上、撒母耳記下、列王紀上、列王紀下、歷代志上、歷代志下、以斯拉記、尼希米記、以斯帖記、約伯記、詩篇、箴言、傳道書、雅歌、以賽亞書、耶利米書、耶利米哀歌、以西結書、但以理書、何西阿書、約珥書、阿摩司書、俄巴底亞書、約拿書、彌迦書、那鴻書、哈巴谷書、西番雅書、哈該書、撒迦利亞書、瑪拉基書)、新約(馬太福音、馬可福音、路加福音、約翰福音、使徒行傳、羅馬書、哥林多前書、哥林多後書、加拉太書、以弗所書、腓立比書、歌羅西書、帖撒羅尼迦前書、帖撒羅尼迦後書、提摩太前書、提摩太後書、提多書、腓利門書、希伯來書、雅各書、彼得前書、彼得後書、約翰一書、約翰二書、約翰三書、猶大書、啟示錄)。"

# ==================== Original Gradio UI Functions ====================

def _launch_demo(args, model, processor):
    # In audio-only mode, only launch the Flask API
    if args.audio_only:
        print("Running in audio-only mode - Gradio UI disabled")
        return

    # Voice settings
    VOICE_LIST = ['Chelsie', 'Ethan']
    DEFAULT_VOICE = 'Chelsie'

    default_system_prompt = 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'

    language = args.ui_language

    def get_text(text: str, cn_text: str):
        if language == 'en':
            return text
        if language == 'zh':
            return cn_text
        return text
   
    def convert_webm_to_mp4(input_file, output_file):
        try:
            (
                ffmpeg
                .input(input_file)
                .output(output_file, acodec='aac', ar='16000', audio_bitrate='192k')
                .run(quiet=True, overwrite_output=True)
            )
            print(f"Conversion successful: {output_file}")
        except ffmpeg.Error as e:
            print("An error occurred during conversion.")
            print(e.stderr.decode('utf-8'))

    def format_history(history: list, system_prompt: str):
        messages = []
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        for item in history:
            if isinstance(item["content"], str):
                messages.append({"role": item['role'], "content": item['content']})
            elif item["role"] == "user" and (isinstance(item["content"], list) or
                                            isinstance(item["content"], tuple)):
                file_path = item["content"][0]

                mime_type = client_utils.get_mimetype(file_path)
                if mime_type.startswith("image"):
                    messages.append({
                        "role":
                        item['role'],
                        "content": [{
                            "type": "image",
                            "image": file_path
                        }]
                    })
                elif mime_type.startswith("video"):
                    messages.append({
                        "role":
                        item['role'],
                        "content": [{
                            "type": "video",
                            "video": file_path
                        }]
                    })
                elif mime_type.startswith("audio"):
                    messages.append({
                        "role":
                        item['role'],
                        "content": [{
                            "type": "audio",
                            "audio": file_path,
                        }]
                    })
        return messages

    def predict(messages, voice=DEFAULT_VOICE):
        print('predict history: ', messages)    

        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
        inputs = inputs.to(model.device).to(model.dtype)

        text_ids, audio = model.generate(**inputs, speaker=voice, use_audio_in_video=True)

        response = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = response[0]
        
        # Extract only the actual response content, removing system/user/assistant tags
        # The response should be the part after the last "assistant\n" tag
        if "assistant\n" in response:
            response = response.split("assistant\n")[-1].strip()
        elif "assistant\n\n" in response:
            response = response.split("assistant\n\n")[-1].strip()
        else:
            # Fallback: just take the last line as before
            response = response.split("\n")[-1]
        
        yield {"type": "text", "data": response}

        audio = np.array(audio * 32767).astype(np.int16)
        wav_io = io.BytesIO()
        sf.write(wav_io, audio, samplerate=24000, format="WAV")
        wav_io.seek(0)
        wav_bytes = wav_io.getvalue()
        audio_path = processing_utils.save_bytes_to_cache(
            wav_bytes, "audio.wav", cache_dir=demo.GRADIO_CACHE)
        yield {"type": "audio", "data": audio_path}

    def media_predict(audio, video, history, system_prompt, voice_choice):
        # First yield
        yield (
            None,  # microphone
            None,  # webcam
            history,  # media_chatbot
            gr.update(visible=False),  # submit_btn
            gr.update(visible=True),  # stop_btn
        )

        if video is not None:
            convert_webm_to_mp4(video, video.replace('.webm', '.mp4'))
            video = video.replace(".webm", ".mp4")
        files = [audio, video]

        for f in files:
            if f:
                history.append({"role": "user", "content": (f, )})

        formatted_history = format_history(history=history,
                                        system_prompt=system_prompt,)


        history.append({"role": "assistant", "content": ""})

        for chunk in predict(formatted_history, voice_choice):
            if chunk["type"] == "text":
                history[-1]["content"] = chunk["data"]
                yield (
                    None,  # microphone
                    None,  # webcam
                    history,  # media_chatbot
                    gr.update(visible=False),  # submit_btn
                    gr.update(visible=True),  # stop_btn
                )
            if chunk["type"] == "audio":
                history.append({
                    "role": "assistant",
                    "content": gr.Audio(chunk["data"])
                })

        # Final yield
        yield (
            None,  # microphone
            None,  # webcam
            history,  # media_chatbot
            gr.update(visible=True),  # submit_btn
            gr.update(visible=False),  # stop_btn
        )

    def chat_predict(text, audio, image, video, history, system_prompt, voice_choice):
        # Process text input
        if text:
            history.append({"role": "user", "content": text})

        # Process audio input
        if audio:
            history.append({"role": "user", "content": (audio, )})

        # Process image input
        if image:
            history.append({"role": "user", "content": (image, )})

        # Process video input
        if video:
            history.append({"role": "user", "content": (video, )})

        formatted_history = format_history(history=history,
                                        system_prompt=system_prompt)

        yield None, None, None, None, history

        history.append({"role": "assistant", "content": ""})
        for chunk in predict(formatted_history, voice_choice):
            if chunk["type"] == "text":
                history[-1]["content"] = chunk["data"]
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip(
                ), history
            if chunk["type"] == "audio":
                history.append({
                    "role": "assistant",
                    "content": gr.Audio(chunk["data"])
                })
        yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), history

    with gr.Blocks() as demo, ms.Application(), antd.ConfigProvider():
        with gr.Sidebar(open=False):
            system_prompt_textbox = gr.Textbox(label="System Prompt",
                                            value=default_system_prompt)
        with antd.Flex(gap="small", justify="center", align="center"):
            with antd.Flex(vertical=True, gap="small", align="center"):
                antd.Typography.Title("Qwen2.5-Omni Demo",
                                    level=1,
                                    elem_style=dict(margin=0, fontSize=28))
                with antd.Flex(vertical=True, gap="small"):
                    antd.Typography.Text(get_text("🎯 Instructions for use:",
                                                "🎯 使用说明："),
                                        strong=True)
                    antd.Typography.Text(
                        get_text(
                            "1️⃣ Click the Audio Record button or the Camera Record button.",
                            "1️⃣ 点击音频录制按钮，或摄像头-录制按钮"))
                    antd.Typography.Text(
                        get_text("2️⃣ Input audio or video.", "2️⃣ 输入音频或者视频"))
                    antd.Typography.Text(
                        get_text(
                            "3️⃣ Click the submit button and wait for the model's response.",
                            "3️⃣ 点击提交并等待模型的回答"))
        voice_choice = gr.Dropdown(label="Voice Choice",
                                choices=VOICE_LIST,
                                value=DEFAULT_VOICE)
        with gr.Tabs():
            with gr.Tab("Online"):
                with gr.Row():
                    with gr.Column(scale=1):
                        microphone = gr.Audio(sources=['microphone'],
                                            type="filepath")
                        webcam = gr.Video(sources=['webcam'],
                                        height=400,
                                        include_audio=True)
                        submit_btn = gr.Button(get_text("Submit", "提交"),
                                            variant="primary")
                        stop_btn = gr.Button(get_text("Stop", "停止"), visible=False)
                        clear_btn = gr.Button(get_text("Clear History", "清除历史"))
                    with gr.Column(scale=2):
                        media_chatbot = gr.Chatbot(height=650, type="messages")

                    def clear_history():
                        return [], gr.update(value=None), gr.update(value=None)

                    submit_event = submit_btn.click(fn=media_predict,
                                                    inputs=[
                                                        microphone, webcam,
                                                        media_chatbot,
                                                        system_prompt_textbox,
                                                        voice_choice
                                                    ],
                                                    outputs=[
                                                        microphone, webcam,
                                                        media_chatbot, submit_btn,
                                                        stop_btn
                                                    ])
                    stop_btn.click(
                        fn=lambda:
                        (gr.update(visible=True), gr.update(visible=False)),
                        inputs=None,
                        outputs=[submit_btn, stop_btn],
                        cancels=[submit_event],
                        queue=False)
                    clear_btn.click(fn=clear_history,
                                    inputs=None,
                                    outputs=[media_chatbot, microphone, webcam])

            with gr.Tab("Offline"):
                chatbot = gr.Chatbot(type="messages", height=650)

                # Media upload section in one row
                with gr.Row(equal_height=True):
                    audio_input = gr.Audio(sources=["upload"],
                                        type="filepath",
                                        label="Upload Audio",
                                        elem_classes="media-upload",
                                        scale=1)
                    image_input = gr.Image(sources=["upload"],
                                        type="filepath",
                                        label="Upload Image",
                                        elem_classes="media-upload",
                                        scale=1)
                    video_input = gr.Video(sources=["upload"],
                                        label="Upload Video",
                                        elem_classes="media-upload",
                                        scale=1)

                # Text input section
                text_input = gr.Textbox(show_label=False,
                                        placeholder="Enter text here...")

                # Control buttons
                with gr.Row():
                    submit_btn = gr.Button(get_text("Submit", "提交"),
                                        variant="primary",
                                        size="lg")
                    stop_btn = gr.Button(get_text("Stop", "停止"),
                                        visible=False,
                                        size="lg")
                    clear_btn = gr.Button(get_text("Clear History", "清除历史"),
                                        size="lg")

                def clear_chat_history():
                    return [], gr.update(value=None), gr.update(
                        value=None), gr.update(value=None), gr.update(value=None)

                submit_event = gr.on(
                    triggers=[submit_btn.click, text_input.submit],
                    fn=chat_predict,
                    inputs=[
                        text_input, audio_input, image_input, video_input, chatbot,
                        system_prompt_textbox, voice_choice
                    ],
                    outputs=[
                        text_input, audio_input, image_input, video_input, chatbot
                    ])

                stop_btn.click(fn=lambda:
                            (gr.update(visible=True), gr.update(visible=False)),
                            inputs=None,
                            outputs=[submit_btn, stop_btn],
                            cancels=[submit_event],
                            queue=False)

                clear_btn.click(fn=clear_chat_history,
                                inputs=None,
                                outputs=[
                                    chatbot, text_input, audio_input, image_input,
                                    video_input
                                ])

                # Add some custom CSS to improve the layout
                gr.HTML("""
                    <style>
                        .media-upload {
                            margin: 10px;
                            min-height: 160px;
                        }
                        .media-upload > .wrap {
                            border: 2px dashed #ccc;
                            border-radius: 8px;
                            padding: 10px;
                            height: 100%;
                        }
                        .media-upload:hover > .wrap {
                            border-color: #666;
                        }
                        /* Make upload areas equal width */
                        .media-upload {
                            flex: 1;
                            min-width: 0;
                        }
                    </style>
                """)

    # Also run Flask API in a background thread
    import threading
    def run_flask():
        flask_host = getattr(args, 'host', args.flask_host)
        flask_port = getattr(args, 'port', args.flask_port)
        print(f"Starting Flask API on {flask_host}:{flask_port}")
        flask_app.run(host=flask_host, port=flask_port, debug=False, threaded=True, use_reloader=False)
    
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    demo.queue(default_concurrency_limit=100, max_size=100).launch(max_threads=100,
                                                                ssr_mode=False,
                                                                share=args.share,
                                                                inbrowser=args.inbrowser,
                                                                server_port=args.server_port,
                                                                server_name=args.server_name,)


DEFAULT_CKPT_PATH = "Qwen/Qwen2.5-Omni-7B"
def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-c',
                        '--checkpoint-path',
                        type=str,
                        default=DEFAULT_CKPT_PATH,
                        help='Checkpoint name or path, default to %(default)r')
    parser.add_argument('--cpu-only', action='store_true', help='Run demo with CPU only')

    parser.add_argument('--flash-attn2',
                        action='store_true',
                        default=False,
                        help='Enable flash_attention_2 when loading the model.')
    parser.add_argument('--share',
                        action='store_true',
                        default=False,
                        help='Create a publicly shareable link for the interface.')
    parser.add_argument('--inbrowser',
                        action='store_true',
                        default=False,
                        help='Automatically launch the interface in a new tab on the default browser.')
    parser.add_argument('--server-port', type=int, default=7860, help='Demo server port.')
    parser.add_argument('--server-name', type=str, default='127.0.0.1', help='Demo server name.')
    parser.add_argument('--ui-language', type=str, choices=['en', 'zh'], default='en', help='Display language for the UI.')
    
    # Add Flask API arguments
    parser.add_argument('--flask-host', type=str, default='127.0.0.1', help='Flask API host')
    parser.add_argument('--flask-port', type=int, default=5000, help='Flask API port')
    parser.add_argument('--idle-timeout', type=int, default=300, help='Idle timeout in seconds before unloading model')
    
    # Additional parameters for compatibility with docker-compose
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind to (for compatibility)')
    parser.add_argument('--port', type=int, default=7860, help='Port to bind to (for compatibility)')
    parser.add_argument('--audio-only', action='store_true', help='Audio only mode (for compatibility)')
    parser.add_argument('--segment-duration', type=int, default=300, help='Audio segment duration in seconds (for compatibility)')
    parser.add_argument('--max-new-tokens', type=int, default=8192, help='Maximum new tokens to generate (for compatibility)')
    parser.add_argument('--temperature', type=float, default=0.1, help='Sampling temperature (for compatibility)')
    parser.add_argument('--repetition-penalty', type=float, default=1.1, help='Repetition penalty (for compatibility)')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _get_args()
    
    # Initialize global config
    model_config = {
        'checkpoint_path': args.checkpoint_path,
        'flash_attn2': args.flash_attn2,
        'cpu_only': args.cpu_only
    }
    
    # Start idle monitor thread
    monitor_thread = threading.Thread(target=idle_monitor, daemon=True)
    monitor_thread.start()
    print(f"[INFO] Idle monitor started (timeout: {args.idle_timeout}s)")
    
    print(f"[INFO] Starting Flask API on {args.flask_host}:{args.flask_port}")
    print(f"[INFO] Available endpoints:")
    print(f"  GET  /health             - Health check")
    print(f"  POST /transcribe         - Transcribe audio (returns text file)")
    print(f"  POST /transcribe/json    - Transcribe audio (returns JSON)")
    
    model, processor = _load_model_processor(args)
    
    # Run Flask API in a background thread regardless of mode
    import threading
    def run_flask():
        flask_host = getattr(args, 'host', args.flask_host)
        flask_port = getattr(args, 'port', args.flask_port)
        print(f"[INFO] Starting Flask API on {flask_host}:{flask_port}")
        flask_app.run(host=flask_host, port=flask_port, debug=False, threaded=True, use_reloader=False)
    
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("[INFO] Flask API thread started")
    
    # In audio-only mode, only start the Flask API without Gradio UI
    if args.audio_only:
        print("[INFO] Running in audio-only mode - launching Flask API only")
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
    else:
        # Normal mode: launch both Gradio UI and Flask API
        _launch_demo(args, model, processor)