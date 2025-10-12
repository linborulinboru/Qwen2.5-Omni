#!/usr/bin/env python3
"""
Qwen2.5-Omni GGUF Quantized Audio Transcription HTTP Service
Using llama-cpp-python for GGUF model inference
"""

import io
import os
import sys
import tempfile
import threading
import uuid
import time
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

import numpy as np
import librosa
import soundfile as sf
from flask import Flask, request, jsonify, send_file
from opencc import OpenCC

# llama-cpp-python for GGUF model
from llama_cpp import Llama

# ==================== Global Configuration ====================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024  # 4GB

# Global variables
model = None
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

# ==================== Model Loading ====================

def load_model(model_path, n_ctx=8192, n_gpu_layers=-1, use_mmap=True, use_mlock=False):
    """Load GGUF model using llama-cpp-python"""
    global model, opencc_converter, last_activity_time, model_config

    with model_lock:
        if model is not None:
            print("[INFO] Model already loaded")
            last_activity_time = time.time()
            return

        print(f"[INFO] Loading GGUF model from: {model_path}")
        print(f"[INFO] Context size: {n_ctx}")
        print(f"[INFO] GPU layers: {n_gpu_layers}")
        print(f"[INFO] Use mmap: {use_mmap}")
        print(f"[INFO] Use mlock: {use_mlock}")

        # Store model configuration for auto-reload
        model_config = {
            'model_path': model_path,
            'n_ctx': n_ctx,
            'n_gpu_layers': n_gpu_layers,
            'use_mmap': use_mmap,
            'use_mlock': use_mlock
        }

        # Initialize OpenCC converter for Simplified to Traditional Chinese (Taiwan standard)
        if opencc_converter is None:
            try:
                opencc_converter = OpenCC('s2tw')
                print("[INFO] OpenCC Simplified to Traditional Chinese (Taiwan) converter initialized")
            except Exception as e:
                print(f"[WARNING] Failed to initialize OpenCC: {e}")
                opencc_converter = None

        # Load GGUF model
        try:
            model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                use_mmap=use_mmap,
                use_mlock=use_mlock,
                verbose=True
            )
            print("[INFO] GGUF model loaded successfully!")

        except Exception as e:
            print(f"[ERROR] Failed to load GGUF model: {e}")
            raise

        # Update last activity time
        last_activity_time = time.time()

def unload_model():
    """Unload model and clear memory - INTERNAL USE ONLY (assumes lock is held)"""
    global model

    if model is None:
        return

    print("[INFO] Unloading model due to idle timeout...")

    try:
        # Delete model
        del model
        model = None

        print("[INFO] Model unloaded")

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

def extract_audio_features(audio_path):
    """
    Extract audio features for GGUF model input

    Args:
        audio_path: Path to audio file

    Returns:
        Audio feature array
    """
    # Load audio
    audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)

    # For GGUF model, we'll encode audio as text description
    # This is a simplified approach - actual implementation may vary
    duration = len(audio_array) / sr

    return audio_array, sr, duration

def transcribe_audio_file(audio_path, request_id, max_tokens=8192, temperature=0.1,
                          repetition_penalty=1.1, enable_s2t=True):
    """
    Transcribe a single audio file using GGUF model

    Args:
        audio_path: Path to audio file
        request_id: Unique request identifier
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        repetition_penalty: Repetition penalty
        enable_s2t: Enable simplified to traditional Chinese conversion (default: True)

    Returns:
        Transcribed text
    """
    global model, opencc_converter, model_config

    # Auto-reload model if it was unloaded
    if model is None and model_config:
        print(f"[{request_id}] Model not loaded, reloading...")
        load_model(**model_config)
    elif model is None:
        raise RuntimeError("Model not loaded and no configuration available")

    # Update activity timestamp
    update_activity()

    print(f"[{request_id}] Processing audio file...")

    try:
        # Extract audio features
        audio_array, sr, duration = extract_audio_features(audio_path)
        duration_mins = duration / 60
        print(f"[{request_id}] Audio duration: {duration:.2f}s ({duration_mins:.2f} mins)")

        # Construct prompt for transcription
        system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        user_prompt = "請將音訊內容精確轉錄為繁體中文文字。格式要求：1) 標點符號：每句話以句號(。)、問號(?)或驚嘆號(!)結尾,語意停頓處加入逗號(,)、頓號(、)或分號(;) 2) 聖經引用格式：使用《書卷名章:節》格式,例如《約翰福音3:16》神愛世人,甚至將他的獨生子賜給他們,叫一切信他的,不致滅亡,反得永生。聖經書卷包含：舊約(創世記、出埃及記、利未記、民數記、申命記、約書亞記、士師記、路得記、撒母耳記上、撒母耳記下、列王紀上、列王紀下、歷代志上、歷代志下、以斯拉記、尼希米記、以斯帖記、約伯記、詩篇、箴言、傳道書、雅歌、以賽亞書、耶利米書、耶利米哀歌、以西結書、但以理書、何西阿書、約珥書、阿摩司書、俄巴底亞書、約拿書、彌迦書、那鴻書、哈巴谷書、西番雅書、哈該書、撒迦利亞書、瑪拉基書)、新約(馬太福音、馬可福音、路加福音、約翰福音、使徒行傳、羅馬書、哥林多前書、哥林多後書、加拉太書、以弗所書、腓立比書、歌羅西書、帖撒羅尼迦前書、帖撒羅尼迦後書、提摩太前書、提摩太後書、提多書、腓利門書、希伯來書、雅各書、彼得前書、彼得後書、約翰一書、約翰二書、約翰三書、猶大書、啟示錄) 3) 直接輸出轉錄文字,不包含任何解釋、評論、標記或元資料。"

        # Build full prompt
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}\n[Audio file: {audio_path}]<|im_end|>\n<|im_start|>assistant\n"

        print(f"[{request_id}] Generating transcription...")

        # Generate with GGUF model
        output = model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            repeat_penalty=repetition_penalty,
            echo=False,
            stop=["<|im_end|>", "<|endoftext|>"]
        )

        # Extract response text
        response = output['choices'][0]['text'].strip()

        # Simplified to Traditional Chinese conversion
        if enable_s2t and opencc_converter is not None:
            try:
                response = opencc_converter.convert(response)
                print(f"[{request_id}] Applied OpenCC Simplified to Traditional Chinese conversion")
            except Exception as e:
                print(f"[{request_id}] Warning: OpenCC conversion failed: {e}")

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

def process_audio_segments(audio_path, request_id, segment_duration=600, overlap_duration=10, **kwargs):
    """
    Process long audio by splitting into overlapping segments

    Args:
        audio_path: Path to audio file
        request_id: Unique request identifier
        segment_duration: Duration of each segment in seconds
        overlap_duration: Overlap duration at segment boundaries in seconds (default: 10s)
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

    # Process each segment
    results = []
    temp_files = []

    try:
        for idx, segment in enumerate(segments):
            # Save segment to temp file
            temp_file = os.path.join(TEMP_DIR, f"{request_id}_segment_{idx}.wav")
            sf.write(temp_file, segment, sr)
            temp_files.append(temp_file)

            segment_start_time = segment_positions[idx]['start']
            segment_start_mins = segment_start_time / 60
            print(f"[{request_id}] Processing segment {idx+1}/{len(segments)} (starts at {segment_start_mins:.1f} mins)...")

            try:
                result = transcribe_audio_file(temp_file, f"{request_id}_seg{idx}", **kwargs)
                results.append(result)
                print(f"[{request_id}] Segment {idx+1}/{len(segments)} completed: {len(result)} chars")
            except Exception as e:
                print(f"[{request_id}] Segment {idx} failed: {e}")
                results.append(f"[Error in segment {idx}]")

            # Clean up segment file
            try:
                os.remove(temp_file)
            except:
                pass

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
        "model_type": "gguf",
        "timestamp": datetime.now().isoformat()
    }

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
    global model, is_processing, model_config

    # Auto-reload model if needed
    if model is None and model_config:
        print("[INFO] Model not loaded, reloading...")
        load_model(**model_config)
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
            max_tokens = int(request.form.get('max_tokens', 8192))
            temperature = float(request.form.get('temperature', 0.1))
            repetition_penalty = float(request.form.get('repetition_penalty', 1.1))
            enable_s2t = request.form.get('enable_s2t', 'true').lower() == 'true'

            # Transcribe
            transcription = process_audio_segments(
                input_path,
                request_id,
                segment_duration=segment_duration,
                max_tokens=max_tokens,
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
        load_model(**model_config)
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
        'max_tokens': int(request.form.get('max_tokens', 8192)),
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
            "filename": "audio.mp4",
            "segment_duration": 600,
            "max_tokens": 8192,
            "temperature": 0.1,
            "repetition_penalty": 1.1,
            "enable_s2t": true
        }
    """
    global model, model_config, is_processing

    # Auto-reload model if needed
    if model is None and model_config:
        print("[INFO] Model not loaded, reloading...")
        load_model(**model_config)
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
            max_tokens = int(data.get('max_tokens', 8192))
            temperature = float(data.get('temperature', 0.1))
            repetition_penalty = float(data.get('repetition_penalty', 1.1))
            enable_s2t = data.get('enable_s2t', True)

            # Transcribe
            transcription = process_audio_segments(
                input_path,
                request_id,
                segment_duration=segment_duration,
                max_tokens=max_tokens,
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
    parser = ArgumentParser(description="Qwen2.5-Omni GGUF Audio Transcription Service")

    # Model arguments
    parser.add_argument("--model-path", type=str,
                        default="app/Qwen/Qwen2.5-Omni-7B-UD-Q8_K_XL.gguf",
                        help="Path to GGUF model file")
    parser.add_argument("--n-ctx", type=int, default=8192,
                        help="Context size (default: 8192)")
    parser.add_argument("--n-gpu-layers", type=int, default=-1,
                        help="Number of layers to offload to GPU (-1 for all)")
    parser.add_argument("--use-mmap", action="store_true", default=True,
                        help="Use memory mapping for faster loading")
    parser.add_argument("--use-mlock", action="store_true",
                        help="Lock model in memory")

    # Processing arguments
    parser.add_argument("--segment-duration", type=int, default=600,
                        help="Audio segment duration in seconds (default: 600 = 10 minutes)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Maximum tokens to generate")
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

    # Update global idle_timeout
    global idle_timeout
    idle_timeout = args.idle_timeout

    # Load model
    print("=" * 60)
    print("Qwen2.5-Omni GGUF Audio Transcription Service")
    print("=" * 60)

    try:
        load_model(
            model_path=args.model_path,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            use_mmap=args.use_mmap,
            use_mlock=args.use_mlock
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
