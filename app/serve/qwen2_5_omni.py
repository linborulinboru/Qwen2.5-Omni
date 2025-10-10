#!/usr/bin/env python3
"""
Qwen2.5-Omni Standard Model Audio Transcription HTTP Service
Using the full precision (non-quantized) model for best quality
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
import torch
from flask import Flask, request, jsonify, send_file

# Standard transformers imports
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_omni_utils import process_mm_info

# ==================== Global Configuration ====================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024  # 4GB

# Global variables
model = None
processor = None
model_lock = threading.Lock()
processing_lock = threading.Lock()
job_status = {}
job_lock = threading.Lock()

# Directories - use absolute paths to ensure correct location
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUTS_DIR = os.path.join(WORK_DIR, "inputs")
TEMP_DIR = os.path.join(WORK_DIR, "temp")
OUTPUTS_DIR = os.path.join(WORK_DIR, "outputs")

os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ==================== Model Loading ====================

def load_model_processor(checkpoint_path, flash_attn2=False, local_model=False):
    """Load standard (non-quantized) model and processor"""
    global model, processor

    with model_lock:
        if model is not None:
            print("[INFO] Model already loaded")
            return

        print(f"[INFO] Loading standard model from: {checkpoint_path}")
        print(f"[INFO] Flash Attention 2: {flash_attn2}")
        print(f"[INFO] Local model: {local_model}")

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load standard model
        attn_impl = "flash_attention_2" if flash_attn2 else "eager"

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            attn_implementation=attn_impl,
            trust_remote_code=True
        )

        # Load processor
        print("[INFO] Loading processor...")
        processor = AutoProcessor.from_pretrained(checkpoint_path)

        # Print GPU memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[INFO] GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

        print("[INFO] Model loaded successfully!")

def unload_model():
    """Unload model and clear CUDA cache"""
    global model, processor

    with model_lock:
        if model is None:
            return

        print("[INFO] Unloading model...")
        del model
        del processor
        model = None
        processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("[INFO] Model unloaded and CUDA cache cleared")

# ==================== Audio Processing ====================

def sanitize_filename(filename):
    """Sanitize filename for security"""
    import re
    # Remove path separators and dangerous characters
    filename = os.path.basename(filename)
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    return filename[:255]  # Limit length

def transcribe_audio_file(audio_path, request_id, max_new_tokens=8192, temperature=0.1,
                          repetition_penalty=1.1, enable_s2t=False):
    """
    Transcribe a single audio file using standard model

    Args:
        audio_path: Path to audio file
        request_id: Unique request identifier
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        repetition_penalty: Repetition penalty
        enable_s2t: Enable simplified to traditional Chinese conversion

    Returns:
        Transcribed text
    """
    global model, processor

    if model is None:
        raise RuntimeError("Model not loaded")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build prompt for transcription
    system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
    user_prompt = "請將音頻內容轉錄為文本。只輸出轉錄結果，不要添加任何解釋或評論。"

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

        # Use process_mm_info to extract audio
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

        print(f"[{request_id}] Audio processed by process_mm_info")
        print(f"[{request_id}] Processing with model...")

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
            text_ids = model.generate(
                **inputs,
                use_audio_in_video=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
            )

        # Decode
        response = processor.batch_decode(
            text_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        print(f"[{request_id}] Raw response length: {len(response)}")

        # Post-process: Remove generation markers
        if '<|im_start|>assistant\n' in response:
            response = response.split('<|im_start|>assistant\n')[-1]

        # Remove ending phrases
        endings_to_remove = [
            "如果你還有其他關於音頻轉寫或者內容理解的問題",
            "如果你還有其他關於這方面的問題",
            "如果需要進一步的幫助",
        ]
        for ending in endings_to_remove:
            if ending in response:
                response = response.split(ending)[0]

        response = response.strip()

        # Simplified to Traditional Chinese conversion (optional)
        if enable_s2t:
            try:
                from opencc import OpenCC
                cc = OpenCC('s2t')
                response = cc.convert(response)
            except Exception as e:
                print(f"[{request_id}] Warning: OpenCC conversion failed: {e}")

        # Clean up CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[{request_id}] Transcription complete: {len(response)} chars")

        return response

    except Exception as e:
        print(f"[{request_id}] ERROR during transcription: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def process_audio_segments(audio_path, request_id, segment_duration=60, **kwargs):
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
    import librosa
    import soundfile as sf

    print(f"[{request_id}] Processing audio file: {audio_path}")

    # Load full audio
    audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(audio_array) / sr

    print(f"[{request_id}] Audio duration: {duration:.2f}s, sr: {sr}")

    # If short enough, process directly
    if duration <= segment_duration:
        print(f"[{request_id}] Audio short enough, processing directly")
        return transcribe_audio_file(audio_path, request_id, **kwargs)

    # Split into segments
    segment_samples = int(segment_duration * sr)
    segments = []

    for i in range(0, len(audio_array), segment_samples):
        segment = audio_array[i:i + segment_samples]
        segments.append(segment)

    print(f"[{request_id}] Split into {len(segments)} segments")

    # Process each segment
    results = []
    temp_files = []

    try:
        for idx, segment in enumerate(segments):
            # Save segment to temp file
            temp_file = os.path.join(TEMP_DIR, f"{request_id}_segment_{idx}.wav")
            sf.write(temp_file, segment, sr)
            temp_files.append(temp_file)

            print(f"[{request_id}] Processing segment {idx+1}/{len(segments)}...")

            try:
                result = transcribe_audio_file(temp_file, f"{request_id}_seg{idx}", **kwargs)
                results.append(result)
            except Exception as e:
                print(f"[{request_id}] Segment {idx} failed: {e}")
                results.append(f"[Error in segment {idx}]")

            # Clean up segment file
            try:
                os.remove(temp_file)
            except:
                pass

        # Combine results
        combined = "\n\n".join(results)
        return combined

    finally:
        # Clean up any remaining temp files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
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
        "model_type": "standard",
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
    global model, processor

    # Ensure model is loaded
    if model is None:
        with model_lock:
            if model is None:
                return jsonify({"error": "Model not loaded yet, please wait"}), 503

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
            print(f"[{request_id}] Starting transcription...")

            # Get parameters
            segment_duration = int(request.form.get('segment_duration', 60))
            max_new_tokens = int(request.form.get('max_new_tokens', 8192))
            temperature = float(request.form.get('temperature', 0.1))
            repetition_penalty = float(request.form.get('repetition_penalty', 1.1))
            enable_s2t = request.form.get('enable_s2t', 'false').lower() == 'true'

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
    global model

    if model is None:
        return jsonify({"error": "Model not loaded yet"}), 503

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
        'segment_duration': int(request.form.get('segment_duration', 60)),
        'max_new_tokens': int(request.form.get('max_new_tokens', 8192)),
        'temperature': float(request.form.get('temperature', 0.1)),
        'repetition_penalty': float(request.form.get('repetition_penalty', 1.1)),
        'enable_s2t': request.form.get('enable_s2t', 'false').lower() == 'true'
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
        try:
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

# ==================== Main ====================

def main():
    parser = ArgumentParser(description="Qwen2.5-Omni Standard Model Audio Transcription Service")

    # Model arguments
    parser.add_argument("--checkpoint-path", type=str, default="Qwen/Qwen2.5-Omni-7B",
                        help="Model checkpoint path or HuggingFace repo")
    parser.add_argument("--local-model", action="store_true",
                        help="Use local model instead of downloading from HF")
    parser.add_argument("--flash-attn2", action="store_true",
                        help="Enable Flash Attention 2")

    # Processing arguments
    parser.add_argument("--segment-duration", type=int, default=60,
                        help="Audio segment duration in seconds (default: 60)")
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
    parser.add_argument("--idle-timeout", type=int, default=600,
                        help="Idle timeout in seconds before unloading model")

    # Audio processing
    parser.add_argument("--audio-only", action="store_true",
                        help="Audio-only mode (no video processing)")

    args = parser.parse_args()

    # Store args globally for routes to access
    app.config['ARGS'] = args

    # Load model
    print("=" * 60)
    print("Qwen2.5-Omni Standard Model Audio Transcription Service")
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

    # Start server
    print(f"\n[INFO] Starting server on {args.host}:{args.port}")
    print(f"[INFO] Endpoints:")
    print(f"  POST /transcribe         - Transcribe audio (returns text file)")
    print(f"  POST /transcribe/json    - Transcribe audio (returns JSON)")
    print(f"  POST /transcribe/srt     - Transcribe audio (returns SRT)")
    print(f"  POST /transcribe/async   - Start async transcription")
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
