"""
Qwen2.5-Omni 標準版本服務
支援 CPU offloading、記憶體優化
"""

import os
import torch
import threading
import uuid
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from argparse import ArgumentParser

# 導入共用模組
from common_utils import (
    OpenCCConverter,
    AudioProcessor,
    FileManager,
    MemoryMonitor,
    IdleChecker
)

# 初始化 Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024  # 4GB

# 全局變量
model = None
processor = None
opencc_converter = None
args_global = None
model_lock = threading.Lock()
idle_checker = None

# 確保目錄存在
current_dir = Path(__file__).parent
INPUTS_DIR = current_dir.parent / "inputs"
TEMP_DIR = current_dir.parent / "temp"
OUTPUTS_DIR = current_dir.parent / "outputs"

INPUTS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


def unload_model():
    """卸載模型並清理緩存"""
    global model, processor, model_lock
    with model_lock:
        if model is not None:
            print(f"[{datetime.now()}] Model inactive, unloading...")
            model = None
            processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[{datetime.now()}] Model unloaded and cache cleared.")


def offload_unused_modules_to_cpu(model, audio_only=True):
    """
    將音頻處理不需要的模組卸載到 CPU RAM

    音頻處理需要:
    - Thinker (Text Decoder + Audio Encoder)

    音頻處理不需要 (卸載到 CPU):
    - Vision Encoder (視覺編碼器)
    - Talker (語音生成器)
    - Token2Wav (音頻解碼器)
    """
    if not audio_only:
        return model

    # 記錄卸載前的 GPU 內存
    if torch.cuda.is_available():
        mem_before = torch.cuda.memory_allocated() / 1024**3

    offloaded_modules = []

    # 1. 卸載 Vision Encoder (視覺編碼器) -> CPU
    if hasattr(model, 'thinker') and hasattr(model.thinker, 'visual'):
        model.thinker.visual = model.thinker.visual.cpu()
        offloaded_modules.append("Vision Encoder")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 2. 卸載 Talker (語音生成器) -> CPU
    if hasattr(model, 'talker') and model.talker is not None:
        model.talker = model.talker.cpu()
        offloaded_modules.append("Talker")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3. 卸載 Token2Wav (音頻解碼器) -> CPU
    if hasattr(model, 'code2wav') and model.code2wav is not None:
        model.code2wav = model.code2wav.cpu()
        offloaded_modules.append("Token2Wav")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 記錄卸載後的 GPU 內存
    if torch.cuda.is_available():
        mem_after = torch.cuda.memory_allocated() / 1024**3
        mem_freed = mem_before - mem_after
        mem_freed_pct = (mem_freed / mem_before) * 100 if mem_before > 0 else 0
        print(f"GPU Memory freed: {mem_freed:.2f} GB ({mem_freed_pct:.1f}%) by offloading {len(offloaded_modules)} modules to CPU")

    return model


def load_model_processor(checkpoint_path, cpu_only=False, flash_attn2=False, audio_only=True):
    """加載模型和處理器 (支持 CPU offloading)"""
    global model, processor
    torch_dtype = torch.bfloat16
    device = 'cuda' if torch.cuda.is_available() and not cpu_only else 'cpu'

    if cpu_only:
        print("Running on CPU, attempting to use bfloat16 data type (requires AMX support).")
    else:
        print(f"Running on {device}, using bfloat16 data type.")

    print(f"[{datetime.now()}] Loading model from {checkpoint_path}...")

    if flash_attn2 and not cpu_only:
        print("Flash Attention 2 is enabled.")
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch_dtype,
            attn_implementation='flash_attention_2'
        )
    else:
        if flash_attn2 and cpu_only:
            print("Warning: Flash Attention 2 is not available on CPU, ignoring.")
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=torch_dtype
        )

    # 明確移動模型到目標設備
    if not cpu_only:
        print(f"Moving model to {device}...")
        model.to(device)
        print("Model moved to GPU.")

    processor = Qwen2_5OmniProcessor.from_pretrained(checkpoint_path)
    print(f"[{datetime.now()}] Model loaded successfully!")

    # 應用 CPU offloading (如果啟用 audio_only 模式)
    if audio_only and not cpu_only:
        model = offload_unused_modules_to_cpu(model, audio_only=True)

    # 手動將 rotary embeddings 移至 GPU
    if not cpu_only:
        print("Ensuring rotary embeddings are on the correct device (safeguard)...")
        if hasattr(model, 'thinker') and hasattr(model.thinker, 'model'):
            if hasattr(model.thinker.model, 'rotary_emb'):
                model.thinker.model.rotary_emb = model.thinker.model.rotary_emb.to(device)

            if hasattr(model.thinker.model, 'layers'):
                for layer in model.thinker.model.layers:
                    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'rotary_emb'):
                        layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)
        print("Rotary embeddings verified on GPU.")

    # 顯示最終 GPU 內存
    if torch.cuda.is_available() and not cpu_only:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\n[Final] GPU Memory: Allocated={allocated:.2f} GB, Reserved={reserved:.2f} GB\n")

    return model, processor


def transcribe_audio_segment(audio_path, request_id, enable_s2t=True):
    """轉錄單個音頻片段"""
    global model, processor, args_global, opencc_converter

    if model is None or processor is None:
        raise ValueError("Model not loaded.")

    try:
        system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        user_prompt = "你的唯一任務是將音頻內容轉錄為文本。請在適當的語義轉折點插入換行符來分隔段落。只輸出帶有語義分段的純粹轉錄文本。不要包含任何介紹性短語、解釋或對話性評論。"

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "audio", "audio": audio_path}]}
        ]

        with model_lock:
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
            device = "cuda" if torch.cuda.is_available() and not args_global.cpu_only else "cpu"
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True
            ).to(device).to(model.dtype)

            text_ids, _ = model.generate(**inputs, use_audio_in_video=True)
            response = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

            if enable_s2t and opencc_converter:
                response = opencc_converter.convert(response)

        print(f"[{request_id}] Transcription result: {response[:100]}..." if len(response) > 100 else f"[{request_id}] Transcription result: {response}")

        return response

    except Exception as e:
        error_str = str(e)
        if "flash_attn" in error_str and "CPU' backend" in error_str:
            clean_error_msg = f"[{request_id}] Error transcribing audio segment: Flash Attention backend issue (CPU not supported for Flash Attention)."
            print(clean_error_msg)
            return f"[Error transcribing segment: Processing error - Flash Attention CPU compatibility issue]"
        else:
            print(f"[{request_id}] Error transcribing audio segment: {e}")
            return f"[Error transcribing segment: {str(e)}]"


def process_full_audio(audio_binary_data, original_filename="audio"):
    """處理完整音頻檔案"""
    global idle_checker

    idle_checker.set_processing(True)
    idle_checker.update_activity()

    try:
        request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        print(f"[{request_id}] Starting audio processing...")

        input_file = INPUTS_DIR / f"{request_id}_{original_filename}"
        with open(input_file, 'wb') as f:
            f.write(audio_binary_data)

        wav_file = TEMP_DIR / f"{request_id}_audio.wav"
        if not AudioProcessor.convert_to_wav(str(input_file), str(wav_file)):
            raise Exception("Failed to convert audio to WAV format")

        segments = AudioProcessor.split_audio(
            str(wav_file),
            request_id,
            segment_duration=60,
            temp_dir=TEMP_DIR
        )

        if not segments:
            raise Exception("Failed to split audio into segments")

        # 處理片段（批次處理以優化 GPU 使用）
        batch_size = getattr(args_global, 'batch_size', 2)
        transcriptions = []

        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            batch_transcriptions = []

            for segment_path in batch:
                transcription = transcribe_audio_segment(segment_path, request_id)
                batch_transcriptions.append(transcription)
                if os.path.exists(segment_path) and segment_path != str(wav_file):
                    os.remove(segment_path)

            transcriptions.extend(batch_transcriptions)

            # 清理 CUDA 緩存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        full_transcription = "\n".join(transcriptions)
        output_file = OUTPUTS_DIR / f"{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_transcription)

        print(f"[{request_id}] Transcription saved to: {output_file.name}")
        print(f"\n[{request_id}] TRANSCRIPTION RESULT:")
        print(full_transcription)
        print(f"[{request_id}] END OF TRANSCRIPTION\n")

        if os.path.exists(wav_file):
            os.remove(wav_file)

        return output_file, full_transcription

    finally:
        idle_checker.set_processing(False)
        idle_checker.update_activity()


# ==================== API 端點 ====================

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """轉錄音頻端點"""
    global model, args_global, model_lock

    try:
        with model_lock:
            if model is None:
                print(f"[{datetime.now()}] Model is not loaded. Loading now...")
                audio_only = not args_global.no_audio_only if hasattr(args_global, 'no_audio_only') else True
                load_model_processor(
                    checkpoint_path=args_global.checkpoint_path,
                    cpu_only=args_global.cpu_only,
                    flash_attn2=args_global.flash_attn2,
                    audio_only=audio_only
                )
                print(f"[{datetime.now()}] Model reloaded.")

        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            audio_data = file.read()
            filename = secure_filename(file.filename)
        elif request.data:
            audio_data = request.data
            filename = "audio"
        else:
            return jsonify({"error": "No audio data received"}), 400

        output_file, transcription = process_full_audio(audio_data, filename)
        return send_file(output_file, mimetype='application/octet-stream', as_attachment=True, download_name='audio.txt')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """健康檢查端點"""
    status = {"status": "ok", "model_loaded": model is not None}

    if model is not None and not args_global.cpu_only:
        try:
            mem_stats = MemoryMonitor.get_memory_stats()
            status["memory"] = mem_stats
        except Exception as e:
            status["memory_error"] = str(e)

        # 檢查哪些模塊在 GPU 上
        if hasattr(model, 'thinker'):
            status["modules_on_gpu"] = {
                "thinker": next(model.thinker.parameters()).device.type == 'cuda',
                "vision_encoder": (
                    next(model.thinker.visual.parameters()).device.type == 'cuda'
                    if hasattr(model.thinker, 'visual') and next(model.thinker.visual.parameters(), None) is not None else 'offloaded'
                ),
                "audio_encoder": (
                    next(model.thinker.audio_tower.parameters()).device.type == 'cuda'
                    if hasattr(model.thinker, 'audio_tower') and next(model.thinker.audio_tower.parameters(), None) is not None else 'N/A'
                ),
                "talker": (
                    next(model.talker.parameters()).device.type == 'cuda'
                    if hasattr(model, 'talker') and model.talker is not None and next(model.talker.parameters(), None) is not None else 'offloaded'
                ),
                "token2wav": (
                    next(model.code2wav.parameters()).device.type == 'cuda'
                    if hasattr(model, 'code2wav') and model.code2wav is not None and next(model.code2wav.parameters(), None) is not None else 'offloaded'
                )
            }

    return jsonify(status), 200


@app.route('/stats', methods=['GET'])
def stats():
    """獲取 GPU 和 RAM 使用情況"""
    try:
        mem_stats = MemoryMonitor.get_memory_stats()

        gpu_info = mem_stats.get("gpu", {})
        cpu_info = mem_stats.get("cpu", {})
        system_info = mem_stats.get("system", {})

        stats_msg = f"GPU: {gpu_info.get('allocated_gb', 0):.2f}GB/{gpu_info.get('reserved_gb', 0):.2f}GB | "
        stats_msg += f"RAM: {cpu_info.get('rss_gb', 0):.2f}GB | "
        stats_msg += f"System: {system_info.get('used_gb', 0):.2f}GB/{system_info.get('total_gb', 0):.2f}GB"

        print(f"[{datetime.now()}] {stats_msg}")

        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "gpu_memory": gpu_info,
            "process_memory": cpu_info,
            "system_memory": system_info,
            "human_readable": stats_msg
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_args():
    """解析命令行參數"""
    parser = ArgumentParser(description='Qwen2.5-Omni HTTP Transcription Service')

    parser.add_argument('-c', '--checkpoint-path', type=str, default='Qwen/Qwen2.5-Omni-7B', help='Checkpoint name or path')
    parser.add_argument('--cpu-only', action='store_true', help='Run with CPU only')
    parser.add_argument('--flash-attn2', action='store_true', default=False, help='Enable flash_attention_2')
    parser.add_argument('--audio-only', action='store_true', default=True, help='Audio-only mode: offload Vision/Talker/Token2Wav to CPU RAM')
    parser.add_argument('--no-audio-only', action='store_true', default=False, help='Disable audio-only mode (keep all modules in GPU)')
    parser.add_argument('--segment-duration', type=int, default=60, help='Audio segment duration in seconds')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for processing audio segments')
    parser.add_argument('--max-new-tokens', type=int, default=8192, help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for sampling')
    parser.add_argument('--repetition-penalty', type=float, default=1.1, help='Repetition penalty')
    parser.add_argument('--idle-timeout', type=int, default=300, help='Idle time in seconds to unload model')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind the server to')
    parser.add_argument('--debug', action='store_true', help='Run Flask in debug mode')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    args_global = args

    # 處理 audio_only 參數
    audio_only = not args.no_audio_only if hasattr(args, 'no_audio_only') else True

    # 初始化 OpenCC
    opencc_converter = OpenCCConverter()

    # 初始化空閒檢查器
    idle_checker = IdleChecker(
        idle_timeout=args.idle_timeout,
        unload_callback=unload_model,
        temp_dir=TEMP_DIR,
        inputs_dir=INPUTS_DIR,
        outputs_dir=OUTPUTS_DIR
    )

    print("\nInitializing model...")
    print(f"Audio-only mode: {'Enabled' if audio_only else 'Disabled'}")

    load_model_processor(
        checkpoint_path=args.checkpoint_path,
        cpu_only=args.cpu_only,
        flash_attn2=args.flash_attn2,
        audio_only=audio_only
    )

    idle_checker.update_activity()
    idle_checker.set_model_loaded(True)

    idle_thread = threading.Thread(target=idle_checker.check_loop, daemon=True)
    idle_thread.start()
    print("✅ Idle checker thread started.")

    print(f"\n✅ Server is ready! Listening on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
