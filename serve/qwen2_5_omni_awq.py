"""
Qwen2.5-Omni AWQ 量化版本服務
支援多種 API 端點、批次處理、異步處理
"""

import io
import os
import sys
import tempfile
import numpy as np
import torch
import importlib.util
import threading
import uuid
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file, abort
from argparse import ArgumentParser
from huggingface_hub import hf_hub_download

# 過濾不必要的日誌輸出
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="The following generation flags are not valid.*")

# AWQ 相關導入
from awq.models.base import BaseAWQForCausalLM
from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# 導入低顯存模式的模型定義
sys.path.insert(0, str(Path(__file__).parent / 'low-VRAM-mode'))
from modeling_qwen2_5_omni_low_VRAM_mode import (
    Qwen2_5OmniDecoderLayer,
    Qwen2_5OmniForConditionalGeneration
)

# 導入共用模組
from common_utils import (
    OpenCCConverter,
    AudioProcessor,
    FileManager,
    MemoryMonitor,
    IdleChecker,
    generate_srt
)
from security import SecurityValidator, add_security_headers

# 初始化 Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024  # 4GB

# 添加安全標頭
@app.after_request
def apply_security_headers(response):
    return add_security_headers(response)

# 全局變量
model = None
processor = None
opencc_converter = None
args_global = None
model_lock = threading.Lock()
idle_checker = None

# 進度追蹤
job_status = {}  # {job_id: {"status": "processing|completed|failed", "progress": 0-100, "result": ..., "error": ...}}
job_lock = threading.Lock()

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


def replace_transformers_module():
    """替換 transformers 模組以使用低顯存模式"""
    original_mod_name = 'transformers.models.qwen2_5_omni.modeling_qwen2_5_omni'
    new_mod_path = str(Path(__file__).parent / 'low-VRAM-mode' / 'modeling_qwen2_5_omni_low_VRAM_mode.py')
    if original_mod_name in sys.modules:
        del sys.modules[original_mod_name]
    spec = importlib.util.spec_from_file_location(original_mod_name, new_mod_path)
    new_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(new_mod)
    sys.modules[original_mod_name] = new_mod


class Qwen2_5_OmniAWQForConditionalGeneration(BaseAWQForCausalLM):
    """AWQ 量化模型包裝類"""
    layer_type = "Qwen2_5OmniDecoderLayer"
    max_seq_len_key = "max_position_embeddings"
    modules_to_not_convert = ["visual"]

    @staticmethod
    def get_model_layers(model: "Qwen2_5OmniForConditionalGeneration"):
        return model.thinker.model.layers

    @staticmethod
    def get_act_for_scaling(module: "Qwen2_5OmniForConditionalGeneration"):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model: "Qwen2_5OmniForConditionalGeneration", device: str):
        model.thinker.model.embed_tokens = model.thinker.model.embed_tokens.to(device)
        model.thinker.visual = model.thinker.visual.to(device)
        model.thinker.audio_tower = model.thinker.audio_tower.to(device)
        model.thinker.visual.rotary_pos_emb = model.thinker.visual.rotary_pos_emb.to(device)
        model.thinker.model.rotary_emb = model.thinker.model.rotary_emb.to(device)
        for layer in model.thinker.model.layers:
            layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)

    @staticmethod
    def get_layers_for_scaling(module: "Qwen2_5OmniDecoderLayer", input_feat, module_kwargs):
        layers = []
        # attention input
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )
        # attention out
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )
        # linear 1
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )
        # linear 2
        layers.append(
            dict(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )
        return layers


def load_model_processor(checkpoint_path, flash_attn2=False, local_model=False):
    """加載 AWQ 量化模型和處理器"""
    global model, processor
    print(f"[{datetime.now()}] Loading AWQ quantized model from {checkpoint_path}...")
    replace_transformers_module()
    device = 'cuda'

    if flash_attn2:
        awq_model = Qwen2_5_OmniAWQForConditionalGeneration.from_quantized(
            checkpoint_path,
            model_type="qwen2_5_omni",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2"
        )
    else:
        awq_model = Qwen2_5_OmniAWQForConditionalGeneration.from_quantized(
            checkpoint_path,
            model_type="qwen2_5_omni",
            torch_dtype=torch.float16
        )

    if local_model:
        spk_path = os.path.join(checkpoint_path, 'spk_dict.pt')
    else:
        spk_path = hf_hub_download(repo_id=checkpoint_path, filename='spk_dict.pt')

    awq_model.model.load_speakers(spk_path)
    awq_model.model.thinker.model.embed_tokens = awq_model.model.thinker.model.embed_tokens.to(device)
    awq_model.model.thinker.visual = awq_model.model.thinker.visual.to(device)
    awq_model.model.thinker.audio_tower = awq_model.model.thinker.audio_tower.to(device)
    awq_model.model.thinker.visual.rotary_pos_emb = awq_model.model.thinker.visual.rotary_pos_emb.to(device)
    awq_model.model.thinker.model.rotary_emb = awq_model.model.thinker.model.rotary_emb.to(device)

    for layer in awq_model.model.thinker.model.layers:
        layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(device)

    processor = Qwen2_5OmniProcessor.from_pretrained(checkpoint_path)
    model = awq_model

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{datetime.now()}] GPU Memory Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    print(f"[{datetime.now()}] AWQ model loaded successfully!")

    return model, processor


def transcribe_audio_batch(audio_paths, request_id, enable_s2t=True):
    """批次轉錄音頻"""
    global model, processor, args_global, model_lock, opencc_converter

    if model is None or processor is None:
        raise ValueError("Model not loaded. Please initialize the model first.")

    try:
        system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        user_prompt = "你的唯一任務是將音頻內容轉錄為文本。請在適當的語義轉折點插入換行符來分隔段落。只輸出帶有語義分段的純粹轉錄文本。不要包含任何介紹性短語、解釋或對話性評論。"

        texts_for_processing = []
        for audio_path in audio_paths:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "audio", "audio": audio_path}]}
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            texts_for_processing.append(text)

        with model_lock:
            inputs = processor(
                text=texts_for_processing,
                audio=audio_paths,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True
            ).to('cuda')

            output = model.generate(
                **inputs,
                use_audio_in_video=True,
                return_audio=False,
                max_new_tokens=args_global.max_new_tokens,
                temperature=args_global.temperature,
                repetition_penalty=args_global.repetition_penalty
            )

            responses = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            processed_responses = []
            gen_marker = "<|im_start|>assistant\n"
            phrases_to_remove = [
                "如果你還有其他關於音頻轉寫或者內容理解的問題,都可以隨時跟我說哦。",
                "如果你還有其他關於這方面的問題或者其他想法,都可以跟我說哦。",
                "如果你還有其他關於這方面的問題,都可以隨時跟我說哦。"
            ]

            for response in responses:
                gen_pos = response.rfind(gen_marker)
                if gen_pos != -1:
                    response = response[gen_pos + len(gen_marker):]
                response = response.strip()

                for phrase in phrases_to_remove:
                    response = response.removesuffix(phrase).strip()

                if enable_s2t and opencc_converter:
                    response = opencc_converter.convert(response)

                processed_responses.append(response)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return processed_responses

    except Exception as e:
        error_str = str(e)
        if "flash_attn" in error_str and "CPU' backend" in error_str:
            clean_error_msg = f"[{request_id}] Error transcribing audio batch: Flash Attention backend issue (CPU not supported for Flash Attention)."
            print(clean_error_msg)
            return [f"[Error transcribing segment: Processing error - Flash Attention CPU compatibility issue]" for _ in audio_paths]
        else:
            print(f"[{request_id}] Error transcribing audio batch: {e}")
            import traceback
            traceback.print_exc()
            return [f"[Error transcribing segment: {str(e)}]" for _ in audio_paths]


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
            segment_duration=args_global.segment_duration,
            temp_dir=TEMP_DIR
        )

        if not segments:
            raise Exception("Failed to split audio into segments")

        batch_size = args_global.batch_size
        segment_batches = [segments[i:i + batch_size] for i in range(0, len(segments), batch_size)]
        transcriptions = []

        for batch in segment_batches:
            batch_transcriptions = transcribe_audio_batch(batch, request_id)
            transcriptions.extend(batch_transcriptions)

            for segment_path in batch:
                if os.path.exists(segment_path) and segment_path != str(wav_file):
                    os.remove(segment_path)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        full_transcription = "\n".join(transcriptions)
        output_file = OUTPUTS_DIR / f"{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_transcription)

        print(f"[{request_id}] Transcription saved to: {output_file.name}")
        print("\n--- BEGIN TRANSCRIPTION ---")
        print(full_transcription)
        print("--- END TRANSCRIPTION ---\n")

        if os.path.exists(wav_file):
            os.remove(wav_file)

        return output_file, full_transcription

    finally:
        idle_checker.set_processing(False)
        idle_checker.update_activity()


# ==================== API 端點 ====================

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """轉錄音頻 - 返回文本檔案"""
    global model, args_global, model_lock

    try:
        SecurityValidator.check_rate_limit()

        with model_lock:
            if model is None:
                print(f"[{datetime.now()}] Model is not loaded. Loading now...")
                load_model_processor(
                    checkpoint_path=args_global.checkpoint_path,
                    flash_attn2=args_global.flash_attn2,
                    local_model=args_global.local_model
                )
                print(f"[{datetime.now()}] Model reloaded.")

        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            filename = SecurityValidator.sanitize_filename(file.filename)
            SecurityValidator.validate_file_size(file)
            SecurityValidator.validate_file_type(file, filename)
            audio_data = file.read()
        elif request.data:
            audio_data = request.data
            filename = "audio.wav"
            SecurityValidator.validate_file_size(audio_data)
            SecurityValidator.validate_file_type(audio_data, filename)
        else:
            return jsonify({"error": "No audio data received"}), 400

        output_file, transcription = process_full_audio(audio_data, filename)
        return send_file(output_file, mimetype='application/octet-stream', as_attachment=True, download_name='audio.txt')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/transcribe/json', methods=['POST'])
def transcribe_json():
    """JSON 格式轉錄 API"""
    global model, args_global, model_lock

    try:
        SecurityValidator.check_rate_limit()

        with model_lock:
            if model is None:
                load_model_processor(
                    checkpoint_path=args_global.checkpoint_path,
                    flash_attn2=args_global.flash_attn2,
                    local_model=args_global.local_model
                )

        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            filename = SecurityValidator.sanitize_filename(file.filename)
            SecurityValidator.validate_file_size(file)
            SecurityValidator.validate_file_type(file, filename)
            audio_data = file.read()
        elif request.data:
            audio_data = request.data
            filename = "audio.wav"
            SecurityValidator.validate_file_size(audio_data)
            SecurityValidator.validate_file_type(audio_data, filename)
        else:
            return jsonify({"error": "No audio data received"}), 400

        output_file, transcription = process_full_audio(audio_data, filename)

        return jsonify({
            "status": "success",
            "transcription": transcription,
            "output_file": output_file.name,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/transcribe/srt', methods=['POST'])
def transcribe_srt():
    """SRT 字幕格式轉錄 API"""
    global model, args_global, model_lock

    try:
        SecurityValidator.check_rate_limit()

        with model_lock:
            if model is None:
                load_model_processor(
                    checkpoint_path=args_global.checkpoint_path,
                    flash_attn2=args_global.flash_attn2,
                    local_model=args_global.local_model
                )

        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            filename = SecurityValidator.sanitize_filename(file.filename)
            SecurityValidator.validate_file_size(file)
            SecurityValidator.validate_file_type(file, filename)
            audio_data = file.read()
        elif request.data:
            audio_data = request.data
            filename = "audio.wav"
            SecurityValidator.validate_file_size(audio_data)
            SecurityValidator.validate_file_type(audio_data, filename)
        else:
            return jsonify({"error": "No audio data received"}), 400

        # 處理音頻獲取分段轉錄
        idle_checker.update_activity()
        idle_checker.set_processing(True)

        try:
            request_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
            input_file = INPUTS_DIR / f"{request_id}_{filename}"
            with open(input_file, 'wb') as f:
                f.write(audio_data)

            wav_file = TEMP_DIR / f"{request_id}_audio.wav"
            if not AudioProcessor.convert_to_wav(str(input_file), str(wav_file)):
                raise Exception("Failed to convert audio to WAV format")

            segments = AudioProcessor.split_audio(
                str(wav_file),
                request_id,
                segment_duration=args_global.segment_duration,
                temp_dir=TEMP_DIR
            )

            if not segments:
                raise Exception("Failed to split audio into segments")

            batch_size = args_global.batch_size
            segment_batches = [segments[i:i + batch_size] for i in range(0, len(segments), batch_size)]
            transcriptions = []

            for batch in segment_batches:
                batch_transcriptions = transcribe_audio_batch(batch, request_id)
                transcriptions.extend(batch_transcriptions)

                for segment_path in batch:
                    if os.path.exists(segment_path) and segment_path != str(wav_file):
                        os.remove(segment_path)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 生成 SRT 格式
            srt_content = generate_srt(transcriptions, segment_duration=args_global.segment_duration)

            # 保存 SRT 文件
            output_file = OUTPUTS_DIR / f"{datetime.now().strftime('%Y%m%d%H%M%S')}.srt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(srt_content)

            if os.path.exists(wav_file):
                os.remove(wav_file)

            return send_file(output_file, mimetype='text/plain', as_attachment=True, download_name='subtitle.srt')

        finally:
            idle_checker.set_processing(False)
            idle_checker.update_activity()

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def process_audio_async(job_id, audio_data, filename):
    """異步處理音頻的後台任務"""
    try:
        with job_lock:
            job_status[job_id] = {"status": "processing", "progress": 0, "result": None, "error": None}

        output_file, transcription = process_full_audio(audio_data, filename)

        with job_lock:
            job_status[job_id] = {
                "status": "completed",
                "progress": 100,
                "result": {
                    "transcription": transcription,
                    "output_file": str(output_file),
                    "timestamp": datetime.now().isoformat()
                },
                "error": None
            }
    except Exception as e:
        with job_lock:
            job_status[job_id] = {
                "status": "failed",
                "progress": 0,
                "result": None,
                "error": str(e)
            }


@app.route('/transcribe/async', methods=['POST'])
def transcribe_async():
    """異步轉錄 API - 立即返回 job_id"""
    try:
        SecurityValidator.check_rate_limit()

        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            filename = SecurityValidator.sanitize_filename(file.filename)
            SecurityValidator.validate_file_size(file)
            SecurityValidator.validate_file_type(file, filename)
            audio_data = file.read()
        elif request.data:
            audio_data = request.data
            filename = "audio.wav"
            SecurityValidator.validate_file_size(audio_data)
            SecurityValidator.validate_file_type(audio_data, filename)
        else:
            return jsonify({"error": "No audio data received"}), 400

        # 生成 job ID
        job_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # 啟動後台線程處理
        thread = threading.Thread(target=process_audio_async, args=(job_id, audio_data, filename))
        thread.daemon = True
        thread.start()

        return jsonify({
            "status": "accepted",
            "job_id": job_id,
            "message": "Audio processing started. Use /status/<job_id> to check progress."
        }), 202

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """查詢轉錄進度"""
    with job_lock:
        if job_id not in job_status:
            return jsonify({"error": "Job not found"}), 404

        status = job_status[job_id].copy()

    return jsonify({
        "job_id": job_id,
        "status": status["status"],
        "progress": status["progress"],
        "result": status.get("result"),
        "error": status.get("error")
    }), 200


@app.route('/health', methods=['GET'])
def health():
    """健康檢查端點"""
    status = {"status": "ok", "model_loaded": model is not None}
    if model is not None and torch.cuda.is_available():
        status["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
        status["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 2)
    return jsonify(status), 200


def get_args():
    """解析命令行參數"""
    parser = ArgumentParser(description='Qwen2.5-Omni HTTP Transcription Service (AWQ Quantized)')
    parser.add_argument('-c', '--checkpoint-path', type=str, default='Qwen/Qwen2.5-Omni-7B-AWQ', help='AWQ model checkpoint path')
    parser.add_argument('--local-model', action='store_true', help='Use local model files')
    parser.add_argument('--flash-attn2', action='store_true', default=False, help='Enable flash_attention_2')
    parser.add_argument('--segment-duration', type=int, default=600, help='Audio segment duration in seconds')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for processing')
    parser.add_argument('--max-new-tokens', type=int, default=8192, help='Max new tokens for generation')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for sampling')
    parser.add_argument('--repetition-penalty', type=float, default=1.1, help='Repetition penalty')
    parser.add_argument('--idle-timeout', type=int, default=300, help='Idle time in seconds to unload model')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind server')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind server')
    parser.add_argument('--debug', action='store_true', help='Run Flask in debug mode')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    args_global = args

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

    print("\nInitializing AWQ quantized model...")
    load_model_processor(
        checkpoint_path=args.checkpoint_path,
        flash_attn2=args.flash_attn2,
        local_model=args.local_model
    )

    idle_checker.update_activity()
    idle_checker.set_model_loaded(True)

    idle_thread = threading.Thread(target=idle_checker.check_loop, daemon=True)
    idle_thread.start()
    print("✅ Idle checker thread started.")

    print(f"\n✅ Server is ready! Listening on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
