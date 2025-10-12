import io
import os
import ffmpeg
import numpy as np
import gradio as gr
import soundfile as sf
import datetime
import time
import threading
import logging
import gc
import torch
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from argparse import ArgumentParser
from opencc import OpenCC

import modelscope_studio.components.base as ms
import modelscope_studio.components.antd as antd
import gradio.processing_utils as processing_utils

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from gradio_client import utils as client_utils
from qwen_omni_utils import process_mm_info

# --- 全域變數和設定 ---
model = None
processor = None
opencc_converter = None  # OpenCC converter for Traditional Chinese
args = None
last_activity_time = time.time()
model_lock = threading.Lock()
processing_lock = threading.Lock()  # 用於序列化請求處理
IDLE_TIMEOUT = 300  # 預設閒置300秒後卸載模型，可由命令列參數覆蓋

# 閒置追蹤
is_processing = False  # 是否正在處理中


# --- 提示詞設定 ---
USER_PROMPT_BIBLE_TRANSCRIPTION = "請將音訊內容精確轉錄為中文文字。格式要求：1) 標點符號：每句話以句號(。)、問號(?)或驚嘆號(!)結尾,語意停頓處加入逗號(,)、頓號(、)或分號(;) 2) 聖經引用格式：使用《書卷名章:節》格式,例如《約翰福音3:16》神愛世人,甚至將他的獨生子賜給他們,叫一切信他的,不致滅亡,反得永生。聖經書卷包含：舊約(創世記、出埃及記、利未記、民數記、申命記、約書亞記、士師記、路得記、撒母耳記上、撒母耳記下、列王紀上、列王紀下、歷代志上、歷代志下、以斯拉記、尼希米記、以斯帖記、約伯記、詩篇、箴言、傳道書、雅歌、以賽亞書、耶利米書、耶利米哀歌、以西結書、但以理書、何西阿書、約珥書、阿摩司書、俄巴底亞書、約拿書、彌迦書、那鴻書、哈巴谷書、西番雅書、哈該書、撒迦利亞書、瑪拉基書)、新約(馬太福音、馬可福音、路加福音、約翰福音、使徒行傳、羅馬書、哥林多前書、哥林多後書、加拉太書、以弗所書、腓立比書、歌羅西書、帖撒羅尼迦前書、帖撒羅尼迦後書、提摩太前書、提摩太後書、提多書、腓利門書、希伯來書、雅各書、彼得前書、彼得後書、約翰一書、約翰二書、約翰三書、猶大書、啟示錄) 3) 直接輸出轉錄文字,不包含任何解釋、評論、標記或元資料。"

# --- 目錄建立 ---
os.makedirs('inputs', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('temp', exist_ok=True)


# --- 模型管理功能 ---
def _load_model_processor_internal():
    """內部函數，用於載入模型和處理器。"""
    global model, processor, args, opencc_converter
    print("[INFO] 開始載入模型...")
    if args.cpu_only:
        device_map = 'cpu'
    else:
        device_map = 'cuda'

    model_kwargs = {
        "torch_dtype": "auto",
        "device_map": device_map
    }
    if args.flash_attn2:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(args.checkpoint_path, **model_kwargs)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.checkpoint_path)
    
    # 初始化 OpenCC 轉換器
    try:
        opencc_converter = OpenCC('s2tw')  # 簡體到台灣繁體
        print("[INFO] OpenCC 轉換器初始化完成 (簡體到台灣繁體)")
    except Exception as e:
        print(f"[WARNING] OpenCC 轉換器初始化失敗: {e}")
        opencc_converter = None

    print("[INFO] 模型載入完成。")
    return model, processor

def unload_model():
    """卸載模型並清除 CUDA 緩存 - 內部使用（假設鎖已被獲取）"""
    global model, processor

    if model is None:
        return

    print("[INFO] 由於閒置超時，正在卸載模型...")

    try:
        # 卸載前將模型組件移至 CPU 以釋放 GPU 記憶體
        if torch.cuda.is_available():
            print("[INFO] 正在將模型組件移至 CPU...")
            try:
                # 移動主要模型組件至 CPU
                if hasattr(model, 'model'):
                    if hasattr(model.model, 'thinker'):
                        model.model.thinker = model.model.thinker.cpu()
                    # 如果是 Qwen2_5OmniForConditionalGeneration 類型
                    elif hasattr(model.model, 'language_model'):
                        model.model.language_model = model.model.language_model.cpu()
                elif hasattr(model, 'cpu'):
                    model.cpu()
            except Exception as e:
                print(f"[WARNING] 將模型移至 CPU 時出錯: {e}")

        # 刪除模型和處理器
        del model
        del processor
        model = None
        processor = None

        # 清除 CUDA 緩存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # 列印記憶體狀態
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[INFO] 卸載後 GPU 記憶體: {allocated:.2f} GB 已分配, {reserved:.2f} GB 已保留")

        print("[INFO] 模型已卸載並清除 CUDA 緩存")

    except Exception as e:
        print(f"[ERROR] 卸載模型時出錯: {e}")
        import traceback
        traceback.print_exc()

def update_activity():
    """更新最後活動時間戳"""
    global last_activity_time
    last_activity_time = time.time()

def idle_monitor():
    """監控閒置時間，如果閒置太久則卸載模型"""
    global model, is_processing, last_activity_time
    idle_check_interval = 30  # 每30秒檢查一次
    idle_timeout = IDLE_TIMEOUT  # 使用命令行參數設置的超時時間

    while True:
        time.sleep(idle_check_interval)

        # 跳過正在處理的情況
        if is_processing:
            continue

        # 跳過沒有活動記錄的情況
        if last_activity_time is None:
            continue

        # 檢查閒置時間
        idle_time = time.time() - last_activity_time

        if idle_time >= idle_timeout:
            # 獲取鎖並卸載模型
            with model_lock:
                # 獲取鎖後再次檢查條件
                if model is not None and not is_processing:
                    print(f"[INFO] 模型閒置 {idle_time:.0f} 秒，正在卸載...")
                    # 調用 unload_model（它本身不獲取鎖）
                    unload_model()
                    last_activity_time = None

def get_model_and_processor():
    """獲取模型和處理器，如果未載入則自動載入。"""
    global model, processor, model_lock
    with model_lock:
        if model is None or processor is None:
            _load_model_processor_internal()
        update_activity()
        return model, processor

def _generate_request_id():
    """產生唯一的請求 ID。"""
    return f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

# --- 核心轉錄邏輯 (重構後) ---
def process_long_audio_yield_transcription(input_path: str, request_id: str):
    """
    產生器函式：分段處理長音訊/影片檔案，並逐段產生(yield)轉錄後的文字。
    """
    global args
    base_filename = os.path.splitext(os.path.basename(input_path))[0]

    try:
        probe = ffmpeg.probe(input_path)
        duration = float(probe['format']['duration'])
        print(f"[{request_id}] 檔案時長: {duration:.2f} 秒")
    except ffmpeg.Error as e:
        import traceback
        print(f"[{request_id}] 無法讀取檔案資訊 {input_path}: {e.stderr.decode('utf8')}")
        traceback.print_exc()
        return

    start_time = 0
    segment_index = 0
    segment_duration_val = args.segment_duration
    total_segments = int(np.ceil(duration / segment_duration_val))

    while start_time < duration:
        segment_duration = min(segment_duration_val, duration - start_time)
        if segment_duration < 1:
            break

        temp_wav_path = os.path.join('temp', f"{base_filename}_seg{segment_index}.wav")
        print(f"[{request_id}] 正在提取第 {segment_index + 1}/{total_segments} 段 ({start_time:.2f}s - {start_time + segment_duration:.2f}s)...")

        try:
            (
                ffmpeg.input(input_path, ss=start_time, t=segment_duration)
                .output(temp_wav_path, ac=1, ar='16000', f='wav')
                .run(quiet=True, overwrite_output=True)
            )
        except ffmpeg.Error as e:
            import traceback
            print(f"[{request_id}] FFmpeg 提取失敗: {e.stderr.decode('utf8')}")
            traceback.print_exc()
            start_time += segment_duration_val
            segment_index += 1
            continue
        
        transcribed_text = ""
        try:
            current_model, current_processor = get_model_and_processor()
            messages = [{"role": "user", "content": [{"type": "text", "text": USER_PROMPT_BIBLE_TRANSCRIPTION}, {"type": "audio", "audio": temp_wav_path}]}]
            text = current_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios, _, _ = process_mm_info(messages, use_audio_in_video=True)
            inputs = current_processor(text=text, audio=audios, return_tensors="pt", padding=True, use_audio_in_video=True).to(current_model.device)
            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "repetition_penalty": args.repetition_penalty,
                "do_sample": True if args.temperature > 0 else False,
                "use_audio_in_video": True
            }
            generated_ids = current_model.generate(**inputs, **gen_kwargs)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
            decoded_list = current_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            transcribed_text = decoded_list[0] if decoded_list else ""

            # 使用 OpenCC 轉換為台灣繁體中文
            if opencc_converter is not None:
                transcribed_text = opencc_converter.convert(transcribed_text)
                print(f"[{request_id}] 已將轉錄結果轉換為台灣繁體中文")

            # --- 範本格式日誌輸出 ---
            print(f"[{request_id}] ====== SEGMENT {segment_index+1}/{total_segments} RESULT (starts at {start_time/60:.1f} mins) ======")
            print(transcribed_text)
            print(f"[{request_id}] ====== END OF SEGMENT {segment_index+1} ({len(transcribed_text)} chars) ======")

        except Exception as e:
            import traceback
            print(f"[{request_id}] 轉錄過程中發生錯誤: {e}")
            traceback.print_exc()
            transcribed_text = f"[第 {segment_index + 1} 段轉錄失敗]"
        finally:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
        
        yield transcribed_text

        start_time += segment_duration_val
        segment_index += 1

def transcribe_file_task(input_path: str, output_filename: str, request_id: str):
    """Curl 任務的背景處理函式。"""
    global is_processing
    
    # 標記為正在處理，防止模型卸載
    is_processing = True
    update_activity()
    
    try:
        print(f"[{request_id}] 開始背景轉錄任務: {input_path}")
        output_txt_path = os.path.join('outputs', output_filename)
        all_segments_text = []
        with open(output_txt_path, 'a', encoding='utf-8') as f:
            for segment_text in process_long_audio_yield_transcription(input_path, request_id):
                f.write(segment_text + '\n')
                all_segments_text.append(segment_text)

        merged_result = "\n".join(all_segments_text)
        
        # --- 範本格式日誌輸出 ---
        print(f"[{request_id}] ====== FINAL COMBINED TRANSCRIPTION ======")
        print(merged_result)
        print(f"[{request_id}] ====== END OF COMBINED TRANSCRIPTION ({len(merged_result)} chars) ======")
        
        print(f"[{request_id}] ====== SAVED TRANSCRIPTION to {output_txt_path} ======")
        print(merged_result)
        print(f"[{request_id}] ====== END OF SAVED TRANSCRIPTION ======")
        
        print(f"[{request_id}] 檔案 {input_path} 處理完成。")
    finally:
        # 標記為處理完成，允許模型卸載
        is_processing = False
        update_activity()

# --- FastAPI 應用和端點 ---
app = FastAPI()

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/transcribe/")
async def create_transcription_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    request_id = _generate_request_id()
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_extension = os.path.splitext(file.filename)[1]
    input_filename = f"{timestamp}{file_extension}"
    input_filepath = os.path.join('inputs', input_filename)
    with open(input_filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    output_txt_filename = f"{timestamp}.txt"
    print(f"[{request_id}] 收到檔案: {file.filename}，已保存至 {input_filepath}。")
    
    # 使用序列化鎖處理請求
    def task_wrapper():
        with processing_lock:
            global is_processing
            is_processing = True
            update_activity()
            try:
                transcribe_file_task(input_filepath, output_txt_filename, request_id)
            finally:
                is_processing = False
                update_activity()
    
    background_tasks.add_task(task_wrapper)
    return {"message": "檔案上傳成功，已開始背景轉錄處理。", "request_id": request_id, "output_file": os.path.join('outputs', output_txt_filename)}

# --- Gradio 應用邏輯 ---
def _launch_demo(args_param):
    global args
    args = args_param
    
    if args.audio_only:
        default_prompt_for_ui = USER_PROMPT_BIBLE_TRANSCRIPTION
        is_interactive_prompt = False
        print("[INFO] 在 audio-only 模式下運行，使用預設的 ASR 提示詞。")
    else:
        default_prompt_for_ui = 'You are Qwen, a virtual human...'
        is_interactive_prompt = True

    def process_gradio_audio(audio_path, history, system_prompt):
        global is_processing
        
        # 標記為正在處理，防止模型卸載
        is_processing = True
        update_activity()
        
        try:
            request_id = _generate_request_id()
            history.append({"role": "user", "content": (audio_path,)})
            history.append({"role": "assistant", "content": ""})
            yield history

            full_transcription = ""
            for segment_text in process_long_audio_yield_transcription(audio_path, request_id):
                full_transcription += segment_text + " "
                history[-1]["content"] = full_transcription.strip()
                yield history
            
            print(f"[{request_id}] ====== FINAL GRADIO TRANSCRIPTION ======")
            print(full_transcription.strip())
            print(f"[{request_id}] ====== END OF GRADIO TRANSCRIPTION ({len(full_transcription.strip())} chars) ======")
        finally:
            # 標記為處理完成，允許模型卸載
            is_processing = False
            update_activity()

    def predict_multimodal(history, system_prompt, voice_choice):
        global is_processing
        
        # 標記為正在處理，防止模型卸載
        is_processing = True
        update_activity()
        
        try:
            request_id = _generate_request_id()
            formatted_history = format_history(history, system_prompt)
            history.append({"role": "assistant", "content": ""})

            for chunk in predict(formatted_history, voice_choice, request_id):
                if chunk["type"] == "text":
                    history[-1]["content"] = chunk["data"]
                    yield history
                if chunk["type"] == "audio":
                    history.append({"role": "assistant", "content": gr.Audio(chunk["data"])})
                    yield history
        finally:
            # 標記為處理完成，允許模型卸載
            is_processing = False
            update_activity()

    def format_history(history, system_prompt):
        messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
        for item in history:
            content, role = item.get("content"), item.get("role")
            if not content or not role: continue
            if isinstance(content, str):
                messages.append({"role": role, "content": content})
            elif role == "user" and (isinstance(content, list) or isinstance(content, tuple)):
                file_path = content[0]
                mime_type = client_utils.get_mimetype(file_path)
                content_item = None
                if mime_type.startswith("image"): content_item = {"type": "image", "image": file_path}
                elif mime_type.startswith("video"): content_item = {"type": "video", "video": file_path}
                elif mime_type.startswith("audio"): content_item = {"type": "audio", "audio": file_path}
                if content_item: messages.append({"role": role, "content": [content_item]})
        return messages

    def predict(messages, voice, request_id):
        current_model, current_processor = get_model_and_processor()
        text = current_processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        inputs = current_processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True).to(current_model.device)
        gen_kwargs = {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature, "repetition_penalty": args.repetition_penalty, "do_sample": True if args.temperature > 0 else False, "use_audio_in_video": True}
        
        if args.audio_only or not audios:
             text_ids = current_model.generate(**inputs, **gen_kwargs)
             audio = None
        else:
             text_ids, audio = current_model.generate(**inputs, speaker=voice, **gen_kwargs)
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, text_ids)]
        decoded_list = current_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = decoded_list[0] if decoded_list else ""
        
        # 使用 OpenCC 轉換為台灣繁體中文
        if opencc_converter is not None and audios:  # 只在有音訊輸入時進行轉換
            response = opencc_converter.convert(response)
            print(f"[{request_id}] 已將轉錄結果轉換為台灣繁體中文")
        
        # --- 範本格式日誌輸出 (for short audio/text) ---
        if audios: # 只在有音訊輸入時打印轉錄日誌
            print(f"[{request_id}] ====== TRANSCRIPTION RESULT ======")
            print(response)
            print(f"[{request_id}] ====== END OF TRANSCRIPTION ({len(response)} chars) ======")

        yield {"type": "text", "data": response}
        
        if audio is not None:
            audio_np = np.array(audio * 32767).astype(np.int16)
            wav_io, audio_path = io.BytesIO(), None
            sf.write(wav_io, audio_np, samplerate=24000, format="WAV")
            wav_io.seek(0)
            audio_path = processing_utils.save_bytes_to_cache(wav_io.getvalue(), "audio.wav", cache_dir=demo.GRADIO_CACHE)
            yield {"type": "audio", "data": audio_path}

    def chat_predict_router(text, audio, image, video, history, system_prompt, voice_choice):
        if audio:
            for updated_history in process_gradio_audio(audio, history, system_prompt):
                yield gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), updated_history
            return

        if text: history.append({"role": "user", "content": text})
        if image: history.append({"role": "user", "content": (image,)})
        if video: history.append({"role": "user", "content": (video,)})
        
        yield gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None), history

        for updated_history in predict_multimodal(history, system_prompt, voice_choice):
            yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), updated_history

    with gr.Blocks() as demo:
        with gr.Sidebar(open=False):
            system_prompt_textbox = gr.Textbox(label="System Prompt", value=default_prompt_for_ui, interactive=is_interactive_prompt)
        # ... UI layout code ... (保持不變)
        with antd.Flex(gap="small", justify="center", align="center"):
            with antd.Flex(vertical=True, gap="small", align="center"):
                antd.Typography.Title("Qwen2.5-Omni Demo", level=1, elem_style=dict(margin=0, fontSize=28))
                with antd.Flex(vertical=True, gap="small"):
                    antd.Typography.Text("🎯 使用說明：", strong=True)
                    antd.Typography.Text("1️⃣ 點擊音訊錄製按鈕，或上傳檔案")
                    antd.Typography.Text("2️⃣ 輸入音訊進行轉錄，或與模型進行多模態對話")
                    antd.Typography.Text("3️⃣ 點擊提交並等待模型的回答")
        voice_choice = gr.Dropdown(label="Voice Choice", choices=['Chelsie', 'Ethan'], value='Chelsie', visible=not args.audio_only)
        
        with gr.Tabs():
            with gr.Tab("Online"):
                with gr.Row():
                    with gr.Column(scale=1):
                        microphone = gr.Audio(sources=['microphone'], type="filepath")
                        webcam = gr.Video(sources=['webcam'], height=400, include_audio=True, visible=not args.audio_only)
                        submit_btn = gr.Button("提交", variant="primary")
                        stop_btn = gr.Button("停止", visible=False)
                        clear_btn = gr.Button("清除歷史")
                    with gr.Column(scale=2):
                        media_chatbot = gr.Chatbot(height=650, type="messages")
                def clear_history(): return [], gr.update(value=None), gr.update(value=None)
                def media_predict(audio, video, history, system_prompt, voice_choice): # 短音訊處理
                    if audio: history.append({"role": "user", "content": (audio,)})
                    if video: 
                        video_path = video.replace('.webm', '.mp4')
                        ffmpeg.input(video).output(video_path, y='-y').run(quiet=True, overwrite_output=True)
                        history.append({"role": "user", "content": (video_path,)})
                    
                    yield gr.update(value=None), gr.update(value=None), history
                    
                    for updated_history in predict_multimodal(history, system_prompt, voice_choice):
                        yield gr.update(value=None), gr.update(value=None), updated_history

                submit_event = submit_btn.click(fn=media_predict, inputs=[microphone, webcam, media_chatbot, system_prompt_textbox, voice_choice], outputs=[microphone, webcam, media_chatbot])
                stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[submit_event], queue=False)
                clear_btn.click(fn=clear_history, inputs=None, outputs=[media_chatbot, microphone, webcam])


            with gr.Tab("Offline"):
                chatbot = gr.Chatbot(type="messages", height=650)
                with gr.Row(equal_height=True):
                    audio_input = gr.Audio(sources=["upload"], type="filepath", label="上傳音訊 (長/短)", elem_classes="media-upload", scale=1)
                    image_input = gr.Image(sources=["upload"], type="filepath", label="上傳圖片", elem_classes="media-upload", scale=1, visible=not args.audio_only)
                    video_input = gr.Video(sources=["upload"], label="上傳影片 (短)", elem_classes="media-upload", scale=1, visible=not args.audio_only)
                text_input = gr.Textbox(show_label=False, placeholder="輸入文字...")
                with gr.Row():
                    submit_btn_offline = gr.Button("提交", variant="primary", size="lg")
                    stop_btn_offline = gr.Button("停止", visible=False, size="lg")
                    clear_btn_offline = gr.Button("清除歷史", size="lg")
                def clear_chat_history(): return [], gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None)
                
                submit_event_offline = gr.on(
                    triggers=[submit_btn_offline.click, text_input.submit], 
                    fn=chat_predict_router, 
                    inputs=[text_input, audio_input, image_input, video_input, chatbot, system_prompt_textbox, voice_choice], 
                    outputs=[text_input, audio_input, image_input, video_input, chatbot]
                )
                stop_btn_offline.click(fn=None, inputs=None, outputs=None, cancels=[submit_event_offline], queue=False)
                clear_btn_offline.click(fn=clear_chat_history, inputs=None, outputs=[chatbot, text_input, audio_input, image_input, video_input])

                gr.HTML("""
                    <style>
                        .media-upload { margin: 10px; min-height: 160px; }
                        .media-upload > .wrap { border: 2px dashed #ccc; border-radius: 8px; padding: 10px; height: 100%; }
                        .media-upload:hover > .wrap { border-color: #666; }
                        .media-upload { flex: 1; min-width: 0; }
                    </style>
                """)
    return demo


def _get_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--checkpoint-path', type=str, default="Qwen/Qwen2.5-Omni-7B", help='Checkpoint name or path, default to %(default)r')
    parser.add_argument('--cpu-only', action='store_true', help='Run demo with CPU only')
    parser.add_argument('--flash-attn2', action='store_true', default=False, help='Enable flash_attention_2 when loading the model.')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Demo server name.')
    parser.add_argument('--port', type=int, default=7860, help='Demo server port.')
    parser.add_argument('--audio-only', action='store_true', help='Run in audio-only mode, hiding video components in the UI.')
    parser.add_argument('--segment-duration', type=int, default=60, help='Duration of each audio segment for transcription in seconds.')
    parser.add_argument('--max-new-tokens', type=int, default=1536, help='Maximum new tokens to generate.')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for sampling.')
    parser.add_argument('--repetition-penalty', type=float, default=1.1, help='Repetition penalty.')
    parser.add_argument('--idle-timeout', type=int, default=300, help='Idle timeout in seconds before unloading the model.')
    parser.add_argument('--ui-language', type=str, choices=['en', 'zh'], default='zh', help='Display language for the UI.')
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = _get_args()
    IDLE_TIMEOUT = cli_args.idle_timeout
    gradio_app = _launch_demo(cli_args)
    app = gr.mount_gradio_app(app, gradio_app, path="/")
    
    # 啟動閒置監控線程
    monitor_thread = threading.Thread(target=idle_monitor, daemon=True)
    monitor_thread.start()
    print(f"[INFO] 閒置監控已啟動 (超時時間: {IDLE_TIMEOUT}s)")
    
    print(f"[INFO] 啟動伺服器於 http://{cli_args.host}:{cli_args.port}")
    print(f"[INFO] Gradio UI 介面位於 http://{cli_args.host}:{cli_args.port}/")
    print(f"[INFO] Curl API 端點位於 http://{cli_args.host}:{cli_args.port}/transcribe/")
    import uvicorn
    uvicorn.run(app, host=cli_args.host, port=cli_args.port)

