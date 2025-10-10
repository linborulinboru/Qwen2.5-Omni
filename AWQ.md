# Qwen2.5-Omni AWQ 服務架構文檔

## 概述

本文檔詳細記錄 Qwen2.5-Omni AWQ 量化版本服務的所有模塊結構、功能和已知問題，用於後續代碼重構參考。

**版本**: AWQ 量化版本
**文件**: `serve/qwen2_5_omni_awq.py`
**目的**: 音頻轉錄 HTTP 服務
**狀態**: ❌ 批次處理存在問題，當前使用逐個處理方式（仍無法正常工作）

---

## 核心依賴項

### 系統依賴
```python
import io, os, sys, tempfile
import numpy as np
import torch
import importlib.util
import threading, uuid
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from argparse import ArgumentParser
from huggingface_hub import hf_hub_download
```

### AWQ 量化相關
```python
from awq.models.base import BaseAWQForCausalLM
from transformers import Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info  # ⚠️ 已知問題
```

### 自定義模組
```python
# 低顯存模式
from modeling_qwen2_5_omni_low_VRAM_mode import (
    Qwen2_5OmniDecoderLayer,
    Qwen2_5OmniForConditionalGeneration
)

# 共用工具
from common_utils import (
    OpenCCConverter,    # 簡繁轉換
    AudioProcessor,     # 音頻處理
    IdleChecker,        # 空閒檢測
    generate_srt        # SRT生成
)

from security import SecurityValidator, add_security_headers
```

---

## 全局配置

### Flask 應用
```python
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 * 1024  # 4GB
```

### 全局變量
| 變量 | 類型 | 用途 |
|------|------|------|
| `model` | AWQ Model | 量化模型實例 |
| `processor` | Qwen2_5OmniProcessor | 文本/音頻處理器 |
| `opencc_converter` | OpenCCConverter | 簡繁轉換 |
| `model_lock` | Lock | 模型訪問鎖 |
| `processing_lock` | Lock | 請求序列化鎖 |
| `idle_checker` | IdleChecker | 空閒監控 |
| `job_status` | dict | 異步任務狀態 |

### 目錄結構
```
inputs/   # 上傳文件
temp/     # 臨時音頻片段
outputs/  # 轉錄結果
```

---

## 核心模塊

### 1. 模型加載

#### `replace_transformers_module()`
替換 transformers 模組以支持低顯存模式。

**工作原理**:
1. 刪除標準模組: `del sys.modules['transformers.models.qwen2_5_omni.modeling_qwen2_5_omni']`
2. 加載自定義模組: `low-VRAM-mode/modeling_qwen2_5_omni_low_VRAM_mode.py`
3. 注入系統: `sys.modules[original_mod_name] = new_mod`

#### `Qwen2_5_OmniAWQForConditionalGeneration`
AWQ 量化模型包裝類。

**關鍵配置**:
```python
layer_type = "Qwen2_5OmniDecoderLayer"
max_seq_len_key = "max_position_embeddings"
modules_to_not_convert = ["visual"]  # 視覺模組不量化
```

**關鍵方法**:
- `get_model_layers()`: 返回可量化層
- `move_embed()`: 移動嵌入層到 CUDA
- `get_layers_for_scaling()`: 配置量化層

#### `load_model_processor(checkpoint_path, flash_attn2, local_model)`
加載模型主流程。

**流程**:
```
1. replace_transformers_module()
2. 加載 AWQ 量化模型
   ├─ flash_attn2=True → Flash Attention 2
   └─ flash_attn2=False → 標準 Attention
3. 加載 speaker dictionary (spk_dict.pt)
4. 移動組件到 CUDA:
   ├─ embed_tokens
   ├─ visual
   ├─ audio_tower
   └─ rotary_emb (所有層)
5. 加載 processor
6. 打印 GPU 內存使用
```

**GPU 內存分配**: ~11GB (7B AWQ 模型)

#### `unload_model()`
卸載模型並清理 CUDA 緩存。由 `IdleChecker` 在空閒超時後自動觸發。

---

### 2. 音頻處理

#### `transcribe_single_audio(audio_path, request_id, enable_s2t)` ⭐
核心轉錄函數。

**當前實現** (使用 librosa):
```python
# 1. 構建 prompt
system_prompt = "You are Qwen..."
user_prompt = "你的唯一任務是將音頻內容轉錄為文本..."

# 2. 使用 librosa 加載音頻（繞過 process_mm_info）
audio_array, sr = librosa.load(audio_path, sr=16000)

# 3. 轉單聲道
if audio_array.ndim > 1:
    audio_array = audio_array.mean(axis=1)

# 4. 包裝格式
audios = [[audio_array]]

# 5. 處理
inputs = processor(
    text=[text],
    audio=audios,
    return_tensors="pt"
).to(device).to(model.dtype)

# 6. 生成
text_ids = model.generate(**inputs, use_audio_in_video=True, ...)

# 7. 解碼
response = processor.batch_decode(text_ids, ...)[0]

# 8. 後處理
- 移除生成標記
- 移除結束語
- 簡繁轉換（optional）

# 9. 清理 CUDA 緩存
```

**已知問題**:
- ❌ 仍然出現 `ValueError: axes don't match array`
- ❌ 無法正常工作
- ⚠️ 問題可能在 processor 內部的 Whisper feature extractor

#### `transcribe_audio_batch(audio_paths, request_id)` ⚠️ 已廢棄
批次處理已放棄，改為循環調用單個處理。

**當前實現**:
```python
results = []
for audio_path in audio_paths:
    try:
        result = transcribe_single_audio(audio_path, request_id)
        results.append(result)
    except Exception as e:
        results.append(f"[Error: {str(e)}]")
return results
```

**原因**: processor 批次處理數據格式不兼容

#### `process_full_audio(audio_binary_data, original_filename)`
完整音頻處理流程。

**流程圖**:
```
1. 生成 request_id: {timestamp}_{uuid}
2. 保存上傳文件 → inputs/
3. 轉換 WAV → temp/{request_id}_audio.wav
4. 分割音頻 → temp/{request_id}_segment_*.wav
   └─ 默認片段長度: 600秒
5. 動態調整批次大小:
   ├─ GPU 剩餘 < 8GB  → batch_size = 2
   ├─ GPU 剩餘 < 16GB → batch_size = 4
   └─ GPU 剩餘 ≥ 16GB → batch_size = 8
6. 分批處理:
   ├─ transcribe_audio_batch()
   ├─ 清理已處理片段
   └─ 清理 CUDA 緩存
7. 合併轉錄結果
8. 保存 → outputs/{timestamp}.txt
9. 清理臨時 WAV 文件
```

**性能監控**:
- 每批後打印 GPU 內存
- CUDA 緩存自動清理

---

### 3. API 端點

#### `POST /transcribe`
返回文本文件下載。

**輸入**:
- `file`: multipart/form-data
- 或 raw binary

**輸出**: `audio.txt` (application/octet-stream)

**安全**: 文件名消毒、大小驗證、類型驗證、速率限制

#### `POST /transcribe/json`
返回 JSON 格式結果。

**輸出**:
```json
{
  "status": "success",
  "transcription": "...",
  "output_file": "20251010123456.txt",
  "timestamp": "2025-10-10T12:34:56.789"
}
```

#### `POST /transcribe/srt`
返回 SRT 字幕文件。

**特殊邏輯**:
- 保留時間戳
- 基於 `segment_duration` 計算時間軸
- 使用 `generate_srt()` 生成格式

#### `POST /transcribe/async`
異步轉錄，立即返回 job_id。

**流程**:
```
1. 驗證輸入
2. 生成 job_id
3. 啟動後台線程
4. 返回 202 Accepted
```

**返回**:
```json
{
  "status": "accepted",
  "job_id": "...",
  "message": "Use /status/<job_id> to check progress."
}
```

#### `GET /status/<job_id>`
查詢異步任務狀態。

**狀態**: `processing` | `completed` | `failed`

#### `GET /health`
健康檢查。

**返回**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "gpu_memory_allocated_gb": 11.05,
  "gpu_memory_reserved_gb": 12.54
}
```

---

### 4. 後處理

#### 文本清理
1. 移除生成標記: `<|im_start|>assistant\n`
2. 移除結束語:
   - "如果你還有其他關於音頻轉寫或者內容理解的問題..."
   - "如果你還有其他關於這方面的問題..."
3. 簡繁轉換 (OpenCC: s2t)

#### SRT 生成
格式:
```srt
1
00:00:00,000 --> 00:10:00,000
第一段轉錄文本

2
00:10:00,000 --> 00:20:00,000
第二段轉錄文本
```

---

### 5. 並發控制

#### 鎖機制
| 鎖 | 用途 |
|----|------|
| `model_lock` | 保護模型加載/卸載 |
| `processing_lock` | 序列化請求處理 |
| `job_lock` | 保護異步任務狀態 |

#### IdleChecker
監控空閒時間，自動卸載模型。

**配置**:
- `idle_timeout`: 300秒
- `unload_callback`: `unload_model()`

**工作方式**:
```
後台線程持續檢查 → 超時 → 卸載模型 → 下次請求重新加載
```

---

## 命令行參數

| 參數 | 默認值 | 說明 |
|------|--------|------|
| `--checkpoint-path` | Qwen/Qwen2.5-Omni-7B-AWQ | 模型路徑 |
| `--local-model` | False | 使用本地模型 |
| `--flash-attn2` | False | Flash Attention 2 |
| `--segment-duration` | 600 | 片段長度(秒) |
| `--batch-size` | 4 | 批次大小 ⚠️已廢棄 |
| `--max-new-tokens` | 8192 | 最大生成 tokens |
| `--temperature` | 0.1 | 採樣溫度 |
| `--repetition-penalty` | 1.1 | 重複懲罰 |
| `--idle-timeout` | 300 | 空閒超時(秒) |
| `--host` | 0.0.0.0 | 監聽地址 |
| `--port` | 5000 | 服務端口 |

---

## 已知問題

### ❌ 批次處理失敗
**錯誤**:
```
ValueError: axes don't match array
ValueError: operands could not be broadcast together
```

**位置**: `transformers/models/whisper/feature_extraction_whisper.py:312`

**操作**: `input_features.transpose(2, 0, 1)`

**根本原因**:
- `process_mm_info` 返回格式與 Whisper feature extractor 不兼容
- 可能是 `qwen-omni-utils` 或 `transformers` 版本問題

**嘗試過的修復**:
1. ❌ 手動填充音頻到相同長度
2. ❌ 多聲道轉單聲道
3. ❌ 調整數據結構 `[[audio1], [audio2], ...]`
4. ❌ 使用 `librosa.load()` 繞過 `process_mm_info`
5. ❌ 移除 `use_audio_in_video` 參數

**當前狀態**: ❌ 完全無法工作

### ⚠️ 性能影響
- 逐個處理無法利用 GPU 批次優勢
- 長音頻處理時間顯著增加

### ⚠️ 內存管理
- 需要頻繁清理 CUDA 緩存
- 長時間運行可能內存碎片化

---

## 重構建議

### 🔴 優先級 1: 修復音頻處理
1. **深入調查 processor**:
   - 查看 `Qwen2_5OmniProcessor` 源代碼
   - 確認正確音頻數據結構
   - 測試不同數據格式

2. **替代方案**:
   - 完全繞過 `process_mm_info`
   - 直接用 `librosa` + `numpy`
   - 自定義音頻預處理

3. **版本測試**:
   - 不同 `transformers` 版本
   - 不同 `qwen-omni-utils` 版本
   - 降級到已知可工作版本

### 🟡 優先級 2: 代碼結構
**分離關注點**:
```
audio_loader.py      # 音頻加載
model_handler.py     # 模型管理
api_routes.py        # Flask 路由
config.py            # 配置
transcriber.py       # 轉錄邏輯
```

**統一錯誤處理**:
- 自定義異常類
- 統一錯誤響應格式
- 詳細錯誤日誌

**改進並發**:
- 隊列系統 (Celery, RQ)
- 真正異步處理
- 支持並發請求

### 🟢 優先級 3: 功能增強
1. 進度追蹤 (WebSocket)
2. 結果緩存
3. 預熱模型
4. 音頻流式處理
5. 多 GPU 支持

---

## 依賴的外部模組

### `common_utils.py`
- `OpenCCConverter`: 簡繁轉換
- `AudioProcessor`: ffmpeg 音頻轉換與分割
- `IdleChecker`: 空閒監控
- `generate_srt`: SRT 生成

### `security.py`
- `sanitize_filename()`: 文件名消毒
- `validate_file_size()`: 大小驗證
- `validate_file_type()`: 類型驗證
- `check_rate_limit()`: 速率限制
- `add_security_headers()`: 安全標頭

### `qwen_omni_utils`
```python
process_mm_info(messages, use_audio_in_video)
# 返回: (audios, images, videos)
# ❌ 問題: 返回格式不兼容
```

---

## 維護日誌

### 2025-10-10
- ❌ 批次處理失敗，改為逐個處理
- ❌ `process_mm_info` 數據格式問題未解決
- ❌ 使用 `librosa` 直接加載仍然失敗
- ✅ 創建此文檔用於重構參考
- ⚠️ 當前版本完全無法工作

**下一步行動**:
1. 查看 `Qwen2_5OmniProcessor` 源代碼
2. 測試不同 `transformers` 版本
3. 考慮完全重寫音頻處理流程
4. 尋找官方示例代碼參考

---

## 技術規格

**運行環境**:
- Python 3.10
- PyTorch 2.8.0
- Transformers 4.52.3
- AWQ 0.2.9
- Librosa 0.11.0

**硬件需求**:
- NVIDIA GPU (12GB+ VRAM)
- CUDA 12.8+
- 32GB+ RAM 推薦

**Docker部署**:
- 基礎鏡像: `nvidia/cuda:12.8.0-devel-ubuntu22.04`
- 端口: 5000
- 掛載: `./app/serve`, `./app/inputs`, `./app/temp`, `./app/outputs`

---

## 結論

當前 AWQ 服務存在嚴重的音頻處理問題，**完全無法正常工作**。批次處理已被放棄，但逐個處理仍然失敗。

問題根源可能在於：
1. `process_mm_info` 與 `Qwen2_5OmniProcessor` 之間的數據格式不匹配
2. Whisper feature extractor 版本兼容性問題
3. AWQ 量化後的模型處理流程與標準版本不同
