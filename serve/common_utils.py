"""
共用工具模組 - Qwen2.5-Omni 服務
包含音頻處理、OpenCC 轉換、檔案清理等共用功能
"""

import os
import time
import ffmpeg
import numpy as np
import torch
import psutil
from pathlib import Path
from datetime import datetime
import logging

# 提高 OpenCC 庫的日誌級別以隱藏 INFO 消息
logging.getLogger("opencc").setLevel(logging.WARNING)

# OpenCC 簡繁轉換
try:
    from opencc import OpenCC
    OPENCC_AVAILABLE = True
except ImportError:
    print("Warning: OpenCC not available. Simplified to Traditional Chinese conversion disabled.")
    OPENCC_AVAILABLE = False


class OpenCCConverter:
    """OpenCC 簡繁轉換器"""

    def __init__(self):
        self.converter = None
        if OPENCC_AVAILABLE:
            try:
                self.converter = OpenCC('s2t')
                print("✅ OpenCC initialized successfully (Simplified to Traditional Chinese)")
            except Exception as e:
                print(f"❌ Failed to initialize OpenCC: {e}")

    def convert(self, text):
        """將簡體中文轉換為繁體中文"""
        if not self.converter:
            return text
        try:
            return self.converter.convert(text)
        except Exception as e:
            print(f"Warning: OpenCC conversion failed: {e}")
            return text


class AudioProcessor:
    """音頻處理工具類"""

    @staticmethod
    def detect_media_format(input_file):
        """檢測媒體文件格式和元數據"""
        try:
            probe = ffmpeg.probe(input_file)
            audio_stream = None
            video_stream = None

            for stream in probe['streams']:
                if stream['codec_type'] == 'audio' and not audio_stream:
                    audio_stream = stream
                elif stream['codec_type'] == 'video' and not video_stream:
                    video_stream = stream

            return {
                'has_audio': audio_stream is not None,
                'has_video': video_stream is not None,
                'audio_stream': audio_stream,
                'video_stream': video_stream,
                'format': probe.get('format', {}),
                'duration': float(probe['format'].get('duration', 0))
            }
        except Exception as e:
            print(f"Error detecting media format: {e}")
            return None

    @staticmethod
    def convert_to_wav(input_file, output_file):
        """
        將各種音頻/視頻格式轉換為 WAV 格式
        支持格式:
        - 音頻: MP3, AAC, FLAC, OGG, M4A, WMA, OPUS, WAV, AIFF, APE
        - 視頻: MP4, AVI, MKV, MOV, FLV, WEBM, WMV, 3GP
        """
        try:
            # 檢測媒體格式
            media_info = AudioProcessor.detect_media_format(input_file)

            if not media_info:
                print(f"Failed to probe media file: {input_file}")
                return False

            if not media_info['has_audio']:
                print(f"No audio stream found in file: {input_file}")
                return False

            # 顯示媒體信息
            if media_info['has_video']:
                print(f"Video file detected, extracting audio stream...")
                video_codec = media_info['video_stream'].get('codec_name', 'unknown')
                print(f"Video codec: {video_codec}")

            audio_codec = media_info['audio_stream'].get('codec_name', 'unknown')
            sample_rate = media_info['audio_stream'].get('sample_rate', 'unknown')
            channels = media_info['audio_stream'].get('channels', 'unknown')
            duration = media_info['duration']

            print(f"Audio codec: {audio_codec}, Sample rate: {sample_rate}, Channels: {channels}, Duration: {duration:.2f}s")

            # 轉換為 WAV 格式 (16kHz, 單聲道, 16-bit PCM)
            (
                ffmpeg
                .input(input_file)
                .output(
                    output_file,
                    acodec='pcm_s16le',  # 16-bit PCM
                    ar='16000',          # 16kHz 採樣率
                    ac=1,                # 單聲道
                    loglevel='error'     # 只顯示錯誤
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            print(f"Successfully converted to WAV: {output_file}")
            return True

        except ffmpeg.Error as e:
            error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
            print(f"FFmpeg conversion error: {error_msg}")
            return False
        except Exception as e:
            print(f"Unexpected error during conversion: {e}")
            import traceback
            traceback.print_exc()
            return False

    @staticmethod
    def split_audio(input_wav, request_id, segment_duration=600, temp_dir=None):
        """將音頻分割成多個片段"""
        segments = []
        try:
            probe = ffmpeg.probe(input_wav)
            duration = float(probe['streams'][0]['duration'])

            # 如果音頻時長小於片段時長，直接返回原檔案
            if duration <= segment_duration:
                return [input_wav]

            num_segments = int(np.ceil(duration / segment_duration))

            for i in range(num_segments):
                start_time = i * segment_duration
                segment_file = Path(temp_dir) / f"{request_id}_segment_{i:04d}.wav"
                actual_duration = min(segment_duration, duration - start_time)

                ffmpeg.input(
                    input_wav,
                    ss=start_time,
                    t=actual_duration
                ).output(
                    str(segment_file),
                    acodec='pcm_s16le',
                    ar='16000',
                    ac=1
                ).run(quiet=True, overwrite_output=True)

                segments.append(str(segment_file))

            return segments
        except Exception as e:
            print(f"[{request_id}] Error splitting audio: {e}")
            return []


class FileManager:
    """檔案管理工具類"""

    @staticmethod
    def cleanup_old_files(directory, max_age_hours=24):
        """清理指定目錄中超過指定時間的文件"""
        try:
            now = datetime.now()
            deleted_count = 0
            total_size = 0

            for file_path in Path(directory).glob('*'):
                if file_path.is_file():
                    file_age = now - datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age.total_seconds() > max_age_hours * 3600:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        deleted_count += 1
                        total_size += file_size

            if deleted_count > 0:
                print(f"[{now}] Cleaned up {deleted_count} files ({total_size / 1024**2:.2f} MB) from {directory}")
        except Exception as e:
            print(f"Error during cleanup of {directory}: {e}")


class MemoryMonitor:
    """記憶體監控工具類"""

    @staticmethod
    def get_memory_stats():
        """獲取詳細的內存統計信息"""
        stats = {}

        if torch.cuda.is_available():
            # GPU 內存
            stats["gpu"] = {
                "allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
                "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count()
            }

        # CPU 內存
        process = psutil.Process()
        mem_info = process.memory_info()
        stats["cpu"] = {
            "rss_gb": round(mem_info.rss / 1024**3, 2),  # Resident Set Size
            "vms_gb": round(mem_info.vms / 1024**3, 2),  # Virtual Memory Size
            "percent": process.memory_percent()
        }

        # 系統內存
        sys_mem = psutil.virtual_memory()
        stats["system"] = {
            "total_gb": round(sys_mem.total / 1024**3, 2),
            "available_gb": round(sys_mem.available / 1024**3, 2),
            "used_gb": round(sys_mem.used / 1024**3, 2),
            "percent": sys_mem.percent
        }

        return stats

    @staticmethod
    def print_memory_summary():
        """打印記憶體使用摘要"""
        mem_stats = MemoryMonitor.get_memory_stats()
        gpu_info = mem_stats.get("gpu", {})
        cpu_info = mem_stats.get("cpu", {})
        system_info = mem_stats.get("system", {})

        stats_msg = f"GPU: {gpu_info.get('allocated_gb', 0):.2f}GB/{gpu_info.get('reserved_gb', 0):.2f}GB | "
        stats_msg += f"RAM: {cpu_info.get('rss_gb', 0):.2f}GB | "
        stats_msg += f"System: {system_info.get('used_gb', 0):.2f}GB/{system_info.get('total_gb', 0):.2f}GB"

        print(f"[{datetime.now()}] {stats_msg}")


class IdleChecker:
    """空閒檢查器 - 用於自動卸載模型"""

    def __init__(self, idle_timeout, unload_callback, temp_dir, inputs_dir, outputs_dir):
        """
        Args:
            idle_timeout: 空閒超時時間（秒）
            unload_callback: 卸載模型的回調函數
            temp_dir: 臨時檔案目錄
            inputs_dir: 輸入檔案目錄
            outputs_dir: 輸出檔案目錄
        """
        self.idle_timeout = idle_timeout
        self.unload_callback = unload_callback
        self.temp_dir = temp_dir
        self.inputs_dir = inputs_dir
        self.outputs_dir = outputs_dir
        self.last_activity_time = None
        self.is_processing = False
        self.model_loaded = False

    def update_activity(self):
        """更新最後活動時間"""
        self.last_activity_time = datetime.now()

    def set_processing(self, is_processing):
        """設置處理狀態"""
        self.is_processing = is_processing

    def set_model_loaded(self, loaded):
        """設置模型加載狀態"""
        self.model_loaded = loaded

    def check_loop(self):
        """後台檢查循環"""
        cleanup_counter = 0
        memory_monitor_counter = 0

        while True:
            time.sleep(60)  # 每60秒檢查一次
            cleanup_counter += 1
            memory_monitor_counter += 1

            # 每5分鐘輸出一次記憶體使用情況
            if memory_monitor_counter >= 5 and self.model_loaded:
                try:
                    MemoryMonitor.print_memory_summary()
                    memory_monitor_counter = 0
                except Exception as e:
                    print(f"[{datetime.now()}] Error getting memory stats: {e}")

            # 檢查模型空閒時間（只有在沒有正在處理請求時才檢查）
            if (self.last_activity_time is not None and
                self.model_loaded and
                not self.is_processing):
                idle_time = datetime.now() - self.last_activity_time
                if idle_time.total_seconds() > self.idle_timeout:
                    self.unload_callback()
                    self.last_activity_time = None
                    self.model_loaded = False

            # 每10分鐘清理一次臨時文件
            if cleanup_counter >= 10:
                FileManager.cleanup_old_files(self.temp_dir, max_age_hours=1)
                FileManager.cleanup_old_files(self.inputs_dir, max_age_hours=24)
                FileManager.cleanup_old_files(self.outputs_dir, max_age_hours=168)
                cleanup_counter = 0


def generate_srt(transcription_segments, segment_duration=600):
    """生成 SRT 字幕格式"""
    srt_content = []
    for i, segment_text in enumerate(transcription_segments, 1):
        start_time = (i - 1) * segment_duration
        end_time = i * segment_duration

        # 轉換為 SRT 時間格式 (HH:MM:SS,mmm)
        start_h, start_m = divmod(start_time, 3600)
        start_m, start_s = divmod(start_m, 60)
        end_h, end_m = divmod(end_time, 3600)
        end_m, end_s = divmod(end_m, 60)

        srt_content.append(f"{i}\n")
        srt_content.append(f"{int(start_h):02d}:{int(start_m):02d}:{int(start_s):02d},000 --> {int(end_h):02d}:{int(end_m):02d}:{int(end_s):02d},000\n")
        srt_content.append(f"{segment_text}\n\n")

    return "".join(srt_content)
