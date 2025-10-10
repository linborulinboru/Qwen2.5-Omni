"""
安全模組 - Qwen2.5-Omni 服務
包含請求頻率限制、檔案驗證、MIME 類型檢查等安全功能
"""

import os
import time
import magic  # python-magic for MIME type detection
from pathlib import Path
from collections import defaultdict
from werkzeug.utils import secure_filename
from flask import request, abort


# ==================== 安全配置 ====================

# 檔案大小限制 (4 GB)
MAX_FILE_SIZE = 4 * 1024 * 1024 * 1024  # 4 GB in bytes

# 允許的 MIME 類型 (音頻和視頻)
ALLOWED_MIME_TYPES = {
    # 音頻格式
    'audio/mpeg',       # MP3
    'audio/mp3',
    'audio/wav',
    'audio/x-wav',
    'audio/wave',
    'audio/flac',
    'audio/x-flac',
    'audio/ogg',
    'audio/opus',
    'audio/aac',
    'audio/x-m4a',
    'audio/mp4',
    'audio/x-ms-wma',
    'audio/webm',
    # 視頻格式
    'video/mp4',
    'video/x-msvideo',   # AVI
    'video/x-matroska',  # MKV
    'video/quicktime',   # MOV
    'video/x-flv',
    'video/webm',
    'video/x-ms-wmv',
    'video/3gpp',
}

# 允許的文件擴展名
ALLOWED_EXTENSIONS = {
    'mp3', 'wav', 'flac', 'ogg', 'opus', 'aac', 'm4a', 'wma', 'webm',
    'mp4', 'avi', 'mkv', 'mov', 'flv', 'wmv', '3gp'
}

# 請求頻率限制 (簡單的滑動窗口)
request_history = defaultdict(list)  # {ip: [timestamp1, timestamp2, ...]}
RATE_LIMIT_WINDOW = 60  # 60 秒窗口
RATE_LIMIT_MAX_REQUESTS = 10  # 每 60 秒最多 10 個請求


class SecurityValidator:
    """安全驗證器"""

    @staticmethod
    def check_rate_limit():
        """檢查請求頻率限制"""
        client_ip = request.remote_addr
        current_time = time.time()

        # 清理過期的請求記錄
        request_history[client_ip] = [
            timestamp for timestamp in request_history[client_ip]
            if current_time - timestamp < RATE_LIMIT_WINDOW
        ]

        # 檢查是否超過限制
        if len(request_history[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
            abort(429, description=f"Rate limit exceeded. Maximum {RATE_LIMIT_MAX_REQUESTS} requests per {RATE_LIMIT_WINDOW} seconds.")

        # 記錄當前請求
        request_history[client_ip].append(current_time)

    @staticmethod
    def validate_file_size(file_data):
        """驗證文件大小"""
        if isinstance(file_data, bytes):
            file_size = len(file_data)
        else:
            file_data.seek(0, 2)  # 移動到文件末尾
            file_size = file_data.tell()
            file_data.seek(0)  # 重置到開頭

        if file_size > MAX_FILE_SIZE:
            abort(413, description=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB")

        return file_size

    @staticmethod
    def validate_file_type(file_data, filename):
        """驗證文件類型 (使用 MIME 檢測和擴展名檢查)"""
        # 檢查文件擴展名
        if '.' in filename:
            ext = filename.rsplit('.', 1)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                abort(400, description=f"File type not allowed. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}")
        else:
            abort(400, description="File must have an extension")

        # 檢查 MIME 類型 (使用 python-magic)
        try:
            if isinstance(file_data, bytes):
                mime_type = magic.from_buffer(file_data, mime=True)
            else:
                # 讀取前 2048 字節用於 MIME 檢測
                file_data.seek(0)
                header = file_data.read(2048)
                file_data.seek(0)
                mime_type = magic.from_buffer(header, mime=True)

            if mime_type not in ALLOWED_MIME_TYPES:
                abort(400, description=f"Invalid file type. Detected MIME type: {mime_type}")

            return mime_type
        except Exception as e:
            print(f"MIME type detection warning: {e}")
            # 如果 magic 檢測失敗,依賴擴展名檢查
            return None

    @staticmethod
    def sanitize_filename(filename):
        """安全化文件名,防止路徑遍歷攻擊"""
        # 使用 werkzeug 的 secure_filename
        safe_name = secure_filename(filename)

        # 額外檢查:移除任何路徑分隔符
        safe_name = safe_name.replace('/', '').replace('\\', '')

        # 限制文件名長度
        if len(safe_name) > 255:
            name, ext = os.path.splitext(safe_name)
            safe_name = name[:250] + ext

        return safe_name


def add_security_headers(response):
    """添加安全響應標頭"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response
