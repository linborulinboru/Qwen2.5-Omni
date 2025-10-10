"""测试音频数据格式"""
import numpy as np
from qwen_omni_utils import process_mm_info

# 模拟单个音频文件的处理
messages_single = [
    {"role": "system", "content": [{"type": "text", "text": "test"}]},
    {"role": "user", "content": [{"type": "text", "text": "test"}, {"type": "audio", "audio": "/app/inputs/20251010165624_6f35ad24_audio.wav"}]}
]

audios, _, _ = process_mm_info(messages_single, use_audio_in_video=True)

print(f"单个音频处理结果:")
print(f"  type(audios): {type(audios)}")
if audios:
    print(f"  len(audios): {len(audios)}")
    for i, audio in enumerate(audios):
        print(f"  audios[{i}].shape: {audio.shape}, dtype: {audio.dtype}, ndim: {audio.ndim}")
else:
    print(f"  audios is None or empty")

# 模拟批次处理
print("\n批次处理（4个音频）:")
batch_messages = []
for i in range(4):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "test"}]},
        {"role": "user", "content": [{"type": "text", "text": "test"}, {"type": "audio", "audio": f"/app/temp/20251010173455_8c5444af_segment_{i}.wav"}]}
    ]
    batch_messages.append(messages)

audios_list = []
for messages in batch_messages:
    audios, _, _ = process_mm_info(messages, use_audio_in_video=True)
    audios_list.append(audios)
    print(f"  messages {len(audios_list)-1}: audios={type(audios)}, len={len(audios) if audios else 0}")
    if audios and len(audios) > 0:
        print(f"    audios[0].shape={audios[0].shape}, dtype={audios[0].dtype}")

print("\n正确的批次格式应该是:")
print("  [[audio_array1], [audio_array2], [audio_array3], [audio_array4]]")
print("\n当前的audios_list:")
print(f"  len(audios_list) = {len(audios_list)}")
for i, audios in enumerate(audios_list):
    print(f"  audios_list[{i}]: type={type(audios)}, len={len(audios) if audios else 0}")
