#!/usr/bin/env python3
"""即時語音辨識預覽 — 開會時即時看到文字輸出。

用法：
    python3 scripts/live_preview.py              # 預設麥克風
    python3 scripts/live_preview.py --device 0   # 指定裝置
    python3 scripts/live_preview.py --lang zh    # 指定語言

配合 ownscribe 使用：
    終端視窗 1: ownscribe              （錄音 + 會後高品質轉錄）
    終端視窗 2: python3 scripts/live_preview.py  （即時字幕預覽）

按 Ctrl+C 停止。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
import queue
from datetime import timedelta

import numpy as np
import sounddevice as sd


def main():
    parser = argparse.ArgumentParser(description="即時語音辨識預覽")
    parser.add_argument("--device", type=int, default=None, help="音訊裝置 index（用 ownscribe devices 查看）")
    parser.add_argument("--lang", type=str, default="zh", help="語言: zh, en, auto")
    parser.add_argument("--model", type=str, default="paraformer-zh-streaming", help="FunASR 串流模型")
    args = parser.parse_args()

    # Apple Silicon 最佳化
    os.environ.setdefault("OMP_NUM_THREADS", "8")

    print("⏳ 載入即時辨識模型中（首次需下載）...")
    from funasr import AutoModel

    model = AutoModel(model=args.model, disable_update=True)
    print("✅ 模型載入完成")
    print()
    print("=" * 60)
    print("🎙️  即時語音辨識中... (Ctrl+C 停止)")
    print("=" * 60)
    print()

    # Streaming 參數
    chunk_size = [0, 10, 5]  # [left, center, right] in 60ms units → 600ms per chunk
    chunk_stride = chunk_size[1] * 960  # 9600 samples @ 16kHz = 600ms
    sample_rate = 16000
    cache = {}

    # 音訊佇列
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    start_time = time.time()

    def audio_callback(indata, frames, time_info, status):
        """sounddevice 回呼：把音訊 chunk 放入佇列"""
        if status:
            print(f"  ⚠️ {status}", file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    # 累積 buffer（sounddevice blocksize 可能跟 chunk_stride 不完全對齊）
    buffer = np.array([], dtype=np.float32)
    line_count = 0

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_stride,
            device=args.device,
            callback=audio_callback,
        ):
            while True:
                # 從佇列取音訊
                try:
                    chunk = audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                buffer = np.concatenate([buffer, chunk])

                # 確保有足夠的樣本
                while len(buffer) >= chunk_stride:
                    audio_chunk = buffer[:chunk_stride]
                    buffer = buffer[chunk_stride:]

                    # FunASR streaming 推理
                    res = model.generate(
                        input=audio_chunk,
                        cache=cache,
                        is_final=False,
                        chunk_size=chunk_size,
                        encoder_chunk_look_back=4,
                        decoder_chunk_look_back=1,
                    )

                    # 輸出辨識結果
                    if res and res[0] and res[0].get("text"):
                        text = res[0]["text"].strip()
                        if text:
                            elapsed = time.time() - start_time
                            ts = str(timedelta(seconds=int(elapsed)))
                            print(f"  [{ts}] {text}")
                            line_count += 1

    except KeyboardInterrupt:
        # 送出最後的 is_final chunk
        if len(buffer) > 0:
            res = model.generate(
                input=buffer,
                cache=cache,
                is_final=True,
                chunk_size=chunk_size,
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1,
            )
            if res and res[0] and res[0].get("text"):
                text = res[0]["text"].strip()
                if text:
                    elapsed = time.time() - start_time
                    ts = str(timedelta(seconds=int(elapsed)))
                    print(f"  [{ts}] {text}")

        print()
        print("=" * 60)
        duration = time.time() - start_time
        print(f"⏱️  總時長: {timedelta(seconds=int(duration))}")
        print(f"📝 辨識句數: {line_count}")
        print("=" * 60)


if __name__ == "__main__":
    main()
