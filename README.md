# ownscribe (Chinese ASR Fork)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![macOS 26+](https://img.shields.io/badge/macOS-26+-black?logo=apple)](https://developer.apple.com/macos/)

> Fork of [paberr/ownscribe](https://github.com/paberr/ownscribe) v0.13.0 — extended with multi-backend Chinese ASR, real-time transcription, and speaker diarization optimized for **Taiwanese Mandarin + English code-switching**.

本地端會議記錄工具，專為**台灣中文 + 中英混合**會議場景設計。即時字幕、錄音保存、說話者辨識、繁體中文輸出 — 全部在你的 Mac 上完成，不上雲。

---

## 與原專案的差異

| | 原版 ownscribe | 本 Fork |
|---|---|---|
| ASR 引擎 | WhisperX（中文 CER ~20%） | **4 後端可選**（最低 CER 3%） |
| 中文優化 | 無 | ✅ 台灣國語 + 中英混合 + 方言 |
| 繁體輸出 | 無 | ✅ 原生繁體 / OpenCC 轉換 |
| 即時字幕 | 無 | ✅ `ownscribe live` |
| 說話者辨識 | pyannote（需 HF token） | ✅ CAM++（免 token） |
| 原有功能 | — | 100% 保留，向後相容 |

---

## 功能特色

- 🎙️ **即時會議模式** — 邊開會邊看繁體字幕，結束後自動產出帶說話者的完整記錄
- 🇹🇼 **台灣國語最佳化** — Breeze-ASR-25 專為台灣中文 + 中英混合訓練
- 👥 **說話者辨識** — CAM++ 自動辨識誰在說話，不需要 HuggingFace token
- 🔒 **完全地端** — 所有音訊、轉錄、模型都在本機，不上傳任何資料
- ⚡ **Apple Silicon 加速** — MPS GPU 加速，M1 Pro 以上流暢運作

---

## ASR 模型比較

| 後端 | 模型 | 中文 CER | 速度 (M5 Pro) | 繁體 | 說話者辨識 | 最適合 |
|------|------|---------|--------------|------|-----------|--------|
| `breeze` ⭐ | MediaTek Breeze-ASR-25 + CAM++ | ~8% | 5.2x (MPS) | ✅ 原生 | ✅ | **台灣中英混合會議** |
| `firered` | FireRedASR2-AED + CAM++ | **3.05%** | 2.3x (MPS) | OpenCC | ✅ | 最高中文準確度 |
| `funasr` | FunASR SenseVoice + CAM++ | 7.81% | 17x (CPU) | OpenCC | ✅ | 快速處理、低資源 |
| `whisperx` | WhisperX + pyannote | ~20% | 13x | ❌ | ✅ | 英文 / 原有行為 |

---

## 系統需求

| 需求 | 最低 | 建議 |
|------|------|------|
| macOS | 26.0+ | 26.0+ |
| 晶片 | Apple M1 | M3 Pro 以上 |
| RAM | 16 GB | 32 GB+ |
| 磁碟 | 20 GB（Breeze 方案） | 30 GB+（全裝） |
| Python | 3.12+ | 3.12+ |

---

## 安裝

### 1. Clone 並建立環境

```bash
git clone git@github.com:nczz/ownscribe.git
cd ownscribe
brew install uv ffmpeg
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[all]"
```

### 2. 安裝中文 ASR 依賴

```bash
# FunASR（即時串流 + SenseVoice + CAM++ 說話者辨識）
uv pip install funasr modelscope opencc-python-reimplemented

# Breeze-ASR-25（台灣中文 + 中英混合）— 需要 transformers
uv pip install transformers
```

### 3. 下載模型

```bash
# 下載到專案 models/ 目錄
python -c "
from huggingface_hub import snapshot_download

# FunASR 基礎元件（VAD + 說話者辨識 + 標點 + 即時串流）
for repo, name in [
    ('FunAudioLLM/SenseVoiceSmall', 'models/sensevoice'),
    ('funasr/fsmn-vad', 'models/fsmn-vad'),
    ('funasr/ct-punc', 'models/ct-punc'),
    ('funasr/campplus', 'models/campplus'),
    ('funasr/paraformer-zh-streaming', 'models/paraformer-zh-streaming'),
]:
    print(f'Downloading {repo}...')
    snapshot_download(repo, local_dir=name)

# Breeze-ASR-25（台灣中文主模型）
print('Downloading Breeze-ASR-25...')
snapshot_download('MediaTek-Research/Breeze-ASR-25', local_dir='models/breeze-asr-25')
print('Done!')
"
```

### 4. 設定

```bash
mkdir -p ~/.config/ownscribe
cat > ~/.config/ownscribe/config.toml << 'EOF'
[audio]
backend = "coreaudio"
mic = true
capture_mode = "all"
silence_timeout = 600

[transcription]
asr_backend = "breeze"
language = ""

[diarization]
enabled = true

[summarization]
enabled = false

[output]
dir = "~/ownscribe"
format = "markdown"
keep_recording = true
EOF
```

### 5. macOS 權限

系統設定 → 隱私權與安全性 → 螢幕錄製 → 啟用你的終端 app（Terminal/iTerm2/VS Code），然後重啟終端。

---

## 使用方式

### 即時會議模式（推薦）

```bash
ownscribe live
```

一個指令同時做：
1. ✅ 即時繁體字幕（paraformer-zh-streaming，~600ms 延遲）
2. ✅ 系統音訊 + 麥克風錄音
3. ✅ Ctrl+C 結束後 → Breeze-ASR-25 精修 + CAM++ 說話者辨識
4. ✅ 輸出 `~/ownscribe/YYYY-MM-DD_HHMM/transcript.md`

```
⏳ 載入即時辨識模型...
✅ 即時辨識模型就緒
📁 錄音將存到: ~/ownscribe/2026-07-13_1400/recording.wav

============================================================
🎙️  即時會議模式 (Ctrl+C 結束)
   ✅ 即時字幕  ✅ 錄音中  ✅ 會後轉錄
============================================================

  [0:00:03] 好各位今天的會議主要討論
  [0:00:06] Q3 產品計畫
  [0:00:09] 核心功能已完成約百分之八十
  ^C

🔄 開始精確轉錄（breeze + 說話者辨識）...
✅ 轉錄完成！
📄 逐字稿: ~/ownscribe/2026-07-13_1400/transcript.md
   說話者: 3 人
   總句數: 42
```

### 轉錄已有錄音

```bash
ownscribe transcribe ~/path/to/meeting.wav
```

### 只看即時字幕（不錄音）

```bash
ownscribe live --no-record
```

### 切換 ASR 後端

編輯 `~/.config/ownscribe/config.toml`：

```toml
[transcription]
asr_backend = "firered"   # 要最高中文準確度時
# asr_backend = "breeze"  # 台灣中英混合（預設）
# asr_backend = "funasr"  # 最快、低資源
```

### 搜尋歷史會議

```bash
ownscribe ask "上次討論的 deadline 是什麼時候？"
```

---

## 模型詳細說明

### Breeze-ASR-25（預設）

MediaTek Research 開發，專為台灣國語 + 中英混合優化。基於 Whisper-large-v2 微調。

- **原生繁體中文輸出**
- 中英混合辨識超強（改善 56% vs Whisper）
- 李宏毅教授指導、NVIDIA Taipei-1 超算訓練
- Apache 2.0 授權
- [HuggingFace](https://huggingface.co/MediaTek-Research/Breeze-ASR-25) | [Paper](https://arxiv.org/abs/2506.11130)

### FireRedASR2-AED（最精確）

小紅書 FireRed Team 開發，2026 年中文 ASR 公開基準冠軍。

- CER 3.05%（超越 Qwen3-ASR、FunASR、Doubao）
- 支援 20+ 中國方言
- 內建 VAD + LID + 標點
- 需要 monkey-patch 才能在 Apple MPS 上跑
- Apache 2.0 授權
- [HuggingFace](https://huggingface.co/FireRedTeam/FireRedASR2-AED) | [GitHub](https://github.com/FireRedTeam/FireRedASR2S)

### FunASR SenseVoice（最快）

阿里巴巴通義實驗室開發，非自回歸架構，CPU 也能 17x realtime。

- CER 7.81%（比 Whisper 好一倍以上）
- CAM++ 說話者辨識內建
- 支援情緒偵測、語言辨識
- MIT 授權
- [GitHub](https://github.com/modelscope/FunASR) | [HuggingFace](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)

---

## 說話者辨識

使用 FunASR 的 **CAM++** 模型，完全不需要 HuggingFace token：

1. FSMN-VAD 切出語音區段
2. CAM++ 對每段提取 speaker embedding（192 維向量）
3. Cosine similarity 聚類（threshold 0.7）
4. 時間戳對齊到 ASR 結果的每個句子

---

## 即時串流辨識

使用 FunASR 的 `paraformer-zh-streaming`：

- 600ms chunk，~600ms 延遲
- CPU 即可即時處理
- 透過 OpenCC `s2twp` 即時轉繁體
- 獨立於會後精修引擎（互不干擾）

---

## 專案結構

```
src/ownscribe/
├── audio/                    錄音層（Core Audio / sounddevice）
├── transcription/            ASR 層（4 後端）
│   ├── base.py                  Transcriber 抽象介面
│   ├── whisperx_transcriber.py  原有 WhisperX
│   ├── funasr_transcriber.py    FunASR SenseVoice + CAM++
│   ├── breeze_transcriber.py    Breeze-ASR-25 + CAM++
│   └── firered_transcriber.py   FireRedASR2-AED + CAM++
├── summarization/            LLM 摘要層
├── output/                   Markdown / JSON 輸出
├── pipeline.py               主 pipeline
├── pipeline_live.py          即時會議 pipeline
├── cli.py                    CLI 入口
└── config.py                 設定 + 模型路徑管理

models/                       模型目錄（gitignored）
```

詳細架構見 [ARCHITECTURE.md](ARCHITECTURE.md)。

---

## 致謝

- **[paberr/ownscribe](https://github.com/paberr/ownscribe)** — 本專案的基礎，由 Pascal Berrang 開發（MIT License）
- **[MediaTek Research](https://huggingface.co/MediaTek-Research)** — Breeze-ASR-25 模型
- **[FireRed Team](https://huggingface.co/FireRedTeam)** — FireRedASR2 模型
- **[FunAudioLLM / Tongyi Lab](https://github.com/modelscope/FunASR)** — FunASR、SenseVoice、CAM++
- **[OpenCC](https://github.com/BYVoid/OpenCC)** — 簡繁轉換

---

## License

MIT — 與原專案相同。見 [LICENSE](LICENSE)。
