# ownscribe (Chinese ASR Fork)

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![macOS 26+](https://img.shields.io/badge/macOS-26+-black?logo=apple)](https://developer.apple.com/macos/)

> Fork of [paberr/ownscribe](https://github.com/paberr/ownscribe) v0.13.0 — extended with multi-backend Chinese ASR, real-time transcription, and speaker diarization optimized for **Taiwanese Mandarin + English code-switching**.

本地端會議記錄工具，專為**台灣中文 + 中英混合**會議場景設計。預設的即時字幕、錄音、轉錄、說話者辨識與摘要都在 Mac 上完成；只有使用者主動選擇遠端 OpenAI-compatible 摘要 endpoint 時，逐字稿才會送到該 endpoint。

---

## 與原專案的差異

| | 原版 ownscribe | 本 Fork |
|---|---|---|
| ASR 引擎 | WhisperX | **4 後端可選** |
| 中文優化 | 無 | ✅ 台灣國語 + 中英混合 + 方言 |
| 繁體輸出 | 無 | ✅ 原生繁體 / OpenCC 轉換 |
| 即時字幕 | 無 | ✅ `ownscribe live` |
| 說話者辨識 | pyannote（需 HF token） | ✅ Community-1（高品質）／CAM++（免 token） |
| 原有功能 | — | 保留原 CLI 與 WhisperX 路徑，新增功能以設定切換 |

---

## 功能特色

- 🎙️ **即時會議模式** — 邊開會邊看繁體字幕，結束後自動產出帶說話者的完整記錄
- 🇹🇼 **台灣國語最佳化** — Breeze-ASR-25 專為台灣中文 + 中英混合訓練
- 👥 **說話者辨識** — Community-1 高品質全局辨識；沒有 HF token 時自動使用 CAM++
- 🔒 **預設完全地端** — 音訊、轉錄與 speaker 模型不會上傳；遠端摘要 backend 是明確的 opt-in
- ⚡ **Apple Silicon 支援** — Breeze／WhisperX 可使用 MPS；FunASR 預設 CPU；FireRed 整合目前使用 CPU

---

## ASR 模型比較

| 後端 | 模型 | 執行裝置 | 繁體 | 說話者辨識 | 最適合 |
|------|------|----------|------|-----------|--------|
| `breeze` ⭐ | MediaTek Breeze-ASR-25 | MPS / CPU | ✅ 原生 | Community-1／CAM++ | **台灣中英混合會議** |
| `firered` | FireRedASR2-AED | CPU | OpenCC | Community-1／CAM++ | 研究與品質比較 |
| `funasr` | FunASR SenseVoice | CPU | OpenCC | Community-1／原生 CAM++ | 快速處理、低資源 |
| `whisperx` | WhisperX | Metal / CPU | ❌ | Community-1／原生 pyannote | 英文 / 原有行為 |

> 各上游發布的 CER 與速度使用不同資料集、硬體及正規化方法，不能直接橫向排名。本專案尚未提供統一 benchmark，因此不宣稱某後端是絕對冠軍。

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
uv sync --all-extras
```

`--all-extras` 會安裝並鎖定中文 ASR、Ollama、OpenAI-compatible 與 FireRed 整合所需套件。只需中文後端可使用 `uv sync --extra chinese`。

### 2. Community-1 存取權（高品質 speaker diarization）

先在 [Community-1 模型頁](https://huggingface.co/pyannote/speaker-diarization-community-1) 接受條款，再建立 read-only Hugging Face token。建議以系統環境變數提供，不要把真實 token 寫進 repo：

```bash
export HF_TOKEN=hf_your_read_only_token
```

沒有 token 仍可使用；`backend = "auto"` 會退回各 ASR backend 的原生 diarization。

### 3. 設定

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
chunk_seconds = 60        # 長錄音分塊處理；降低數值可進一步壓低峰值 RAM，最小 30

[diarization]
enabled = true
backend = "auto"          # 有 HF_TOKEN 時使用 Community-1，否則使用 ASR backend 原生方案
device = "mps"
min_speakers = 1
max_speakers = 8
telemetry = false
speaker_threshold = 0.7               # 僅原生 CAM++ 使用
window_seconds = 600
window_overlap_seconds = 30
community_speaker_threshold = 0.55
segmentation_batch_size = 4
embedding_batch_size = 8

[summarization]
enabled = false

[output]
dir = "~/ownscribe"
format = "markdown"
keep_recording = true
EOF
```

`min_speakers = 0` 表示自動偵測；若確定至少有幾人可設定下限。`max_speakers = 8` 同時是跨 window 的全局安全上限，可避免噪音造成 speaker labels 爆增。

Community-1 預設使用 MPS。M5 Pro 實測在停用 CPU fallback 時輸出與 CPU 完全一致，Metal warm-up 後純推論約快 3.5 倍；若環境或未來依賴版本不支援 MPS，可明確設定 `device = "cpu"`。`segmentation_batch_size` 與 `embedding_batch_size` 越大不保證越快，README 的 4/8 是目前 M5 Pro 實測值。

### 4. 預先下載模型（建議）

設定完成後，以正式 pipeline 的 warmup 路徑下載目前選用的 ASR、alignment、Community-1 與本機摘要模型：

```bash
ownscribe warmup --with-diarization
```

模型會寫入設定的 cache/model 目錄；下載完成後 Community-1 可離線執行。若使用 `backend = "community"` 但沒有 token，程式會在載入 ASR 前直接失敗；`auto` 則會安全 fallback。

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
3. ✅ Ctrl+C 結束後 → 使用設定的 ASR backend 精修，並遵守 `diarization.enabled`
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

Starting accurate transcription (breeze, with speaker diarization)...
✅ 轉錄完成！
📄 逐字稿: ~/ownscribe/2026-07-13_1400/transcript.md
   說話者: 3 人
   總句數: 42
```

### 轉錄已有錄音

```bash
ownscribe transcribe ~/path/to/meeting.wav
```

### 常用管理命令

```bash
ownscribe warmup --with-diarization  # 預抓目前設定使用的模型
ownscribe config                     # 開啟使用者設定檔
ownscribe resume ~/ownscribe/...     # 繼續未完成的 meeting pipeline
ownscribe summarize transcript.md    # 只重新產生摘要
ownscribe ask "上次的決議是什麼？"    # 搜尋歷史會議
```

### 只看即時字幕（不錄音）

```bash
ownscribe live --no-record
```

### 切換 ASR 後端

編輯 `~/.config/ownscribe/config.toml`：

```toml
[transcription]
asr_backend = "firered"   # FireRed 品質比較／研究整合
# asr_backend = "breeze"  # 台灣中英混合（本 fork 建議）
# asr_backend = "funasr"  # 最快、低資源
```

### 搜尋歷史會議

```bash
ownscribe ask "上次討論的 deadline 是什麼時候？"
```

---

## 模型詳細說明

### Breeze-ASR-25（本 fork 建議）

MediaTek Research 開發，專為台灣國語 + 中英混合優化。基於 Whisper-large-v2 微調。

- **原生繁體中文輸出**
- 針對台灣國語與中英 code-switching 訓練；實際品質應以自己的會議語料驗證
- 李宏毅教授指導、NVIDIA Taipei-1 超算訓練
- Apache 2.0 授權
- [HuggingFace](https://huggingface.co/MediaTek-Research/Breeze-ASR-25) | [Paper](https://arxiv.org/abs/2506.11130)

### FireRedASR2-AED（實驗性外部整合）

小紅書 FireRed Team 開發。FireRedASR2S 不是本套件的一部分，必須另外取得相容 checkout，且目前只啟用官方可支援的 CPU 路徑；不再以全域 `.cuda()` monkey-patch 模擬 MPS。

```toml
[transcription]
asr_backend = "firered"
firered_repo = "/absolute/path/to/FireRedASR2S"
```

- 上游提供多項公開 benchmark；請勿與不同資料集上的數字直接比較
- 支援 20+ 中國方言
- 內建 VAD + LID + 標點
- Apache 2.0 授權
- [HuggingFace](https://huggingface.co/FireRedTeam/FireRedASR2-AED) | [GitHub](https://github.com/FireRedTeam/FireRedASR2S)

### FunASR SenseVoice（最快）

阿里巴巴通義實驗室開發，採非自回歸架構；實際速度取決於音訊、硬體與 pipeline 設定。

- 上游提供 CER 與速度 benchmark；本專案不將跨資料集數字作為品質保證
- CAM++ 說話者辨識內建
- 支援情緒偵測、語言辨識
- MIT 授權
- [GitHub](https://github.com/modelscope/FunASR) | [HuggingFace](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)

### WhisperX（相容原版）

保留原版 WhisperX transcription、alignment 與 pyannote native diarization。適合英文或既有 Whisper 工作流程；設定 Community-1 時，WhisperX 只負責 ASR/alignment，speaker timeline 由統一 diarization 階段產生。

---

## 說話者辨識

高品質預設是本機執行的 [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)。設定 `HF_TOKEN` 並接受模型條款後，`backend = "auto"` 會讓所有 ASR backend 共用同一套 Community-1 speaker timeline；錄音不會上傳。模型下載後可離線執行，telemetry 預設關閉。

```bash
export HF_TOKEN=hf_your_read_only_token
```

若沒有 token，`auto` 會保留原生方案：Breeze／FireRed 使用 FSMN-VAD + CAM++、FunASR 使用上游 CAM++，WhisperX 使用既有 pyannote pipeline。也可明確設定 `backend = "native"`。

| `diarization.backend` | 行為 |
|---|---|
| `auto`（預設） | 有 `HF_TOKEN` 使用 Community-1；沒有則使用 ASR backend 原生方案 |
| `community` | 強制 Community-1；缺少 token 時在 ASR 前失敗 |
| `native` | 強制 CAM++／WhisperX 原生方案，不載入 Community-1 |

Community-1 路徑使用：

1. speaker-change／overlap-aware segmentation
2. WeSpeaker embeddings 與 VBx clustering
3. exclusive diarization 對齊 ASR word／segment timestamps
4. 重疊 window 的 timeline 優先對接；未出現在 overlap 的 speaker 再以全局 embedding centroid 合併

原生 CAM++ 路徑則使用：

1. FSMN-VAD 切出語音區段
2. CAM++ 對每段提取 speaker embedding（192 維向量）
3. Cosine similarity 聚類（threshold 0.7）
4. 時間戳對齊到 ASR 結果的每個句子

## 長錄音與記憶體

所有 ASR 後端都以 bounded chunks 解碼，不會再把完整錄音一次載入 RAM。預設每次最多處理 60 秒音訊，可在 `[transcription]` 以 `chunk_seconds` 調整，最小值為 30 秒。

Community-1 也不會載入整場錄音：預設使用 600 秒 window 與 30 秒 overlap，前一窗釋放後才讀取下一窗，只跨窗保留 speaker embeddings 與 timeline。可用 `diarization.window_seconds` 調低記憶體上限。ASR 模型會在 Community-1 載入前釋放，避免兩套大型模型同時常駐。

## 處理架構

一般轉錄、`resume` 與 `live` 會後精修共用同一條準確模式：

```text
錄音／既有音檔
  → ASR backend 以 60 秒 bounded chunks 轉錄
  → 釋放 ASR 模型與 MPS cache
  → Community-1 以 600 秒 bounded overlapping windows diarize
  → overlap timeline + speaker embeddings 統一跨窗身份
  → exclusive timeline 對齊 words／segments
  → 輸出逐字稿
  → 可選的本機或遠端摘要
```

`backend = "native"` 時不執行獨立 Community-1 階段，speaker labels 由所選 ASR backend 的原生實作產生。

## 摘要與隱私邊界

`summarization.backend` 支援：

| backend | 執行位置 | 資料行為 |
|---|---|---|
| `local`（預設） | 內建 llama.cpp | 逐字稿留在本機 |
| `ollama` | 設定的 Ollama host | 傳送到該 host；使用 localhost 時仍是本機 |
| `openai` | 設定的 OpenAI-compatible endpoint | 逐字稿會傳送到該 endpoint |

音訊與 speaker diarization 不會送往上述摘要 endpoint；遠端摘要只接收產生摘要所需的逐字稿文字。

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
│   ├── models.py                TranscriptResult / Segment / Word
│   ├── utils.py                 bounded chunks/windows 與 speaker helpers
│   ├── community_diarizer.py    Community-1 + 跨 window speaker reconciliation
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
- **[pyannote](https://github.com/pyannote/pyannote-audio)** — Community-1 speaker diarization pipeline
- **[WeSpeaker](https://github.com/wenet-e2e/wespeaker)** — Community-1 speaker embeddings
- **[OpenCC](https://github.com/BYVoid/OpenCC)** — 簡繁轉換

---

## License

MIT — 與原專案相同。見 [LICENSE](LICENSE)。
