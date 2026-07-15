# OwnScribe — 專案架構與開發狀態

> 最後更新：2026-07-15

## 專案概述

OwnScribe 是一個 **macOS 本地端會議記錄工具**，fork 自 [paberr/ownscribe](https://github.com/paberr/ownscribe) v0.13.0，新增了多個中文 ASR 後端和即時轉錄功能。

**核心改動目標**：將 ASR 層從 WhisperX 擴展為可抽換的多後端架構，大幅提升中文辨識品質，同時保持原有設計不被破壞。

## 目前驗證狀態

- 已提交基準：`906e5b0 Harden Chinese transcription backends`
- 目前工作樹：四後端 bounded-memory 長音訊分塊處理已實作且驗證，待提交
- 自動測試：263 passed
- 靜態檢查：Ruff 與 `git diff --check` 通過
- 打包：sdist 與 wheel build 通過
- 尚未完成的實證：固定台灣會議語料 benchmark、實體大型模型長時間 soak test

---

## 架構總覽

```
src/ownscribe/
├── audio/                        ← 【未動】錄音層
│   ├── base.py                      AudioRecorder 抽象基類
│   ├── coreaudio.py                 macOS Core Audio Tap（系統音訊+麥克風）
│   └── sounddevice_recorder.py      fallback
├── transcription/                ← 【擴充】多後端 ASR
│   ├── base.py                      Transcriber 抽象基類（介面：transcribe(path) → TranscriptResult）
│   ├── models.py                    TranscriptResult / Segment / Word 資料模型
│   ├── whisperx_transcriber.py      原有：WhisperX + pyannote（英文佳）
│   ├── funasr_transcriber.py        新增：FunASR SenseVoice + native optional CAM++
│   ├── breeze_transcriber.py        新增：Breeze-ASR-25 + CAM++（台灣國語+中英混合，原生繁體）
│   └── firered_transcriber.py       新增：FireRedASR2-AED + optional CAM++
├── summarization/                ← 【未動】LLM 摘要層
│   ├── base.py / llama_cpp_summarizer.py / ollama_summarizer.py / openai_summarizer.py
│   └── prompts.py
├── output/                       ← 【未動】輸出格式
│   ├── markdown.py / json_output.py
├── pipeline.py                   主流程與 4-backend factory
├── pipeline_live.py              ← 新增：即時會議 pipeline（串流字幕+錄音+會後精修）
├── cli.py                        ← 【微調】新增 `ownscribe live` 子指令
├── config.py                     設定、backend 參數與 model resolver
├── progress.py                   TUI 進度
└── search.py                     會議搜尋

scripts/
└── live_preview.py               ← 獨立即時字幕腳本（已被 pipeline_live.py 取代）

models/                           ← 模型目錄（gitignored）
├── breeze-asr-25/                   實體：MediaTek Breeze-ASR-25（15GB）
├── sensevoice → ~/.cache/funasr/... FunASR SenseVoice-Small
├── fsmn-vad → ~/.cache/funasr/...   語音活動偵測
├── ct-punc → ~/.cache/funasr/...    標點還原
├── campplus → ~/.cache/funasr/...   CAM++ 說話者辨識
├── paraformer-zh-streaming → ...    即時串流模型
├── firered-asr2-aed                 FireRed model cache or project-local model
├── firered-vad → ...
├── firered-lid → ...
└── firered-punc → ...

swift/                            ← 【未動】Core Audio 錄音 helper（macOS native binary）
```

---

## ASR 後端比較

| 設定值 `asr_backend=` | 模型 | 執行裝置 | 繁體輸出 | 說話者辨識 | 適合場景 |
|---|---|---|---|---|---|
| `"breeze"` | Breeze-ASR-25 + optional CAM++ | MPS / CPU | ✅ 原生 | 可選 | 台灣中英混合會議 |
| `"firered"` | FireRedASR2-AED + optional CAM++ | CPU | ❌ 需 OpenCC | 可選 | 外部研究整合 |
| `"funasr"` | SenseVoice + optional CAM++ | CPU | ❌ 需 OpenCC | 原生整合 | 快速處理 |
| `"whisperx"` ← 程式預設 | Whisper + optional pyannote | Metal / CPU | ❌ | 可選 | 英文場景 |

上游 CER 與速度不是在同一 benchmark 下產生，架構文件不將它們作為可直接比較的產品保證。

---

## 即時會議模式 (`ownscribe live`)

```
啟動 → 載入 paraformer-zh-streaming（即時辨識）
     → 開始 Core Audio 錄音（系統音+麥克風）
     → 終端即時顯示繁體字幕（OpenCC s2twp 轉換）

Ctrl+C → 停止錄音
       → 會後精修：使用設定的 asr_backend，並遵守 diarization.enabled
       → 輸出 ~/ownscribe/YYYY-MM-DD_HHMM/transcript.md
```

---

## 設定檔

`~/.config/ownscribe/config.toml`

```toml
[audio]
backend = "coreaudio"
mic = true
capture_mode = "all"
silence_timeout = 600

[transcription]
asr_backend = "breeze"        # breeze / firered / funasr / whisperx
funasr_model = "sensevoice"
language = ""
models_dir = "~/.cache/ownscribe/models"  # 本地模型搜尋目錄
firered_repo = ""                    # FireRedASR2S checkout 的明確路徑
chunk_seconds = 60                    # bounded-memory 音訊視窗，最小 30 秒

[diarization]
enabled = true
speaker_threshold = 0.7

[summarization]
enabled = false               # 待裝 Ollama 後啟用

[output]
dir = "~/ownscribe"
format = "markdown"
keep_recording = true
```

---

## 模型路徑管理

統一透過 `config.py` 的 `resolve_model_path("名稱")` 解析：
1. 先找 `專案/models/名稱`（跟隨 symlinks）
2. 再找設定的 models_dir
3. fallback 為已註冊的 canonical ModelScope/HuggingFace ID（自動下載）

## 長音訊記憶體模型

`iter_audio_chunks()` 使用 `soundfile.SoundFile.read(frames)` 逐塊讀取、downmix 與必要的 resample。四個 ASR 後端的峰值 audio RAM 因此與 `chunk_seconds` 成正比，不再與整場會議長度成正比。Breeze/FireRed 會累積低維 speaker embeddings 後做跨 chunk centroid clustering；FunASR/WhisperX 的上游 diarization 是 chunk-local，因此 speaker label 會加入 chunk 前綴，避免錯誤合併不同人物。

---

## 外部依賴

| 工具 | 用途 | 安裝方式 |
|------|------|---------|
| `yap` | Apple SpeechAnalyzer CLI（即時字幕替代方案） | `brew install yap` |
| `ffmpeg` | 音訊格式轉換 | `brew install ffmpeg` |
| `uv` | Python 套件管理 | `brew install uv` |
| FireRedASR2S | ASR 程式碼 | 使用 `transcription.firered_repo` 明確設定；程式啟動時驗證 |

---

## 環境

- macOS 26.4.1 / Apple M5 Pro / 64GB
- Python 3.12（via uv venv）
- PyTorch 2.8.0（MPS 支援）
- 虛擬環境：`~/Projects/ownscribe/.venv/`

---

## 待辦 / 已知問題

### 待做
- [ ] 安裝 Ollama + qwen3:8b，啟用會後摘要（行動項/決議/總結）
- [ ] 建立固定台灣會議語料與一致正規化方式的可重現 benchmark
- [ ] 加入 CLI `--asr-backend` flag 讓命令列可臨時切換
- [x] 新增中文 backend 的依賴、輸入、設定及錯誤路徑 contract tests

### 已知問題
- FunASR 需 1.3.12 以上，使用其已修正的 SenseVoice + CAM++ 原生 `sentence_info` 路徑
- Breeze-ASR-25 沒有標點輸出，需要 LLM 後處理或接 FireRedPunc
- FireRed 僅使用官方 CPU/CUDA 契約；本專案不再修改全域 Torch `.cuda()` 行為

---

## 快速開始（給下次對話用）

```bash
cd ~/Projects/ownscribe
source .venv/bin/activate

# 確認設定
cat ~/.config/ownscribe/config.toml

# 即時會議
ownscribe live

# 轉錄已有錄音
ownscribe transcribe ~/ownscribe/2026-07-11_1150/recording.wav

# 切換後端（編輯設定）
ownscribe config
```

---

## 檔案改動清單（相對於上游 v0.13.0）

**新增：**
- `src/ownscribe/transcription/funasr_transcriber.py`
- `src/ownscribe/transcription/breeze_transcriber.py`
- `src/ownscribe/transcription/firered_transcriber.py`
- `src/ownscribe/pipeline_live.py`
- `scripts/live_preview.py`
- `ARCHITECTURE.md`（本文件）

**修改：**
- `.gitignore` — 加入 `models/`
- `src/ownscribe/cli.py` — 新增 `live` 子指令
- `src/ownscribe/config.py` — 新增 `FunASRConfig`、`resolve_model_path()`、`TranscriptionConfig.asr_backend/funasr_model/models_dir`
- `src/ownscribe/pipeline.py` — `_create_transcriber()` 支援 4 後端

**主要測試覆蓋：**
- `tests/test_chinese_backends.py` — backend factory、模型解析、bounded chunks、stereo、speaker clustering、atomic output
- `tests/test_transcription.py` — WhisperX lifecycle、alignment、diarization API 相容性與 chunk integration
- 其餘 `tests/` — CLI、pipeline、search、summarization、output、progress 與錄音行為
