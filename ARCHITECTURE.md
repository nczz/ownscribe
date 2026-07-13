# OwnScribe — 專案架構與開發狀態

> 最後更新：2026-07-13

## 專案概述

OwnScribe 是一個 **macOS 本地端會議記錄工具**，fork 自 [paberr/ownscribe](https://github.com/paberr/ownscribe) v0.13.0，新增了多個中文 ASR 後端和即時轉錄功能。

**核心改動目標**：將 ASR 層從 WhisperX 擴展為可抽換的多後端架構，大幅提升中文辨識品質，同時保持原有設計不被破壞。

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
│   ├── funasr_transcriber.py        新增：FunASR SenseVoice + CAM++（中文 CER 7.81%）
│   ├── breeze_transcriber.py        新增：Breeze-ASR-25 + CAM++（台灣國語+中英混合，原生繁體）
│   └── firered_transcriber.py       新增：FireRedASR2-AED + CAM++（中文 CER 3.05%，最精確）
├── summarization/                ← 【未動】LLM 摘要層
│   ├── base.py / llama_cpp_summarizer.py / ollama_summarizer.py / openai_summarizer.py
│   └── prompts.py
├── output/                       ← 【未動】輸出格式
│   ├── markdown.py / json_output.py
├── pipeline.py                   ← 【微調】_create_transcriber() 工廠函式支援 4 後端
├── pipeline_live.py              ← 新增：即時會議 pipeline（串流字幕+錄音+會後精修）
├── cli.py                        ← 【微調】新增 `ownscribe live` 子指令
├── config.py                     ← 【微調】新增 FunASRConfig、resolve_model_path()
├── progress.py                   ← 【未動】TUI 進度
└── search.py                     ← 【未動】會議搜尋

scripts/
└── live_preview.py               ← 獨立即時字幕腳本（已被 pipeline_live.py 取代）

models/                           ← 模型目錄（gitignored）
├── breeze-asr-25/                   實體：MediaTek Breeze-ASR-25（15GB）
├── sensevoice → ~/.cache/funasr/... FunASR SenseVoice-Small
├── fsmn-vad → ~/.cache/funasr/...   語音活動偵測
├── ct-punc → ~/.cache/funasr/...    標點還原
├── campplus → ~/.cache/funasr/...   CAM++ 說話者辨識
├── paraformer-zh-streaming → ...    即時串流模型
├── firered-asr2-aed → ~/Projects/FireRedASR2S/...
├── firered-vad → ...
├── firered-lid → ...
└── firered-punc → ...

swift/                            ← 【未動】Core Audio 錄音 helper（macOS native binary）
```

---

## ASR 後端比較

| 設定值 `asr_backend=` | 模型 | 中文 CER | 速度 (M5 Pro) | 繁體輸出 | 說話者辨識 | 適合場景 |
|---|---|---|---|---|---|---|
| `"breeze"` ← **目前預設** | Breeze-ASR-25 + CAM++ | ~8% (台灣國語) | 5.2x MPS | ✅ 原生 | ✅ CAM++ | 台灣中英混合會議 |
| `"firered"` | FireRedASR2-AED + CAM++ | 3.05% | 2.3x MPS | ❌ 需 OpenCC | ✅ CAM++ | 最高中文準確度 |
| `"funasr"` | SenseVoice + CAM++ | 7.81% | 17x CPU | ❌ 需 OpenCC | ✅ 內建 | 快速處理 |
| `"whisperx"` | Whisper + pyannote | ~20% | 13x | ❌ | ✅ pyannote | 英文場景 |

---

## 即時會議模式 (`ownscribe live`)

```
啟動 → 載入 paraformer-zh-streaming（即時辨識）
     → 開始 Core Audio 錄音（系統音+麥克風）
     → 終端即時顯示繁體字幕（OpenCC s2twp 轉換）

Ctrl+C → 停止錄音
       → 會後精修：用設定的 asr_backend（預設 breeze）+ CAM++ 說話者辨識
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
models_dir = "~/.cache/ownscribe/models"  # 不再使用，路徑統一由 resolve_model_path 解析

[diarization]
enabled = true

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
3. fallback 為 ModelScope/HuggingFace ID（自動下載）

---

## 外部依賴

| 工具 | 用途 | 安裝方式 |
|------|------|---------|
| `yap` | Apple SpeechAnalyzer CLI（即時字幕替代方案） | `brew install yap` |
| `ffmpeg` | 音訊格式轉換 | `brew install ffmpeg` |
| `uv` | Python 套件管理 | `brew install uv` |
| FireRedASR2S | ASR 程式碼 | `~/Projects/FireRedASR2S/`（需在 PYTHONPATH） |

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
- [ ] `ownscribe live` 的即時串流還有 tqdm 進度條洩漏（需確認修復效果）
- [ ] FireRedASR2 在 MPS 上 `torchaudio::forced_align` 未實作，時間戳精度降低
- [ ] 考慮把 FireRedASR2S 的模型也直接移到專案 `models/` 下（目前用 symlink 指向外部）
- [ ] 加入 CLI `--asr-backend` flag 讓命令列可臨時切換
- [ ] 寫 unit test 覆蓋新增的 transcriber

### 已知問題
- SenseVoice + CAM++ 搭配時有 bug（`distribute_spk` TypeError），所以 funasr backend 的說話者辨識是 SenseVoice 內建的 sentence_info，不是獨立 CAM++
- Breeze-ASR-25 沒有標點輸出，需要 LLM 後處理或接 FireRedPunc
- FireRedASR2 的 `.cuda()` 硬編碼需要 monkey-patch 才能走 MPS

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

**未動：**
- `audio/`、`summarization/`、`output/`、`progress.py`、`search.py`、`swift/`、`tests/`
