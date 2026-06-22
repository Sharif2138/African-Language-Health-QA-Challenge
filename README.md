# Multilingual Health Question Answering in Low-Resource African Languages

All notebooks have a link to open directly in google colab

Final project for Machine Learning Techniques I — Zindi Competition submission.

This project fine-tunes multilingual sequence-to-sequence models (mT5, flan-t5) with LoRA adapters to answer health-related questions in low-resource African languages including Luganda, Amharic, Akan, Swahili, and English variants across Uganda, Ethiopia, Ghana, and Kenya.

---

## Competition

**Zindi:** [Multilingual Health Question Answering in Low-Resource African Languages Challenge](https://zindi.africa)

**Evaluation metrics:** ROUGE-1 F1, ROUGE-L F1, LLM-as-a-Judge

---

## Repository Structure
├── Exp1.ipynb          # EDA + preprocessing pipeline + retrieval baseline

├── Exp2_3.ipynb        # mT5-small baseline + decoder start token fix

├── Exp4.ipynb          # Proper baseline with sufficient data (500/lang)

├── Exp5.ipynb          # Long instruction prompt experiment

├── Exp6_7_8.ipynb      # Short prompt + LoRA rank ablation + mT5-base

├── Exp9_10_11.ipynb    # Full dataset scale-up + flan-t5 + final mT5-base

└── README.md

---

## Experiment Overview

| Exp | Notebook | Model | Key Change |
|-----|----------|-------|------------|
| 1 | Exp1.ipynb | Retrieval (mpnet) | EDA + cosine similarity baseline |
| 2 | Exp2_3.ipynb | mT5-small + LoRA r=4 | First generative attempt |
| 3 | Exp2_3.ipynb | mT5-small + LoRA r=4 | Decoder start token fix + no_repeat_ngram |
| 4 | Exp4.ipynb | mT5-small + LoRA r=4 | 500 samples/lang, true baseline |
| 5 | Exp5.ipynb | mT5-small + LoRA r=4 | Long instruction prompt |
| 6 | Exp6_7_8.ipynb | mT5-small + LoRA r=4 | Short prefix prompt |
| 7 | Exp6_7_8.ipynb | mT5-small + LoRA r=16 | Higher LoRA rank ablation |
| 8 | Exp6_7_8.ipynb | mT5-base + LoRA r=4 | Larger model on small data |
| 9 | Exp9_10_11.ipynb | mT5-small + LoRA r=4 | Full dataset (29,751 samples) |
| 10 | Exp9_10_11.ipynb | flan-t5-base + LoRA r=8 | Different model family |
| 11 | Exp9_10_11.ipynb | mT5-base + LoRA r=8 | Lower lr + warmup + full data — **best result** |

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Sharif2138/African-Language-Health-QA-Challenge
.git
```

### 2. Install dependencies

All notebooks are designed to run on **Google Colab** with a T4 GPU (free tier) or **Kaggle** (recommended for Exp 9–11 due to longer session limits).

Run the first cell in any notebook to install required packages:

```bash
!pip install -q torchao==0.16.0
!pip install transformers datasets peft evaluate sentencepiece accelerate rouge_score
```

### 3. Mount Google Drive

All notebooks save checkpoints, cleaned datasets, and plots to Google Drive. Mount Drive in Colab:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Create the following folders in your Drive before running:
MyDrive/

├── datasets/       # cleaned CSVs saved here

├── plots/          # EDA plots saved here

└── checkpoints/    # model checkpoints saved here

### 4. Upload competition data

Download `train.csv`, `val.csv`, and `test.csv` from the Zindi competition page and upload them to `/content/drive/MyDrive/datasets/` or directly to the Colab session.

### 5. Run notebooks in order

Start with **Exp1.ipynb** — it runs the full preprocessing pipeline and saves `train_clean.csv`, `val_clean.csv`, and `test_clean.csv` which are required by all later experiments.
Exp1.ipynb  →  Exp2_3.ipynb  →  Exp4.ipynb  →  Exp5.ipynb  →  Exp6_7_8.ipynb  →  Exp9_10_11.ipynb

Each notebook can also be run independently as long as the cleaned CSVs are available.

---

## Reproducing the Best Result (Experiment 11)

Open `Exp9_10_11.ipynb` on Kaggle or Colab with a GPU runtime and navigate to the **Experiment 11** section. Key configuration:

```python
MODEL_NAME       = "google/mt5-base"
LORA_R           = 8
LORA_ALPHA       = 16
LEARNING_RATE    = 1e-4
WARMUP_STEPS     = 200
MAX_INPUT_LEN    = 192
MAX_TARGET_LEN   = 256
EPOCHS           = 3
BATCH_SIZE       = 8
PROMPT_TEMPLATE  = "health qa {language}: {question}"
```

The notebook generates a `submission.csv` ready for direct upload to Zindi.

---

## Preprocessing Pipeline (Exp1.ipynb)

The preprocessing pipeline runs before any training experiment:

1. **Unicode normalization** — NFKC normalization across all scripts
2. **Whitespace cleaning** — remove tabs, newlines, collapse spaces
3. **Empty record removal** — drop rows with blank questions
4. **Exact duplicate removal** — drop identical (input, output, subset) triplets
5. **Leakage removal** — remove training questions that appear in validation set

Output files: `train_clean.csv`, `val_clean.csv`, `test_clean.csv`

---

## Key Findings

- **mT5 requires explicit `decoder_start_token_id = tokenizer.pad_token_id`** — without this, the model generates `<extra_id_0>` sentinel tokens and loops indefinitely (Exp 2 → Exp 3)
- **Short prompts outperform long instruction prompts** — `"health qa {language}: {question}"` adds only 4–5 tokens vs 25+ for verbose templates, preventing question truncation (Exp 5 → Exp 6)
- **LoRA rank must match dataset size** — r=16 overfit on 4,000 samples; r=8 was optimal on 29,751 samples (Exp 7 → Exp 11)
- **mT5-base outperforms mT5-small** even on limited data (Exp 8)
- **flan-t5-base underperforms on low-resource African languages** despite strong English instruction-following capability (Exp 10)
- **Learning rate 3e-4 causes gradient explosion on the full dataset** — reducing to 1e-4 with 200 warmup steps fixed convergence (Exp 9 → Exp 11)

---

## Hardware Used

| Experiments | Platform | GPU |
|-------------|----------|-----|
| Exp 1–8 | Google Colab (free) | NVIDIA T4 |
| Exp 9–11 | Kaggle (free) | NVIDIA T4 x1 |

---

## Author

**Sharif** — Machine Learning Techniques I, Final Project
