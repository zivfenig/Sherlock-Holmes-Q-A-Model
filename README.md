# Sherlock Holmes Q&A Model

Fine-tuning Gemma 3 12B on the Sherlock Holmes canon in two stages:
1. **Part 1** — Continued pre-training with LoRA to adapt the model to Doyle's Victorian prose
2. **Part 2** — Supervised Fine-Tuning (SFT) to teach the model to answer questions grounded in the canon

## Results Summary

### Part 1 — Continued Pre-training

| Model | Perplexity |
|---|---|
| Base model (no training) | 23.07 |
| Baseline LoRA (4B) | 9.82 |
| Best ablation (warmup_short) | 9.60 |
| Gemma 3 12B + QLoRA | 6.80 |

All trained configurations achieve ~57% perplexity reduction over the base model. The main finding: corpus size is the bottleneck, not hyperparameter choice.

### Part 2 — SFT

| Configuration | Token F1 | BERTScore |
|---|---|---|
| Part 1 model, zero-shot (lower bound) | 0.192 | 0.677 |
| Part 1 model, 3-shot | 0.277 | 0.787 |
| SFT final, zero-shot | 0.274 | 0.796 |
| SFT final, 3-shot (best system) | 0.296 | 0.800 |

Exact match is 0 across all configurations — the model paraphrases rather than reproducing answers verbatim.

## Key Findings

**Part 1:**
- Training itself dominates — any reasonable configuration achieves similar perplexity on a small corpus
- LoRA rank, learning rate, and target modules had minimal effect
- QLoRA matches LoRA quality at ~4x less memory
- Model scale is the most impactful lever: 12B + QLoRA (PPL 6.80) vs 4B + QLoRA (PPL 9.75)

**Part 2:**
- SFT adds 43% relative F1 improvement over zero-shot Part 1 model
- Most surprising: Part 1 3-shot prompting slightly outperforms SFT zero-shot — for a 12B model, in-context learning is a strong competitor to parameter updates on small datasets
- Adding MLP layers to LoRA targets (attn_plus_mlp) improved QA performance, unlike Part 1 where it had no effect on perplexity
- Best system combines domain pre-training + SFT + 3-shot prompting

## Dataset

### Part 1
Nine canonical Sherlock Holmes books by Arthur Conan Doyle downloaded from Project Gutenberg:

| # | Title | Gutenberg ID |
|---|---|---|
| 1 | A Study in Scarlet | 244 |
| 2 | The Sign of the Four | 2097 |
| 3 | The Hound of the Baskervilles | 2852 |
| 4 | The Valley of Fear | 3289 |
| 5 | The Adventures of Sherlock Holmes | 1661 |
| 6 | The Memoirs of Sherlock Holmes | 834 |
| 7 | The Return of Sherlock Holmes | 108 |
| 8 | His Last Bow | 2350 |
| 9 | The Case-Book of Sherlock Holmes | 69700 |

Book 2 held out as validation (~6.2% of corpus). 836 training chunks of 1024 tokens.

### Part 2
622 QA pairs total across 7 categories:

| Source | Count |
|---|---|
| Provided (LiteraryQA + GutenQA) | 120 |
| Provided (synthetic) | 383 |
| Our generated | 119 |

Train: 500 pairs, Val: 61 pairs, Test: 61 pairs (stratified by category).

## Models

- **Base model**: `google/gemma-3-12b-pt`
- **Part 1**: QLoRA continued pre-training on Sherlock Holmes corpus
- **Part 2**: Fresh LoRA adapter (attn+mlp, r=8) on top of merged Part 1 model

Both models are gated on HuggingFace. Accept the Gemma license at https://huggingface.co/google/gemma-3-12b-pt before running.

## Experiments Tracked

Full training history available on Weights and Biases:
- Part 1: https://wandb.ai/zivfenig-ben-gurion-university-of-the-negev/sherlock-IBM-Course-Assignment-1
- Part 2: https://wandb.ai/zivfenig-ben-gurion-university-of-the-negev/sherlock-IBM-Course-Assignment-2

## Setup

```bash
conda create -n sherlock_env python=3.11 -y
conda activate sherlock_env
pip install torch transformers peft datasets bitsandbytes accelerate trl wandb \
            matplotlib pandas requests sentence-transformers bert-score
```

Save your HuggingFace token:
```bash
echo "your_hf_token" > ~/.hf_key
chmod 600 ~/.hf_key
```

Save your W&B token:
```bash
echo "your_wandb_key" > ~/.wandb_key
chmod 600 ~/.wandb_key
```

## Repository Structure
