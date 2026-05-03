# LLM Text Generation Lab Work

This repository contains the Python code for a lab work on Text Generation with Large Language Models

## Aim

The aim of this lab work is to understand how a Large Language Model generates text by predicting the next token step by step.

## Input Phrase

```text
Life in Lithuania is
```

## Methods Used

The lab work compares several text generation methods:

* Greedy decoding
* Beam search
* Beam search with repetition penalty
* Temperature sampling
* Top-K sampling
* Top-P / nucleus sampling

## Requirements

Install the required packages:

```bash
pip install transformers
pip install hf_xet
pip install accelerate
```

If PyTorch is not installed, install it from the official PyTorch website according to your system.

## How to Run

```bash
python llm_text_generation_experiment.py
```

## Output

The program displays token information, possible next-token predictions, and generated text outputs for each method. The results are also saved in:

```text
generation_results.json
```

## Summary

This lab work shows that deterministic methods produce more stable text, while sampling-based methods create more varied outputs. However, high randomness can make the generated text less coherent.