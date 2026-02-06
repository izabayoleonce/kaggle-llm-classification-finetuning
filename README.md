# 🤖 LLM Classification Finetuning - LMSYS Chatbot Arena
# kaggle-llm-classification-finetuning
Supervised Fine-Tuning (SFT) pipeline for classifying LLM responses using PyTorch and Hugging Face Transformers.


![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📋 Project Overview

This repository contains the solution code for the Kaggle competition **"LLM Classification Finetuning"** (LMSYS - Chatbot Arena Human Preference Predictions). 

Large language models (LLMs) are rapidly entering our lives, but ensuring their responses resonate with users is critical for successful interaction. This competition presents a unique opportunity to tackle this challenge with real-world data and help bridge the gap between LLM capability and human preference.

### The Challenge
The goal is to predict which response a user will prefer in a head-to-head battle between two chatbots powered by large language models. The challenge aligns with the concept of "reward models" or "preference models" in reinforcement learning from human feedback (RLHF).

> **Context:** Previous research has identified limitations in directly prompting an existing LLM for preference predictions. These limitations often stem from biases such as favoring responses presented first (position bias), being overly verbose (verbosity bias), or exhibiting self-promotion (self-enhancement bias).

## 🎯 Objective

Develop a machine learning model that effectively predicts user preferences to assist in developing LLMs that can tailor responses to individual user preferences, ultimately leading to more user-friendly AI systems.

**Evaluation Metric:** Submissions are evaluated on the **Log Loss** between the predicted probabilities and the ground truth values.

## 🛠️ Methodology & Approach

The solution implements a **Supervised Fine-Tuning (SFT)** approach using Pre-trained Language Models (PLMs) for Sequence Classification.

### Key Techniques:
* **Models Evaluated:**
    * `microsoft/deberta-v3-base`: Selected for its strong performance on NLU tasks.
    * `roberta-base`: Used as a comparative baseline.
* **Input Formatting:** The prompt and both responses are concatenated into a single sequence with special separators to allow the model to learn the relationship between the context and the answers:
    ```text
    PROMPT: {prompt} [SEP] RESPONSE A: {response_a} [SEP] RESPONSE B: {response_b}
    ```
* **Training:**
    * Utilized **Hugging Face `Trainer` API** for efficient training loop management.
    * Dynamic padding using `DataCollatorWithPadding`.
    * Model selection based on the lowest validation Log Loss.

## 📂 Project Structure

```bash
├── llm-classification-finetuning.ipynb  # Main Jupyter Notebook containing training & inference pipeline
├── submission.csv                       # Final predictions (generated)
└── README.md                            # Project documentation
