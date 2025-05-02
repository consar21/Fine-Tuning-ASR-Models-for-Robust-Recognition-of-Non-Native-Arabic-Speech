# Fine-Tuning-ASR-Models-for-Robust-Recognition-of-Non-Native-Arabic-Speech



# Fine-Tuning ASR Models on Native Arabic Speech Data

## Table of Contents
- [Overview](#overview)
- [Team Information](#team-information)
- [Learning Outcomes](#learning-outcomes)
- [Project Objectives](#project-objectives)
- [Literature Review](#literature-review)
- [Methodology](#methodology)
  - [Model Architecture](#model-architecture)
  - [Implementation Framework](#implementation-framework)
  - [Dataset Description](#dataset-description)
- [Experimental Design](#experimental-design)
- [Results and Analysis](#results-and-analysis)
- [Discussion](#discussion)
- [Conclusion and Future Work](#conclusion-and-future-work)
- [References](#references)

---

## Overview
This project explores fine-tuning automatic speech recognition (ASR) models on native Arabic speech data to improve transcription accuracy for native speakers. The focus is on adapting pre-trained models using domain-specific data and evaluating performance using WER/CER metrics.

---

## Team Information
- **Team Leader**: [Sara Ghanem Alqahtani ] - [2210002845]  
- **Team Member 2**: [Njood Tawfiq Alrushaid] - [2210003029]  
- **Team Member 3**: [Atheer Abdullah Alkhudair] - [2210003371]
- **Team Member 4**: [Sarah Mohammed Alsulaim] - [2210003352 ]
-  **Team Member 5**: [Shahad Khalid Alsadah ] - [2210003178]  
- **Group Number**: [6]

---

## Learning Outcomes
- Conduct a literature review and synthesize current research in ASR and Generative AI.
- Apply theoretical and practical skills to train and evaluate deep learning models.
- Communicate technical findings effectively through written and oral formats.

---

## Project Objectives
- Fine-tune an ASR model (e.g., Whisper or Wav2Vec2) using native Arabic speech datasets.
- Evaluate performance improvements using WER and other ASR metrics.
- Identify bottlenecks in adapting ASR for regional or phonetic variations.

---

## Literature Review
- Overview of current ASR models used for Arabic (Whisper, Wav2Vec2, XLS-R, etc.)
- Comparison of training strategies for low-resource or morphologically rich languages.
- Challenges in transcribing dialectal Arabic and how existing research handles them.

---

## Methodology

### Model Architecture
- Wave2vec Model
  - Wav2Vec 2.0
is a cutting-edge self-supervised model developed by Facebook AI for automatic speech recognition (ASR) tasks. It processes raw audio waveforms to extract rich speech features, significantly reducing the need for extensive labeled datasets. The model architecture is composed of two main components: a feature encoder that converts raw audio into latent representations, and a transformer based context network that models long range dependencies within the speech signal [1].

  - Wav2Vec 2.0 XLSR-300M
Wav2Vec 2.0 XLS-R 300M is a multilingual extension of the Wav2Vec 2.0 model, specifically built to perform well across a wide range of languages. Developed by Facebook AI, it contains around 300 million parameters and is trained on an extensive multilingual dataset. The model employs self-supervised learning on raw audio to extract robust speech features through a feature encoder and a transformer based context network. Its architecture is fine-tuned to deliver accurate speech recognition and transcription in various languages[2]. 

<img width="468" alt="image" src="https://github.com/user-attachments/assets/883ef391-a63d-4104-9175-4c365a3c3e8c" />


  -  Whisper Model
    
Whisper is an advanced automatic speech recognition (ASR) system developed by OpenAI, designed to handle both multilingual and multitask speech processing. Unlike conventional ASR models that need significant fine-tuning on specific datasets, Whisper is trained on a massive dataset comprising 680,000 hours of multilingual and multitask speech data. This broad training allows the model to perform effectively in zero-shot scenarios where it can carry out tasks without prior task specific training. Leveraging its multilingual training, Whisper can transcribe or translate previously unseen languages. It also demonstrates strong resilience to variations in accents, background noise, and spontaneous speech, making it highly versatile across different audio environments [4].

### Implementation Framework
- Framework: Hugging Face Transformers + PyTorch
- Hardware: [e.g., Colab Pro, 16GB RAM, GPU Tesla T4]
- Tools: torchaudio, datasets, WER evaluation scripts

### Dataset Description
- Name: [e.g., ArabicSpeechCorpus, Custom Dataset]
- Size: [e.g., 3,000 samples, 10 hours]
- Format: WAV + text
- Preprocessing: Resampling, normalization, silence trimming
- Train/Val/Test Split: [e.g., 80/10/10]

---

## Experimental Design

This section outlines the training methodology, manual fine-tuning process, evaluation metrics, and baseline comparisons for our ASR models fine-tuned on native Arabic speech data. Two architectures were explored: Whisper-small and two versions of Wav2Vec2.

### 1. Training Methodology
We adopted a manual fine-tuning approach where model training was conducted iteratively based on intermediate results. Each model was fine-tuned using the L1-KSU speech datasets, and evaluated using Word Error Rate (WER).

Training steps included:
- Audio normalization and resampling to 16kHz
- Manual model checkpoint tracking across different training stages
- Validation-based tuning with iterative hyperparameter adjustments

All experiments were conducted using PyTorch and Hugging Face Transformers on GPU-enabled environments.

### 2. Hyperparameter Selection and Tuning
Hyperparameters were selected based on validation WER and adjusted manually. The best-performing configurations for each model were:

| Model Variant                | Epochs | Batch Size | Learning Rate | Notes                          |
|-----------------------------|--------|------------|----------------|--------------------------------|
| Whisper-small               | 30     | 16         | 3e-4           |  
| AndrewMcDowell/wav2vec2-xls-r-300m-arabic         | 8      | 4          | 2e-5           |                
| phantomcoder1996/wav2vec2-large-xls-r-300m-arabic             | 6      | 8          | 1e-5           |              

> Note: All experiments included logging of **training loss** and **validation WER** for early stopping decisions.

### 3. Baseline Models for Comparison
We used the original pre-trained versions as our baselines (no fine-tuning applied). The table below highlights the best result for each model after fine-tuning.

| Model Name                        | Baseline WER | Fine-Tuned WER |
|----------------------------------|--------------|----------------|
| Whisper-small                    | 51.46%       | 0.37%         |
| AndrewMcDowell/wav2vec2-xls-r-300m-arabic             | 72.39%      | 2.52%        |
| phantomcoder1996/wav2vec2-large-xls-r-300m-arabic            | 82.60%       | 3.89%         |

This comparison quantifies the improvement achieved through task-specific training.

### 4. Evaluation Metrics
The evaluation was based exclusively on:
- **Word Error Rate (WER)**: Primary metric to evaluate transcription accuracy.
- **Training Loss**: Monitored throughout to avoid overfitting and guide fine-tuning decisions.

Evaluation was performed on a held-out test set of native Arabic audio samples using Hugging Face’s `evaluate` and custom scoring scripts.

### 5. Ablation Studies Design
We conducted multiple experiments to understand the impact of different training configurations, including:

- **Model Variant Comparison**: Whisper-small vs two Wav2Vec2 versions
- **Layer Freezing**: Tested freezing encoder layers vs full fine-tuning
- **Learning Rate Impact**: Compared high vs low learning rates
- **Dataset Subsampling**: Fine-tuned on reduced data sizes to observe WER drop

The best-performing configurations from each ablation were selected for the final comparison table above.

---

## Results and Analysis
| Model | Dataset | WER (%) | CER (%) |
|-------|---------|---------|---------|
| Pre-trained | Native Test | XX | XX |
| Fine-tuned  | Native Test | XX | XX |

- Qualitative results: transcription examples
- Charts and confusion matrices (add if available)
- Limitations and failure examples

---

## Discussion
- Comparison to state-of-the-art systems
- Insights from fine-tuning on dialect-rich speech
- Impact on Arabic NLP and accessibility

---

## Conclusion and Future Work
- Summary of main contributions and findings
- Areas of improvement: larger datasets, dialect-specific modeling
- Future directions: code-switching handling, speaker adaptation

---


