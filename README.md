# Fine-Tuning-ASR-Models-for-Robust-Recognition-of-Non-Native-Arabic-Speech



# Fine-Tuning ASR Models on Native Arabic Speech Data

## Table of Contents
- [Overview](#overview)
- [Team Information](#team-information)
- [Learning Outcomes](#learning-outcomes)
- [Introduction](#introduction)
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
## Introduction 
Generative AI has opened new possibilities in speech processing by enabling systems to learn from patterns and adapt to diverse real-world inputs. In this project, we apply generative principles to fine-tune Automatic Speech Recognition (ASR) models for improved recognition of non-native Arabic speech. Rather than relying solely on native speech data, we adapt pre-trained models using domain-specific samples that reflect common mispronunciations. This allows the ASR system to better handle spontaneous and imperfect speech from learners of Arabic.

Arabic holds a unique cultural and spiritual significance as the language of the Quran and a symbol of unity for millions of Muslims worldwide. Despite its global importance, Arabic is considered one of the most difficult languages to learn due to its rich phonetic system particularly emphatic, uvular, and pharyngeal sounds which often lead to mispronunciations that change meaning. This challenge is intensified by the lack of adaptive tools that address pronunciation variation, especially in real-time interactions. Traditional ASR models, which are typically trained on native speaker data, struggle with these errors and often yield inaccurate transcriptions for non-native speakers.

To bridge this gap, our project focuses on fine-tuning ASR models to improve robustness and accuracy for non-native Arabic speech. By training the model to recognize common phonetic variations, we aim to enhance its performance across different speaker profiles. This effort supports the goals of Saudi Vision 2030, which emphasizes cultural accessibility and the promotion of Arabic as a global language. Our objectives include reducing word and character error rates (WER/CER), increasing the inclusivity of ASR systems, and enabling clearer communication for learners and visitors in Arabic-speaking environments.


---

## Project Objectives
- Fine-tune an ASR model (e.g., Whisper or Wav2Vec2) using native Arabic speech datasets.
- Evaluate performance improvements using WER and other ASR metrics.
- Identify bottlenecks in adapting ASR for regional or phonetic variations.

---

## Literature Review
ASR systems have advanced significantly with the integration of deep learning and large-scale datasets. However, their performance still drops when encountering non-native speech, which often involves unfamiliar accents, phoneme substitutions, and pronunciation errors. This section reviews recent approaches that address these challenges by using fine-tuning, augmentation, or alignment-free methods to enhance recognition accuracy for diverse speakers.

A study by Korzekwa et al. proposed the use of generative techniques—such as phoneme-to-phoneme (P2P), text-to-speech (T2S), and speech-to-speech (S2S) conversions to simulate non-native pronunciation errors in English. These synthetic datasets improved the detection of lexical stress and pronunciation issues, with the S2S model achieving a 41% AUC improvement, highlighting the effectiveness of generative augmentation in capturing real learner variability.

Anantha et al. introduced DTW-SiameseNet, a dual-model approach that aligns user speech with reference pronunciations using Dynamic Time Warping (DTW) and a Siamese neural network. This model eliminated the need for dictionary updates and showed improved performance in detecting mispronunciations across multiple languages, with about 6% higher accuracy than traditional alignment-based methods.

Another work by Liao et al. tackled ASR output errors by proposing ASR Post-processing for Readability (APR). Their system leveraged Transformer-based models, including UniLM and RoBERTa, to clean ASR transcriptions without altering the speaker’s meaning. The results demonstrated significant improvements in BLEU and GLEU scores, suggesting these models can effectively handle disfluencies and misrecognitions in real-world speech.

According to a paper by Alkhamissi and Shoufan (2022) explored ASR performance on Arabic speech using deep learning models trained on diverse dialects. Their work revealed that even state-of-the-art ASR models underperform on certain Arabic phonemes, especially those with pharyngeal or emphatic features common in non-native speech. Their results emphasized the need for domain-adapted Arabic ASR models that account for dialectal and accentual variation.

Lastly, Radfar et al. applied fine-tuning techniques to OpenAI’s Whisper model, showing that even small-scale updates using accent-specific data could significantly reduce Word Error Rates (WER) on non-native speech. Their study confirms that pre-trained models can generalize better when adapted to pronunciation patterns that deviate from native norms.

Despite these advancements, a clear research gap remains for Arabic specific ASR systems tailored to non-native speakers, particularly for handling spontaneous, unscripted input and phonetic irregularities. Our project aims to address this by fine-tuning ASR models on Arabic speech that reflects realistic pronunciation errors, focusing on robustness and inclusivity in language learning and communication.

###  Table 1: Literature Review Summary – ASR Models and Non-Native Speech

| Study                  | Focus                                                                                         | Key Contribution                                                                                   |
|------------------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Korzekwa et al. (2023) | Generative data augmentation (P2P, S2S) for non-native speech error simulation                | 41% AUC improvement in error detection using synthetic non-native data                             |
| Anantha et al. (2021)  | DTW-SiameseNet for pronunciation error detection without reference dictionaries              | 6% improvement in pronunciation accuracy with language-agnostic detection                          |
| Liao et al. (2022)     | Transformer-based ASR post-processing for improving transcript readability                    | Enhanced BLEU/GLEU scores by correcting grammar/disfluency in ASR output                           |
| Waheed et al. (2023)   | Dialect-aware Arabic ASR trained on diverse accents and phonemes                              | Improved Arabic ASR performance across dialects and speaker variability                            |
| Gandhi (2022)          | Fine-tuning Whisper for multilingual ASR including non-native accents                         | Reduced WER on accented speech using low-resource fine-tuning of Whisper                           |

---

## Methodology

### Model Architecture
- Wave2vec Model
  - Wav2Vec 2.0
is a cutting-edge self-supervised model developed by Facebook AI for automatic speech recognition (ASR) tasks. It processes raw audio waveforms to extract rich speech features, significantly reducing the need for extensive labeled datasets. The model architecture is composed of two main components: a feature encoder that converts raw audio into latent representations, and a transformer based context network that models long range dependencies within the speech signal [1].

  - Wav2Vec 2.0 XLSR-300M
Wav2Vec 2.0 XLS-R 300M is a multilingual extension of the Wav2Vec 2.0 model, specifically built to perform well across a wide range of languages. Developed by Facebook AI, it contains around 300 million parameters and is trained on an extensive multilingual dataset. The model employs self-supervised learning on raw audio to extract robust speech features through a feature encoder and a transformer based context network. Its architecture is fine-tuned to deliver accurate speech recognition and transcription in various languages[2]. 

<img width="468" alt="image" src="https://github.com/user-attachments/assets/883ef391-a63d-4104-9175-4c365a3c3e8c" />

*Figure 1 overview of the Wav2Vec 2.0 XLS-R 300M pipeline [3] .*

  -  Whisper Model
    
Whisper is an advanced automatic speech recognition (ASR) system developed by OpenAI, designed to handle both multilingual and multitask speech processing. Unlike conventional ASR models that need significant fine-tuning on specific datasets, Whisper is trained on a massive dataset comprising 680,000 hours of multilingual and multitask speech data. This broad training allows the model to perform effectively in zero-shot scenarios where it can carry out tasks without prior task specific training. Leveraging its multilingual training, Whisper can transcribe or translate previously unseen languages. It also demonstrates strong resilience to variations in accents, background noise, and spontaneous speech, making it highly versatile across different audio environments [4].

### Implementation Framework
-  Wave2vec:
  
  The model was implemented in Google Colab using Python along with the Hugging Face Transformers library. Fine-tuning of the Wav2Vec2.0 XLS-R 300M model was carried out on a custom dataset, utilizing a 16 kHz audio processing pipeline and a custom data collator for dynamic input padding. Training was handled using the Trainer API, configured with a batch size of 16, a learning rate of 3e-4, and a total of 30 training epochs. To optimize GPU usage, techniques such as gradient accumulation, mixed-precision training (fp16), and gradient checkpointing were employed. Model performance was assessed using the Word Error Rate (WER), calculated with the help of the jiwer and evaluate libraries.

-  Whisper :

  The Whisper-based ASR system was developed using Python in the Google Colab environment, leveraging the Hugging Face Transformers library. The model openai/whisper-small was adapted for Arabic speech transcription and fine-tuned using the L2-KSU dataset. The implementation utilized the WhisperProcessor, WhisperTokenizer, and WhisperFeatureExtractor to process 16 kHz mono audio inputs and prepare them for model consumption. Training was conducted using the Seq2SeqTrainer API with key settings including a batch size of 16, a learning rate of 3e-4, and 30 training epochs. To optimize performance and resource efficiency, techniques such as mixed-precision training (fp16), gradient accumulation, and gradient checkpointing were employed. Model evaluation was based on Word Error Rate (WER), with regular evaluations and checkpoint saving every 500 steps. The final model was validated through Hugging Face’s ASR pipeline to assess its accuracy on new audio samples.



### Dataset Description
- L2-KSU Dataset 

  In this study, the L2-KSU dataset served as the primary resource for training and evaluating the ASR models. It contains 4086 audio recordings totaling 6 hours and 6 minutes, each accompanied by labeled transcriptions that include both standard and mispronounced forms. The dataset was collected from 80 adult speakers , 40 native and 40 non-native Arabic speakers with an equal gender distribution. The audio content comprises Quranic verses and Modern Standard Arabic (MSA) sentences, with a focus on phonetically challenging sounds for non-native speakers, such as /ʕ/ (ﻉ) and /ħ/ (ﺡ), to help improve the model’s sensitivity to pronunciation errors. Following the methodology of [5], the data was split by speaker: 60 participants were used for training (including both native and non-native speakers), while the remaining 20 non-native speakers were assigned to the test set. This speaker-based division was designed to minimize speaker-specific bias and evaluate model performance on unfamiliar voices, enhancing the system’s robustness and generalizability. Further details of this split are provided in Table 3.

**Table 2.** L2-KSU Dataset Description
| **Field**                 | **Description**                                                                 |
|--------------------------|---------------------------------------------------------------------------------|
| Language                 | Arabic                                                                          |
| Speaker                  | 40 native Arabic, 40 African L1 (learn Arabic)                                  |
| Non-Native Nationalities | West and Central African countries                                              |
| Level                    | Advanced for native speakers <br> Beginner to advanced learners for non-native speakers |
| Data                     | Read sentences                                                                  |
| Size                     | 4086 utterances, 6 h, and 6 min                                                 |
| Age                      | Adult                                                                           |
| Labelling                | Arabic script (utf-8)                                                           |


**Table 3.** Details of the L2-KSU Dataset Setup

| **Subset** | **No. of Speakers** | **Speaker Type** | **Utterances** | **Hours**    |
|------------|---------------------|------------------|----------------|--------------|
| Train      | 60                  | N + NN           | 3273           | 4 h. 45 min  |
| Test       | 20                  | NN               | 813            | 1 h. 22 min  |




 - Data Preprocessing


   As part of the dataset preparation process outlined in [5], audio recordings were enhanced through noise and silence removal using the PRAAT software, which significantly improved signal clarity particularly by eliminating low-frequency background noise. This refinement helped make phonetic features more distinguishable, thereby improving the quality of data for training speech recognition models. To ensure compatibility with ASR systems, stereo audio files were converted to mono and resampled at 16 kHz. For feature extraction, audio waveforms were transformed into log-mel spectrograms by first generating spectrograms and then applying Mel-scale filters, followed by a logarithmic transformation to reduce dynamic range. These final features effectively capture key time frequency patterns such as pitch and phoneme structures, which are essential for accurate speech recognition and linguistic analysis.



---

## Experimental Design

This section outlines the training methodology, manual fine-tuning process, evaluation metrics, and baseline comparisons for our ASR models fine-tuned on native Arabic speech data. Two architectures were explored: Whisper-small and two versions of Wav2Vec2.

### 1. Training Methodology
We adopted a manual fine-tuning approach where model training was conducted iteratively based on intermediate results. Each model was fine-tuned using the L2-KSU speech datasets, and evaluated using Word Error Rate (WER).

Training steps included:
- Feature extraction using log-Mel spectrograms .
- Tokenization and text normalization using model-specific processors.
- Use of a data collator for dynamic padding and label alignment .

All experiments were conducted using PyTorch and Hugging Face Transformers on a Cloud GPU-enabled environment (Google Colab Pro+).


### 2. Hyperparameter Selection and Tuning
Hyperparameters were selected based on validation WER and adjusted manually. The best-performing configurations for each model were:

| Model Variant                | Epochs | Batch Size | Learning Rate | Notes                          |
|-----------------------------|--------|------------|----------------|--------------------------------|
| Whisper-small               | 30     | 16         | 3e-4           |  
| AndrewMcDowell/wav2vec2-xls-r-300m-arabic         | 30      | 16          | 3e-4           |                
| phantomcoder1996/wav2vec2-large-xls-r-300m-arabic             | 30      | 16          | 3e-4         |              

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

Evaluation was performed on a **held-out 10% test split of non-native Arabic audio samples**, using Hugging Face’s `evaluate` library . This helped assess how well the fine-tuned models generalized beyond the Arabic native dataset they were trained on.

### 5. Ablation Studies Design
We conducted multiple experiments to understand the impact of different training configurations, including:

- **Model Variant Comparison**: Whisper-small vs two Wav2Vec2 versions
- **Layer Freezing**: Tested freezing encoder layers vs full fine-tuning
- **Learning Rate Impact**: Compared high vs low learning rates
- **Data Augmentation**: Applied augmentation techniques (e.g., time-stretching, noise injection) to increase training data variability and improve generalization.

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
## References
[1] NeuroSys, Wav2vec 2.0: A framework for self-supervised learning of speech repre-sentations, 2023. [Online]. Available: https://neurosys.com/blog/wav2vec-2-0-framework.

[2] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “Wav2vec 2.0: A framework for self-supervised learning of speech representations,” arXiv preprint arXiv:2006.11477, 2020. 
[Online]. Available: https://arxiv.org/abs/2006.11477.

[3] F. AI, Facebook/wav2vec2-xls-r-300m, 2023. [Online]. Available: https://huggingface.co/facebook/wav2vec2-xls-r-300m.

[4] [Online]. Available: https://openai.com/index/whisper.

[5] N. Alrashoudi, H. Al-Khalifa, and Y. Alotaibi, “Improving mispronunciation detec-tion and diagnosis for non-native learners of the arabic language,” Discover Com-puting, vol. 28, no. 1, p. 1, 2025.


