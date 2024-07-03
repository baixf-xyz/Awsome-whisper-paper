以下是优化后的 Markdown 文档，包含了关于 Whisper 的论文及其相关主题的详细信息：

---

# Awsome-whisper-paper

Papers about Whisper. If you feel there are papers with related topics missing or any mistake in this list, do not hesitate to let us know (via issues or pull requests).

## Whisper

- [[Blog]](https://openai.com/blog/whisper)
- [[Paper]](https://arxiv.org/abs/2212.04356)
- [[Model card]](https://github.com/openai/whisper/blob/main/model-card.md)
- [[Colab example]](https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb)

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitasking model that can perform multilingual speech recognition, speech translation, and language identification.

### Approach

![Approach](https://raw.githubusercontent.com/openai/whisper/main/approach.png)

A Transformer sequence-to-sequence model is trained on various speech processing tasks, including multilingual speech recognition, speech translation, spoken language identification, and voice activity detection. These tasks are jointly represented as a sequence of tokens to be predicted by the decoder, allowing a single model to replace many stages of a traditional speech-processing pipeline. The multitask training format uses a set of special tokens that serve as task specifiers or classification targets.

## Related Papers

### Model Architecture
- [Language independent end-to-end architecture for joint language identification and speech recognition](https://www.merl.com/publications/docs/TR2017-003.pdf) (Mitsubishi Electric Research Laboratories MERL; ASRU 2017)
- [Reproducing Whisper-Style Training Using an Open-Source Toolkit and Publicly Available Data](https://arxiv.org/abs/2306.13771) (ASRU 2023, Carnegie Mellon University, Yifan Peng, Jinchuan Tian)
- [Teach me with a Whisper: Enhancing Large Language Models for Analyzing Spoken Transcripts using Speech Embeddings](https://arxiv.org/abs/2311.07014) (arXiv 2023, Hasan F, Li Y, Foulds J, et al.; University of Maryland Baltimore County, IBM Research AI)
- [Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling](https://arxiv.org/abs/2311.00430) (arXiv 2023, Gandhi S, von Platen P, Rush A M.; Hugging Face)

### Multilingual ASR, Transfer Learning, and Fine-tuning
- [Chinese ASR and NER Improvement Based on Whisper Fine-Tuning](https://ieeexplore.ieee.org/document/9758210) (ICACT 2023, Huawei)
- [N-Shot Benchmarking of Whisper on Diverse Arabic Speech Recognition](https://www.isca-speech.org/archive/interspeech_2023/talafha23_interspeech.html) (Interspeech 2023, The University of British Columbia, Canada, Bashar Talafha, Abdul Waheed, Muhammad Abdul-Mageed)
- [Whisper Encoder Features for Infant Cry Classification](https://www.isca-speech.org/archive/interspeech_2023/). (Interspeech 2023, DA-IICT)
- [Adaptation of Whisper Models to Child Speech Recognition](https://www.isca-speech.org/archive/interspeech_2023/). (Interspeech 2023, University of Galway, Galway, Ireland, Xishabh Jain, Andrei Barcovschi, Mariam Yahayah Yiwere, Peter Corcoran, Horia Cucu)
- [Zero-Shot Domain-Sensitive Speech Recognition with Prompt-Conditioning Fine-Tuning](https://ieeexplore.ieee.org/document/9455454) (ASRU 2023, July 2023, MediaTek Research)
- [Adapting OpenAI’s Whisper for Speech Recognition on Code-Switch Mandarin-English SEAME and ASRU2019 Datasets](https://arxiv.org/abs/2311.17382) (arXiv 2023, Yuhang Yang; Hunan University)
- [LoRA-Whisper Parameter-Efficient and Extensible Multilingual ASR](https://www.isca-speech.org/archive/interspeech_2023/) (Interspeech 2024, Shanghai Jiao Tong University)

### VAD
- [WhisperX: Time-Accurate Speech Transcription of Long-Form Audio](https://www.isca-speech.org/archive/interspeech_2023/) (INTERSPEECH 2023, Bain M, Huh J, Han T, et al.; University of Oxford)

### Classification
- [A Whisper Transformer for Audio Captioning Trained with Synthetic Captions and Transfer Learning](https://arxiv.org/abs/2308.03229) (arXiv 2023, Masaryk University)
- [Multi-Resolution Approach to Identification of Spoken Languages and to Improve Overall Language Diarization System Using Whisper Model](https://www.isca-speech.org/archive/interspeech_2023/) (Interspeech 2023, Augnito)
- [Whisper-AT: Noise-Robust Automatic Speech Recognizers are Also Strong General Audio Event Taggers](https://www.isca-speech.org/archive/interspeech_2023/) (Interspeech 2023, MIT)

### Hot Words and Keywords
- [Prompt Tuning for Speech Recognition on Unknown Spoken Name Entities](https://www.isca-speech.org/archive/interspeech_2023/) (Submitted to Interspeech 2024)
- [Can Whisper Perform Speech-Based In-Context Learning?](https://ieeexplore.ieee.org/document/9506421) (ICASSP 2024, THU, Siyin Wang, Chao-Han Huck Yang, Ji Wu, Chao Zhang)
- [Keyword-Guided Adaptation of Automatic Speech Recognition](https://arxiv.org/abs/2406.00123) (aiOla Research, Israel, arxiv 2024)

### Target Speaker Recognition (TSE)
- [Extending Whisper with Prompt Tuning to Target-Speaker ASR](https://ieeexplore.ieee.org/document/9506421) (ICASSP 2024, Shandong University)

### Zero-Shot and In-Context Learning (ICL)
- [Zero-Shot Domain-Sensitive Speech Recognition with Prompt-Conditioning Fine-Tuning](https://ieeexplore.ieee.org/document/9455454) (ASRU 2023, July 2023, MediaTek Research)
- [Can Whisper Perform Speech-Based In-Context Learning](https://ieeexplore.ieee.org/document/9506421) (ICASSP 2024, Siyin Wang, Chao-Han Huck Yang, Ji Wu, Chao Zhang; THU)
- [Prompting the Hidden Talent of Web-Scale Speech Models for Zero-Shot Task Generalization](https://www.isca-speech.org/archive/interspeech_2023/) (Interspeech 2023, The University of Texas at Austin, Carnegie Mellon University; Peng P, Yan B, Watanabe S, et al.)

### Audio-Vision Speech Recognition (AVSR)
- [Visual Speech Recognition for Low-Resource Languages with Automatic Labels from Whisper Model](https://ieeexplore.ieee.org/document/9506421) (ICASSP 2024, KAIST, South Korea, CMU)

### Generative Error Correction (GER) and LLM
- [Salmonn: Towards Generic Hearing Abilities for Large Language Models](https://arxiv.org/abs/2310.13289) (arXiv 2023, Thu, Bytedance)
- [Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models](https://arxiv.org/abs/2311.07919) (arXiv 2023, Alibaba)
- [HyPoradise: An Open Baseline for Generative Speech Recognition with Large Language Models](https://papers.nips.cc/paper/2023/hash/9e10b4ef91cb244f3a3dd3b3a5302c50-Abstract.html) (NeurIPS 2023, NTU)
- [Generative Speech Recognition Error Correction with Large Language Models and Task-Activating Prompting](https://ieeexplore.ieee.org/document/9506421) (Amazon, ASRU 2023)
- [Generative Error Correction for Code-Switching Speech Recognition Using Large Language Models](https://arxiv.org/abs/2310.13013) (Chen C, Hu Y, Yang C H H, et al.; NTU; arXiv 2023)
- [Whispering LLaMA: A Cross-Modal Generative Error Correction Framework for Speech Recognition](https://aclanthology.org/2023.emnlp-main.100) (EMNLP 2023, Radhakrishnan S, Yang C H H, Khan S A, et al.)
- [Listen Again and Choose the Right Answer: A New Paradigm for Automatic Speech Recognition with Large Language Models](https://ieeexplore.ieee.org/document/9506421) (Yuchen Hu; NTU)
- [Integrated into Large Language Models (LLMs) for GER-Based Speech Recognition Tasks](https://openreview.net/forum?id=L3e8DXt8fG) (Chenchen, ICLR 2024, NTU)
- [Large Language Models are Efficient Learners of Noise-Robust Speech Recognition](https://openreview.net/forum?id=L3e8DXt8fG) (NTU, ICLR 2024)

### Other
- [WhiSLU: End-to-End Spoken Language Understanding with Whisper](https://www.isca-speech.org/archive/interspeech_2023/) (Interspeech 2023, Huawei)

---
