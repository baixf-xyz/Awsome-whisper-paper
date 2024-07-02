# Awsome-whisper-paper

Papers about Whisper. If you feel there are papers with related topics missing or any mistake in this list, do not hesitate to let us know (via issues or pull requests). 

---


## Whisper

[[Blog]](https://openai.com/blog/whisper)
[[Paper]](https://arxiv.org/abs/2212.04356)
[[Model card]](https://github.com/openai/whisper/blob/main/model-card.md)
[[Colab example]](https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb)
Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitasking model that can perform multilingual speech recognition, speech translation, and language identification.


### Approach

![Approach](https://raw.githubusercontent.com/openai/whisper/main/approach.png)

A Transformer sequence-to-sequence model is trained on various speech processing tasks, including multilingual speech recognition, speech translation, spoken language identification, and voice activity detection. These tasks are jointly represented as a sequence of tokens to be predicted by the decoder, allowing a single model to replace many stages of a traditional speech-processing pipeline. The multitask training format uses a set of special tokens that serve as task specifiers or classification targets.



## Related Papers

### Model Archicture
* Language independent end-to-end architecture for joint language identification and speech recognition(Mitsubishi Electric Research Laboratories MERL；ASRU2017)
* Reproducing Whisper-Style Training Using an Open-Source Toolkit and Publicly Available Data(ASRU 2023，Carnegie Mellon University，Yifan Peng, Jinchuan Tian)
* Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling(arXiv:2311.00430, 2023.;Gandhi S, von Platen P, Rush A M. ;Hugging Face)

### Mutiligual ASR , Transfer Learning and Fine-tuning
* N-Shot Benchmarking of Whisper on Diverse Arabic Speech Recognition（Interspeech2023，The University of British Columbia, Canada，Bashar Talafha, Abdul Waheed, Muhammad Abdul-Mageed）
* Adaptation of Whisper models to child speech recognition(Interspeech2023，University of Galway, Galway, Ireland，Xishabh Jain, Andrei Barcovschi, Mariam Yahayah Yiwere, Peter Corcoran, Horia Cucu)
* Zero-Shot Domain-Sensitive Speech Recognition with Prompt-Conditioning Fine-Tuning.(ASRU2023, July 2023，MediaTek Research)
* Adapting openai’s whisper for speech recognition on code-switch mandarin-english seame and asru2019 datasets(arXiv:2311.17382, 2023;Yuhang Yang;Hunan University)
* LoRA-Whisper Parameter-Efficient and Extensible Multilingual ASR(Interspeech 2024, Shanghai Jiao Tong University)

### VAD
* WhisperX: Time-Accurate Speech Transcription of Long-Form Audio (INTERSPEECH 2023,Bain M, Huh J, Han T, et al. University of Oxford)


### Hot Word
* Can whisper perform speech-based in-context learning?(ICASSP 2024;THU;Siyin Wang, Chao-Han Huck Yang, Ji Wu, Chao Zhang)

### Target speaker recognization (TSE)
* Extending Whisper with prompt tuning to target-speaker ASR(ICASSP 2024,Shandong University)

### Zero-shot and In Context Learning(ICL)
* Zero-Shot Domain-Sensitive Speech Recognition with Prompt-Conditioning Fine-Tuning.(ASRU2023, July 2023，MediaTek Research)
* Can Whisper perform speech-based in-context learning(ICASSP2024；Siyin Wang, Chao-Han Huck Yang, Ji Wu, Chao Zhang；THU)
* Prompting the Hidden Talent of Web-Scale Speech Models for Zero-Shot Task Generalization（Interspeech 2023；The University of Texas at Austin、Carnegie Mellon University；Peng P, Yan B, Watanabe S, et al.)

### Generative Error Correction(GER) and LLM
* Salmonn: Towards generic hearing abilities for large language models(arXiv:2310.13289,Thu，Bytedance)
* Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models(arXiv: 2311.07919；Alibaba)
* HyPoradise: An Open Baseline for Generative Speech Recognition with Large Language Models.(arXiv preprint arXiv:2309.15701, 2023.;NTU；Chen C, Hu Y, Yang C H H, et al.)
* Generative speech recognition error correction with large language models and task-activating prompting(Amazon;ASRU2023)
* Generative error correction for code-switching speech recognition using large language models(Chen C, Hu Y, Yang C H H, et al.；NTU；arXiv:2310.13013, 2023.）
* Whispering LLaMA: A Cross-Modal Generative Error Correction Framework for Speech Recognition (EMNLP2023，Radhakrishnan S, Yang C H H, Khan S A, et al.)
* Listen Again and Choose the Right Answer: A New Paradigm for Automatic Speech Recognition with Large Language Models（Yuchen Hu;NTU)
* Integrated into Large Language Models (LLMs) for GER-based speech recognition tasks(chenchen，ICLR 2024, NTU)
* Large Language Models are Efficient Learners of Noise-Robust Speech Recognition(NTU;ICLR2024)



