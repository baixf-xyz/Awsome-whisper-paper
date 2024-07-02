![image](https://github.com/baixf-xyz/Awsome-whisper-paper/assets/69945562/5ea52d1b-63df-4efb-81ff-87272fac8cab)# Awsome-whisper-paper

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


### Mutiligual ASR , Transfer Learning and Fine-tuning
* LoRA-Whisper Parameter-Efficient and Extensible Multilingual ASR(Interspeech 2024, Shanghai Jiao Tong University)

### Target speaker recognization (TSE)
* Extending Whisper with prompt tuning to target-speaker ASR(ICASSP 2024,Shandong University)


### GER and LLM
* Salmonn: Towards generic hearing abilities for large language models(arXiv:2310.13289,Thu，Bytedance)
* Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models(arXiv: 2311.07919；Alibaba)
* Generative speech recognition error correction with large language models and task-activating prompting(Amazon;ASRU2023)
* Listen Again and Choose the Right Answer: A New Paradigm for Automatic Speech Recognition with Large Language Models（Yuchen Hu;NTU)
* Large Language Models are Efficient Learners of Noise-Robust Speech Recognition(NTU;ICLR2024)



