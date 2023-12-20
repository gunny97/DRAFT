# DRAFT (EMNLP 2023 Findings)
DRAFT: Dense Retrieval Augmented Few-shot Topic classifier Framework

Keonwoo Kim · Younggun Lee

https://arxiv.org/abs/2312.02532

## Abstract
With the growing volume of diverse information, the demand for classifying arbitrary topics has become increasingly critical. To address this challenge, we introduce DRAFT, a simple framework designed to train a classifier for few-shot topic classification. DRAFT uses a few examples of a specific topic as queries to construct Customized dataset with a dense retriever model. Multi-query retrieval (MQR) algorithm, which effectively handles multiple queries related to a specific topic, is applied to construct the Customized dataset. Subsequently, we fine-tune a classifier using the Customized dataset to identify the topic. To demonstrate the efficacy of our proposed approach, we conduct evaluations on both widely used classification benchmark datasets and manually constructed datasets with 291 diverse topics, which simulate diverse contents encountered in real-world applications. DRAFT shows competitive or superior performance compared to baselines that use in-context learning, such as GPT-3 175B and InstructGPT 175B, on few-shot topic classification tasks despite having 177 times fewer parameters, demonstrating its effectiveness.


<p align="center">
<img src=".\png\DRAFT_figure.png" height = "350" alt="" align=center />
</p>
(a) Overall pipeline of DRAFT. DRAFT receives n queries as input, and a trained classifier is only used in the test phase. 

(b) Illustration of MQR in two-dimensional space. A circle represents the normalized embedding space of texts in Data Collection. For each query, passages only within an angle size θ, calculated as a threshold from n query vectors, are retrieved as positive samples, while others are classified as negative samples.

## Main Result
Table presents the evaluation results of the few-shot classification tasks on diverse topics. We evaluate the F1 score for all 291 subtopics and aggregate the results based on the five major categories. Among the baselines with billions of parameters, except for InstructGPT 175B 1-shot, DRAFT, only with millions of parameters, demonstrates superior performance compared to the others in all categories. When considering the average rankings across five major categories, DRAFT achieves the highest rank of 1.4, followed by InstructGPT 175B 1-shot with an average rank of 1.6, implying DRAFT’s optimality.
<p align="center">
<img src=".\png\DRAFT_results.png" height = "450" alt="" align=center />
</p>

## Citation
If you find this repo useful, please cite our paper. 

```
@inproceedings{
anonymous2023memto,
title={{MEMTO}: Memory-guided Transformer for Multivariate Time Series Anomaly Detection},
author={Junho Song, Keonwoo Kim, Jeonglyul Oh, Sungzoon Cho},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=UFW67uduJd}
}
```
