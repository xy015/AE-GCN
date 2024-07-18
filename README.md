# AE-GCN
Code for the article 'High-risk Factor Prediction in Lung Cancer Using Thin CT Scans: An Attention-Enhanced Graph Convolutional Network Approach'
## Introduction
In medical scenarios, identifying high-risk factors in early-stage lung cancers is vital for surgical planning. Currently, surgeons rely on intraoperative pathological analyses for decision-making. However,preoperative determination, such as using CT images, remains a significant challenge. 

In this paper, we introduce the AE-GCN model, a novel approach combining attentional features with spatial information to tackle this challenge. We develop a novel graph construction method integrating nodule positionalinformation with GCN, enabling more efficient and faster model training. Experimental results demonstrated that the proposed model outperforms many of the previous benchmarks. The AE-GCN model enables thoracic surgeons to assess the high-risk probability of pulmonary nodules using only CT images,aiding in preoperative surgical planning.

The effectiveness of our proposed method is demonstrated using real-world data collected from a multi-center cohort.
## Model
Figure 1 shows the flowchart of the proposed AE-GCN.
![image](https://github.com/xy015/AE-GCN/blob/main/Model.png?raw=true)
The input is an array of CT images with regard to a specific patient, with nodule centers manually labelled, in advance. Module A represents the CBAM feature extractor, utilizing both channel and spatial attention mechanism on a pre-trained VGG (a zoom-in view of the attention mechanism is provided below). Module B encompasses both the graph construction and GCN classification. The final nodule-level prediction is obtained by averaging the slice-level probabilities.
## Requirements
The code is written in Python and requires the following packages:
* Python 3.8.12
* Tensorflow 2.6.0 
* Keras 2.3.1
* Matplotlib 3.4.3
* Numpy 1.19.5
* Pandas 1.3.4
* Sklearn 0.19.3
* Scipy 1.7.1
* tqdm 4.66.4
* tf_geometric 0.1.6


