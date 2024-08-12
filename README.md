# M-ENAS: Evolutionary neural architecture search for the automatic diagnosis of MDD using multi-modal MRI imaging
This repository contains the code and data for the paper "M-ENAS: Evolutionary neural architecture search for the automatic diagnosis of MDD using multi-modal MRI imaging". The project explores the integration of multimodal MRI data (sMRI, DTI, fMRI) using a novel neural architecture search method to improve diagnostic performance for MDD.

![Image](https://github.com/TTLi1996/M-ENAS/blob/main/M-ENAS.jpg)

# Highlights

·	We proposed a M-ENAS framework for the automatic diagnosis of MDD.

·	The M-ENAS method effectively extracts and fuses multimodal neuroimaging features.

·	M-ENAS achieves competitive performance on a private and an open-access dataset.

·	Our findings show that somatomotor network regions are crucial in MDD diagnosis.

# Summary
Major depressive disorder (MDD) is a prevalent mental disorder that poses a serious threat to human life and health. Various neuroimaging techniques offer quantifiable evidence from different perspectives for diagnosing MDD. However, existing computer-aided diagnosis models for MDD are typically based on the researchers’ experience and intuition. This approach is time-consuming and may hinder the discovery of the globally optimal network architecture. To address this issue, we proposed an Evolutionary Neural Architecture Search (M-ENAS) framework for the automatic diagnosis of MDD using multi-modal Magnetic Resonance Imaging (MRI). M-ENAS determines the optimal weights and network architecture through a two-stage architecture search method. Specifically, a One-shot NAS strategy was used to train the supernet weights during the supernet training stage, and a self-defined evolutionary learning search strategy was applied during the exploration of the optimal network architecture. Moreover, multi-modal neuroimaging features were integrated and recognized through feature concatenation and a fully connected layers with a softmax activation function. Lastly, we evaluated the performance of M-ENAS on both a private and an open-access dataset, demonstrating that M-ENAS outperforms existing self-defined state-of-the-art methods. Additionally, our findings reveal that the somatomotor network regions exhibit significant differences in the diagnosis of MDD compared to other subnetworks, which provide new insights into the neural mechanisms of depression.

# Dataset
In this work, The proposed model was comprehensively evaluated using both a private dataset and an open-access dataset.The dataset can be found at the following link: 
1. a private dataset: https://drive.google.com/drive/folders/1bVTjJMWI6IRAYurPGiaUDTntwMfJN8Mm?usp=sharing
2. an opne-access dataset: https://rfmri.org/REST-meta-MDD

# Requirements
The experiments related to this study were compiled using PyTorch-1.13 and executed on an NVIDIA A100 GPU, running on Ubuntu 20.04.
