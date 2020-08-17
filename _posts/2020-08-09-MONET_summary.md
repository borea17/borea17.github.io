---
title: "MONet: Unsupervised Scene Decomposition and Representation"
abb_title: "MONet: Unsupervised Scene Decomposition and Representation"
permalink: "/paper_summaries/multi-object_network"
author: "Markus Borea"
tags: [machine learning, unsupervised object detection, generalization, varational autoencoder]
published: false 
toc: true
toc_sticky: true
toc_label: "Table of Contents"
---

NOTE: THIS IS CURRENTLY WIP

[Burgess et al. (2019)](https://arxiv.org/abs/1901.11390) developed
the Multi-Object Network (MONet) as an end-to-end trainable model to
decompose images into meaningful entities such as objects. Notably,
the whole training process is unsupervised, i.e., there are no labeled
segmentations, handcrafted bounding boxes or whatsoever. In essence,
their model combines a Variational Auto-Encoder (VAE) with a recurrent
attention network (*segmentation network*) to spatially decompose
scenes into binary attention masks (over which the VAE needs to
reconstruct masked regions) and latent representations of each masked
region. As a proof of concept, they show that their model could learn
disentangled representations in a common latent code (i.e.,
representations of object features in latent space) and object
segmentations (i.e., attention masks on the original image) on
non-trivial 3D scenes. 

## Model Description

MONet builds upon the inductive bias that the world (or rather
*simple images* of the world) can often be approximated as a composition of
individual objects with the same underlying structure (i.e., different
instantiations of the same class). To put this into practice, [Burgess et al. (2019)](https://arxiv.org/abs/1901.11390) developed a
compositional generative model architecture incorporating two kinds of
neural networks:

* **Attention Network**: Its purpose is to deliver binary attention
  masks $\textbf{m}\_k$ for the image such that the whole image is
  completely spatially decomposed into $K$ parts, i.e., $\sum_{k=1}^K
  \textbf{m}_k = \textbf{1}$. Ideally, each mask focuses on a
  semantically meaningful element/segment of the image, i.e., object. 
  Thus, it may also be understood as a *segmentation network*.
  
* **Component VAE**: 


| ![Schematic of MONet](/assets/img/04_MONet/MONet_schematic.png "Schematic of MONet") |
| :--         |
| (left) Schematic of the Spatial Broadcast VAE. In the decoder, we broadcast (tile) the latent code $\textbf{z}$ of size $k$ to the image width $w$ and height $h$, and concatenate two "coordinate" channels. This is then fed to an unstrided convolutional decoder. (right) Pseudo-code of the spatial operation. Taken from [Watters et al. (2019)](https://arxiv.org/abs/1901.07017).|


**Motivation**: 
  - semantically meaningful decompositions
  - push optimization towards decomposition
  - hypothesis: network that build up scenes compositionally (by
    operating at the level of structurally similar scene elements)
    performs better than trained on entire image
  - wrong masks (segmentation) leads to bad performance, i.e., worse
    reconstruction accuracy 
    => optimization pushes towards meaningful decomposition
  - processing elements of secnes in a way that can exploit any common
    structure of the data mkaes more efficient use of neural network's capacity
  
  
  
**Motivation**:

- might be helpful to think from the otherside:
=> Having multiple objects => encode each of them in the same latent
space
=> the world/image is then composed as sum of these objects

Therefore, MONet incorporates
two kind of neural networks 




The main idea



## Learning the Model
