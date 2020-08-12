---
title: "MONet: Unsupervised Scene Decomposition and Representation"
abb_title: "MONet: Unsupervised Scene Decomposition and Representation"
permalink: "/blog/multi-object_network"
author: "Markus Borea"
tags: [machine learning, unsupervised object detection, generalization, varational autoencoder]
published: false
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
*images* of the world) can often be approximated as a composition of
individual components (e.g., objects). To put this into practice,
[Burgess et al. (2019)](https://arxiv.org/abs/1901.11390) developed a
compositional generative model architecture incorporating two kind of
neural networks:
* **Attention Network**
* **Component VAE**

- might be helpful to think from the otherside:
=> Having multiple objects => encode each of them in the same latent
space
=> the world/image is then composed as sum of these objects

Therefore, MONet incorporates
two kind of neural networks 




The main idea



## Learning the Model
