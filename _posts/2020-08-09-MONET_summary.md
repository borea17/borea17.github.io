---
title: "MONet: Unsupervised Scene Decomposition and Representation"
abb_title: "MONet: Unsupervised Scene Decomposition and Representation"
permalink: "/paper_summaries/multi-object_network"
author: "Markus Borea"
tags: ["unsupervised learning", "object detection", "generalization", "varational autoencoder"]
published: false
toc: true
toc_sticky: true
toc_label: "Table of Contents"
---

NOTE: THIS IS CURRENTLY WIP

[Burgess et al. (2019)](https://arxiv.org/abs/1901.11390) developed
the **Multi-Object Network (MONet)** as an end-to-end trainable model to
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
instantiations of the same class). To put this into practice, [Burgess
et al. (2019)](https://arxiv.org/abs/1901.11390) developed a 
compositional generative model architecture incorporating two kinds of
neural networks that are trained in tandem:

* **Attention Network**: Its purpose is to deliver binary attention
  masks $\textbf{m}\_k$ for the image such that the whole image is
  completely spatially decomposed into $K$ parts, i.e., $\sum_{k=1}^K
  \textbf{m}_k = \textbf{1}$. Ideally, after training each mask focuses on a
  semantically meaningful element/segment of the image. 
  Thus, it may also be understood as a *segmentation network*.
  
  To allow for a variable number of attention masks, [Burgess et al.
  (2019)](https://arxiv.org/abs/1901.11390) used a
  recurrent neural network $\alpha_{\boldsymbol{\psi}}$ for the
  decomposition. Therein an auto-regressive process is defined for the
  ongoing state. 
  This state is called "scope" $\textbf{s}_k \in \\{0, 1\\}^{W\times
  H}$ (image width $W$ and height $H$) as it is
  used to track the image parts that remain to be explained, i.e., the
  scope for the next state is given by 

  $$
     \textbf{s}_{k+1} = \textbf{s}_k \odot \left(\textbf{1} -
  \underbrace{\alpha_{\boldsymbol{\psi}} \left( \textbf{x};
  \textbf{s}_{k} \right)}_{\{0,1\}^{W \times H}} \right)
  $$

  with the first scope $\textbf{s}_0 = \textbf{1}$ ($\odot$ denotes
  element-wise multiplication). The attention
  masks are given by
  
  $$
    \textbf{m}_k  = \begin{cases} \textbf{s}_{k-1} \odot
    \alpha_{\boldsymbol{\psi}} \left( \textbf{x}; \textbf{s}_{k-1}
    \right) & \forall k < K \\ 
    \textbf{s}_{k-1} & k=K \end{cases}
  $$
  
* **Component VAE**: Its purpose is to represent each masked region in a
  common latent code, i.e., each segment is encoded by the same
  VAE[^1]. The encoder distribution $q\_{\boldsymbol{\phi}}
  \left(\textbf{z}_k | \textbf{x}, \textbf{m}_k\right)$
  is conditioned both on the input image $\textbf{x}$ and the corresponding attention mask
  $\textbf{m}_k$. I.e., instead of feeding each masked region into the
  network, [Burgess et al. (2019)](https://arxiv.org/abs/1901.11390)
  use the whole image $\textbf{x}$ concatenated with the corresponding
  attention mask $\textbf{m}_k$. As a result, we get $K$ different
  latent codes $\textbf{z}_k$ (termed "slots") which represent the
  features of each object (masked region) in a common latent/feature
  space across all objects.
  
  The decoder distribution $p\_{\boldsymbol{\theta}}$ is required to reconstruct the image
  $\widetilde{\textbf{x}} \sim p\_{\boldsymbol{\theta}} \left( \textbf{x} | \textbf{z}_k \right)$
  and the binary attention masks[^2] $\widetilde{\textbf{m}}_k \sim p\_{\boldsymbol{\theta}}
  \left(\textbf{c} | \textbf{z}_k \right)$ from these latent codes.
  Note that $p\_{\boldsymbol{\theta}} \left(\textbf{c} | \textbf{z}_k
  \right)$ defines the mask distribution of the Component VAE, whereas
  $q\_{\boldsymbol{\psi}} \left(\textbf{c} | \textbf{x}\right)$
  denotes the mask distribution of the attention network[^3]. 
  
  Importantly, each of the $k$ reconstructions is multiplied with the
  corresponding binary attention mask $\textbf{m}_k$, i.e., 
  
  $$
     \mathcal{L}_k = \textbf{m}_k \odot p_{\boldsymbol{\theta}} \left(
     \textbf{x} | \textbf{z}_k
     \right).
  $$
  
  The reconstruction accuracy of the whole image is given by
  
  $$
  \text{Reconstruction Accuracy} = \log \left(\sum_{k=1}^K
  \mathcal{L}_k \right) = \log \left( \sum_{k=1}^K \textbf{m}_k \odot p_{\boldsymbol{\theta}} \left(\textbf{x} | \textbf{z}_k \right)\right),
  $$
  
  where the sum can be understood as a full reconstruction of the
  image conditioned on the latent codes $\textbf{z}_k$ and the
  attention masks $\textbf{m}_k$. This accuracy is unconstrained
  outside of the masked regions. 
  
The whole model is end-to-end trainable with the following loss
function 

$$
\mathcal{L}\left(\boldsymbol{\phi}; \boldsymbol{\theta};
\boldsymbol{\psi}; \textbf{x} \right) = - \log \sum_{k=1}^K
$$
  
  
[^1]: Encoding each segment through the same VAE can be understood as
    an architectural prior on common structure within individual
    objects. 
  
[^2]: [Burgess et al. (2019)](https://arxiv.org/abs/1901.11390) do not
    explain why the Component VAE should also model the attention
    masks. Note however that this allows for better generalization,
    e.g., shape/class variation depends on attention mask. 


[^3]: For completeness $\textbf{c} \in \\{1, \dots, K\\}$ denotes a
    categorical variable to indicate the probability that pixels
    belong to a particular component $k$, i.e., $\textbf{m}_k =
    p(\textbf{c} = k)$. 
    
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
  
  
  
  
  Note that instead of
  feeding each masked region into the network, the whole image
  $\textbf{x}$ is used and the reconstructions $p\_{\boldsymbol{\theta}}
  \left( \textbf{x} | \textbf{z}_k \right)$ are multiplied with the
  binary attention masks. 
  
  
  


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




## Implementation


## Drawbacks of Paper

* static scene decomposition
* only works on simply images in which multiple objects of the same
class occur
* even simple images high training times

--------------------------------------------------------------------------------------------
