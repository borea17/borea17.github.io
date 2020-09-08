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
their model combines a Variational Auto-Encoder
([VAE](https://borea17.github.io/paper_summaries/auto-encoding_variational_bayes))
with a recurrent attention network
([U-Net](https://borea17.github.io/paper_summaries/u_net)
*segmentation network*) to spatially decompose
scenes into attention masks (over which the VAE needs to
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
compositional generative model architecture in which scenes are
spatially decomposed into parts that have to be individually modelled
through a common representation code. The architecture incorporates two
kind of neural networks that are trained in tandem:

* **Attention Network**: Its purpose is to deliver attention
  masks $\textbf{m}\_k$ for the image such that the whole image is
  completely spatially decomposed into $K$ parts, i.e., $\sum_{k=1}^K
  \textbf{m}_k = \textbf{1}$. Ideally, after training each mask focuses on a
  semantically meaningful element/segment of the image. 
  Thus, it may also be understood as a *segmentation network*.
  
  To allow for a variable number of attention masks, [Burgess et al.
  (2019)](https://arxiv.org/abs/1901.11390) use a
  recurrent neural network $\alpha_{\boldsymbol{\psi}}$ for the
  decomposition. Therein, an auto-regressive process is defined for the
  ongoing state. 
  This state is called "scope" $\textbf{s}_k \in [0, 1]^{W\times
  H}$ (image width $W$ and height $H$) as it is
  used to track the image parts that remain to be explained, i.e., the
  scope for the next state is given by 

  $$
     \textbf{s}_{k+1} = \textbf{s}_k \odot \left(\textbf{1} -
  \underbrace{\alpha_{\boldsymbol{\psi}} \left( \textbf{x};
  \textbf{s}_{k} \right)}_{[0,1]^{W \times H}} \right)
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
  
  By construction, we get that 
  
  $$
  \begin{align}
    &\textbf{s}_{k} = \textbf{s}_{k+1} + \textbf{m}_{k + 1} =
    \textbf{s}_{k+2} + \textbf{m}_{k+2} + \textbf{m}_{k+1} \\ 
    \textbf{1}=&\textbf{s}_0 =
    \textbf{s}_{K-1} + \sum_{k=1}^{K-1} \textbf{m}_{k} = \sum_{k=1}^K \textbf{m}_k,
  \end{align}
  $$

  i.e., at each recursion the remaining part to be explained
  $\textbf{s}_{k}$ is divided into a segmentation mask
  $\textbf{m}\_{k+1}$ and a new scope $\textbf{s}\_{k+1}$ such that
  with $\textbf{s}_0=\textbf{1}$ the entire image is explained by the
  resulting segmentation masks, i.e., $\sum\_{k=1}^K \textbf{m}_k = \textbf{1}$.
    
  
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
  and the attention masks[^2] $\widetilde{\textbf{m}}_k \sim p\_{\boldsymbol{\theta}}
  \left(\textbf{c} | \textbf{z}_k \right)$ from these latent codes.
  Note that $p\_{\boldsymbol{\theta}} \left(\textbf{c} | \textbf{z}_k
  \right)$ defines the mask distribution of the Component VAE, whereas
  $q\_{\boldsymbol{\psi}} \left(\textbf{c} | \textbf{x}\right)$
  denotes the mask distribution of the attention network[^3]. 
  
  Importantly, each of the $k$ reconstruction distributions is
  multiplied with the corresponding attention mask
  $\textbf{m}_k$, i.e., 

  $$
  \text{Reconstruction Distribution}_k = \textbf{m}_k \odot
     p_{\boldsymbol{\theta}} \left(\textbf{x} | \textbf{z}_k \right). 
  $$

  The reconstruction accuracy (decoder log likelihood, see my post on
  [VAEs](https://borea17.github.io/paper_summaries/auto-encoding_variational_bayes#model-description))
  of the whole image is given by 
  
  $$
  \text{Reconstruction Accuracy} = \log \left( \sum_{k=1}^K \textbf{m}_k \odot p_{\boldsymbol{\theta}} \left(\textbf{x} | \textbf{z}_k \right)\right),
  $$
  
  where the sum can be understood as the reconstruction distribution
  of the whole image (mixture of components) conditioned on the latent
  codes $\textbf{z}_k$ and the attention masks $\textbf{m}_k$.
  [Burgess et al. (2019)](https://arxiv.org/abs/1901.11390) chose a
  Gaussian distribution with fixed variance as the decoder
  distribution $p\_{\boldsymbol{\theta}} \left( \textbf{x} |
  \textbf{z}_k \right) \sim \mathcal{N} \left(
  \boldsymbol{\mu}_k (\boldsymbol{\theta}), \sigma_k^2 \textbf{I}_d \right)$, hence the term can be
  rewritten as follows 
  
  $$
  \begin{align}
  \text{Recons. Acc.} &= \log \left(  \sum_{k=1}^K \sum_{i=1}^N
  m_{k, i} \frac {1}{ \sqrt{2 \pi}^{d} \cdot
  \sqrt{\left(\sigma_k^2\right)^{d}}} \exp \left( - \frac { \left[ x_i -
  \mu_{k,i} (\boldsymbol{\theta}) \right]^2 } {2 \sigma_k^2}  \right)
  \right)\\
  &= \log \frac {1}{\sqrt{2\pi}^{d}} \left( \sum_{k=1}^K \sum_{i=1}^N \exp \left( \log \frac
  {m_{k, i}}{\sqrt{\left( \sigma_k^2 \right)^{d}}}
  \right) \exp \left( - \frac {\left[x_i -
  \mu_{k,i} (\boldsymbol{\theta}) \right]^2 } {2 \sigma_k^2}
  \right)\right)\\
  &\propto \log \left( \sum_{k=1}^K \sum_{i=1}^N \exp \left(
  \left(\sigma_k^2\right)^{- \frac {d}{2}}
  \log m_{k, i} -  \frac { \left[x_i -
  \mu_{k, i} (\boldsymbol{\theta}) \right]^2 } {2 \sigma_k^2}  \right)  \right),
  \end{align}
  $$
  
  where $d$ denotes the dimensionalty of the latent space and $i$
  enumerates the pixel space. Note that
  the accuracy is unconstrained outside of the masked regions for each
  reconstruction[^4]. 
  
[^4]: In practice, the reconstruction accuracy becomes incalculable for binary
    attention masks, since $\log 0$ is undefined. However, the
    reconstruction accuracy is effectively unconstrained outside of
    masked regions, since for $\lim \log m\_{k,i} \rightarrow -\infty$
    the reconstruction accuracy for the corresponding slot also
    approaches negative infinity (resulting in vanishing gradients).
  
The figure below summarizes the whole architecture of the model by
showing the individual components (attention network, component VAE)
and their interaction.

| ![Schematic of MONet](/assets/img/04_MONet/MONet_schematic.png "Schematic of MONet") |
| :--         |
| Schematic of MONet. (a) The overall compositional generative model architecture is represented by showing schematically how the attention network and the component VAE interact with the ground truth image. (There is a small mistake, the last attention mask should be $\textbf{s}_{K-1}$ instead of $\textbf{s}_K$). (b) The attention network is used for a recursive decomposition process to generate attention masks $\textbf{m}_k$. (c) The Component VAE takes as input the image $\textbf{x}$ and the corresponding attention mask $\textbf{m}_k$ and reconstructs both. Taken from [Burgess et al. (2019)](https://arxiv.org/abs/1901.11390). |

The whole model is end-to-end trainable with the following loss
function 

$$
\begin{align}
\mathcal{L}\left(\boldsymbol{\phi}; \boldsymbol{\theta};
\boldsymbol{\psi}; \textbf{x} \right) &= \underbrace{- \log \left( \sum_{k=1}^K \textbf{m}_k \odot p_{\boldsymbol{\theta}} \left(\textbf{x} |
\textbf{z}_k \right)\right)}_{\text{Reconstruction Error }
\widetilde{\textbf{x}} \odot \textbf{m}_k} + \beta
\underbrace{D_{KL} \left( \prod_{k=1}^K q_{\boldsymbol{\phi}} \left(\textbf{z}_k |
\textbf{x}, \textbf{m}_k\right) || p(\textbf{z})
\right)}_{\text{Regularization Term}}\\
&+ \gamma \underbrace{D_{KL} \left( q_{\boldsymbol{\psi}} \left( \textbf{c} |
\textbf{x} \right) || p_{\boldsymbol{\theta}} \left( \textbf{c} | \{
\textbf{z}_k \} \right) \right)}_{\text{Reconstruction Error } \widetilde{\textbf{m}}_k},
\end{align}
$$

where the first term measures the reconstruction error of the
fully reconstructed image (sum) as mentioned above. The second term is
the KL divergence between the variational posterior
approximation factorized across slots, i.e., $q\_{\boldsymbol{\phi}}
\left( \textbf{z} | \textbf{x} \right) = \prod_{k=1}^K
q\_{\boldsymbol{\phi}} \left(\textbf{z}_k| \textbf{x},
\textbf{m}_k\right)$, and the prior of the latent distribution
$p(\textbf{z})$. As this term pushes the encoder distribution to be
close to the prior distribution, it is commonly referred to as
*regularization term*. It is weighted by the tuneable
hyperparameter $\beta$ to encourage learning of disentanglement latent
representions, see [Higgins et al.
(2017)](https://deepmind.com/research/publications/beta-VAE-Learning-Basic-Visual-Concepts-with-a-Constrained-Variational-Framework).
Note that the first two terms are derived from the standard VAE loss.  
The third term is the KL divergence between the attention mask
distribution generated by the attention network
$q\_{\boldsymbol{\psi}} \left( \textbf{c} | \textbf{x} \right)$ and
the component VAE $p\_{\boldsymbol{\theta}}
\left(\textbf{c} |\\{\textbf{z}_k\\} \right)$, i.e., it forces these
distributions to lie close to each other. It could be understood as
the reconstructions error of the VAE's attention masks
$\widetilde{\textbf{m}}_k$, as it forces them to lie close to the
attention masks $\textbf{m}_k$ of the attention network. Note however
that the attention network itself is trainable, thus the network could
also react by pushing the attention mask distribution towards the
reconstructed mask distribution of the VAE. $\gamma$ is a tuneable
hypeterparameter to modulate the importance of this term, i.e.,
increasing $\gamma$ results in close distributions.
  
  
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
    
**Motivation**: The model aims to produce semantically meaningful
decompositions in terms of segmentation and latent space attributes.
Previous work such as the [Spatial Broadcast
decoder](https://borea17.github.io/paper_summaries/spatial_broadcast_decoder)
has shown that VAEs are extensively capable of decomposing *simple*
single-object scenes into disentangled latent space representations.
However, even *simple* multi-object scenes are far more challenging to
encode due to their complexity. [Burgess et al.
(2019)](https://arxiv.org/abs/1901.11390) hypothesize that exploiting
the compositional structure of scenes (inductive bias) may help to
reduce this complexity. Instead of decomposing the entire multi-object
scene in one sweep, MONet breaks the image in multiple ($K$) tasks which it
decomposes with the same VAE[^5]. As a result, the segmentation should 
produce similar tasks (structurally similar scene elements) such that
the VAE is capable of solving all tasks. The authors argue that
optimization should push towards a meaningful decomposition.
Furthermore, they empirically validate their hypothesis by showing
that for the *Objects Room* dataset the reconstruction error is much
lower when the ground truth attention masks are given compared to an
*all-in-one* (single sweep) or a *wrong* masks situation.

Adding some more motivation: It might be helpful to think about the
data-generating process: Commonly, *artificial* multi-object scenes are
created by adding each object successively to the image. Assuming that
each of these objects is generated from the same class with different
instantiations (i.e., different color/shape/size/...), it seems most
natural to recover this process by decomposing the image and then
decoding each part.

[^5]: Philosophical note: Humans also tend to work better when focusing on one task at a time. 


## Implementation

[Burgess et al. (2019)](https://arxiv.org/abs/1901.11390) tested MONet
on three different multi-object scene datasets (*Objects Room*, *CLEVR*,
*Multi-dSprites*) and showed that their model could successively learn to
<!-- decompose scenes into semantically meaningful parts (i.e., produce -->
<!-- meaningful segmentation masks), to represent each segmented object in a -->
<!-- common (nearly disentangled) latent code, and to generalize to unseen -->
<!-- scene configurations without any supervision.  -->

* decompose scenes into semantically meaningful parts, i.e.,
  produce meaningful segmentation masks,
* represent each segmented object in a common (nearly disentangled) latent
  code, and 
* generalize to unseen scene configurations

without any supervision. Notably, MONet can handle a variable number
of objects by producing latent codes that map to an empty scene,
see image below. Furthermore, it turned out that MONet is also able to
deal with occlusions: In the CLEVR dataset the unmasked
reconstructions could even recover occluded objects. [Burgess et al.
(2019)](https://arxiv.org/abs/1901.11390) argue that this indicates
how `MONet is learning from and constrained by the structure of the
data`.


| ![MONet Paper Results](/assets/img/04_MONet/paper_results.png "MONet Paper Results") |
| :--         |
| MONet Paper Results: Decomposition on *Multi-dSprties* and *CLEVR* images. First row shows the input image, second and third row the corresponding reconstructions and segmentations by MONet (trained for 1,000,000 iterations). The last three rows show the unmasked component reconstructions from some chosen slots (indicated by $S$). Red arrows highlight occluded regions of shapes that are completed as full shapes. Taken from [Burgess et al. (2019)](https://arxiv.org/abs/1901.11390). |

The following reimplementation intends to be 



### Data Generation

### Model Implementation

* **Attention Network**: For the implementation of the attention
  network, it is worth to look at the reconstruction error in more
  detail:

* **Component VAE**:
  * latent distribution Gaussian with diagonal covariance
  * decoder: Spatial Broadcast Decoder, parametrises 
    means of pixel-wise independent distributions 
    

## Drawbacks of Paper

* static scene decomposition (valuable prior is ignored)
* only works on simple images in which multiple objects of the same
class occur
* even simple images require high training times
* lots of hyperparameters (network architectures, $\beta$, $\gamma$, optimization)

--------------------------------------------------------------------------------------------