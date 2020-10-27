---
title: "Attend, Infer, Repeat: Fast Scene Understanding with Generative Models"
permalink: "/paper_summaries/AIR"
author: "Markus Borea"
tags: ["unsupervised learning", "object detection", "generalization"]
published: false
toc: true
toc_sticky: true
toc_label: "Table of Contents"
type: "paper summary"
---

NOTE: THIS IS CURRENTLY WIP!

[Eslami et al. (2016)](https://arxiv.org/abs/1603.08575) introduce the
**Attend-Infer-Repeat (AIR)** framework as an end-to-end trainable
generative model capable of decomposing multi-object scenes into its 
constituent objects in an unsupervised learning setting. AIR builds
upon the inductive bias that real-world scenes can be understood as a
composition of (locally) self-contained objects. Therefore, AIR uses a
structured probabilistic model whose parameters are obtained by
inference/optimization. As the name suggests, the image decomposition
process can be abstracted into three steps: 

* **Attend**: Firstly, the model uses a [Spatial
  Transformer (ST)](https://borea17.github.io/paper_summaries/spatial_transformer)
  to focus on a specific region of the image, i.e., crop the image.
* **Infer**: Secondly, the cropped image is encoded by a [Variational
  Auto-Encoder (VAE)](https://borea17.github.io/paper_summaries/auto-encoding_variational_bayes).
  Note the same VAE is used for every cropped image.
* **Repeat**: Lastly, these steps are repreated until the full image
  is described.

Notably, the model can handle a variable number of objects by treating
inference as an iterative process. As a proof of concept, they show
that AIR could successfully learn to decompose multi-object scenes in 
multiple datasets (multiple MNIST, Sprites, Omniglot, 3D scenes).

| ![Paper Results](/assets/img/010_AIR/paper_results.gif "Paper Results") |
| :--  |
|  **Paper Results**. Taken from [this presentation](https://www.youtube.com/watch?v=4tc84kKdpY4). Note that the aim of unsupervised representation learning is to obtain good representations rather than perfect reconstructions.  |


<!-- {% capture code %}{% raw %}def this_is_a_test(asdb): -->
<!--     print(answer) -->
<!-- {% endraw %}{% endcapture %} -->
<!-- {% include code.html code=code lang="python" %} -->


<!-- - aim: good representations for downstream tasks -->
<!-- - scheme for efficient variational inference in latent spaces of -->
<!--   variable dimensionality -->
<!-- Motivated by human perception -->
<!-- - produce representations that are more useful for downstream tasks  -->
<!-- - structured models for image understanding -->
<!-- - standard VAEs lack interpretations of latent space (unstructured** -->
<!-- - AIR imposes structure on its representation through the generative -->
<!--   model/process rather than supervision from labels -->
<!-- - aiming to obtain good representations rather than good reconstructions -->


## Model Description

AIR is a rather sophisticated framework with some non-trivial
subtleties. For the sake of clarity, the following description is
organized as follows:  
Firstly, a high-level overview of the main ideas is given. Secondly,
the transition from these ideas into a mathmatical formulation
(ignoring difficulties) is described. Lastly, the main difficulties
are highlighted and how [Eslami et al.
(2016)](https://arxiv.org/abs/1603.08575) proposed to tackle them.

### High-Level Overview

In essence, the model can be understood as a special VAE architecture
in which an image $\textbf{x}$ is encoded to some kind of latent
distribution from which we sample the latent representation
$\\{\textbf{z}\\}$ which then can be decoded into an reconstructed
image $\widetilde{\textbf{x}}$, see image below. The main idea by [Eslami et al.
(2016)](https://arxiv.org/abs/1603.08575) consists of imposing
additional structure in the model using the inductive bias that
real-world scenes can often be approximated as multi-object scenes,
i.e., compositions of several (variable number) objects. Additionally,
they assume that all of these objects live in the same domain, i.e.,
each object is an instantiation from the same class.

| ![Standard VAE Architecture](/assets/img/010_AIR/standard_VAE.png "Standard VAE Architecture") |
| :---       |
| **Standard VAE Architecture**. AIR can be understood as a modified VAE architecture. |

To this end, [Eslami et al. (2016)](https://arxiv.org/abs/1603.08575)
replace the encoder with an recurrent, variable-length inference
network to obtain a group-structured latent representation.
Each group $\textbf{z}^{(i)}$ should ideally correspond to one object
where the entries can be understood as the compressed attributes of
that object (e.g., type, appearance, pose). The main purpose of the
inference network is to explain the whole scene by iteratively
updating what remains to be explained, i.e., each step is conditioned
on the image and on its knowledge of previously explained objects, see
image below. Since they assume that each object lives in the same
domain, the decoder is applied group-wise, i.e., each vector
$\textbf{z}^{(i)}$ is fed through the same decoder network, see image
below. 

| ![VAE with Recurrent Inference Network](/assets/img/010_AIR/VAE_inference.png "VAE with Recurrent Inference Network") |
| :---       |
| **VAE with Recurrent Inference Network**. A group-structured latent representation is obtained by replacing the encoder with a recurrent, variable-length inference network. This network should ideally attend to one object at a time and is conditioned on the image $\textbf{x}$ and its knowledge of previously epxlained objects $\textbf{h}$, $\textbf{z}$. |

[Eslami et al. (2016)](https://arxiv.org/abs/1603.08575) put
additional structure to the model by dividing the latent space of
each object into `what`, `where` and `pres`. As the names suggest,
$\textbf{z}^{(i)}\_{\text{what}}$ corresponds to the objects
appearance, while $\textbf{z}^{(i)}\_{\text{where}}$ gives information
about the position and scale. $\textbf{z}\_{\text{pres}}^{(i)}$ is
a binary variable describing whether an object is present, it is
rather a helper variable to allow for a variable number of objects to
be detected (going to be explained in the [Mathematical Model
section](https://borea17.github.io/paper_summaries/AIR#mathematical-model)).
To disentangle `what`from `where`


| ![Schematic of AIR](/assets/img/010_AIR/AIR_model.png "Schematic of AIR") |
| :---       |
| **Schematic of AIR**. A group-structured latent representation is obtained by replacing the encoder with a recurrent, variable-length inference network. This network should ideally attend to one object at a time and is conditioned on the image $\textbf{x}$ and its knowledge of previously epxlained objects $\textbf{h}$, $\textbf{z}$. |




Note that the latent
space is shared across all objects by using the same VAE for each
attended image part (i.e., object). But how does the network know on
which image part it should attend? And how can it chose a variable
number of crops?  

[Eslami et al. (2016)](https://arxiv.org/abs/1603.08575) propose

**Generative Process**: The assumed generative process (depicted
below) 



[Eslami et al. (2016)](https://arxiv.org/abs/1603.08575) define AIR as
a generative model in which images $\textbf{x}$ are composed of
a variable number of objects 

are decomposed into variable number
of entities. 


- iterative, variable-length inference network that attends to one
  object $\textbf{h}^i$ at a time



Each entity (e.g., digit) shall be encoded and decoded by
the same VAE[^1].


[^1]

### Mathematical Model


**Modeling Assumption**:

$$
 p_{\boldsymbol{\theta}} (\textbf{x}) = \sum_{i=1}^N p_N (i) \int
 p_{\boldsymbol{\theta}}^z \left( \textbf{z} | n \right) \cdot
 p_{\boldsymbol{\theta}}^x \left( \textbf{x} | \textbf{z}\right)
$$

### Difficulties 



These entities shall be encoded by a latent representation 
$\textbf{z}\_{\text{what}}^{(i)}$ respectively, 


described as a composition of
individual objects with the same underlying structure
$\textbf{z}_{\text{what}}$ (i.e., different instantiations from the
same class). 




[Eslami et al. (2016)](https://arxiv.org/abs/1603.08575) describe 
scenes as a composition of individual objects $i$ with the same
underlying structure

$\textbf{z}^{(i)}$ where each entry corresponds
to  (i.e., different instantiations of the same class). 


| ![Schematic of AIR](/assets/img/010_AIR/air_model.png "Schematic of AIR") |
| :--  |
|  **Schematic of AIR**. Taken from [Eslami et al. (2016)](https://arxiv.org/abs/1603.08575).|


<!-- - scene interpretation via learned, amortized inference -->
<!-- - impose structure through partly-/fully-specified generative models -->


<!-- - model structure enforces that a scene is formed by a variable number -->
<!--   of entities that appear at different locations -->

Bayesian perspective of scene interpretation
-> treating scene interpretation as inference in a generative model

Modeling assumption: multi-object scenes can be structured into groups
of $\textbf{z}^{(i)}$ (each group describes attributes of one of the
objects in the scene)




**Motivation**:

## Implementation


## Drawbacks

* 


## Acknowledgements

This [blog post](http://akosiorek.github.io/ml/2017/09/03/implementing-air.html) by Adam Kosiorek was very helpful.

--------------------------------------------------------------------------------------
