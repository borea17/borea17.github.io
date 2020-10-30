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
the transition from these ideas into a mathematical formulation
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
about the position and scale. $\text{z}\_{\text{pres}}^{(i)}$ is
a binary variable describing whether an object is present, it is
rather a helper variable to allow for a variable number of objects to
be detected (going to be explained in the [Difficulties section](https://borea17.github.io/paper_summaries/AIR#difficulties)).

To disentangle `what`from `where`, the inference network extracts
attentions crops $\textbf{x}^{(i)}\_{\text{att}}$ of the image
$\textbf{x}$ based on a three-dimensional vector
$\textbf{z}^{(i)}\_{\text{where}} \left( \textbf{h}^{(i)} \right)$
which specifies the affine parameters $(s^{(i)}, t_x^{(i)}, t_y^{(i)})$ of the attention
transformation[^1]. These attention crops are then put through a
standard VAE to encode the latent `what`-vector
$\textbf{z}^{(i)}\_{\text{what}}$. Note that each attention crop is
put through the same VAE, thereby consistency between compressed
object attributes is achieved (i.e., each object is an instantiation
of the same class). 

On the decoder side, the reconstructed attention crop
$\widetilde{\textbf{x}}^{(i)}\_{\text{att}}$ is transformed to
$\widetilde{\textbf{x}}^{(i)}$ using the information from
$\textbf{z}^{(i)}\_{\text{where}}$. $\widetilde{\textbf{x}}^{(i)}$ can
be understood as a reconstructed image of the $i$-th object in the
original image $\textbf{x}$. Note that
$\text{z}^{(i)}\_{\text{pres}}$ is used to decide whether the
contribution of $\widetilde{\textbf{x}}^{(i)}\_{\text{att}}$ is added
to the otherwise empty canvas $\widetilde{\textbf{x}}^{(i)}$.

The schematic below summarizes the whole AIR architecture.

[^1]: Visualization of a standard attention transformation, for more
    details refer to my [Spatial Transformer
    description](https://borea17.github.io/paper_summaries/spatial_transformer#model-description).
    
    | <img width="800" height="512" src='/assets/img/09_spatial_transformer/attention_transform.gif'> |
    | :--  |
    |  This transformation is more constrained with only 3-DoF. Therefore it only allows cropping, translation and isotropic scaling to be applied to the input feature map.|

| ![Schematic of AIR](/assets/img/010_AIR/AIR_model2.png "Schematic of AIR") |
| :---:      |
| **Schematic of AIR**                                          |

**Creation of Attention Crops and Inverse Transformation**: As stated
before, a [Spatial
  Transformer (ST)](https://borea17.github.io/paper_summaries/spatial_transformer) module is used to produce the
attention crops using a standard attention transformation. Remind that
this means that the regular grid $\textbf{G} = \\{\begin{bmatrix}
x_k^t & y_k^t \end{bmatrix}^{\text{T}} \\}$ defined on the output is
transformed into a new sampling grid $\widetilde{\textbf{G}} = \\{\begin{bmatrix}
x_k^s & y_k^s \end{bmatrix}^{\text{T}} \\}$ defined
on the input. The latent vector $\textbf{z}^{(i)}\_{\text{where}}$ can
be used to build the attention transformation matrix, i.e.,

$$
  \textbf{A}^{(i)} = \begin{bmatrix} s^{(i)} & 0 & t_x^{(i)} \\
   0 & s^{(i)} & t_y^{(i)} \\ 0 & 0 & 1\end{bmatrix}, \quad \quad \quad
  \begin{bmatrix} x_k^s \\ y_k^s \\ 1 \end{bmatrix} = \textbf{A}^{(i)}
  \begin{bmatrix} x_k^t \\ y_k^t \\ 1\end{bmatrix} 
$$

This is nothing new, but how do we map the reconstructed attention
crop $\tilde{\textbf{x}}^{(i)}\_{\text{att}}$ back to the original
image space, i.e., how can we produce $\widetilde{\textbf{x}}^{(i)}$
from $\widetilde{\textbf{x}}^{(i)}\_{\text{att}}$ and
$\textbf{z}^{(i)}\_{\text{where}}$? The answer is pretty simple, we
use the (pseudo)inverse[^2] of the formerly defined attention transformation
matrix, i.e., 

$$
\begin{bmatrix} x_k^s \\ y_k^s \\ 1 \end{bmatrix} = \left(\textbf{A}^{(i)}\right)^{+}
  \begin{bmatrix} x_k^t \\ y_k^t \\ 1\end{bmatrix}
$$

where  $\left(\textbf{A}^{(i)}\right)^{+}$ denotes the Moore-Penrose
inverse of $\textbf{A}^{(i)}$, and the regular grid $\textbf{G} = \\{\begin{bmatrix}
x_k^t & y_k^t \end{bmatrix}^{\text{T}} \\}$ is now defined on the
original image space[^3]. Below is a self-written interactive
visualization where $\widetilde{\textbf{x}}^{(i)}\_{\text{att}} =
\textbf{x}^{(i)}\_{\text{att}}$. It shows nicely that the whole
process can abstractly be understood as cutting of a crop from the
original image and placing the reconstructed version with the inverse
scaling and shifting on an otherwise empty (black) canvas. The code and visualization can be
found [here](https://github.com/borea17/InteractiveTransformations).

| <img width="800" height="404" src='/assets/img/010_AIR/transformation.gif'> |
| :--: |
|  **Interactive Transformation Visualization** |

[^2]: Using the pseudoinverse (or Moore-Penrose inverse) is beneficial
    to allow inverse mappings even if $\textbf{A}^{(i)}$ becomes
    non-invertible, e.g., if $s^{(i)} = 0$.
    
[^3]: Note that we assume normalized coordinates with same
    resolutions such that the notation is not completely messed up. 

### Mathematical Model

While the former model description gave an overview about the
inner workings and ideas of AIR, the following section introduces the
probabilistic model over which AIR operates. Similar to the [VAE
paper](https://borea17.github.io/paper_summaries/auto-encoding_variational_bayes)
by [Kingma and Welling (2013)](https://arxiv.org/abs/1312.6114),
[Eslami et al. (2016)](https://arxiv.org/abs/1603.08575) introduce a
modeling assumption for the generative process and variational
approximation to allow for joint optimization of the inference
(encoder) and generator (decoder) parameters.

In contrast to standard VAEs, the modeling assumption for the
generative process is more structured in AIR, see image below. It
assumes that:  

1. The number of objects $n$ is sampled from some discrete prior
distribution $p_N$ (e.g., Binomial distribution) with maximum value
$N$. 
2. The latent scene descriptor $\textbf{z} =
\left(\textbf{z}^{(1)}, \textbf{z}^{(2)}, \dots, \textbf{z}^{(n)}
\right)$ (length depends on sampled $n$) is sampled from a scene model
$\textbf{z} \sim p\_{\boldsymbol{\theta}}^{z} \left( \cdot | n
\right)$, where each vector $\textbf{z}^{(i)}$ describes the
attributes of one object in the scene. Furthermore, [Eslami et al.
(2016)](https://arxiv.org/abs/1603.08575) assume that
$\textbf{z}^{(i)}$ are independent for each possible $n$, i.e.,
$p\_{\boldsymbol{\theta}}^{z} \left( \textbf{z} | n \right) =
\prod_{i=1}^n p\_{\boldsymbol{\theta}}^z \left( \textbf{z}^{(i)}\right)$.
3. $\textbf{x}$ is generated by sampling from the conditional
distribution $p_{\boldsymbol{\theta}}^{x} \left( \textbf{x} |
\textbf{z} \right)$. 

As a result, the marginal likelihood of an image
given the generative model parameters can be stated as follows 

$$
 p_{\boldsymbol{\theta}} (\textbf{x}) = \sum_{n=1}^N p_N (n) \int
 p_{\boldsymbol{\theta}}^z \left( \textbf{z} | n \right) p_{\boldsymbol{\theta}}^x \left( \textbf{x} | \textbf{z}\right)
$$

| ![Generative Model VAE vs AIR](/assets/img/010_AIR/VAE_vs_AIR.png "Generative Model VAE vs AIR") |
| :--  |
|  **Generative Model VAE vs AIR**. Note that for a given dataset $\textbf{X} = \\{ \textbf{x}^{(i)}\\}\_{i=1}^{L}$ the marginal likelihood of the whole dataset can be computed via $p\_{\boldsymbol{\theta}} ( \textbf{X} ) = \prod\_{i=1}^{L} p\_{\boldsymbol{\theta}} \left( \textbf{x}^{(i)} \right) $. |

**Learning by optimizing the ELBO**: Since the integral is intractable
for most models, [Eslami et al. (2016)](https://arxiv.org/abs/1603.08575) introduce an amortized[^4]
variational approximation
$q\_{\boldsymbol{\phi}} \left(\textbf{z}, n | \textbf{x}\right)$
for the true posterior $p\_{\boldsymbol{\theta}}\left(\textbf{z}, n
|\textbf{x}\right)$.
From here on, the steps are very similar to the [VAE paper](https://borea17.github.io/paper_summaries/auto-encoding_variational_bayes)
by [Kingma and Welling (2013)](https://arxiv.org/abs/1312.6114): The
objective of minimizing the KL divergence between the parameterized
variational approximation (using a neural network) and the true (but
unknown) posterior $p\_{\boldsymbol{\theta}}\left(\textbf{z}, n
|\textbf{x}\right)$
is approximated by maximizing the evidence lower bound 
([ELBO](https://borea17.github.io/ML_101/probability_theory/evidence_lower_bound)): 

$$
\mathcal{L} \left( \boldsymbol{\theta}, \boldsymbol{\phi};
\textbf{x}^{(i)} \right) = \underbrace{- D_{KL} \left( q_{\boldsymbol{\phi}}
\left( \textbf{z}, n | \textbf{x}^{(i)}\right) || p_{\boldsymbol{\theta}}
(\textbf{z}, n)\right)}_{\text{Regularization Term}} + \underbrace{\mathbb{E}_{q_{\boldsymbol{\phi}} \left(
\textbf{z}, n | \textbf{x}^{(i)} \right)} \left[ \log
p_{\boldsymbol{\theta}} \left( \textbf{x}^{(i)} | \textbf{z}, n
\right) \right]}_{\text{Reconstruction Accuracy}},
$$

where $p\_{\boldsymbol{\theta}} \left( \textbf{x}^{(i)} | \textbf{z},
 n \right)$ is a parameterized probabilistic decoder[^5] (using a neural
 network) and $p\_{\boldsymbol{\theta}} (\textbf{z}, n) =
 p\_{\boldsymbol{\theta}} \left(\textbf{z} | n \right)
 p \left( n \right)$ is prior on the joint
 probability of $\textbf{z}$ and $n$ that we need to define a priori.
 As a result, the optimal parameters $\boldsymbol{\theta}$,
 $\boldsymbol{\phi}$ can be learnt jointly by optimizing (maximizing)
 the ELBO. 




[^4]: `Amortized` variational approximation means basically
    `parameterized` variational approximation, i.e., we introduce a
    parameterized function $q\_{\boldsymbol{\phi}} \left( \textbf{z},
    n | \textbf{x}\right)$ (e.g., neural network parameterized by
    $\boldsymbol{\phi}$) that maps from an image $\textbf{x}$ to the
    distribution parameters for number of objects $n$ and their latent
    representation $\textbf{z}$, see [this excellent answer on
    variational
    inference](https://www.quora.com/What-is-amortized-variational-inference). 

[^5]: The probabilistic decoder $p\_{\boldsymbol{\theta}} \left(
    \textbf{x}^{(i)} | \textbf{z}, n \right)$ is also just an
    approximation to the true generative process. However, note that
    for each $\textbf{x}^{(i)}$ we know how the reconstruction should
    look like. I.e., if $p\_{\boldsymbol{\theta}} \left(
    \textbf{x}^{(i)} | \textbf{z}, n \right)$ approximates the true
    generative process, we can optimize it by maximizing its
    expectation for given $\textbf{z}$ and $n$ sampled from the
    approximate true posterior $q\_{\boldsymbol{\phi}} \left(
    \textbf{z}, n | \textbf{x}^{(i)}\right)$.  
 
 
### Difficulties 

Yet, we still haven't everything we need to implement the model. In
the last section, I introduced the amortized approximation of the true
posterior $q\_{\boldsymbol{\phi}} \left(\textbf{z}, n |
\textbf{x}^{(i)}\right)$ as a neural network. But how can it extract
a variable number of objects $n$ from scenes? 


## Implementation


## Drawbacks

* 


## Acknowledgements

This [blog post](http://akosiorek.github.io/ml/2017/09/03/implementing-air.html) by Adam Kosiorek was very helpful.

--------------------------------------------------------------------------------------
