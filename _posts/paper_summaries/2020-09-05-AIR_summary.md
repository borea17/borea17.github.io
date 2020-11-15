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
  is described or the maximum number of repetitions is reached.

Notably, the model can handle a variable number of objects
(upper-bounded) by treating inference as an iterative process. As a
proof of concept, they show that AIR could successfully learn to
decompose multi-object scenes in multiple datasets (multiple MNIST,
Sprites, Omniglot, 3D scenes).


| <img width="900" height="512" src='/assets/img/010_AIR/paper_results.gif'> |
| :--  |
|  **Paper Results**. Taken from [this presentation](https://www.youtube.com/watch?v=4tc84kKdpY4). Note that the aim of unsupervised representation learning is to obtain good representations rather than perfect reconstructions.  |


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

    | <img width="900" height="512" src='/assets/img/09_spatial_transformer/attention_transform.gif'> |
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
  \begin{bmatrix} x_k^t \\ y_k^t \\ 1\end{bmatrix} \stackrel{s\neq
  0}{=} \begin{bmatrix} \frac {1}{s^{(i)}} & 0 & - \frac{t_x^{(i)}}{s} \\
   0 & \frac {1}{s^{(i)}} & -\frac {t_y^{(i)}}{s} \\ 0 & 0 &
   1\end{bmatrix}\begin{bmatrix} x_k^t \\ y_k^t \\ 1\end{bmatrix},
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

| <img width="900" height="404" src='/assets/img/010_AIR/transformation.gif'> |
| :--: |
|  **Interactive Transformation Visualization** |

[^2]: Using the pseudoinverse (or Moore-Penrose inverse) is beneficial
    to allow inverse mappings even if $\textbf{A}^{(i)}$ becomes
    non-invertible, i.e., if $s^{(i)} = 0$.

[^3]: Note that we assume normalized coordinates with same
    resolutions such that the notation is not completely messed up.

### Mathematical Model

While the former model description gave an overview about the
inner workings and ideas of AIR, the following section introduces the
probabilistic model over which AIR operates. Similar to the [VAE
paper](https://borea17.github.io/paper_summaries/auto-encoding_variational_bayes)
by [Kingma and Welling (2013)](https://arxiv.org/abs/1312.6114),
[Eslami et al. (2016)](https://arxiv.org/abs/1603.08575) introduce a
modeling assumption for the generative process and use a variational
approximation for the true posterior of that process to allow for
joint optimization of the inference (encoder) and generator (decoder)
parameters.

In contrast to standard VAEs, the modeling assumption for the
generative process is more structured in AIR, see image below. It
assumes that:

1. The number of objects $n$ is sampled from some discrete prior
distribution $p_N$ (e.g., geometric distribution) with maximum value
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

In the former explanation, it was assummed that we could easily define
some parameterized probabilistic encoder $q\_{\boldsymbol{\phi}} \left(
\textbf{z}, n | \textbf{x}^{(i)} \right)$ and
decoder $p\_{\boldsymbol{\theta}} \left( \textbf{x}^{(i)} |
\textbf{z}, n \right)$ using neural networks. However, there are some
obstacles in our way:

- How can we infer a variable number of objects $n$? Actually, we
  would need to evaluate $p_N \left(n | \textbf{x}\right) = \int
  q\_{\boldsymbol{\phi}} \left(\textbf{z}, n | \textbf{x} \right)
  d \textbf{z}$ for all $n=1,\dots, N$ and then sample from the
  resulting distribution.
  <!-- Depending on the maximum number of objects -->
  <!-- $N$, this would quickly become computationally inefficient. -->

- The number of objects $n$ is clearly a discrete variable. How can we
  backprograte if we sample from a discrete distribution?

- What priors should we choose? Especially, the prior for the number of
  objects in a scene $n \sim p_N$ is unclear.

- What the `first` or `second` object in a scene constitutes is
  somewhat arbitrary. As a result, object assigments
  $\begin{bmatrix} \textbf{z}^{(1)} & \dots & \textbf{z}^{(n)}
  \end{bmatrix} =\textbf{z} \sim q\_{\boldsymbol{\phi}} \left(\textbf{z} |
  \textbf{x}^{(i)}, n \right)$
  should be exchangeable and the decoder $p\_{\boldsymbol{\theta}} \left( \textbf{x}^{(i)} |
\textbf{z}, n \right)$ should be permutation invariant in terms of
  $\textbf{z}^{(i)}$. Thus, the latent representation needs to
  preserve some strong symmetries.

[Eslami et al. (2016)](https://arxiv.org/abs/1603.08575) tackle these
challenges by defining inference as an iterative process using a
recurrent neural network (RNN) that is run for $N$ steps (maximum number of
objects). As a result, the number of objects $n$ can be encoded in the latent
distribution by defining the approximated posterior as follows

$$
  q_{\boldsymbol{\phi}} \left( \textbf{z}, \textbf{z}_{\text{pres}} |
  \textbf{x} \right) = q_{\boldsymbol{\phi}} \left(
  z_{\text{pres}}^{(n+1)} = 0 | \textbf{z}^{(1:n)} , \textbf{x}\right)
  \prod_{i=1}^n q_{\boldsymbol{\phi}} \left( \textbf{z}^{(i)} ,
  z_{\text{pres}}^{(i)}=1 | \textbf{x}, \textbf{z}^{(1:i-1)}\right),
$$

where $z\_{\text{pres}}^{(i)}$ is an introduced binary variable sampled from a
Bernoulli distribution $z\_{\text{pres}}^{(i)} \sim \text{Bern} \left(
p\_{\text{pres}}^{(i)} \right)$ whose probability $p\_{\text{pres}}^{(i)}$ is
predicted at each iteration step. Whenever $z_{\text{pres}}^{(i)}=0$ the
inference process stops and no more objects can be described, i.e., we enforce $z_{\text{pres}}^{(i+1)}=0$ for all
subsequent steps such that the vector $\textbf{z}\_{\text{pres}}$ looks as follows

$$
\textbf{z}_{\text{pres}} = \begin{bmatrix} \smash[t]{\overbrace{\begin{matrix}1 & 1 & \dots &
1\end{matrix}}^{n \text{ times}}}   & 0 &\dots & 0 \end{bmatrix}
$$


Thus, $z\_{\text{pres}}^{(i)}$ may be understood as an *interruption
variable*. Recurrence is required to avoid explaining the same object
twice.

**Backpropagation for Discrete Variables**: While we can easily draw
samples from a Bernoulli distribution $z\_{\text{pres}}^{(i)} \sim \text{Bern} \left(
p\_{\text{pres}}^{(i)} \right)$,
backpropagation turns out to be problematic. Remind that for continuous
variables such as Gaussian distributions parameterized by mean and
variance (e.g., $\textbf{z}^{(i)}\_{\text{what}}$,
$\textbf{z}^{(i)}\_{\text{where}}$) there is the reparameterization
trick to circumvent this problem. However, any reparameterization of
discrete variables includes discontinuous operations through which we
cannot backprograte. Thus, [Eslami et al.
(2016)](https://arxiv.org/abs/1603.08575)  use a variant of the [score-function
estimator](https://borea17.github.io/ML_101/probability_theory/score_function_estimator)
as a gradient estimator. More precisely, the reconstruction accuracy
gradient w.r.t. $\textbf{z}\_{\text{pres}}$ is approximated by the
score-function estimator, i.e.,

$$
\begin{align}
\nabla_{\boldsymbol{\phi}}\mathbb{E}_{q_{\boldsymbol{\phi}} \left(
\textbf{z}, n | \textbf{x}^{(i)} \right)} \left[ \log
p_{\boldsymbol{\theta}} \left( \textbf{x}^{(i)} | \textbf{z}, n
\right) \right] &= \mathbb{E}_{q_{\boldsymbol{\phi}} \left(
\textbf{z}, n | \textbf{x}^{(i)} \right)} \left[ \log
p_{\boldsymbol{\theta}} \left( \textbf{x}^{(i)} | \textbf{z}, n
\right) \nabla_{\boldsymbol{\phi}} q_{\boldsymbol{\phi}} \left(
\textbf{z}, n | \textbf{x}^{(i)} \right) \right] \\
&\approx \frac {1}{N} \sum_{k=1}^N \log p_{\boldsymbol{\theta}} \left( \textbf{x}^{(i)} | \left(\textbf{z}, n\right)^{(k)}
\right) \nabla_{\boldsymbol{\phi}} q_{\boldsymbol{\phi}} \left(
 \left(\textbf{z}, n\right)^{(k)} | \textbf{x}^{(i)} \right)\\
 &\quad \text{with} \quad \left(\textbf{z}, n\right)^{(k)} \sim q_{\boldsymbol{\phi}} \left(
\textbf{z}, n | \textbf{x}^{(i)} \right)
\end{align}
$$

[Eslami et al. (2016)](https://arxiv.org/abs/1603.08575) note that in
this raw form the gradient estimate `is likely to have high variance`.
To reduce variance, they use `appropriately structured neural
baselines` citing a paper from [Minh and Gregor,
2014](https://arxiv.org/abs/1402.0030). Without going into too much detail,
appropriately structured neural baselines build upon the idea of [variance
reduction in score function estimators](https://borea17.github.io/ML_101/probability_theory/better_score_function_estimator)
by introducing a scalar baseline $\lambda$ as follows

$$
\begin{align}
&\nabla_{\boldsymbol{\phi}} \mathbb{E}_{q_{\boldsymbol{\phi}} \left(
\textbf{z}_{\text{pres}} | \textbf{x}^{(i)} \right)} \left[ \log
p_{\boldsymbol{\theta}} \left( \textbf{x}^{(i)} | \textbf{z}, n
\right) \right] = \mathbb{E}_{q_{\boldsymbol{\phi}} \left(
\textbf{z}, n | \textbf{x}^{(i)} \right)} \left[ \Big(
f_{\boldsymbol{\theta}} \left( \textbf{x}, \textbf{z} \right) - \lambda  \Big)
\nabla_{\boldsymbol{\phi}} q_{\boldsymbol{\phi}} \left(\textbf{z}, n |
\textbf{x}^{(i)} \right) \right]\\
&\text{with} \quad f_{\boldsymbol{\theta}} \left( \textbf{x}, \textbf{z} \right)
= \log p_{\boldsymbol{\theta}} \left( \textbf{x}^{(i)} | \textbf{z}, n \right), \quad
\text{since} \quad\mathbb{E}_{q_{\boldsymbol{\phi}} \left(
\textbf{z}_{\text{pres}} | \textbf{x}^{(i)} \right)} \left[ \nabla_{\boldsymbol{\phi}} q_{\boldsymbol{\phi}} \left(\textbf{z}, n |
\textbf{x}^{(i)} \right) \right] = \textbf{0}.
\end{align}
$$

[Minh and Gregor, 2014](https://arxiv.org/abs/1402.0030) propose to use a
data-dependent neural baseline $\lambda_{\boldsymbol{\psi}} (\textbf{x})$ that
is trained to match its target $f\_{\boldsymbol{\theta}}$. For further reading,
[pyro's SVI part III](https://pyro.ai/examples/svi_part_iii.html#Reducing-Variance-with-Data-Dependent-Baselines)
is a good starting point.


**Prior Distributions**: Before we take a closer look on the prior distribution,
it will be helpful to rewrite the regularization term

$$
\begin{align}
D_{KL} & \left(q_{\boldsymbol{\phi}} \left(\textbf{z}, n | \textbf{x}^{(i)}
\right) || p_{\boldsymbol{\theta}} \left( \textbf{z}, n\right) \right) = D_{KL}
\left( \prod_{i=1}^n q_{\boldsymbol{\phi}} \left(\textbf{z}^{(i)}| \textbf{x},
\textbf{z}^{(1:i-1)} \right) || \prod_{i=1}^n p_{\boldsymbol{\theta}} \left(
\textbf{z}^{(i)} \right) \right)\\
&\stackrel{\text{independent dists.}}{=} \sum_{i=1}^n D_{KL} \left[
\prod_{k=1}^{3} q_{\boldsymbol{\phi}} \left(\textbf{z}^{(i)}_k |  \textbf{x},
\textbf{z}^{(1:i-1)} \right) || \prod_{k=1}^3 p_{\boldsymbol{\theta}} \left(
\textbf{z}^{(i)}_k \right)  \right]\\
&\stackrel{\text{independent dists.}}{=} \sum_{i=1}^n \sum_{k\in \{\text{pres},
\text{where}, \text{what}\}} D_{KL} \left[ q_{\boldsymbol{\phi}} \left(\textbf{z}^{(i)}_k| \textbf{x},
\textbf{z}^{(1:i-1)} \right) || p_{\boldsymbol{\theta}} \left(
\textbf{z}^{(i)}_k \right)  \right]
\end{align}
$$

Note that we assume that each $\textbf{z}_k^{(i)}$ is sampled independently from
their respective distribution such that products could equally be rewritten as
concatenated vectors. Clearly, there are three different prior distributions
that we need to define in advance:

* $p\_{\boldsymbol{\theta}} \left(\textbf{z}\_{\text{what}}^{(i)} \right) \sim
  \mathcal{N} \left(\textbf{0}, \textbf{I} \right)$: A centerd isotropic Gaussian prior
  is a typical choice in standard VAEs and has proven to be effective[^6].
  Remind that the `what`-VAE should ideally receive patches of standard MNIST digits.

[^6]: On a standard MNIST dataset, [Kingma and Welling
    (2013)](https://arxiv.org/abs/1312.6114) successfully used the centered
    isotropic multivariate Gaussian as a prior distribution.

* $p\_{\boldsymbol{\theta}} \left(\textbf{z}\_{\text{where}}^{(i)}\right) \sim
  \mathcal{N} \left( \boldsymbol{\mu}\_{\text{w}} ,
  \boldsymbol{\sigma}\_{\text{w}}^2 \textbf{I}  \right)$: In this distribution,
  we can encode prior knowledge about the objects locality, i.e.,
  average size and location of objects and their standard deviations.

* $p\_{\boldsymbol{\theta}} \left(\textbf{z}\_{\text{pres}}^{(i)}\right) \sim
  \text{Bern} (p\_{\text{pres}})$: [Eslami et
al. (2016)](https://arxiv.org/abs/1603.08575) used an anealing geometric
  distribution as a prior on the number of objects[^7], i.e., the success
  probability decreases from a value close to 1 to some small value close to 0
  during the course of the training. The intuitive idea behind this process is
  to encourage the model to explore the use of objects (in the initial phase),
  and then to constrain the model to use as few objects as possible (trade-off
  between number of objects and reconstruction accuracy).

  For simplicity, we use a fixed Bernoulli distribution for each step as
  suggested in [the pyro tutorial](https://pyro.ai/examples/air.html#In-practice) with
  $p\_{\text{pres}} = 0.01$, i.e., we will constrain the number of objects from
  the beginning. Note that we encourage the use of objects by an empty scene
  initialization in the `what`-decoder (also inspired by
  [pyro](https://pyro.ai/examples/air.html#In-practice)).


[^7]: That [Eslami et al. (2016)](https://arxiv.org/abs/1603.08575) used an
    anealing geometric distribution is not mentioned in their paper, however
    Adam Kosiorek emailed the authors and received that information, see [his
    blog post]([this blog
    post](http://akosiorek.github.io/ml/2017/09/03/implementing-air.html)).



## Implementation

The following reimplementation aims to reproduce the results of the
multi-MNIST experiment, see image below. We will make some adaptations inspired
by [this pyro tutorial](https://pyro.ai/examples/air.html) and [this pytorch
reimplementation](https://github.com/addtt/attend-infer-repeat-pytorch) from
Andrea Dittadi. As a result, the following reimplementation receives a huge
speed up in terms of convergence time and can be trained in less than 20 minutes on a
Nvidia Tesla K80 (compared to 2 days on a Nvidia Quadro K4000 GPU by [Eslami et
al. (2016)](https://arxiv.org/abs/1603.08575)).

As noted by [Eslami et al. (2016)](https://arxiv.org/abs/1603.08575), their
model successfully learned to count the number of digits and their location in
each image (i.e., appropriate attention windows) without any supervision.
Furthermore, the scanning policy of the inference network (i.e., object
assignment policy) converges to spatially divided regions where the direction of
the spatial border seems to be random (dependent on random initialization).
Lastly, the model also learned that it never needs to assign a third object (all
images in the training dataset contained a maximum of two digits).
<!-- This ensures that different regions are -->
<!-- assigned as different objects. -->

| ![Multi-MNIST Paper Results](/assets/img/010_AIR/paper_results.png "Multi-MNIST Paper Results") |
| :--  |
|  **Paper Results of Multi-MNIST Experiment**. Taken from [Eslami et al. (2016)](https://arxiv.org/abs/1603.08575). |

[Eslami et al. (2016)](https://arxiv.org/abs/1603.08575) argue that the
the structure of AIR puts an important inductive bias onto explaining
multi-object scenes by using two adversaries:
* AIR wants to explain the scene, i.e., the reconstruction error should be
  minimized.
* AIR is penalized for each instantiated object due to the KL divergence.
  Furthermore, the `what`-VAE puts an additional prior of instantiating similar
  objects.

### Multi-MNIST Dataset

The multi-MNIST datasets consists of $50 \times 50$ gray-scale images containing
zero, one or two non-overlapping random MNIST digits with equal probability, see
image below. This dataset can easily be generated by taking a blank $50 \times
50$ canvas and positioning a random number of digits (drawn uniformly from MNIST
dataset) onto it. To ensure that MNIST digits ($28\times28$) will not overlap,
we scale them to $24\times 24$ and then position them such that the centers of
two MNIST digits do not overlap. Note that some small overlap may occur which we
simply accept. At the same time, we record the number of digits in each
generated image to measure the count accuracy during training.

| ![Multi-MNIST Dataset Examples](/assets/img/010_AIR/multi-MNIST_dataset.png "Multi-MNIST Dataset Examples") |
| :---: |
|  **Multi-MNIST Dataset Examples**. |

{% capture code %}{% raw %}import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset

CANVAS_SIZE = 50                # canvas in which 0/1/2 MNIST digits are put
MNIST_SIZE = 24                 # size of original MNIST digits (resized)

def generate_dataset(num_images, SEED=1):
    """generates multiple MNIST dataset with 0, 1 or 2 non-overlaping digits

    Args:
        num_images (int): number of images inside dataset

    Returns:
        multiple_MNIST (torch dataset)
    """
    data = torch.zeros([num_images, 1, CANVAS_SIZE, CANVAS_SIZE])

    original_MNIST = datasets.MNIST('./data', train=True, download=True,
        transform=transforms.Compose([
          transforms.Resize(size=(MNIST_SIZE, MNIST_SIZE)),
          transforms.ToTensor()]))
    # sample random digits and positions
    np.random.seed(SEED)
    pos_positions = np.arange(int(MNIST_SIZE/2),CANVAS_SIZE - int(MNIST_SIZE/2))

    mnist_indices = np.random.randint(len(original_MNIST), size=(num_images, 2))
    num_digits = np.random.randint(3, size=(num_images))
    positions_0 = np.random.choice(pos_positions, size=(num_images, 2),
                                   replace=True)

    for i_data in range(num_images):
        if num_digits[i_data] > 0:
            # add random digit at random position
            random_digit = original_MNIST[mnist_indices[i_data][0]][0]
            x_0, y_0 = positions_0[i_data][0], positions_0[i_data][1]
            x = [x_0-int(MNIST_SIZE/2), x_0+int(MNIST_SIZE/2)]
            y = [y_0-int(MNIST_SIZE/2), y_0+int(MNIST_SIZE/2)]
            data[i_data,:,y[0]:y[1],x[0]:x[1]] += random_digit
            if num_digits[i_data] == 2:
                # add second non overlaping random digit
                random_digit = original_MNIST[mnist_indices[i_data][1]][0]
                impos_x_pos = np.arange(x_0-int(MNIST_SIZE/2),
                                        x_0+int(MNIST_SIZE/2))
                impos_y_pos = np.arange(y_0-int(MNIST_SIZE/2),
                                        y_0+int(MNIST_SIZE/2))
                x_1 = np.random.choice(np.setdiff1d(pos_positions, impos_x_pos),
                                       size=1)[0]
                y_1 = np.random.choice(np.setdiff1d(pos_positions, impos_y_pos),
                                       size=1)[0]
                x = [x_1-int(MNIST_SIZE/2), x_1+int(MNIST_SIZE/2)]
                y = [y_1-int(MNIST_SIZE/2), y_1+int(MNIST_SIZE/2)]
                data[i_data,:,y[0]:y[1],x[0]:x[1]] += random_digit
    labels = torch.from_numpy(num_digits)
    return TensorDataset(data.type(torch.float32), labels){% endraw %}{% endcapture %}
{% include code.html code=code lang="python" %}


### Model Implementation

For the sake of clarity, the model implementation is divided into its constitutive
parts:

* `what`-**VAE implementation**: The `what`-VAE can be implemented as an
  independent class that receives an image patch and outputs its reconstruction
  as well as its latent distribution parameters. Note that we could also compute
  the KL divergence and reconstruction error within that class, however we will
  put the whole loss computation in another function to have everything in one
  place. As shown in a [previous
  summary](https://borea17.github.io/paper_summaries/auto-encoding_variational_bayes#vae-implementation),
  two fully connected layers with ReLU non-linearity in between suffice for
  decent reconstruction of MNIST digits.

  We have additional prior knowledge about the output distribution: It should
  only be between 0 and 1. It is always useful to put as much prior knowledge as
  possible into the architecture, but how to achieve this?

  * *Clamping*: The most intuitive idea would be to simply clamp the network
  outputs, however this is a bad idea as gradients wont propagate if the outputs
  are outside of the clamped region.

  * *Network Initialization*: Another approach would be to simply initialize the
    weights and biases of the output layer to zero such that further updates
    push the outputs into the positive direction. However, as the reconstruction
    of the whole image in AIR is a sum over multiple reconstructions, this turns
    out to be a bad idea as well. I tried it and the `what`-VAE produces
    negative outputs which it compensates with another object that has outputs
    greater than 1.

  * *Sigmoid Layer*: This is a typical choice in classification problems and is
    commonly used in VAEs when the decoder approximates a Bernoulli
    distribution. However, it should be noted that using MSE loss (Gaussian
    decoder) with a sigmoid is generally bad practice due to the
    vanishing/saturating gradients (explained [here](https://borea17,gitbub.io/ML_101/probability_theory/sigmoid_loss)).
    WHY DOES IT WORK SO GOOD HERE?

  * *Sigmoid Layer with Bias*:

  {% capture code %}{% raw %}from torch import nn

  WINDOW_SIZE = MNIST_SIZE        # patch size (in one dimension) of what-VAE
  Z_WHAT_HIDDEN_DIM = 400         # hidden dimension of what-VAE
  Z_WHAT_LATENT_DIM = 20          # latent dimension of what-VAE


  class VAE(nn.Module):
      """simple VAE class with a Gaussian encoder (mean and diagonal variance
      structure) and a Gaussian decoder with fixed variance

      Attributes:
          encoder (nn.Sequential): encoder network for mean and log_var
          decoder (nn.Sequential): decoder network for mean (fixed var)
      """

      def __init__(self, bias=-5.):
          super(VAE, self).__init__()
          self.encoder = nn.Sequential(
              nn.Linear(WINDOW_SIZE**2, Z_WHAT_HIDDEN_DIM),
              nn.ReLU(),
              nn.Linear(Z_WHAT_HIDDEN_DIM, Z_WHAT_LATENT_DIM*2),
          )

          out_layer = nn.Linear(Z_WHAT_HIDDEN_DIM, WINDOW_SIZE**2)
          out_layer.weight.data = nn.Parameter(
            torch.zeros(WINDOW_SIZE**2, Z_WHAT_HIDDEN_DIM)
          )
          out_layer.bias.data = nn.Parameter(torch.zeros(WINDOW_SIZE**2))
          self.decoder = nn.Sequential(
              nn.Linear(Z_WHAT_LATENT_DIM, Z_WHAT_HIDDEN_DIM),
              nn.ReLU(),
              nn.Linear(Z_WHAT_HIDDEN_DIM, WINDOW_SIZE**2),
          )
          self.bias = bias
          return

      def forward(self, x_att_i):
          z_what_i, mu_E_i, log_var_E_i = self.encode(x_att_i)
          x_tilde_att_i = self.decode(z_what_i)
          return x_tilde_att_i, z_what_i, mu_E_i, log_var_E_i

      def encode(self, x_att_i):
          batch_size = x_att_i.shape[0]
          # get encoder distribution parameters
          out_encoder = self.encoder(x_att_i.view(batch_size, -1))
          mu_E_i, log_var_E_i = torch.chunk(out_encoder, 2, dim=1)
          # sample noise variable for each batch
          epsilon = torch.randn_like(log_var_E_i)
          # get latent variable by reparametrization trick
          z_what_i = mu_E_i + torch.exp(0.5*log_var_E_i) * epsilon
          return z_what_i, mu_E_i, log_var_E_i

      def decode(self, z_what_i):
          # get decoder distribution parameters
          x_tilde_att_i = self.decoder(z_what_i)
          x_tilde_att_i = torch.sigmoid(x_tilde_att_i + self.bias)
          x_tilde_att_i = x_tilde_att_i.view(-1, 1, WINDOW_SIZE, WINDOW_SIZE)
          return x_tilde_att_i{% endraw %}{% endcapture %}
  {% include code.html code=code lang="python" %}

* **Recurrent Inference Network**: [Eslami et al.
  (2016)](https://arxiv.org/abs/1603.08575) used a standard recurrent neural
  network (RNN) which in each step $i$ computes

  $$
   \left(\underbrace{p^{(i)}_{\text{pres}}, \boldsymbol{\mu}_{\text{where}},
   \boldsymbol{\sigma}^2_{\text{where}}}_{\boldsymbol{\omega}^{(i)}},
   \textbf{h}^{(i)} \right) = RNN \left(\textbf{x},
   \underbrace{\text{z}_{\text{pres}}^{(i-1)}, \textbf{z}_{\text{what}}^{(i-1)},
   \textbf{z}_{\text{where}}^{(i-1)}}_{\textbf{z}^{(i-1)}}, \textbf{h}^{(i-1)}
   \right),
  $$

  i.e., the distribution parameters of $\text{z}\_{\text{pres}}^{(i)}\sim
  \text{Bern}\left( p^{(i)}\_{\text{pres}} \right)$ and
  $\textbf{z}\_{\text{where}}^{(i)} \sim \mathcal{N} \left( \boldsymbol{\mu}\_{\text{where}},
   \boldsymbol{\sigma}^2\_{\text{where}}\textbf{I}\right)$, and the next hidden
  state $\textbf{h}^{(i)}$.


  A simple 3 layer (fully-connected) network should be
  enough for this task.

  To speedup convergence, we initialize useful the distribution parameters:

  * $p\_{\text{pres}}^{(i)}=0.7$: This encourages AIR to use objects in the
    beginning of the training.



{% capture code %}{% raw %}Z_PRES_LATENT_DIM = 1               # latent dimension of z_pres
Z_WHERE_LATENT_DIM = 3              # latent dimension of z_where
RNN_HIDDEN_STATE_DIM = 256          # hidden state dimension of RNN
P_PRES_INIT = [1.]                  # initialization of p_pres
MU_WHERE_INIT = [3.0, 0., 0.]       # initialization of z_where mean
LOG_VAR_WHERE_INIT = [-3.,-3.,-3.]  # initialization of z_where log var


class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()

        RNN_INPUT_SIZE = (CANVAS_SIZE**2 + RNN_HIDDEN_STATE_DIM +
                          Z_WHAT_LATENT_DIM + Z_PRES_LATENT_DIM +
                          Z_WHERE_LATENT_DIM)
        RNN_OUTPUT_SIZE = (RNN_HIDDEN_STATE_DIM + Z_PRES_LATENT_DIM +
                           2*Z_WHERE_LATENT_DIM)

        output_layer = nn.Linear(RNN_HIDDEN_STATE_DIM, RNN_OUTPUT_SIZE)
        self.rnn = nn.Sequential(
            nn.Linear(RNN_INPUT_SIZE, RNN_HIDDEN_STATE_DIM),
            nn.ReLU(),
            nn.Linear(RNN_HIDDEN_STATE_DIM, RNN_HIDDEN_STATE_DIM),
            nn.ReLU(),
            output_layer
        )
        # initialize distribution parameters (first 7 entries)
        output_layer.weight.data[0:7] = nn.Parameter(
            torch.zeros(1 + 2*3, RNN_HIDDEN_STATE_DIM)
        )
        output_layer.bias.data[0:7] = nn.Parameter(
            torch.tensor(P_PRES_INIT + MU_WHERE_INIT + LOG_VAR_WHERE_INIT)
        )
        return

    def forward(self, x, z_im1, h_im1):
        batch_size = x.shape[0]
        rnn_input = torch.cat((x.view(batch_size, -1), z_im1, h_im1), dim=1)
        rnn_output = self.rnn(rnn_input)
        omega_i = rnn_output[:, 0:7]
        h_i = rnn_output[:, 7::]
        # omega_i[:, 0] corresponds to z_pres probability
        omega_i[:, 0] = torch.sigmoid(omega_i[:, 0])
        return omega_i, h_i{% endraw %}{% endcapture %}
{% include code.html code=code lang="python" %}

* **AIR Implementation**: Lastly, we put everything together to obtain the whole
  AIR model. To better understand what's happening, let's take a closer look
  on the two main functions:

  * `forward(x)`:

  * `compute_loss(x)`


* **Training Procedure**

<!-- where -->
<!-- https://mr-easy.github.io/2020-04-16-kl-divergence-between-2-gaussian-distributions/  -->
<!-- https://pytorch.org/docs/stable/_modules/torch/distributions/kl.html#kl_divergence -->

**From Inference to Generation**: In the generative model, we assume that the
number of objects $n$ is drawn from a geometrical distribution $n\sim
\text{Geom} (\rho)$. However, the inference network produces a 1-dimensional
Bernoulli variable $\text{z}^{(i)}\_{\text{pres}}$ at every time-step $i$, i.e.,

$$
  q_{\boldsymbol{\phi}} \left( \text{z}_{\text{pres}}^{(i)} = 1 | \textbf{x},
  \textbf{z}^{(1:i-1)} \right) = \text{z}_{\text{pres}}^{(i-1)} \cdot
  \text{z}_{\text{pres}}^{(i)} \quad \text{with} \quad
  \text{z}_{\text{pres}}^{(i)} \sim \text{Binom} \left(
  p_{\boldsymbol{\phi}}^{(i)} \left( \textbf{x}, \textbf{z}^{(1:i-1)} \right) \right)
$$

such that at the end $n$ is encoded in the $\textbf{z}\_{\text{pres}}$ variable.
As we need to compute the KL-divergence between the geometrical prior
distribution and the infered geometrical distribution, we need to ask how do we
transform the Bernoulli distribution probabilities $p\_{\boldsymbol{\phi}}^{(i)}
\left( \textbf{x}, \textbf{z}^{(1:i-1)} \right)$ into a success probability of
the infered geometric distribution $\widetilde{\rho}$.

To do this, [Eslami et al. (2016)](https://arxiv.org/abs/1603.08575) assume that

$$
p_{\boldsymbol{\phi}}^{(i)}
\left( \textbf{x}, \textbf{z}^{(1:i-1)} \right) = \frac {\mu_{n\ge (i)}}
{\mu_{n\ge (i-1)}}
$$


each Bernoulli distribution probability

### Results


### Notes

* introduced a lot of prior knowledge to obtain good results

## Drawbacks

*


<!-- Possible Enhancement -->
<!-- * scheduled learning => divide learning into parts where we learn useful -->
<!--   reconstructions and and part where we learn useful attention crops -->


## Acknowledgements

The [blog post](http://akosiorek.github.io/ml/2017/09/03/implementing-air.html)
by Adam Kosiorek, the [pyro tutorial on
AIR](https://pyro.ai/examples/air.html) and [the pytorch
implementation](https://github.com/addtt/attend-infer-repeat-pytorch) by Andrea
Dittadi are great resources and helped very much to understand the details of
the paper.

--------------------------------------------------------------------------------------
