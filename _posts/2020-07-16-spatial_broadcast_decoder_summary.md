---
title: "Spatial Broadcast Decoder: A Simple Architecture for Learning Disentangled Representations in VAEs"
author: "Markus Borea"
tags: [machine learning, variational autoencoder, disentanglement, generalization]
---

[Watters et al. (2019)](https://arxiv.org/abs/1901.07017) introduce
the *Spatial Broadcast Decoder (SBD)* as an architecture for the
decoder in VAEs to improve disentanglement in the latent
space<sup>[1](#myfootnote1)</sup>, reconstruction accuracy and
generalization in limited datasets  (i.e., held-out regions in data
space). Motivated by the limitations of deconvolutional layers (i.e.,
[checkerboard
artifacts](https://distill.pub/2016/deconv-checkerboard/) and spatial
discontinuities) in traditional 
decoders, these upsampling layers are replaced by a tiling operation
in the Spatial Broadcast decoder. Furthermore, explicit spatial
information (inductive bias) is appended in the form of $x-$ and
$y-$coordinate channels in order to simplify the optimization problem
in terms of position variation. 
As a proof of concept, they tested the model on the colored sprites dataset (a
dataset with known factors of variation such as position, size, shape), Chairs
and 3D Object-in-Room datasets (datasets without positional variation), a dataset
with small objects and a dataset with dependent factors. They could show that
the Spatial Broadcast decoder can be used complementary or as an improvement to
state-of-the-art disentangling techniques.


## Model Description

As stated in the title, the model architecture of the Spatial Broadcast decoder
is very simple: Take a standard VAE decoder and replace all upsampling
deconvolutional layers by tiling the latent code $\textbf{z}$ across the original
image space, appending fixed coordinate channels and applying an convolutional
network with $1 \times 1$ stride, see the figure below.

| ![Schematic of the Spatial Broadcast VAE](/assets/img/03_SBD/sbd.png "Schematic of the Spatial Broadcast VAE") |
| :--         |
| (left) Schematic of the Spatial Broadcast VAE. In the decoder, we broadcast (tile) the latent code $\textbf{z}$ of size $k$ to the image width $w$ adn height $h$, and concatenate two "coordinate" channels. This is then fed to an unstrided convolutional decoder. (right) Pseudo-code of the spatial operation.<sup>[2](#myfootnote2)</sup>|

#### Motivation



## Learning the Model

### Implementation


------------------------------------------------------------

<a name="myfootnote1">1</a>: As outlined by [Watters et al.
  (2019)](https://arxiv.org/abs/1901.07017), there is "yet no
  consensus on the definition of a disentangled representation".
  However, in their paper they focus on *feature
  compositionality* (i.e., composing a scene in terms of independent
  features such as color and object) and refer to it as
  *disentangled representation*.  
<a name="myfootnote2">2</a>: Taken from the original
paper of [Watters et al. (2019)](https://arxiv.org/abs/1901.07017).  

-----------------------------------------------------------

#### Acknowledgement

[Daniel Daza's](https://dfdazac.github.io/) blog was really helpful
and the presented code is highly inspired by his [VAE-SBD implementation](https://github.com/dfdazac/vaesbd).
