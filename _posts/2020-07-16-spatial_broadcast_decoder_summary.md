---
title: "Spatial Broadcast Decoder: A Simple Architecture for Learning Disentangled Representations in VAEs"
abb_title: "Spatial Broadcast Decoder"
permalink: "/blog/spatial_broadcast_decoder"
author: "Markus Borea"
tags: [machine learning, variational autoencoder, disentanglement, generalization]
---

NOTE: THIS IS CURRENTLY WIP

[Watters et al. (2019)](https://arxiv.org/abs/1901.07017) introduce
the *Spatial Broadcast Decoder (SBD)* as an architecture for the
decoder in Variational Auto-Encoders
[(VAEs)](https://borea17.github.io/blog/auto-encoding_variational_bayes/) 
to improve 
disentanglement in the latent
space<sup>[1](#myfootnote1)</sup>, reconstruction accuracy and
generalization in limited datasets  (i.e., held-out regions in data
space). Motivated by the limitations of deconvolutional layers in traditional decoders,
these upsampling layers are replaced by a tiling operation in the Spatial
Broadcast decoder. Furthermore, explicit spatial information (inductive bias) is
appended in the form of coordinate channels leading to a simplified optimization
problem and improved positional generalization. As a proof of concept, they
tested the model on the colored sprites dataset (known factors of
variation such as position, size, shape), Chairs and 3D Object-in-Room datasets
(no positional variation), a dataset with small objects and a
dataset with dependent factors. They could show that the Spatial Broadcast
decoder can be used complementary or as an improvement to state-of-the-art
disentangling techniques.


## Model Description

As stated in the title, the model architecture of the Spatial Broadcast decoder
is very simple: Take a standard VAE decoder and replace all upsampling
deconvolutional layers by tiling the latent code $\textbf{z}$ across the original
image space, appending fixed coordinate channels and applying an convolutional
network with $1 \times 1$ stride, see the figure below.

| ![Schematic of the Spatial Broadcast VAE](/assets/img/03_SBD/sbd.png "Schematic of the Spatial Broadcast VAE") |
| :--         |
| (left) Schematic of the Spatial Broadcast VAE. In the decoder, we broadcast (tile) the latent code $\textbf{z}$ of size $k$ to the image width $w$ adn height $h$, and concatenate two "coordinate" channels. This is then fed to an unstrided convolutional decoder. (right) Pseudo-code of the spatial operation. Taken from [Watters et al. (2019)](https://arxiv.org/abs/1901.07017)|

#### Motivation


The presented architecture is mainly motivated by two reasons:
1. **Deconvolution layers cause optimization difficulties**: [Watters
   et al. (2019)](https://arxiv.org/abs/1901.07017) argue that
   upsampling deconvolutional layers should be avoided, since these
   are prone to produce checkerboard
   artifacts, i.e., a
   checkerboard pattern can be identified on the resulting images
   (when looking closer), see figure below. These artifacts constrain
   the reconstruction accuracy and [Watters et al.
   (2019)](https://arxiv.org/abs/1901.07017) hypothesize that the
  resulting effects may raise problems for learning a disentangled
  representation in the latent space. 
    
    | ![Checkerboard Artifacts](/assets/img/03_SBD/cherckerboard_artifacts.png "Checkerboard Artifacts") |
    | :--         |
    | A checkerboard pattern can often be identified in artifically generated images that use deconvolutional layers. <br>Taken from [Odena et al. (2016)](https://distill.pub/2016/deconv-checkerboard/) (very worth reading).|
  
2. **Appended coordinate channels improve positional generalization and
  optimization**: Previous work by [Liu et al.
  (2018)](https://arxiv.org/abs/1807.03247) showed that standard
  convolution/deconvolution networks (CNNs) perform badly when trying to learn trivial
  coordinate transformations (e.g., learning a mapping from Cartesian space
  into one-hot pixel space or vice versa). This behavior may seem
  counterintuitive (easy task, small dataset), however the feature of translational
  equivariance (i.e., shifting an object in the input equally shifts its
  representation in the output) in CNNs<sup>[2](#myfootnote2)</sup>
  hinders learning this task: The filters have by design no 
  information about their position. Thus, the network must learn to decode
  spatial information without having them resulting in a complicated function
  which makes optimization difficult. E.g., changing the input coordinate
  slighlty might push the resulting function in a completelty different
  direction.

    **CoordConv Solution**: To overcome this problem, [Liu et al.
  (2018)](https://arxiv.org/abs/1807.03247) propose to
  append coordinate channels before convolution and term the resulting layer
  *CoordConv*, see figure below. In principle, this layer can
  learn to use or discard translational equivariance and
  keeps the other advantages of convolutional layers (fast computations, few
  parameters). Under this modification learning coordinate transformation
  problems works out of the box with perfect generalization in less time (150
  times faster) and less memory (10-100 times fewer parameters).
  As coordinate transformations are implicitely needed in a variaty of tasks (such as
  producing bounding boxes in object detection) using CoordConv instead of
  standard convolutions might increase the performance of several other models. 

    | ![CoordConv Layer](/assets/img/03_SBD/CoordConv.png "CoordConv Layer") |
    | :--         |
    | Comparison of 2D convolutional and CoordConv layers. <br>Taken from [Liu et al. (2018)](https://arxiv.org/abs/1807.03247). |

    **Positional Generalization**: Appending fixed coordinate channels is
  mainly beneficial in datasets in which same objects may appear at distinct
  positions (i.e., there is positional variation). The main idea is that
  rendering an object at a specific position without spatial information (i.e.,
  standard convolution/deconvolution) results in a very complicated function. In
  contrast,the Spatial Broadcast decoder architecture can
  leverage the spatial information to reveal objects easily: E.g., by convolving
  the positions in the latent space with the fixed coordinate channels and
  applying a threshold operation. Thus, [Watters
   et al. (2019)](https://arxiv.org/abs/1901.07017) argue that the
  Spatial Broadcast decoder architecture puts an inductive bias of dissociating 
  positional from non-positional features in the latent distribution.  
  Datasets without positional variation in turn seem unlikely to benefit from this
  architecture. However, [Watters et al.
  (2019)](https://arxiv.org/abs/1901.07017) showed that the Spatial 
  Broadcast decoder could still help in these datasets and attribute this to the
  replacement of deconvolutional layers. 
    
## Learning the Model

Basically, the Spatial Broadcast decoder is a function approximator for
probabilistic decoder in a VAE. Thus, learning the model works exactly
as in VAEs (see my
[post](https://borea17.github.io/blog/auto-encoding_variational_bayes/)):
The optimal parameters are learned jointly 
by training the VAE using the AEVB algorithm. 

### Implementation



------------------------------------------------------------

<a name="myfootnote1">1</a>: As outlined by [Watters et al.
  (2019)](https://arxiv.org/abs/1901.07017), there is "yet no
  consensus on the definition of a disentangled representation".
  However, in their paper they focus on *feature
  compositionality* (i.e., composing a scene in terms of independent
  features such as color and object) and refer to it as
  *disentangled representation*.  
<a name="myfootnote2">2</a>: In typical image classification
    problems, translational equivariance is highly valued since it ensures that
    if a filter detects an object (e.g., edges), it will detect it irrespective of its
    position.  

-----------------------------------------------------------

#### Acknowledgement

[Daniel Daza's](https://dfdazac.github.io/) blog was really helpful
and the presented code is highly inspired by his [VAE-SBD implementation](https://github.com/dfdazac/vaesbd).
