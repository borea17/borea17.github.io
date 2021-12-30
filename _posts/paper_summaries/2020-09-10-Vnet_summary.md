---
title: "V-Net: Fully Convolutional Neural Networks for Volumentric Medical Image Segmentation"
permalink: "/paper_summaries/v-net"
author: "Markus Borea"
tags: ["image segmentation"]
published: false
toc: true
toc_sticky: true
toc_label: "Table of Contents"
type: "paper summary"
---

NOTE: THIS IS CURRENTLY WIP!

[Milletari et al. (2016)](https://arxiv.org/abs/1606.04797) introduce
the fully convolutional V-Net architecture as an extension to the
[U-Net](https://borea17.github.io/paper_summaries/u_net) architecture
proposed by [Ronneberger et al.
(2015)](https://arxiv.org/abs/1505.04597) to allow for volumentric
image segmentation. The incentive for their idea comes from the fact
that medical data often consists of 3D volumes (e.g., MRI volumes
below). Clearly, using volume-based instead of slice-based methods is
more adequate in such cases as slices fail to capture valuable
information about the (volumentric) neighborhood. Similar to the U-Net
architecture, their model essentially consists of a V-shaped fully
convolutional neural network with skip connections between blocks to
capture context information, while allowing for precise localizations.
Additionally, [Milletari et al.
(2016)](https://arxiv.org/abs/1606.04797) introduce a new objective
function based on the Dice coefficient to better deal with class
imbalances. As a proof of concept, they trained V-Net end-to-end on prostate
MRI volumes from the [PROMISE2012
challenge](https://promise12.grand-challenge.org/evaluation/challenge/leaderboard/)
with accurate and fast results (though they could not beat the top
results).

| ![Slices of MRI Volumes](/assets/img/08_Vnet/MRI_examples.png "Slices of MRI Volumes") |
| :--  |
| Slices from MRI volumes depicting prostate. This data is part of the PROMISE2012 challenge dataset. Taken from [Milletari et al. (2016)](https://arxiv.org/abs/1606.04797). |


## Model Description

The V-Net architecture builds upon the ideas of
[U-Net](https://borea17.github.io/paper_summaries/u_net): Using a
*contractive path* (left side) to learn and extract necessary features
and add an *expanisve path* (right side) in which the spatial support
is expanded to allow for precise localizations. In addition skip
connections between both paths are added such that features of each
level can be easily propagated towards the localization task. However,
there are some differences (besides using 3D convolutions instead of
standard 2D convolutions) to the U-Net architecture: 

- **Different number of convolutional layers within each block**: In
  the standard U-Net architecture, each block/stage consisted of two
  convolutional layers with ReLU layers in between. In contrast, in
  V-Net each block comprises one to three convolutional layers with
  the number increasing along the encoder path. Note that the number
  convolutional layers is symmetric in both paths, i.e., decreases
  along the decoder path. 

- **Residual Function within each block**: Inspired by [He et al.
  (2015)](https://arxiv.org/abs/1512.03385), each block is formulated
  as a residual function by defining the output as the sum of the
  unprocessed and proccesed (through convolutions and non-linearities)
  input. [He et al. (2015)](https://arxiv.org/abs/1512.03385) give
  empirical evidence that such layers are easier to optimize and [Milletari et al.
(2016)](https://arxiv.org/abs/1606.04797) confirm this hypothesis by
  their empirical observations (comparing the performance with a
  similar network that does not learn residual functions).

- **PReLU non-linearty**: [Milletari et al.
(2016)](https://arxiv.org/abs/1606.04797) chose to use a parametric
ReLU (**PReLU**) as non-linearity throughout the network whereas [Ronneberger et al.
(2015)](https://arxiv.org/abs/1505.04597) proposed to use ReLUs.

- **No pooling**: Motivated by [Springenberg et al.
  (2015)](https://arxiv.org/abs/1412.6806) (who basically found that
  `max-pooling can simply be replaced by a convolutional layer with
  increased stride without loss in accuracy on several image
  recognition benchmarks`), pooling operations are replaced by
  convolutional layers with kernel size $2 \times 2 \times 2$ and
  stride $2$. [Milletari et al.
(2016)](https://arxiv.org/abs/1606.04797) note that there are other
  works discouraging max-pooling (see [Geoffrey
    Hinton's comment about
    pooling](https://www.reddit.com/r/MachineLearning/comments/2lmo0l/ama_geoffrey_hinton/clyj4jv/))
  and that replacing pooling with convolutions can be beneficial in
  terms of memory consumption during training (the reason being that
  during backpropagation pooling requires additional memory allocation
  to assign pointers from the output of each pooling layer back to the
  corresponding input).
  
The full V-Net architecture is depicted below:
  
| ![V-Net Architecture](/assets/img/08_Vnet/v_net_architecture.png "V-Net Architecture") |
| :--  |
| **V-Net architecture**. Input dimension denoted as $128 \times 128 \times 64$. Taken from [Milletari et al. (2016)](https://arxiv.org/abs/1606.04797). |

Interestingly, [Milletari et al.
(2016)](https://arxiv.org/abs/1606.04797) report the theoretical
receptive field[^1] of each network layer, see table below. It can be
seen that after `L-Stage 4` the receptive field ($172\times 172 \times
172$) already captures the content of the whole input volume
($128\times 128 \times 64$). They hypothesize that this is important
to segment poorly visible anatomy as features computed after `L-Stage 4`
perceive the whole input at once and have spatial support that is
larger than the input size. These features may be used to impose
global constraints during segmentation.

| ![Receptive Fields](/assets/img/08_Vnet/receptive_fields.png "Receptive Fields") |
| :--  |
| Theoretical receptive fields of the network. Taken from [Milletari et al. (2016)](https://arxiv.org/abs/1606.04797). |


[^1]: The receptive field in CNNs can be defined as the size of the
    input region that produces/affects an output feature. Clearly, for
    the first layer it is defined by the kernel size, i.e., $5 \times
    5 \times 5$. For subsequent layers, it is a bit more complicated,
    however can be calculated recurrently, see [Araujo et al.
    (2019)](https://distill.pub/2019/computing-receptive-fields/).
    There is also a neat
    [tool](https://fomoro.com/research/article/receptive-field-calculator#5,1,1,SAME;2,2,1,SAME;5,1,1,SAME;5,1,1,SAME)
    that may help to understand what's going on.


### Dice Loss Layer

Class imbalances are a typical problem in image segmentation tasks,
especially in medical image segmentation: Often the anatomy of
interest (first class) is only occupying a small fraction of the image
while the background (second class) makes up most of the image. As a
result, training with a standard loss function such as binary cross
entropy causes the optimization to get trapped into a
local minima in which it often misclassifies the anatomy of interest
as background (at least partially). 

To compensate for this imbalance, a typical solution is to reweight the
cross entropy. E.g., in the U-Net paper [Ronneberger et al.
(2015)](https://arxiv.org/abs/1505.04597) define a special
weight map in the [loss
function](https://borea17.github.io/paper_summaries/u_net#model-implementation). [Milletari et al.
(2016)](https://arxiv.org/abs/1606.04797) take a different path based
on the dice coefficient.

--------------------------------

>**Recap Dice Coefficient** $D$ 
>
>The Dice coefficient $D$ is a symmetric similiarity metric
>between two sets $A$ and $B$ and is defined as two times the area of
>overlap divided by the total sizes of both sets, i.e.,
>
>$$
>D = \frac {2 | A \cup B |} {|A| + |B|}.
>$$
>
>The image below summarizes the basic idea where each circle corresponds
>to one set, respectively. By defintion, $D\in [0, 1]$ where
>$D=0$ means no overlap or no similiarity and $D=1$ signifies complete
>overlap or complete similiarity. 
>
>| ![Dice Coefficient Illustration](/assets/img/08_Vnet/dice_illustration2.png "Dice Coefficient Illustration") |
>| :---: |
>| **Dice Coefficient Illustration**. Taken from [this post](https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2). |
>
>In the case where both sets are binary vectors,
>the dice coefficient can be written in terms of vector operations 
>
>$$
>D = \frac {2 \left(\textbf{p} \cdot \textbf{g}\right)}{|\textbf{p}|^2 + |\textbf{g}|^2} =\frac {2 \sum_{i=1}^N p_i g_i}{\sum_{i=1}^{N} p_i^2 + \sum_{i=1}^{N}g _i^2},
>$$
>
>where $\textbf{g} \in \\{0, 1\\}^{N}$ shall denote the ground truth vector,
>$\textbf{p} \in \\{0, 1\\}^{N}$ the prediction vector and $i$ enumerates
>the pixel/voxel space. Note that in this case the area of overlap is
>simply two times the sum of all true predicted values, i.e., $
>2 \sum\_{\\{i| p_i=1\\}} g_i$. 

**Dice loss layer**: [Milletari et al.
(2016)](https://arxiv.org/abs/1606.04797) noticed that the dice
coefficient in the above formulation can be easily differentiated with
respect to each prediction $p_j$, i.e., 

$$
\frac {\partial D}{\partial p_j} = 2 \left[ \frac {g_j \left(
\sum_{i=1}^N p_i^2 +  \sum_{i=1}^N g_i^2\right) - 2 p_j \left(
\sum_{i=1}^N p_i g_i \right)} {\left( \sum_{i=1}^N p_i^2 +
\sum_{i=1}^N g_i^2 \right)^2}\right]
$$

Thus, the negative dice coefficient is an appropriate loss function.
[Milletari et al. (2016)](https://arxiv.org/abs/1606.04797) argue that
using the dice coefficient (as defined between two binary volumes,
although the predictions are not binary) eliminates the need for
reweighting each pixel as the dice coefficient naturally handles class
imbalances: Remember that it kind of measures the relative overlap
between truth and prediction, i.e., the dice coefficient is equal
irrespective of the object size when the same (relative) amount is
correctly classified (see this
[post](https://stats.stackexchange.com/a/438532)). [Milletari et al.
(2016)](https://arxiv.org/abs/1606.04797) observed much better results
using the dice loss compared to weighted binary cross entropy.

Nowadays, dice loss has become a standard in image segmentation tasks. 

## Implementation

Similar to the U-Net publication, [Milletari et al.
(2016)](https://arxiv.org/abs/1606.04797) open-sourced their original
[V-Net implementation](https://github.com/faustomilletari/VNet)

### Dataset

### Data Augmentation

### Model Implementation

### Results



--------------------------------------------------------------------------------------------------------
