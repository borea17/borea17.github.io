---
title: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
abb_title: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
permalink: "/paper_summaries/u_net"
author: "Markus Borea"
tags: ["image segmentation"]
published: false
toc: true
toc_sticky: true
toc_label: "Table of Contents"
---

NOTE: THIS IS CURRENTLY WIP

[Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597)
introduced a novel neural network architecture to generate better and
faster semantic segmentations (i.e., class label assigend to each
pixel) in the area of biomedical image processing (see figure below
for an example). In essence, their model consists of a U-shaped
convolutional neural network (CNN) with skip connections between
blocks to capture context information, while allowing for precise
localizations. In addition to the network architecture, they describe
some data augmentation methods to use available data more
efficiently. By the time the paper was published, the proposed
architecture won several segmentation challenges in the field of
biomedical engineering, outperforming state-of-the-art models by a large
margin. Due to its success and efficiency, U-Net has become a
standard architecture when it comes to image segmentations tasks even
in the non-biomedical area (e.g., [image-to-image
translation](https://arxiv.org/abs/1611.07004), [neural style
transfer](https://arxiv.org/abs/1706.03319), [Multi-Objetct
Network](https://arxiv.org/abs/1901.11390)).

| ![Semantic Segmentation Example](/assets/img/05_Unet/semantic_segmentation.png "Semantic Segmentation Example") |
| :--  |
| Example of a biomedical image segmentation task in which dental x-ray images should be segmented: <br> (**a**) Raw dental image. (**b**) Ground truth segmentation, each color represents some class (e.g., red=pulp, blue=caries). <br>Taken from [ISBI 2015 Challenge on Computer-Automated Detection of Caries in Bitewing Radiography](http://www-o.ntust.edu.tw/~cweiwang/ISBI2015/challenge2/index.html) |

## Model Description

U-Net builds upon the ideas of `Fully Convolutional Networks (FCNs) for Semantic
Segmentation` by [Long et al. (2015)](https://arxiv.org/abs/1411.4038) who
successfully trained FCNs (including prediction, upsampling layers
and skip connections) end-to-end (pixels-to-pixels) on semantic
segmentation tasks. U-Net is basically a modified version of the FCN
by making the architecture more symmetric, i.e., adding a more
powerful expansive path. [Ronneberger et al.
(2015)](https://arxiv.org/abs/1505.04597) argue that this modification
yields more precise segmentations even with few training samples due
to its capacity to propagate more context information to higher
resolution layers. 

**FCN architecture**: The main idea of the FCN architecture is to take
a standard classification network (such as VGG-16), discard the final
classifier layer, convert fully connected layers into convolutions
(i.e., prediction layers) and add skip connections to (some) pooling
layers, see figure below. The skip connections consist of a prediction
($1 \times 1$ convolutional layer with channel dimension equal to
number of possible classes) and a deconvolutional (upsampling) layer.

| ![Example FCN Architecture](/assets/img/05_Unet/FCN_example.png "Example FCN Architecture") |
| :--  |
| Example of FCN Architecture. VGG-16 net is used as feature learning part. Numbers under the cubes indicate the number of output channels. The prediction layer is itself a $1 \times 1$ convolutional layer (the final output consists only of 6 possible classes). A final softmax layer is added to output a normalized classification per pixel. Taken from [Tai et al. (2017)](https://arxiv.org/abs/1610.01732) |

**U-Net architecture**: The main idea of the U-Net architecture is to 
build an encoder-decoder FCN with skip connections between
corresponding blocks, see figure below. The left side of U-Net, i.e.,
*contractve path* or *encoder*, is very similar to the left side of
the FC architecture above. The right side of U-Net, i.e., *expansive
path* or *decoder*, differs due to its number of feature channels and the
convolutional + ReLu layers. Note that the input image size is
greater than the output segmentation size, i.e., the network only segments
the inner part of the image[^1].

[^1]: A greater input image than output segmentation size makes sense since
    the network has no information about the surrounding of the input image.


| ![U-Net Architecture](/assets/img/05_Unet/u_net_architecture.png "U-Net Architecture") |
| :--  |
| U-Net architecture as proposed by [Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597).  |

**Motivation**: Semantic segmentation of images can be divided into two tasks 
* **Context Information Retrieval**: Global information about the
  different parts of the image, e.g., in a CNN classification
  network after training there might be some feature representation
  for *nose*, *eyes* and *mouth*. Depending on the feature combination at hand, the
  network may classify the image as *human* or *not human*.
* **Localization of Context Information**: In addition to `what`,
  localization ensures `where`. Semantic segmentation is only possible
  when content information can be localized. Note: In image 
  classification, we are often not interested in `where`[^2].
  
[Long et al. (2015)](https://arxiv.org/abs/1411.4038) argue that
CNNs during classification tasks must learn useful feature
representations, i.e., classification nets are capable to solve the *context
information retrieval* task. Fully connected layers are inappropriate
for semantic segmentation as they throw away the principle of
localization. These two arguments motivate the use of FCNs that take
the feature representation part of classification nets and convert
fully connected layers into convolutions. During the *contractive*
path, information gets compressed into coarse appearance/context
information. However, in this process the dimensionality of the input
is reduced massively. Skip connections are introduced to combine coarse,
semantic information of deeper layers with finer, appearance
information of early layers. Thereby, the *localization* task is addressed.

[Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597) extend
these ideas by essentially increasing the capacity of the decoder
path. The symmetric architecture allows to combine fine context
information (left side, less compact) with coarse, semantic
information (right side) more effectively such that context
information can be better propagated to higher resolution layers (top right).
As the result, more precise segmentations can be retrieved even with
few training examples, indicating that the optimization problem is
better posed in U-Nets.
  
  
[^2]: Actually, CNNs should put more emphasis on the `where` or rather
    the local relation between context information, see [Geoffrey
    Hinton's comment about
    pooling](https://www.reddit.com/r/MachineLearning/comments/2lmo0l/ama_geoffrey_hinton/clyj4jv/).
 
## Implementatation

### Data Generation

### Model Implementation

### Visualization Functions

### Results

-----------------------------------------------------------------------------------------------
