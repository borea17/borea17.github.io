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
margin. Due to its success and efficiency, U-net has become a
standard architecture when it comes to image segmentations tasks even
in the non-biomedical area (e.g., [image-to-image
translation](https://arxiv.org/abs/1611.07004), [neural style
transfer](https://arxiv.org/abs/1706.03319), [Multi-Objetct
Network](https://arxiv.org/abs/1901.11390)).

| ![Semantic Segmentation Example](/assets/img/05_Unet/semantic_segmentation.png "Semantic Segmentation Example") |
| :--  |
| Example of a biomedical image segmentation task in which dental x-ray images should be segmented: <br> (**a**) Raw dental image. (**b**) Ground truth segmentation, each color represents some class (e.g., red=pulp, blue=caries). <br>Taken from [ISBI 2015 Challenge on Computer-Automated Detection of Caries in Bitewing Radiography](http://www-o.ntust.edu.tw/~cweiwang/ISBI2015/challenge2/index.html) |

## Model Description

U-net builds upon the ideas of `Fully Convolutional Networks (FCNs) for Semantic
Segmentation` by [Long et al. (2015)](https://arxiv.org/abs/1411.4038) who
successfully trained FCNs (including predction, upsampling layers
and skip connections) end-to-end (pixels-to-pixels) on semantic
segmentation tasks. U-net is basically a modified version of the FCN
by making the architecture more symmetric, i.e., adding an expansive
path with skip connections to the contractive network path.
[Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597) argue
that this modification yields more precise segmentations even with
fewer training samples due to its capacity to propagate more context
information to higher resolution layers (note that U-net has a larger
number of feature channels in the expansive path).

**FCN architecture**: The main idea of the FCN architecture is to take
a standard classification network (such as VGG-16), discard the final
classifier layer, convert fully connected layers into convolutions and
add prediction ($1 \times 1$ convolutional layer with channel
dimension equal to number of possible classes) + deconvolution
(upsampling) layers to (some) pooling layers, see figure below for an
example FCN architecture. 

| ![Example FCN Architecture](/assets/img/05_Unet/FCN_example.png "Example FCN Architecture") |
| :--  |
| Example of FCN Architecture. VGG-16 net is used as feature learning part. Numbers under the cubes indicate the number of output channels. The prediction layer is itself a $1 \times 1$ convolutional layer (the final output consists only of 6 possible classes). A final softmax layer is added to output a normalized classification per pixel. Taken from [Tai et al. (2017)](https://arxiv.org/abs/1610.01732) |

**U-net architecture**: The main idea of the U-net architecture is to 
build an encoder-decoder FCN with skip connections between
corresponding blocks, see figure below. Note that the input image size is
greater than the output segmentation size, i.e., the network only segments
the inner part of the image[^1].

[^1]: A greater input image than output segmentation size makes sense since
    the network has no information about the surrounding of the input image.


| ![U-Net Architecture](/assets/img/05_Unet/u_net_architecture.png "U-Net Architecture") |
| :--  |
| U-Net architecture as proposed by [Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597).  |

**Motivation**:
- feature learning part

- reinterprete classification nets as fully convolutional and
  fine-tune from their learned representations

- tension between semantics and location: global infromation resolves
  what while local information resolves where
- skip architecture to combine deep, coarse, semantic information and
  shallow, fine, appearance information

Motivated by the success and efficiency of standard CNNs

- segmantic segmentation
- U-shaped CNN
- encoder (contraction/downsampling)
- decoder (expansion/upsampling)


 
## Implementatation


-----------------------------------------------------------------------------------------------
