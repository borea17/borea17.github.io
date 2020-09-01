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
path. The symmetric architecture allows to combine low level feature
maps (left side, fine information) with high level feature maps (right side,
coarse information) more effectively such that context
information can be better propagated to higher resolution layers (top right).
As the result, more precise segmentations can be retrieved even with
few training examples, indicating that the optimization problem is
better posed in U-Nets.
  
[^2]: Actually, CNNs should put more emphasis on the `where** or rather
    the local relation between context information, see [Geoffrey
    Hinton's comment about
    pooling](https://www.reddit.com/r/MachineLearning/comments/2lmo0l/ama_geoffrey_hinton/clyj4jv/).
 
## Implementatation

[Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597)
demonstrated U-Net application results for three different
segmentation tasks and open-sourced their original
[U-Net
implementation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
(or rather the ready trained network). The whole training process and data
augmentation procedures are not provided (except for overlap-tile
segmentation). The following reimplementation aims to give an
understanding of the whole paper (data augmentation and training
process included), while beeing as simple as possible. Note that there
are lots of open-source U-Net reimplementations out there, however
most of them are already modified versions.

### EM Dataset

Only the first task of the three different U-Net applications is
reimplemented: The segmentation of neuronal structures in electron
microscopic (EM) recordings. The traning data consists of 30 images
($512 \times 512$ pixels with 8-bit grayscale) from the ventral nerve
cord of some species of fruit flies together with the corresponding 30
binary segmentation masks (white pixels for segmented objects, black
for the rest), see below. The dataset formed part of the 2D EM
segmentation challenge at the ISBI 2012 conference. Although the workshop
competition is done, the challenge remains open for new contributions.
Further details about the data can be found at the [ISBI Challenge
website](http://brainiac2.mit.edu/isbi_challenge/), where also the training
and test data can be downloaded (after registration).

<center>
<table>
<tbody>

<tr>
 <td><img src='/assets/img/05_Unet/EM_dataset.gif' ></td>
</tr>

<tr>
 <td>EM training data. Taken from <a href="http://brainiac2.mit.edu/isbi_challenge/">ISBI Challenge</a>.</td>
</tr>

</tbody>
</table>
</center>

For the sake of simplicity, I open-sourced the dataset in
my [Datasets github repository](https://github.com/borea17/Datasets).
Assuming you cloned the repository, the following function can be used to
load the training data set.

```python
from PIL import Image
import torch
import torchvision.transforms as transforms


def load_dataset():
    num_img, img_size = 30, 512
    # initialize
    imgs = torch.zeros(num_img, 1, img_size, img_size)
    labels = torch.zeros(num_img, 1, img_size, img_size)
    # fill tensors with data
    for index in range(num_img):
        cur_name = str(index) + '.png'
        
        img_frame = Image.open('./Datasets/EM_2012/train/image/' + cur_name)
        label_frame = Image.open('./Datasets/EM_2012/train/label/' + cur_name)
        
        imgs[index] = transforms.ToTensor()(img_frame).type(torch.float32)
        labels[index] = transforms.ToTensor()(label_frame).type(torch.float32)
    return imgs, labels


imgs, labels = load_dataset()
```

### Data Augmentation

Training neural networks on image data typically requires large
amounts of data to make the model robust (i.e., avoid overfitting) and
accurate (i.e., avoid underfitting). However, data scarcity is a common
problem in biomedical segmentation tasks, since data generation is
expensive and time consuming. In such cases, **data augmentation**
offers a solution by generating additional data (using plausible
transformations) to expand the training dataset. In most image
segmentation tasks the function to be learned has some 
transformation-invariance properties (e.g., translating the input
should result in a translated output). The data augmentation applied by
[Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597) can be
divided into four parts:


* **Overlap-tile strategy** is used to divide an arbitrary large image
  into several overlaping parts (each forming an input and label to
  the training algorithm). Remind that the input to the neural network is
  greater than the output, in case of the EM dataset the input is even
  greater than the whole image. Therefore, [Ronneberger et al.
  (2015)](https://arxiv.org/abs/1505.04597) expand the images
  by mirroring at the sides. The overlap-tile strategy is shown below.
  Depending on the `stride` (i.e., how much the next rectangle is
  shifted to the right), the training dataset is enlarged by a factor
  greater than 4.
  
  | ![Overlap-Tile Strategy](/assets/img/05_Unet/overlap_tile_self.png "Overlap-Tile Strategy") |
  | :--  |
  | Overlap-Tile Strategy for seamless segmentation of arbitrary large images. Blue area depicts input to neural network, yellow area corresponds to the prediction area. Missing input is extrapolated by mirroring (white lines left). The number of tiles depends on the `stride` length (here: `stride=124`). Image created with `visualize_overlap_tile_strategy` (code presented at the end of this section). |

* **Affine transformations** are mathematically defined as
  geometric transformations preserving lines and parallelisms, e.g.,
  scaling, translation, rotation, reflection or any mix of them.
  [Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597) state
  that in case of microscopical image data mainly translation and
  rotation invariance (as affine transformation invariances)
  are desired properties of the resulting function. Note that the
  overlap-tile strategy itself leads to some translation
  invariance. 

* **Elastic deformations** are basically distinct affine
  transformations for each pixel. The term is probably derived from
  physics in which an elastic deformation describes a temporary change
  in shape (e.g., length) of an elastic material (due to induced
  force). The transformation result looks similar to the physics
  phenomenon, see image below. [Ronneberger et al.
  (2015)](https://arxiv.org/abs/1505.04597) noted that elastic
  deformations seem to be a key concept for successfully training with
  few samples. A possible reason may be that the elastic deformations
  increase the model's generalization capabilities as the resulting
  images have more variability than with affine transformations. 


  | ![Elastic Deformation Visualization](/assets/img/05_Unet/deformation_with_grid2.png "Elastic Deformation Visualization") |
  | :-- |
  | Elastic deformation visualization. Left side shows input and label data before deformation is applied. Right side shows the corresponding data after deformation. The grid is artificially added to emphasize that label and image are deformed in the same way. Image created with `visualize_deformation` (code presented at the end of this section).  |

  Implementing elastic deformations basically consists of generating
  random displacement fields, convolving these with a Gaussian filter
  for smoothening, scaling the result by a predefined factor to
  control the intensity and then computing the new pixel values for
  each displacement vector (using interpolation within the old grid),
  see Best Practices for CNNs by [Simard et al.
  (2003)](https://www.researchgate.net/publication/220860992_Best_Practices_for_Convolutional_Neural_Networks_Applied_to_Visual_Document_Analysis)
  for more details.
  

* **Color variations**:


The whole data augmentation process is put into a self written Pytorch `Dataset`
class, see code below. 

**Data Augmentation**

- data scarce
- data augmentation:
 - improve performance of model by generating additional useful data.
   distribution to be learned has some transformation-invariance
   properties such as rotation
   
* **affine transformation**: - preserves lines and parallelism (e.g.,
  scaling, translation, rotation, reflection)

* **elastic deformations**:

### Model Implementation

### Results

-----------------------------------------------------------------------------------------------
