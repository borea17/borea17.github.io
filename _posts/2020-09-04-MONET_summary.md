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
  
  The decoder distribution $p\_{\boldsymbol{\theta}}$ is required to
  reconstruct the image component $\widetilde{\textbf{x}}_k \sim p\_{\boldsymbol{\theta}} \left( \textbf{x} | \textbf{z}_k \right)$
  and the attention masks[^2] $\widetilde{\textbf{m}}_k \sim p\_{\boldsymbol{\theta}}
  \left(\textbf{c} | \textbf{z}_k \right)$ from these latent codes.
  Note that $p\_{\boldsymbol{\theta}} \left(\textbf{c} | \textbf{z}_k
  \right)$ defines the mask distribution of the Component VAE, whereas
  $q\_{\boldsymbol{\psi}} \left(\textbf{c} | \textbf{x}\right)$
  denotes the mask distribution of the attention network[^3]. 
  
  Importantly, each of the $k$ component reconstruction distributions is
  multiplied with the corresponding attention mask
  $\textbf{m}_k$, i.e., 

  $$
  \text{Reconstruction Distribution}_k = \textbf{m}_k \odot
     p_{\boldsymbol{\theta}} \left(\textbf{x} | \textbf{z}_k \right). 
  $$

  The negative (decoder) log likelihood *NLL* (can be understood as
  the *reconstruction error*, see my post on
  [VAEs](https://borea17.github.io/paper_summaries/auto-encoding_variational_bayes#model-description)) 
  of the whole image is given by 
  
  $$
  \text{NLL} = - \log \left( \sum_{k=1}^K \textbf{m}_k \odot p_{\boldsymbol{\theta}} \left(\textbf{x} | \textbf{z}_k \right)\right),
  $$
  
  where the sum can be understood as the reconstruction distribution
  of the whole image (mixture of components) conditioned on the latent
  codes $\textbf{z}_k$ and the attention masks $\textbf{m}_k$. Note
  that the reconstructions $\widetilde{\textbf{x}}_k \sim
  p\_{\boldsymbol{\theta}} \left( \textbf{x} | \textbf{z}_k\right)$
  are unconstrained outside of the masked regions (i.e., where
  $m\_{k,i} = 0$).
  
  
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
\textbf{z}_k \right)\right)}_{\text{Reconstruction Error between }
\widetilde{\textbf{x}} \text{ and } \textbf{x}} + \beta
\underbrace{D_{KL} \left( \prod_{k=1}^K q_{\boldsymbol{\phi}} \left(\textbf{z}_k |
\textbf{x}, \textbf{m}_k\right) || p(\textbf{z})
\right)}_{\text{Regularization Term for Distribution of }\textbf{z}_k}\\
&+ \gamma \underbrace{D_{KL} \left( q_{\boldsymbol{\psi}} \left( \textbf{c} |
\textbf{x} \right) || p_{\boldsymbol{\theta}} \left( \textbf{c} | \{
\textbf{z}_k \} \right) \right)}_{\text{Reconstruction Error between }
\widetilde{\textbf{m}}_k \text{ and } \textbf{m}_k},
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
decomposes with the same VAE[^4]. As a result, the segmentation should 
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

[^4]: Philosophical note: Humans also tend to work better when focusing on one task at a time. 


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
reconstructions could even recover occluded objects, see image below.
[Burgess et al. (2019)](https://arxiv.org/abs/1901.11390) argue that
this indicates how `MONet is learning from and constrained by the
structure of the data`.

| ![MONet Paper Results](/assets/img/04_MONet/paper_results.png "MONet Paper Results") |
| :--         |
| MONet Paper Results: Decomposition on *Multi-dSprties* and *CLEVR* images. First row shows the input image, second and third row the corresponding reconstructions and segmentations by MONet (trained for 1,000,000 iterations). The last three rows show the unmasked component reconstructions from some chosen slots (indicated by $S$). Red arrows highlight occluded regions of shapes that are completed as full shapes. Taken from [Burgess et al. (2019)](https://arxiv.org/abs/1901.11390). |

The following reimplementation aims to reproduce some of these results
while providing an in-depth understanding of the model architecture.
Therefore, a dataset that is similar to the *Multi-dSprites* dataset
is created, then the whole model (as faithfully as possible close to
the original architecture) is reimplemented and trained in Pytorch and
lastly, some useful visualizations of the trained model are created. 

### Data Generation

A dataset that is similar in spirit to the *Multi-dSprites* will be
generated. [Burgess et al. (2019)](https://arxiv.org/abs/1901.11390)
generated this dataset by sampling $1-4$ images randomly from the
binary [dsprites dataset](https://github.com/deepmind/dsprites-dataset)(consisting of
$737,280$ images), colorizing these by sampling from a uniform random
RGB color and compositing those (with occlusion) onto a uniform random
RGB background. 

To reduce training time, we going to generate a much simpler dataset
of $x$ images with two non-overlaping objects (`square` or `circle`) and a
fixed color space (`red`, `green` or `aqua`) for these objects, see
image below. The dataset is generated by sampling uniformly random
from possible latent factors, i.e., random non-overlaping
positions for the two objects, random object constellations and random
colors from color space, see code below image. 


| ![Examples of Dataset](/assets/img/04_MONet/self_dataset.png "Examples of Dataset") |
| :--:        |
| Visualization of self-written dataset. |

```python
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import TensorDataset
import torch


def generate_img(x_positions, y_positions, shapes, colors, img_size, size=20):
    """Generate an RGB image from the provided latent factors
    
    Args:
        x_position (list): normalized x positions (float)
        y_position (list): normalized y positions (float)
        shapes (list): shape can only be 'circle' or 'square'
        colors (list): color names or rgb strings
        img_size (int): describing the image size (img_size, img_size)
        size (int): size of shape
        
    Returns:
        torch tensor [3, img_size, img_size] (dtype=torch.float32)
    """
    # creation of image
    img = Image.new('RGB', (img_size, img_size), color='black')
    # map (x, y) position to pixel coordinates
    for x, y, shape, color in zip (x_positions, y_positions, shapes, colors):
        #x_position = (img_size - 2 - size) * x
        #y_position = (img_size - 2 - size) * y
        # define coordinates
        x_0, y_0 = x, y
        x_1, y_1 = x + size, y + size
        # draw shapes
        img1 = ImageDraw.Draw(img)
        if shape == 'square':
            img1.rectangle([(x_0, y_0), (x_1, y_1)], fill=color)
        elif shape == 'circle':       
            img1.ellipse([(x_0, y_0), (x_1, y_1)], fill=color)
    return transforms.ToTensor()(img).type(torch.float32)

def generate_dataset(n_samples, obj_size, colors, SEED=1):
    """simplified version of the multi dsprites dataset without overlap
    
    Args:
           n_samples (int): number of images to generate
           obj_size (int): size of objects 
           colors (list): possible colors
           SEED (int): generation of dataset is a random process
    
    Returns:
           data: torch tensor [n_samples, 3, img_size, img_size]
    """
    img_size = 64
    num_objs = 2
    shapes = ['square', 'circle']
    pos_positions = np.arange(64 - obj_size - 1)
    
    data = torch.empty([n_samples, 3, img_size, img_size])
    
    np.random.seed(SEED)
    positions_0 = np.random.choice(pos_positions, size=(n_samples, 2),
                                   replace=True)
    
    rand_colors = np.random.choice(colors, size=(n_samples, num_objs),
                                   replace=True)
    rand_shapes = np.random.choice(shapes, size=(n_samples, num_objs),
                                   replace=True)   
    for index in range(n_samples):
        x_0, y_0 = positions_0[index][0], positions_0[index][1]
        # make sure that there is no overlap
        impos_x_pos = np.arange(x_0 - obj_size, x_0 + obj_size + 1)
        impos_y_pos = np.arange(y_0 - obj_size, y_0 + obj_size + 1)
        x_1 = np.random.choice(np.setdiff1d(pos_positions, impos_x_pos), size=1)
        y_1 = np.random.choice(np.setdiff1d(pos_positions, impos_y_pos), size=1)
        
        x_positions, y_positions = [x_0, x_1], [y_0, y_1]
        shapes = rand_shapes[index]
        colors = rand_colors[index]
        
        img = generate_img(x_positions, y_positions, shapes, colors, 
                           img_size, obj_size)
        data[index] = img
    return data
```

### Model Implementation

MONet is a rather sophisticated model composing two powerful neural network
architectures in a reasonable way. One major downside of such complex
models is that they comprise lots of hyperparamters from which much
remains unknown such as sensitivity to small pertubations (e.g.,
changing layers within network architectures or parameters $\beta$,
$\gamma$). Therefore, the model
implementation aims to be as close as possible to the original model.
Note that [Burgess et al. (2019)](https://arxiv.org/abs/1901.11390)
did not publish their implementation.

For the sake of simplicity, this section is divided into four parts:

* **Attention Network**: The architecture of the recurrent neural
  network $\boldsymbol{\alpha}_{\psi}$ is described in appendix B.2 of
  [Burgess et al. (2019)](https://arxiv.org/abs/1901.11390).
  Basically, it consists of a slightly modified
  [U-Net](https://borea17.github.io/paper_summaries/u_net)
  architecture that (at the $k$th step) takes as input the
  concatenation of the image $\textbf{x}$ and the current scope mask
  in log units $\log \textbf{s}_k$. The output of the modified U-Net
  is a one channel image $\textbf{o} \in ]-\infty, + \infty[^{W\times
  H}$ in which each entry can be interpreted as the logits probability
  for $\boldsymbol{\alpha}_k$. A sigmoid layer[^5] is used to transform
  these logits into probabilities, additionally these probabilities
  are transformed into log probabilities, i.e.,
  $\log\left(\boldsymbol{\alpha}_k\right)$ and $\log \left( 1 - \boldsymbol{\alpha}_k\right)$.
  <!-- The output of the modified U-Net assigns each -->
  <!-- pixel of the image two channels $\textbf{o}^{(k)} \in ]-\infty, + -->
  <!-- \infty[^{W \times H}$ with $k=\\{1, 2\\}$ where each channel can be understood as -->
  <!-- the logits probability for $\boldsymbol{\alpha}_k$. A pixel-wise log softmax layer is used to transform -->
  <!-- these logits into the log probabilities, i.e., $\log -->
  <!-- (\boldsymbol{\alpha}_k)$ and $\log (1 - \boldsymbol{\alpha}_k)$. -->
  From the model description above, it follows 
  
  $$
  \begin{align}
    \textbf{s}_{k+1} &= \textbf{s}_k \odot \left( 1 -
    \boldsymbol{\alpha}_k \right) \quad &&\Leftrightarrow  \quad \log
    \textbf{s}_{k+1} = \log \textbf{s}_k + \log \left(1 - \boldsymbol{\alpha}_k \right)\\
    \textbf{m}_{k+1} &= \textbf{s}_{k} \odot \boldsymbol{\alpha}_k \quad
  &&\Leftrightarrow \quad \log \textbf{m}_{k+1} = \log \textbf{s}_{k} + \log \boldsymbol{\alpha}_k,
  \end{align}
  $$
  
  i.e., the output of the $k$th step can be computed by simply adding the
  log current scope $\log \textbf{s}_k$ to each log probability. As a
  result, the next log attention mask $\log \textbf{m}\_{k+1}$ and next
  log scope $\log \textbf{s}\_{k+1}$ can be retrieved. Note that using
  log units instead of standard units is beneficial as it ensures
  numerical stability while simplifying the optimization due
  to an increased learning signal. <!-- or simpliyfing the loss function computation -->
  
  The code below summarizes the network architecture, [Burgess et al.
  (2019)](https://arxiv.org/abs/1901.11390) did not state the channel
  dimensionality within the U-Net blocks explicitely. However, as they
  mentioned to use a `U-Net blueprint`, it is assumed that they use
  the same dimensionality as in the original [U-Net
  paper](https://borea17.github.io/paper_summaries/u_net). To reduce
  training time and memory capacity, the following implementation caps
  the channel dimensionality in the encoder to 256 output channels.
    
  ```python
  from torch import nn
  
  
  class AttentionNetwork(nn.Module):
      """Attention Network class for use in MONet, 
      consists of a slightly modified standard U-Net blueprint as described by
      Burgess et al. (2019) in Appendix B.2,
      to reduce complexity:
          - maximum channel dim in encoder is set to 256 (original U-Net is 1024)

      Attributes:
          encoder_blocks (list): 5 Unet blocks of encoder path
          decoder_blocks (list): 5 Unet blocks of decoder path
          bottleneck_block: bottleneck is a 3-layer MLP with ReLUs in-between
          max_pool: down sample using max pool layer
          upsample: upsample layer using nearest neighbor-resizing
      """

      def __init__(self):
          super().__init__()
          self.encoder_blocks = nn.ModuleList([
              AttentionNetwork._block(4, 64),      # [batch_size, 64, 64, 64]
              AttentionNetwork._block(64, 128),    # [batch_size, 128, 32, 32]
              AttentionNetwork._block(128, 256),   # [batch_size, 256, 16, 16]
              AttentionNetwork._block(256, 256),   # [batch_size, 256, 8, 8]
              AttentionNetwork._block(256, 256),   # [batch_size, 256, 4, 4]
          ])
          self.max_pool = nn.MaxPool2d(kernel_size=(2,2))
          self.bottleneck = nn.Sequential(
              nn.Flatten(),                        # [batch_size, 4096]
              nn.Linear(4096, 128),                # [batch_size, 128]
              nn.ReLU(),
              nn.Linear(128, 128),                 # [batch_size, 128]
              nn.ReLU(),
              nn.Linear(128, 128),                 # [batch_size, 128]
              nn.ReLU()              # reshaped into [batch_size, 8, 4, 4]
          )
          # input channels are sum of skip connection and output of last block
          self.decoder_blocks = nn.ModuleList([
              AttentionNetwork._block(264, 256),   # [batch_size, 256, 8, 8]
              AttentionNetwork._block(512, 256),   # [batch_size, 256, 8, 8]
              AttentionNetwork._block(512, 128),   # [batch_size, 512, 16, 16]
              AttentionNetwork._block(256, 64),    # [batch_size, 512, 32, 32]
              AttentionNetwork._block(128, 64)     # [batch_size, 64, 64, 64]
          ])
          self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
          self.prediction = nn.Conv2d(64, 2, kernel_size=(1,1), stride=1)
          self.log_softmax = nn.LogSoftmax(dim=1)
          return

      def forward(self, image, log_scope):
          # concatenate image and current scope along channels
          inp = torch.cat((image, log_scope), 1)   # [batch_size, 4, 64, 64]
          ########################## U-Net Computation ##########################
          # go through encoder path and store intermediate results
          skip_tensors = []
          for index, encoder_block in enumerate(self.encoder_blocks):
              out = encoder_block(inp)
              skip_tensors.append(out)
              # no resizing in the last block
              if index < len(self.encoder_blocks) - 1:
                  inp = self.max_pool(out)
          # feed last skip tensor through bottleneck
          out_MLP = self.bottleneck(out)           # [batch_size, 128]
          # reshape final output to match last skip tensor
          out = out_MLP.view(-1, 8, 4, 4)          # [batch_size, 8, 4, 4]
          # go through decoder path and use skip tensors
          for index, decoder_block in enumerate(self.decoder_blocks):
              inp = torch.cat((skip_tensors[-1 - index], out), 1)
              # no resizing in the last block
              if index == len(self.decoder_blocks) - 1:
                  out = decoder_block(inp)
              else:
                  out = self.upsample(decoder_block(inp))       
          alpha_logits = self.prediction(out)      # [batch_size, 2, 64, 64]
          ########################################################################
          # compute log (alpha) and log (1 - alpha) using pixel wise logsoftmax
          log_alpha = self.log_softmax(alpha_logits)
          # compute log_new_mask, log_new_scope (logarithm rules)
          log_new_mask = log_scope + log_alpha[:, 0:1]
          log_new_scope = log_scope + log_alpha[:, 1:2]
          return [log_new_mask, log_new_scope]

      @staticmethod
      def _block(in_channels, out_channels):
          """each block consists of 3x3 bias-free convolution with stride 1,
          followed by instance normalisation with a learned bias term
          followed by a ReLU activation,
          padding is added to keep image dimension

          Args:
              in_channels (int): number of input channels for first convolution
              out_channels (int): number of output channels for both convolutions

          Returns:
              u_net_block (sequential): U-Net block as defined by Burgess et al.

          """
          u_net_block = nn.Sequential(
              nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=1, 
                        bias=False, padding=1),
              nn.InstanceNorm2d(num_features=out_channels, affine=True),
              nn.ReLU(),
              #nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=1, 
              #          bias=False, padding=1),
              #nn.InstanceNorm2d(num_features=out_channels, affine=True),
              #nn.ReLU(),
          )
          return u_net_block 
  ```


[^5]: [Burgess et al. (2019)](https://arxiv.org/abs/1901.11390) state
    that they use a log softmax layer, however this would only be
    possible if there were two channels at the output. Note that in
    binary classification, a pixel-wise softmax layer (two channel) can be
    transformed into a sigmoid layer (one channel) by using the
    difference between the two channels as input: 
    $$
    \begin{align}
      \text{Softmax} (x_1) &= \frac {1} {1 + \exp(x_2 - x_1)} = \frac
    {1} {1 + \exp(t)} = \text{Sigmoid} (t), \\
      \text{Softmax} (x_2) &= \frac {1}{1 + \exp(x_1 - x_2)} = \frac
    {1}{1+\exp(-t)} = 1 - \text{Sigmoid}(t).
    \end{align}
    $$

* **Component VAE**: The architectures for the encoder
  $q\_{\boldsymbol{\phi}}$ and decoder $p\_{\boldsymbol{\theta}}$
  neural networks are described in appendix B.1 of [Burgess et al.
  (2019)](https://arxiv.org/abs/1901.11390). Basically, the encoder
  $q\_{\boldsymbol{\phi}}(\textbf{z}_k | \textbf{x}, \textbf{m}_k)$ is a typical CNN that takes the
  concatentation of an image $\textbf{x}$ and a segmentation mask
  $\textbf{m}_k$ to compute the mean $\boldsymbol{\mu}\_{E, k}$ and
  logarithmed variance $\boldsymbol{\sigma}^2\_{E,k}$ of the Gaussian latent
  space distribution (*encoder distribution*) $\mathcal{N} \left(
  \boldsymbol{\mu}\_{E, k}, \boldsymbol{\sigma}^2\_{E,k} \textbf{I}
  \right)$. Sampling from this distribution is avoided by using
  the reparametrization trick, i.e., the latent variable
  $\textbf{z}_k$ is expressed as a deterministic variable[^6] 
  
  $$
    \textbf{z}_k = \boldsymbol{\mu}_{E, k} +
    \boldsymbol{\sigma}^2_{E,k} \odot \boldsymbol{\epsilon} \quad
    \text{where} \quad \boldsymbol{\epsilon} \sim \mathcal{N}\left(
    \textbf{0}, \textbf{I}
    \right).
  $$
  
  The component VAE uses a [Spatial Broadcast
  decoder](https://borea17.github.io/paper_summaries/spatial_broadcast_decoder) $p\_{\boldsymbol{\theta}}$
  to transform the latent vector $\textbf{z}_k$ into the reconstructed
  image component $\widetilde{\textbf{x}}_k \sim p\_{\boldsymbol{\theta}}
  \left(\textbf{x} | \textbf{z}_k \right)$ and mask
  $\widetilde{\textbf{m}}_k \sim p\_{\boldsymbol{\theta}} \left(
  \textbf{c}|\textbf{z}_k \right)$. [Burgess et al.
  (2019)](https://arxiv.org/abs/1901.11390) chose independent Gaussian
  distributions with fixed variances for each pixel as the
  reconstructed image component distributions
  $p\_{\boldsymbol{\theta}} \left( x_i | \textbf{z}_k \right) \sim
  \mathcal{N} \left(\mu\_{k,i} (\boldsymbol{\theta}), \sigma_k^2
  \right)$ and independent Bernoulli distributions for each pixel as
  the reconstructed mask distributions $p\left(c\_{k, i}| \textbf{z}_k
  \right) \sim \text{Bern} \left( p\_{k,i}
  (\boldsymbol{\theta})\right)$. I.e., the decoder output is a 4 channel
  image from which the first three channels correspond to the 3 RGB
  channels for the means of the image components
  $\boldsymbol{\mu}_k$ and the last channel corresponds to the logits probabilities
  of the Bernoulli distribution. These logits are converted into
  probabilties $\textbf{p}_k$ using a sigmoid layer.
  
  ```python
  class Encoder(nn.Module):
      """"Encoder class for use in Component VAE of MONet, 
      input is the image and a binary attention mask (same dimension as image)

      Args:
          latent_dim: dimensionality of latent distribution

      Attributes:
          encoder_conv: convolution layers of encoder
          MLP: 2 layer MLP, output parametrises mu and log var of latent_dim Gaussian
      """

      def __init__(self, latent_dim):
          super().__init__()
          self.latent_dim = latent_dim

          self.encoder_conv = nn.Sequential(
              # shape: [batch_size, 4, 64, 64]
              nn.Conv2d(4,  32, kernel_size=(3,3), stride=2, padding=1),
              nn.ReLU(),
              # shape: [batch_size, 32, 32, 32]
              nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1),
              nn.ReLU(),
              # shape: [batch_size, 32, 16, 16]
              nn.Conv2d(32, 64, kernel_size=(3,3), stride=2, padding=1),
              nn.ReLU(),
              # shape: [batch_size, 64, 8, 8]
              nn.Conv2d(64, 64, kernel_size=(3,3), stride=2, padding=1),
              nn.ReLU(),
              # shape: [batch_size, 64, 4, 4],
              nn.Flatten(),
              # shape: [batch_size, 1024]
          )
          self.MLP = nn.Sequential(
              nn.Linear(1024, 256),
              nn.ReLU(),
              nn.Linear(256, 2*self.latent_dim),
          )
          return

      def forward(self, x, m):
          # concatenate x and m along color channels
          inp = torch.cat((x, m), dim=1)
          # shape [batch_size, 4, 64, 64]
          out_conv = self.encoder_conv(inp)
          # shape [batch_size, 1024]
          out_MLP = self.MLP(out_conv)
          # shape [batch_size, 32] 
          mu, log_var = torch.chunk(out_MLP, 2, dim=1)
          return [mu, log_var]


  class SpatialBroadcastDecoder(nn.Module):
      """SBD class for use in Component VAE of MONet,
        a Gaussian distribution with fixed variance (identity times fixed 
        variance as covariance matrix) used as the decoder distribution

      Args:
          latent_dim: dimensionality of latent distribution

      Attributes:
          img_size: image size (necessary for tiling)
          decoder_convs: convolution layers of decoder (also upsampling)
          sigmoid: sigmoid layer
      """

      def __init__(self, latent_dim):
          super().__init__()
          self.img_size = 64
          self.latent_dim = latent_dim
          # "input width and height of CNN both 8 larger than target output"
          x = torch.linspace(-1, 1, self.img_size + 8)
          y = torch.linspace(-1, 1, self.img_size + 8)
          x_grid, y_grid = torch.meshgrid(x, y)
          # reshape into [1, 1, img_size, img_size] and save in state_dict
          self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
          self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))

          self.decoder_convs = nn.Sequential(
              # shape [batch_size, latent_dim + 2, 72, 72]
              nn.Conv2d(in_channels=self.latent_dim+2, out_channels=32,
                        stride=(1, 1), kernel_size=(3,3)),           
              nn.ReLU(),
              # shape [batch_size, 32, 70, 70]
              nn.Conv2d(in_channels=32, out_channels=32, stride=(1,1), 
                        kernel_size=(3, 3)),
              nn.ReLU(),
              # shape [batch_size, 32, 68, 68]
              nn.Conv2d(in_channels=32, out_channels=32, stride=(1,1), 
                        kernel_size=(3, 3)),
              nn.ReLU(),
              # shape [batch_size, 32, 66, 66] 
              nn.Conv2d(in_channels=32, out_channels=32, stride=(1,1), 
                        kernel_size=(3, 3)),
              # shape [batch_size, 4, 64, 64]
              nn.ReLU(),
              nn.Conv2d(in_channels=32, out_channels=4, stride=(1,1), 
                        kernel_size=(1, 1)),
          )
          self.sigmoid = nn.Sigmoid()
          return

      def forward(self, z):
          batch_size = z.shape[0]
          # reshape z into [batch_size, latent_dim, 1, 1]
          z = z.view(z.shape + (1, 1))
          # tile across image [batch_size, latent_im, img_size+8, img_size+8]
          z_b = z.repeat(1, 1, self.img_size + 8, self.img_size + 8)
          # upsample x_grid and y_grid to [batch_size, 1, img_size+8, img_size+8]
          x_b = self.x_grid.expand(batch_size, -1, -1, -1)
          y_b = self.y_grid.expand(batch_size, -1, -1, -1)
          # concatenate vectors [batch_size, latent_dim+2, img_size+8, img_size+8]
          z_sb = torch.cat((z_b, x_b, y_b), dim=1)
          # apply convolutional layers mu_D [batch_size, 4, 64, 64]
          mu_D = self.decoder_convs(z_sb)
          # split into means of x and logits of m
          mu_x, logits_m = torch.split(mu_D, 3, dim=1)
          # convert logits into probabilities using Sigmoid
          p_m = self.sigmoid(logits_m)
          return [mu_x, p_m]


  class ComponentVAE(nn.Module):
      """Component VAE class for use in MONet

      Args:
          latent_dim: dimensionality of latent distribution

      Attributes:
          encoder: encoder neural network object (Encoder class)
          decoder: decoder neural network object (SpatialBroadcastDecoder class)
          normal_dist: unit normal distribution to sample noise variable
      """

      def __init__(self, latent_dim):
          self.encoder = Encoder(latent_dim)
          self.decoder = SpatialBroadcastDecoder(latent_dim)
          self.normal_dist = MultivariateNormal(torch.zeros(latent_dim), 
                                                torch.eye(latent_dim))
          return

      def forward(self, image, log_mask):
          batch_size = image.shape[0]
          # get encoder distribution parameters
          [mu_E, log_var_E] = self.encoder(image, log_mask)
          # sample noise variable for each batch
          epsilon = self.normal_dist.sample(sample_shape=(batch_size, )
                                            ).to(image.device)
          # get latent variable by reparametrization trick
          z = mu_E + torch.exp(0.5*log_var_E) * epsilon
          # get decoder distribution parameters
          mu_x, p_m = self.decoder(z)
          return [mu_E, log_var_E, z,  mu_x, p_m]
  ```

[^6]: This is explained in more detail in my [VAE](https://borea17.github.io/paper_summaries/auto-encoding_variational_bayes) post. For simplicity, we are setting the number of (noise variable) samples $L$ per datapoint to 1 (see equation $\displaystyle \widetilde{\mathcal{L}}$ in [*Reparametrization Trick*](https://borea17.github.io/paper_summaries/auto-encoding_variational_bayes#model-description) paragraph). Note that [Kingma and Welling (2013)](https://arxiv.org/abs/1312.6114) stated that in their experiments setting $L=1$ sufficed as long as the minibatch size was large enough.
  
  
 * **Monet Implementation**: The compositional structure is achieved
    by looping for $K$ steps over the image and combining the attention
    network with the component VAE to compute the loss $\mathcal{L}$.
    Remind that the loss function is given by

    $$
    \begin{align}
    \mathcal{L}\left(\boldsymbol{\phi}; \boldsymbol{\theta};
    \boldsymbol{\psi}; \textbf{x} \right) &= \underbrace{- \log \left( \sum_{k=1}^K \textbf{m}_k \odot p_{\boldsymbol{\theta}} \left(\textbf{x} |
    \textbf{z}_k \right)\right)}_{\text{Reconstruction Error between }
    \widetilde{\textbf{x}} \text{ and } \textbf{x}} + \beta
    \underbrace{D_{KL} \left( \prod_{k=1}^K q_{\boldsymbol{\phi}} \left(\textbf{z}_k |
    \textbf{x}, \textbf{m}_k\right) || p(\textbf{z})
    \right)}_{\text{Regularization Term for Distribution of }\textbf{z}_k}\\
    &+ \gamma \underbrace{D_{KL} \left( q_{\boldsymbol{\psi}} \left( \textbf{c} |
    \textbf{x} \right) || p_{\boldsymbol{\theta}} \left( \textbf{c} | \{
    \textbf{z}_k \} \right) \right)}_{\text{Reconstruction Error between }
    \widetilde{\textbf{m}}_k \text{ and } \textbf{m}_k}.
    \end{align}
    $$
    Each of these three terms can be written in a more explicit form such that
    the implementation becomes trivial:

    1. *Reconstruction Error between $\widetilde{\textbf{x}}$ and
       $\textbf{x}$*: This term is also known as the negative log
       likelihood (NLL) of the whole reconstructed image. [Burgess et
       al. (2019)](https://arxiv.org/abs/1901.11390) chose independent
       Gaussian distributions with fixed variance for each pixel as
       the decoder distribution $p\_{\boldsymbol{\theta}} \left(
       x\_{i} | \textbf{z}_k \right) \sim \mathcal{N} \left(\mu\_{k,
       i}(\boldsymbol{\theta}), \sigma_k^2 \right)$, hence the term
       can be rewritten as follows  

        $$
        \begin{align}
          \text{NLL} &= -\sum_{i=1}^N \log \left(  \sum_{k=1}^K 
        m_{k, i} \left(\boldsymbol{\psi} \right) \frac {1}{ \sqrt{2 \pi \sigma_k^2} } \exp \left( - \frac { \left[ x_i -
        \mu_{k,i} (\boldsymbol{\theta}) \right]^2 } {2 \sigma_k^2}  \right)
        \right)\\
        &= -\sum_{i=1}^N \log \left( \sum_{k=1}^K \exp \left( \log \frac
        {m_{k, i} \left(\boldsymbol{\psi} \right)}{\sqrt{2\pi \sigma_k^2}}
        \right) \exp \left( - \frac {\left[x_i -
        \mu_{k,i} (\boldsymbol{\theta}) \right]^2 } {2 \sigma_k^2}
        \right)\right)\\
        &= -\sum_{i=1}^N \log \left( \sum_{k=1}^K \exp \left(
        \log \big[m_{k, i} \left(\boldsymbol{\psi} \right) \big] - \frac {\log 2\pi \sigma_k^2} {2} - \frac { \big[x_i -
        \mu_{k, i} (\boldsymbol{\theta}) \big]^2 } {2 \sigma_k^2}  \right)
        \right), 
        \end{align}
        $$

        where $i$ enumerates the pixel space ($N=W\cdot H$). The inner
        sum (index $k$) results in a reconstructed image distribution,
        the outer sum (index $i$) computes the log likelihood of each
        pixel independently and sums them to retrieve the
        reconstruction accuracy of the whole image. Note that the
        sum (for fixed $k$) is unconstrained outside of the masked regions for
        each reconstruction[^7]. 
        
    2. *Regularization Term for Distribution of $\textbf{z}_k$*: The
       coding space is regularized using the KL divergence between the 
       latent (posterior) distribution factorized across slots with
       the latent prior distribution weighted with the hyperparameter
       $\beta$. [Burgess et
       al. (2019)](https://arxiv.org/abs/1901.11390) chose a unit
       Gaussian distribution as the latent prior $p(\textbf{z}) \sim
       \mathcal{N} \left(\textbf{0}, \textbf{I} \right)$, hence the KL
       divergence can be rewritten as follows 
       
       $$
       \begin{align}
          D_{KL} \left( \prod_{k=1}^K q_{\boldsymbol{\phi}} \left(\textbf{z}_k \right) || p(\textbf{z}) \right) &=
       D_{KL} \left( \prod_{k=1}^K \mathcal{N}
       \left(\boldsymbol{\mu}_k (\boldsymbol{\phi}),
       \boldsymbol{\sigma}^2_k (\boldsymbol{\phi}) \textbf{I} \right) ||
       \mathcal{N} \left(\textbf{0}, \textbf{I} \right) \right)\\
       &= \int \left(\prod_{k=1}^K \mathcal{N}
       \left(\textbf{z};\boldsymbol{\mu}_k (\boldsymbol{\phi})
       \right) \right) 
       \end{align}
       $$
       
       <!-- \left(\prod_{k=1}^K \mathcal{N} -->
       <!-- \left(\textbf{z};\boldsymbol{\mu}_k (\boldsymbol{\phi}) -->
       <!-- \right) \log \mathcal{N} \left( \textbf{z}; \textbf{0}, \textbf{I}\right) -->

    3. Reconstruction Error
  
  

[^7]: In practice, the reconstruction error for a fixed component
    (fixed $k$) becomes incalculable for binary attention masks, since $\log 0$ is
    undefined. However, the reconstruction error is effectively
    unconstrained outside of masked regions, since for $\lim \log
    m\_{k,i} \rightarrow -\infty$ the reconstruction error for the
    corresponding pixel and slot approaches 0. 
  
 
 
 <!-- - fusing both classse -->
 <!-- - compositional structure  -->
 
 
 
  
  
  
  


  * latent distribution Gaussian with diagonal covariance
  * decoder: Spatial Broadcast Decoder, parametrises 
    means of pixel-wise independent distributions 

* **MONet implementation**

* **Training Procedure**
    
### Results

## Drawbacks of Paper

* static scene decomposition (valuable prior is ignored)
* only works on simple images in which multiple objects of the same
class occur
* even simple images require high training times
* lots of hyperparameters (network architectures, $\beta$, $\gamma$, optimization)
* [see this](https://github.com/ChenYutongTHU/Learning-to-manipulate-individual-objects-in-an-image-Implementation)

--------------------------------------------------------------------------------------------
