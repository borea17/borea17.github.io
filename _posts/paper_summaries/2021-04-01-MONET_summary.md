---
title: "MONet: Unsupervised Scene Decomposition and Representation"
abb_title: "MONet: Unsupervised Scene Decomposition and Representation"
permalink: "/paper_summaries/multi-object_network"
author: "Markus Borea"
tags: ["unsupervised learning", "object detection", "generalization", "varational autoencoder"]
published: true
toc: true
toc_sticky: true
toc_label: "Table of Contents"
type: "paper summary"
---

[Burgess et al. (2019)](https://arxiv.org/abs/1901.11390) developed
the **Multi-Object Network (MONet)** as an end-to-end trainable model to
decompose images into meaningful entities such as objects. Similar to
[AIR](https://borea17.github.io/paper_summaries/AIR), the whole training process
is unsupervised, i.e., there are no labeled segmentations, handcrafted bounding
boxes or whatsoever. In essence, their model combines a Variational Auto-Encoder
([VAE](https://borea17.github.io/paper_summaries/auto-encoding_variational_bayes))
with a recurrent attention network
([U-Net](https://borea17.github.io/paper_summaries/u_net) *segmentation
network*) to spatially decompose scenes into attention masks (over which the VAE
needs to reconstruct masked regions) and latent representations of each masked
region. In contrast to AIR, MONet does not contain a fully generative model and
its latent space is less structured. As a proof of concept, they show that their
model could learn disentangled representations in a common latent code (i.e.,
representations of object features in latent space) and object segmentations
(i.e., attention masks on the original image) on non-trivial 3D scenes.

## Model Description

MONet builds upon the inductive bias that the world (or rather
*simple images* of the world) can often be approximated as a composition of
individual objects with the same underlying structure (i.e., different
instantiations of the same class). To put this into practice, [Burgess
et al. (2019)](https://arxiv.org/abs/1901.11390) developed a
conditional generative sampling scheme in which scenes are
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
  This state is called *scope* $\textbf{s}_k \in [0, 1]^{W\times
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

  The negative (decoder) log likelihood *NLL* (can be interpreted as
  the *reconstruction error*, see my post on
  [VAEs](https://borea17.github.io/paper_summaries/auto-encoding_variational_bayes#model-description))
  of the whole image is given by

  $$
  \text{NLL} = - \log \left( \sum_{k=1}^K \textbf{m}_k \odot p_{\boldsymbol{\theta}} \left(\textbf{x} | \textbf{z}_k \right)\right),
  $$

  where the sum can be understood as the reconstruction distribution
  of the whole image (mixture of components) conditioned on the latent
  codes $\textbf{z}_k$ and the attention masks $\textbf{m}_k$. Clearly, the
  reconstructions $\widetilde{\textbf{x}}_k \sim p\_{\boldsymbol{\theta}} \left(
  \textbf{x} | \textbf{z}_k\right)$ are unconstrained outside of the masked
  regions (i.e., where $m\_{k,i} = 0$).

  Note that they use a prior for the latent codes $\textbf{z}_k$, but not for
  the attentions masks $\textbf{m}_k$. Thus, the model is not fully generative,
  but rather a conditional generative model.


The figure below summarizes the whole architecture of the model by
showing the individual components (attention network, component VAE)
and their interaction.

| ![Schematic of MONet](/assets/img/04_MONet/MONet_schematic.png "Schematic of MONet") |
| :--         |
| **Schematic of MONet**. A recurrent attention network is used to obtain the attention masks $\textbf{m}^{(i)}$. Afterwards, a group structured representation is retrieved by feeding each concatenation of $\textbf{m}^{(i)}, \textbf{x}$ through the same VAE with encoder parameters $\boldsymbol{\phi}$ and decoder parameters $\boldsymbol{\theta}$. The outputs of the VAE are the unmasked image reconstructions $\widetilde{\textbf{x}}^{(i)}$ and mask reconstructions $\widetilde{\textbf{m}}^{(i)}$. Lastly, the reconstructed image is assembled using the deterministic attention masks $\textbf{m}^{(i)}$ and the sampled unmasked image reconstructions $\widetilde{\textbf{x}}^{(i)}$. |

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
decomposes with the same VAE[^4]. Restricting the model complexity of
the decoder (e.g., by using few layers), forces the model to produce
segmentation with similar tasks, i.e., segmentations over structurally
similar scene elements such that the VAE is capable of solving all
tasks (note that this is a hypothesis). The authors argue that
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
| **MONet Paper Results**: Decomposition on *Multi-dSprties* and *CLEVR* images. First row shows the input image, second and third row the corresponding reconstructions and segmentations by MONet (trained for 1,000,000 iterations). The last three rows show the unmasked component reconstructions from some chosen slots (indicated by $S$). Red arrows highlight occluded regions of shapes that are completed as full shapes. Taken from [Burgess et al. (2019)](https://arxiv.org/abs/1901.11390).  |

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

To reduce training time, we are going to generate a much simpler dataset
of $x$ images with two non-overlaping objects (`square` or `circle`) and a
fixed color space (`red`, `green` or `aqua`) for these objects, see
image below. The dataset is generated by sampling uniformly random
from possible latent factors, i.e., random non-overlaping
positions for the two objects, random object constellations and random
colors from color space, see code below image.


| ![Examples of Dataset](/assets/img/04_MONet/self_dataset.png "Examples of Dataset") |
| :--:        |
| Visualization of self-written dataset. |

{% capture code %}{% raw %}from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import TensorDataset
import torch


def generate_dataset(n_samples, SEED=1):
    ############### CONFIGURATION ###############
    canvas_size=64
    min_obj_size, max_obj_size = 12, 20
    min_num_obj, max_num_obj = 0, 2
    shapes = ["circle", "square"]
    colors = ["red", "green", "aqua"]
    #############################################
    data = torch.empty([n_samples, 3, canvas_size, canvas_size])
    labels = torch.empty([n_samples, 1, canvas_size, canvas_size])

    pos_positions = np.arange(canvas_size - max_obj_size - 1)

    np.random.seed(SEED)
    rnd_positions = np.random.choice(pos_positions, size=(n_samples, 2), replace=True)
    rnd_num_objs = np.random.randint(min_num_obj, max_num_obj + 1, size=(n_samples))
    rnd_obj_sizes = np.random.randint(min_obj_size, max_obj_size + 1, 
                                      size=(n_samples, max_num_obj))
    rnd_shapes = np.random.choice(shapes, size=(n_samples, max_num_obj), replace=True)
    rnd_colors = np.random.choice(colors, size=(n_samples, max_num_obj), replace=True)
    for i_data in range(n_samples):
        x_0, y_0 = rnd_positions[i_data]
        num_objs = rnd_num_objs[i_data]
        if num_objs > 1:
            # make sure that there is no overlap
            max_obj_size = max(rnd_obj_sizes[i_data])
            impos_x_pos = np.arange(x_0 - max_obj_size, x_0 + max_obj_size + 1)
            impos_y_pos = np.arange(y_0 - max_obj_size, y_0 + max_obj_size + 1)
            x_1 = np.random.choice(np.setdiff1d(pos_positions, impos_x_pos), size=1)
            y_1 = np.random.choice(np.setdiff1d(pos_positions, impos_y_pos), size=1)
        else:
            x_1 = 0
            y_1 = 0

        # current latent factors
        num_objs = rnd_num_objs[i_data]
        x_positions, y_positions = [x_0, x_1], [y_0, y_1]
        obj_sizes = rnd_obj_sizes[i_data]
        shapes = rnd_shapes[i_data]
        colors = rnd_colors[i_data]

        # create img and label tensors
        img, label = generate_img_and_label(
            x_pos=x_positions[:num_objs],
            y_pos=y_positions[:num_objs],
            shapes=shapes[:num_objs],
            colors=colors[:num_objs],
            sizes=obj_sizes[:num_objs],
            img_size=canvas_size
        )
        data[i_data] = img
        labels[i_data] = label
    dataset = TensorDataset(data, labels)
    return dataset


def generate_img_and_label(x_pos, y_pos, shapes, colors, sizes, img_size):
    """generates a img and corresponding segmentation label mask
    from the provided latent factors

    Args:
        x_pos (list): x positions of objects
        y_post (list): y positions of objects
        shapes (list): shape can only be `circle` or `square`
        colors (list): colors of object
        sizes (list): object sizes

    Returns:
        img (torch tensor): generated image represented as tensor
        label (torch tensor): corresponding semantic segmentation mask
    """
    out_img = Image.new("RGB", (img_size, img_size), color="black")
    labels = []
    # add objects
    for x, y, shape, color, size in zip(x_pos, y_pos, shapes, colors, sizes):
        img = Image.new("RGB", (img_size, img_size), color="black")
        # define end coordinates
        x_1, y_1 = x + size, y + size
        # draw new image onto black image
        img1 = ImageDraw.Draw(img)
        img2 = ImageDraw.Draw(out_img)
        if shape == "square":
            img1.rectangle([(x, y), (x_1, y_1)], fill=color)
            img2.rectangle([(x, y), (x_1, y_1)], fill=color)
        elif shape == "circle":
            img1.ellipse([(x, y), (x_1, y_1)], fill=color)
            img2.ellipse([(x, y), (x_1, y_1)], fill=color)
        labels.append((transforms.ToTensor()(img).sum(0) > 0).unsqueeze(0))
    out_image = transforms.ToTensor()(out_img).type(torch.float32)
    out_label = torch.zeros(1, img_size, img_size)
    for i_object in range(len(labels)):
        out_label[labels[i_object]] = i_object + 1
    return out_image, out_label{% endraw %}{% endcapture %}
{% include code.html code=code lang="python" %}

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
  $\text{logits }\boldsymbol{\alpha}_k$. A sigmoid layer can be used to
  transform these logits into probabilities, i.e.,

  $$
  \begin{align}
    \boldsymbol{\alpha}_k &= \text{Sigmoid} \left(\text{logits }
    \boldsymbol{\alpha}_k \right) = \frac {1} {1 + \exp\left(- \text{logits }
    \boldsymbol{\alpha}_k \right)}\\
    1 - \boldsymbol{\alpha}_k &= 1 - \text{Sigmoid} \left(\text{logits }
    \boldsymbol{\alpha}_k \right) = \frac {\exp\left(- \text{logits }
    \boldsymbol{\alpha}_k \right) } { 1 + \exp\left(- \text{logits }
    \boldsymbol{\alpha}_k \right)}
  \end{align}
  $$

  Additionally, [Burgess et al.
  (2019)](https://arxiv.org/abs/1901.11390) transform these
  probabilties into logaritmic units, i.e.,

  $$
  \begin{align}
     \log \boldsymbol{\alpha}_k &= - \log \left( 1 + \exp\left(- \text{logits }
    \boldsymbol{\alpha}_k \right)\right)=\text{LogSigmoid }\left(
  \text{logits } \boldsymbol{\alpha}_k \right)\\
     \log \left(1 - \boldsymbol{\alpha}_k\right) &= - \text{logits }
  \boldsymbol{\alpha}_k + \log \boldsymbol{\alpha}_k,
  \end{align}
  $$

  i.e., a LogSigmoid layer can be used (instead of a sigmoid
  layer with applying logarithm to both outputs) to speed up the computations.
  <!-- A sigmoid layer[^5] is used to transform -->
  <!-- these logits into probabilities, additionally these probabilities -->
  <!-- are transformed into log probabilities, i.e., -->
  <!-- $\log\left(\boldsymbol{\alpha}_k\right)$ and $\log \left( 1 - \boldsymbol{\alpha}_k\right)$. -->
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
  the channel dimensionality in the encoder to 64 output channels.

{% capture code %}{% raw %}import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """U-Net architecture with blocks proposed by Burgess et al. (2019)

    Attributes:
        encoder_blocks (list): u_net blocks of encoder path
        decoder_blocks (list): u_net blocks of decoder path
        bottleneck_MLP (list): bottleneck is a 3-layer MLP with ReLUs
        out_conv (nn.Conv2d): convolutional classification layer
    """

    def __init__(self):
        super().__init__()
        self.encoder_blocks = nn.ModuleList(
            [
                UNet._block(4, 16),              # [batch_size, 16, 64, 64]
                UNet._block(16, 32),             # [batch_size, 32, 32, 32]
                UNet._block(32, 64),             # [batch_size, 64, 16, 16]
                UNet._block(64, 64),             # [batch_size, 64, 8, 8]
                UNet._block(64, 64),             # [batch_size, 75, 4, 4]
            ]
        )
        self.bottleneck_MLP = nn.Sequential(
            nn.Flatten(),                        # [batch_size, 512*4*4]
            nn.Linear(64*4*4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),                 # [batch_size, 512*4*4]
            nn.ReLU(),
            nn.Linear(128, 64*4*4),              # [batch_size, 512*4*4]
            nn.ReLU(),             # reshaped into [batch_size, 512, 4, 4]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                UNet._block(128, 64),             # [batch_size, 64, 4, 4]
                UNet._block(128, 64),             # [batch_size, 64, 8, 8]
                UNet._block(128, 32),             # [batch_size, 32, 16, 16]
                UNet._block(64, 16),              # [batch_size, 32, 32, 32]
                UNet._block(32, 16),              # [batch_size, 64, 64, 64]
            ]
        )

        self.out_conv = nn.Conv2d(16, 1, kernel_size=(1,1), stride=1)
        return

    def forward(self, x):
        # go through encoder path and store intermediate results
        skip_tensors = []
        for index, encoder_block in enumerate(self.encoder_blocks):
            out = encoder_block(x)
            skip_tensors.append(out)
            # no resizing in the last block
            if index < len(self.encoder_blocks) - 1:  # downsample
                x = F.interpolate(
                    out, scale_factor=0.5, mode="nearest", 
					recompute_scale_factor=False
                )
        last_skip = out
        # feed last skip tensor through bottleneck
        out_MLP = self.bottleneck_MLP(last_skip)
        # reshape output to match last skip tensor
        out = out_MLP.view(last_skip.shape)
        # go through decoder path and use skip tensors
        for index, decoder_block in enumerate(self.decoder_blocks):
            inp = torch.cat((skip_tensors[-1 - index], out), 1)
            out = decoder_block(inp)
            # no resizing in the last block
            if index < len(self.decoder_blocks) - 1:  # upsample
                out = F.interpolate(out, scale_factor=2, mode="nearest")
        prediction = self.out_conv(out)
        return prediction

    @staticmethod
    def _block(in_channels, out_channels):
        """U-Net block as described by Burgess et al. (2019)"""
        u_net_block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(num_features=out_channels, affine=True),
            nn.ReLU(),
        )
        return u_net_block


class AttentionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet()
        return

    def forward(self, x, num_slots):
        log_s_k = torch.zeros_like(x[:, 0:1, :, :])
        # initialize list to store intermediate results
        log_m = []
        for slot in range(num_slots - 1):
            inp = torch.cat((x, log_s_k), 1)
            alpha_logits = self.unet(inp)  # [batch_size, 1, image_dim, image_dim]
            # transform into log probabilties log (alpha_k) and log (1 - alpha_k)
            log_alpha_k = F.logsigmoid(alpha_logits)
            log_1_m_alpha_k = -alpha_logits + log_alpha_k
            # compute log_new_mask, log_new_scope
            log_new_mask = log_s_k + log_alpha_k
            log_new_scope = log_s_k + log_1_m_alpha_k
            # store intermediate results in list
            log_m.append(log_new_mask)
            # update log scope
            log_s_k = log_new_scope
        log_m.append(log_s_k)
        # convert list to tensor [batch_size, num_slots, 1, image_dim, image_dim]
        log_m = torch.cat(log_m, dim=1).unsqueeze(2)
        return log_m{% endraw %}{% endcapture %}
{% include code.html code=code lang="python" %}


<!-- [^5]: [Burgess et al. (2019)](https://arxiv.org/abs/1901.11390) state -->
<!--     that they use a log softmax layer, however this would only be -->
<!--     possible if there were two channels at the output. Note that in -->
<!--     binary classification, a pixel-wise softmax layer (two channel) can be -->
<!--     transformed into a sigmoid layer (one channel) by using the -->
<!--     difference between the two channels as input:  -->
<!--     $$ -->
<!--     \begin{align} -->
<!--       \text{Softmax} (x_1) &= \frac {1} {1 + \exp(x_2 - x_1)} = \frac -->
<!--     {1} {1 + \exp(t)} = \text{Sigmoid} (t), \\ -->
<!--       \text{Softmax} (x_2) &= \frac {1}{1 + \exp(x_1 - x_2)} = \frac -->
<!--     {1}{1+\exp(-t)} = 1 - \text{Sigmoid}(t). -->
<!--     \end{align} -->
<!--     $$ -->

* **Component VAE**: The architectures for the encoder
  $q\_{\boldsymbol{\phi}}$ and decoder $p\_{\boldsymbol{\theta}}$
  neural networks are described in appendix B.1 of [Burgess et al.
  (2019)](https://arxiv.org/abs/1901.11390). Basically, the encoder
  $q\_{\boldsymbol{\phi}}(\textbf{z}_k | \textbf{x}, \textbf{m}_k)$ is a typical CNN that takes the
  concatentation of an image $\textbf{x}$ and a segmentation mask in
  logaritmic units $\log \textbf{m}_k$ as input to compute the mean
  $\boldsymbol{\mu}\_{E, k}$ and
  logarithmed variance $\boldsymbol{\sigma}^2\_{E,k}$ of the Gaussian latent
  space distribution $\mathcal{N} \left(
  \boldsymbol{\mu}\_{E, k}, \text{diag}\left(\boldsymbol{\sigma}^2\_{E,k} \right)
  \right)$. Sampling from this distribution is avoided by using
  the reparametrization trick, i.e., the latent variable
  $\textbf{z}_k$ is expressed as a deterministic variable[^5]

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
  of the Bernoulli distribution $\text{logits }\textbf{p}_k$.
  <!-- These logits are converted into -->
  <!-- probabilties $\textbf{p}_k$ using a sigmoid layer. -->

{% capture code %}{% raw %}class CNN_VAE(nn.Module):
    """simple CNN-VAE class with a Gaussian encoder (mean and diagonal variance
    structure) and a Gaussian decoder with fixed variance 
    (decoder is implemented as a Spatial Broadcast decoder) 

    Attributes
        latent_dim (int): dimension of latent space
        encoder (nn.Sequential): encoder network for mean and log_var
        decoder (nn.Sequential): spatial broadcast decoder  for mean (fixed var)
        x_grid (torch tensor): appended x coordinates for spatial broadcast decoder
        y_grid (torch tensor): appended x coordinates for spatial broadcast decoder
    """

    def __init__(self):
        super(CNN_VAE, self).__init__()
        self.latent_dim = 8
        self.encoder = nn.Sequential(
            # shape: [batch_size, 4, 64, 64]
            nn.Conv2d(4, 32, kernel_size=(3,3), stride=2, padding=1),
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
            nn.Linear(64*4*4, 256),
            nn.ReLU(),
            nn.Linear(256, 2*self.latent_dim),
        )
        # spatial broadcast decoder configuration
        img_size = 64
        # "input width and height of CNN both 8 larger than target output"
        x = torch.linspace(-1, 1, img_size + 8)
        y = torch.linspace(-1, 1, img_size + 8)
        x_grid, y_grid = torch.meshgrid(x, y)
        # reshape into [1, 1, img_size, img_size] and save in state_dict
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape).clone())
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape).clone())
        self.decoder = nn.Sequential(
             # shape [batch_size, latent_dim + 2, 72, 72]
            nn.Conv2d(in_channels=self.latent_dim+2, out_channels=16,
                      stride=(1, 1), kernel_size=(3,3)),
            nn.ReLU(),
            # shape [batch_size, 16, 70, 70]
            nn.Conv2d(in_channels=16, out_channels=16, stride=(1,1),
                      kernel_size=(3, 3)),
            nn.ReLU(),
            # shape [batch_size, 16, 68, 68]
            nn.Conv2d(in_channels=16, out_channels=16, stride=(1,1),
                      kernel_size=(3, 3)),
            nn.ReLU(),
            # shape [batch_size, 16, 66, 66]
            nn.Conv2d(in_channels=16, out_channels=16, stride=(1,1),
                      kernel_size=(3, 3)),
            # shape [batch_size, 4, 64, 64]
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=4, stride=(1,1),
                      kernel_size=(1, 1)),
        )
        return

    def forward(self, x):
        [z, mu_E, log_var_E] = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z, mu_E, log_var_E

    def encode(self, x):
        out_encoder = self.MLP(self.encoder(x))
        mu_E, log_var_E = torch.chunk(out_encoder, 2, dim=1)
        # sample noise variable for each batch
        epsilon = torch.randn_like(log_var_E)
        # get latent variable by reparametrization trick
        z = mu_E + torch.exp(0.5 * log_var_E) * epsilon
        return [z, mu_E, log_var_E]

    def decode(self, z):
        batch_size = z.shape[0]
        # reshape z into [batch_size, latent_dim, 1, 1]
        z = z.view(z.shape + (1, 1))
        # tile across image [batch_size, latent_im, 64+8, 64+8]
        z_b = z.repeat(1, 1, 64 + 8, 64 + 8)
        # upsample x_grid and y_grid to [batch_size, 1, 64+8, 64+8]
        x_b = self.x_grid.repeat(batch_size, 1, 1, 1)
        y_b = self.y_grid.repeat(batch_size, 1, 1, 1)
        # concatenate vectors [batch_size, latent_dim+2, 64+8, 64+8]
        z_sb = torch.cat((z_b, x_b, y_b), dim=1)
        # apply convolutional layers mu_D
        mu_D = self.decoder(z_sb)
        return mu_D

	
class ComponentVAE(CNN_VAE):
    """Component VAE class for use in MONet as proposed by Burgess et al. (2019)

    Attributes:
        #################### CNN_VAE ########################
        encoder (nn.Sequential): encoder network for mean and log_var
        decoder (nn.Sequential): decoder network for mean (fixed var)
        img_dim (int): image dimension along one axis
        expand_dim (int): expansion of latent image to accomodate for lack of padding
        x_grid (torch tensor): appended x coordinates for spatial broadcast decoder
        y_grid (torch tensor): appended x coordinates for spatial broadcast decoder
        #####################################################
        img_channels (int): number of channels in image
    """

    def __init__(self,):
        super().__init__()
        self.img_channels = 3
        return

    def forward(self, image, log_mask, deterministic=False):
        """
        parellize computation of reconstructions

        Args:
            image (torch.tensor): input image [batch, img_channels, img_dim, img_dim]
            log_mask (torch.tensor): all seg masks [batch, slots, 1, img_dim, img_dim]

        Returns:
            mu_z_k (torch.tensor): latent mean [batch, slot, latent_dim]
            log_var_z_k (torch.tensor): latent log_var [batch, slot, latent_dim]
            z_k (torch.tensor): latent log_var [batch, slot, latent_dim]
            x_r_k (torch.tensor): img reconstruction 
				[batch, slot, img_chan, img_dim, img_dim]
            logits_m_r_k (torch.tensor): mask recons. [batch, slot, 1, img_dim, img_dim]
        """
        num_slots = log_mask.shape[1]
        # create input [batch_size*num_slots, image_channels+1, img_dim, img_dim]
        x = ComponentVAE._prepare_input(image, log_mask, num_slots)
        # get encoder distribution parameters [batch*slots, latent_dim]
        [z_k, mu_z_k, log_var_z_k] = self.encode(x)
        if deterministic:
            z_k = mu_z_k
        # get decoder dist. parameters [batch*slots, image_channels, img_dim, img_dim]
        [x_r_k, logits_m_r_k] = self.decode(z_k)
        # convert outputs into easier understandable shapes
        [mu_z_k, log_var_z_k, z_k, x_r_k, logits_m_r_k] = ComponentVAE._prepare_output(
            mu_z_k, log_var_z_k, z_k, x_r_k, logits_m_r_k, num_slots
        )
        return [mu_z_k, log_var_z_k, z_k, x_r_k, logits_m_r_k]

    def decode(self, z):
        """
        Args:
            z (torch.tensor): [batch_size*num_slots, latent_dim]

        Returns:
            mu_x (torch.tensor): [batch*slots, img_channels, img_dim, img_dim]
            logits_m (torch.tensor): [batch*slots, 1, img_dim, img_dim]

        """
        mu_D = super().decode(z)
        # split into means of x and logits of m
        mu_x, logits_m = torch.split(mu_D, self.img_channels, dim=1)
        # enforce positivity of mu_x
        mu_x = mu_x.abs()
        return [mu_x, logits_m]

    @staticmethod
    def _prepare_input(image, log_mask, num_slots):
        """
        Args:
            image (torch.tensor): input image [batch, img_channels, img_dim, img_dim]
            log_mask (torch.tensor): all seg masks [batch, slots, 1, img_dim, img_dim]
            num_slots (int): number of slots (log_mask.shape[1])

        Returns:
            x (torch.tensor): input image [batch*slots, img_channels+1, img_dim, img_dim]
        """
        # prepare image [batch_size*num_slots, image_channels, img_dim, img_dim]
        image = image.repeat(num_slots, 1, 1, 1)
        # prepare log_mask [batch_size*num_slots, 1, img_dim, img_dim]
        log_mask = torch.cat(log_mask.squeeze(2).chunk(num_slots, 1), 0)
        # concatenate along color channel
        x = torch.cat((image, log_mask), dim=1)
        return x

    @staticmethod
    def _prepare_output(mu_z_k, log_var_z_k, z_k, x_r_k, logits_m_r_k, num_slots):
        """
        convert output into an easier understandable format

        Args:
            mu_z_k (torch.tensor): [batch_size*num_slots, latent_dim]
            log_var_z_k (torch.tensor): [batch_size*num_slots, latent_dim]
            z_k (torch.tensor): [batch_size*num_slots, latent_dim]
            x_r_k (torch.tensor): [batch_size*num_slots, img_channels, img_dim, img_dim]
            logits_m_r_k (torch.tensor): [batch_size*num_slots, 1, img_dim, img_dim]
            num_slots (int): number of slots (log_mask.shape[1])

        Returns:
            mu_z_k (torch.tensor): [batch, slot, latent_dim]
            log_var_z_k (torch.tensor): [batch, slot, latent_dim]
            z_k (torch.tensor): [batch, slot, latent_dim]
            x_r_k (torch.tensor): [batch, slots, img_channels, img_dim, img_dim]
            logits_m_r_k (torch.tensor): [batch, slots, 1, img_dim, img_dim]
        """
        mu_z_k = torch.stack(mu_z_k.chunk(num_slots, dim=0), 1)
        log_var_z_k = torch.stack(log_var_z_k.chunk(num_slots, dim=0), 1)
        z_k = torch.stack(z_k.chunk(num_slots, dim=0), 1)
        x_r_k = torch.stack(x_r_k.chunk(num_slots, dim=0), 1)
        logits_m_r_k = torch.stack(logits_m_r_k.chunk(num_slots, dim=0), 1)
        return [mu_z_k, log_var_z_k, z_k, x_r_k, logits_m_r_k]{% endraw %}{% endcapture %}
{% include code.html code=code lang="python" %}

[^5]: This is explained in more detail in my [VAE](https://borea17.github.io/paper_summaries/auto-encoding_variational_bayes) post. For simplicity, we are setting the number of (noise variable) samples $L$ per datapoint to 1 (see equation $\displaystyle \widetilde{\mathcal{L}}$ in [*Reparametrization Trick*](https://borea17.github.io/paper_summaries/auto-encoding_variational_bayes#model-description) paragraph). Note that [Kingma and Welling (2013)](https://arxiv.org/abs/1312.6114) stated that in their experiments setting $L=1$ sufficed as long as the minibatch size was large enough.


 * **MONet Implementation**: The compositional structure is achieved
    by looping for $K$ steps over the image and combining the attention
    network with the component VAE. While attention masks and latent
    codes can be generated easily (during test time), computing the
    loss $\mathcal{L}$ is more complicated. Remind that the loss
    function is given by

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
       i}(\boldsymbol{\theta}), \sigma_k^2 \right)$. 

	   <!-- , hence the term -->
       <!-- can be rewritten as follows -->

       <!--  $$ -->
       <!--  \begin{align} -->
       <!--    \text{NLL} &= -\sum_{i=1}^N \log \left(  \sum_{k=1}^K -->
       <!--  m_{k, i} \left(\boldsymbol{\psi} \right) \frac {1}{ \sqrt{2 \pi \sigma_k^2} } \exp \left( - \frac { \left[ x_i - -->
       <!--  \mu_{k,i} (\boldsymbol{\theta}) \right]^2 } {2 \sigma_k^2}  \right) -->
       <!--  \right)\\ -->
       <!--  &= -\sum_{i=1}^N \log \left( \frac {1} {\sqrt{2\pi}} -->
       <!--  \sum_{k=1}^K \exp \left( -->
       <!--  \log \frac { m_{k, i} \left(\boldsymbol{\psi} \right) } {\sigma_k} -->
       <!--  - \frac { \big[x_i - \mu_{k, i} (\boldsymbol{\theta}) \big]^2 } {2 \sigma_k^2}  \right) -->
       <!--  \right)\\ -->
       <!--  &= \frac {N \log 2\pi}{2} \\%- \sum_{i=1}^N - \frac {\log 2 \pi}{2}\\ -->
       <!--  & \quad -\sum_{i=1}^N \log \left( \sum_{k=1}^K \exp \left( -->
       <!--  \log \big[m_{k, i} \left(\boldsymbol{\psi} \right)\big]  - -->
       <!--  \log \sigma_k - \frac { \big[x_i - \mu_{k, i} (\boldsymbol{\theta}) \big]^2 -->
       <!--  } {2 \sigma_k^2}  \right) \right), -->
       <!--  \end{align} -->
       <!--  $$ -->

       <!--  where $i$ enumerates the pixel space ($N=W\cdot H$). The inner -->
       <!--  sum (index $k$) results in a reconstructed image distribution, -->
       <!--  the outer sum (index $i$) computes the log likelihood of each -->
       <!--  pixel independently and sums them to retrieve the -->
       <!--  reconstruction accuracy of the whole image. The term inside -->
       <!--  the exponent is unconstrained outside of the masked regions[^6] -->
       <!--  (for each reconstruction, i.e., fixed $k$). Note that [Burgess -->
       <!--  et al. (2019)](https://arxiv.org/abs/1901.11390) define the -->
       <!--  variances of the decoder distribution for each component as -->
       <!--  follows -->

       <!--  $$ -->
       <!--    \sigma_k^2 = \begin{cases} \sigma_{bg}^2 & \text{if } k=1 -->
       <!--  \quad \text{(background variance)} \\ -->
       <!--  \sigma_{fg}^2 & \text{if } k>1 \quad \text{(foreground variance)}\end{cases} -->
       <!--  $$ -->

    2. *Regularization Term for Distribution of $\textbf{z}_k$*: The
       coding space is regularized using the KL divergence between the
       latent (posterior) distribution $q\_{\boldsymbol{\phi}} \left( \textbf{z}_k \right) \sim
       \mathcal{N} \left( \boldsymbol{\mu}_k, \left(\boldsymbol{\sigma}_k^2\right)^{\text{T}}
       \textbf{I} \right)$ factorized across slots and the latent prior distribution weighted with the hyperparameter
       $\beta$. The product of multiple Gaussians is itself a
       Gaussian, however it is rather complicated to compute the new
       mean and covariance matrix of this Gaussian. Fortunately, each
       $\textbf{z}_k$ is sampled independently from the corresponding
       latent distribution $q\_{\boldsymbol{\phi}}(\textbf{z}_k)$,
       thus we can generate the new mean and covariance by
       concatenation (see [this
       post](https://stats.stackexchange.com/a/308137)), i.e.,

       $$
         q(\textbf{z}_1, \dots, \textbf{z}_K) = \prod_{k=1}^K q_{\boldsymbol{\phi}} \left(\textbf{z}_k
       \right)  = q\left( \begin{bmatrix} \textbf{z}_1 \\ \vdots
         \\ \textbf{z}_K  \end{bmatrix}\right) = \mathcal{N} \left(
         \underbrace{
         \begin{bmatrix} \boldsymbol{\mu}_1 \\ \vdots
          \\ \boldsymbol{\mu}_K \end{bmatrix}}_{
          \widehat{\boldsymbol{\mu}}}, \underbrace{\text{diag}\left(
         \begin{bmatrix} \boldsymbol{\sigma}_1^2 \\  \vdots\\
         \boldsymbol{\sigma}_K^2  \end{bmatrix}
         \right)}_{ \left(\widehat{\boldsymbol{\sigma}}^2\right)^{\text{T}} \textbf{I}}\right)
       $$

       [Burgess et al. (2019)](https://arxiv.org/abs/1901.11390) chose
       a unit Gaussian distribution as the latent prior $p(\textbf{z})
       \sim \mathcal{N} \left(\textbf{0}, \textbf{I} \right)$ with
       $\text{dim}(\textbf{0}) = \text{dim}(\hat{\boldsymbol{\mu}})$.
       The KL divergence between those two Gaussian distributions
       can be calculated in closed form (see Appendix B of [Kingma and Welling (2013)](https://arxiv.org/abs/1312.6114))

       $$
       \begin{align}
          D_{KL} \left( \prod_{k=1}^K q_{\boldsymbol{\phi}}
       \left(\textbf{z}_k \right) || p(\textbf{z}) \right) &= -\frac
       {1}{2} \sum_{j=1}^{K \cdot L} \left(1 + \log \left(
       \widehat{\sigma}_j^2 \right) - \widehat{\mu}_j^2 - \widehat{\sigma}_j^2 \right),
       \end{align}
       $$

       where $L$ denotes the dimensionality of the latent space.

    3. *Reconstruction Error between $\widetilde{\textbf{m}}_k$ and
       $\textbf{m}_k$*: Remind that the attention network
       $\boldsymbol{\alpha}\_{\boldsymbol{\psi}}$ produces $K$
       segmentation masks in logaritmic units, i.e., $\log
       \textbf{m}_k$. By construction $\sum\_{k=1}^K \textbf{m}_k =
       \textbf{1}$, i.e., concatentation of the attention masks
       $\textbf{m} = \begin{bmatrix} \textbf{m}_1 & \dots &
       \textbf{m}_K \end{bmatrix}^{\text{T}}$ can be interpreted as a
       pixel-wise categorical distribution[^7]. Similarly,
       concatenating the logits probabilties of the component VAE and
       applying a pixel-wise softmax, i.e.,

       $$
       \widetilde{\textbf{m}} = \begin{bmatrix} \widetilde{\textbf{m}}_1 \\ \vdots \\
       \widetilde{\textbf{m}}_K \end{bmatrix} = \text{Softmax}\left(\begin{bmatrix} \text{logits }\textbf{p}_1 \\ \vdots \\
       \text{logits }\textbf{p}_K \end{bmatrix}\right),
       $$

       transforms the logits outputs of the component VAE into a pixel-wise
       categorical distribution. Thus, the KL-divergence can be
       calculated as follows

       $$
       \begin{align}
         D_{KL} \left( q_{\boldsymbol{\psi}} \left( \textbf{c} |
    \textbf{x} \right) || p_{\boldsymbol{\theta}} \left( \textbf{c} | \{
    \textbf{z}_k \} \right) \right) &=
         \sum_{i=1}^{H\cdot W} D_{KL} \left( {\textbf{m}}_i || \widetilde{\textbf{m}}_i \right) \\
         &= \sum_{i=1}^{H\cdot W} \textbf{m}_i \odot \left(\log \textbf{m}_i - \log \widetilde{\textbf{m}}_i \right),
       \end{align}
       $$

       where $i$ denotes the pixel space, i.e., $\textbf{m}_i \in [0,
       1]^{K}$. To make the computation more efficient, we directly
       compute the reconstructed segmentations in logaritmic units
       using pixel-wise logsoftmax, i.e.,

       $$
       \log \widetilde{\textbf{m}} = \text{LogSoftmax}\left(\begin{bmatrix} \text{logits }\textbf{p}_1 \\ \vdots \\
       \text{logits }\textbf{p}_K \end{bmatrix}\right).
       $$

{% capture code %}{% raw %}import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


class MONet(pl.LightningModule):
    """Multi-Object Network class as described by Burgess et al. (2019)
    
    Atributes:
        n_samples (int): number of samples in training dataset
        attention_network (AttentionNetwork)
        component_VAE (ComponentVAE)
        ############## loss specific ##############
        bg_var (float): background variance
        fg_var (float): foreground variance
        beta (float): hyperparamater for loss
        gamma (float): hyperparameter for loss
        ###########################################
        ############ training specific ############
        num_slots_train (int): number of slots used during training time
        lr (float): learning rate
        batch_size (int): batch size used during training
        log_every_k_epochs (int): how often current result img should be logged
        ###########################################
    """

    def __init__(self, n_samples):
        super(MONet, self).__init__()
        self.n_samples = n_samples
        self.attention_network = AttentionNetwork()
        self.component_VAE = ComponentVAE()
        # initialize all biases to zero
        self.attention_network.apply(MONet.weight_init)
        self.component_VAE.apply(MONet.weight_init)
        ############## loss specific ##############
        self.num_slots_train = 3
        self.bg_var, self.fg_var = 0.09**2, 0.11**2
        self.beta = 0.5
        self.gamma = 0.5
        ###########################################
        ############ training specific ############
        self.lr, self.batch_size = 0.0001, 64
        self.log_every_k_epochs = 1
        # Initialise pixel output standard deviations (NLL calculation)
        var = self.fg_var * torch.ones(1, self.num_slots_train, 1, 1, 1)
        var[0, 0, 0, 0, 0] = self.bg_var  # first step
        self.register_buffer("var", var)
        self.save_hyperparameters()
        return

    def forward(self, x, num_slots):
        """
        defines the inference procedure of MONet, i.e., computes the latent
        space and keeps track of useful metrics

        Args:
            x (torch.tensor): image [batch_size, img_channels, img_dim, img_dim]
            num_slots (int): number of slots

        Returns:
            out (dict): output dictionary containing
                log_m_k (torch.tensor) [batch, slots, 1, img_dim, img_dim]
                    (logarithmized attention masks of attention_network)
                mu_k (torch.tensor) [batch, slots, latent_dim]
                    (means of component VAE latent space)
                log_var_k (torch.tensor) [batch, slots, latent_dim]
                    (logarithmized variances of component VAE latent space)
                x_r_k (torch.tensor) [batch, slots, img_channels, img_dim, img_dim]
                    (slot-wise VAE image reconstructions)
                logits_m_r_k (torch.tensor) [batch, slots, 1, img_dim, img_dim]
                    (slot-wise VAE mask reconstructions in logits)
                x_tilde (torch.tensor) [batch, img_channels, img_dim, img_dim]
                    (reconstructed image using x_r_k and log_m_k)
        """
        # compute all logarithmized masks (iteratively)
        log_m_k = self.attention_network(x, num_slots)
        # compute all VAE reconstructions (parallel)
        [mu_z_k, log_var_z_k, z_k, x_r_k, logits_m_r_k] = self.component_VAE(x, 
                                                                             log_m_k.exp())
        # store output in dict
        output = dict()
        output["log_m_k"] = log_m_k
        output["mu_z_k"] = mu_z_k
        output["log_var_z_k"] = log_var_z_k
        output["z_k"] = z_k
        output["x_r_k"] = x_r_k
        output["logits_m_r_k"] = logits_m_r_k
        output["x_tilde"] = (log_m_k.exp() * x_r_k).sum(axis=1)
        return output
    
    
    ########################################
    #########  TRAINING FUNCTIONS  #########
    ########################################

    def training_step(self, batch, batch_idx):
        x, labels = batch  # labels are not used here (unsupervised)
        output = self.forward(x, self.num_slots_train)        
        ############ NLL \sum_k m_k log p(x_k) #############################
        NLL = (
            output["log_m_k"].exp() * 
            (((x.unsqueeze(1) - output["x_r_k"]) ** 2 / (2 * self.var)))
        ).sum(axis=(1, 2, 3, 4))
        # compute KL divergence of latent space (component VAE) per batch
        KL_div_VAE = -0.5 * (
            1 + output["log_var_z_k"] - output["mu_z_k"] ** 2 
            - output["log_var_z_k"].exp()
        ).sum(axis=(1, 2))
        # compute KL divergence between masks
        log_m_r_k = output["logits_m_r_k"].log_softmax(dim=1)
        KL_div_masks = (output["log_m_k"].exp() * (output["log_m_k"] - log_m_r_k)).sum(
            axis=(1, 2, 3, 4)
        )
        # compute loss
        loss = (NLL.mean() + self.beta * KL_div_VAE.mean() 
                + self.gamma * KL_div_masks.mean())
        # log results in TensorBoard
        step = self.global_step
        self.logger.experiment.add_scalar("loss/NLL", NLL.mean(), step)
        self.logger.experiment.add_scalar("loss/KL VAE", KL_div_VAE.mean(), step)
        self.logger.experiment.add_scalar("loss/KL masks", KL_div_masks.mean(), step)
        self.logger.experiment.add_scalar("loss/loss", loss, step)
        return {"loss":loss, "x": x}

    def training_epoch_end(self, outputs):
        """this function is called after each epoch"""
        step = int(self.current_epoch)
        if (step + 1) % self.log_every_k_epochs == 0:
            # log some images, their segmentations and reconstructions
            n_samples = 7
            
            last_x = outputs[-1]["x"]
            i_samples = np.random.choice(range(len(last_x)), n_samples, False)
            images = last_x[i_samples]
            
            fig_rec = self.plot_reconstructions_and_decompositions(images, 
                                                                   self.num_slots_train)
            self.logger.experiment.add_figure("image and reconstructions", 
                                              fig_rec, global_step=step)
        return
    
    ########################################
    ######### TRAINING SETUP HOOKS #########
    ########################################

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        return optimizer
    
    @staticmethod
    def weight_init(m):
        """initialize all bias to zero"""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        return
    
    ########################################
    ####### PLOT AND HELPER FUNCTIONS ######
    ########################################

    @staticmethod
    def convert_masks_indices_to_mask_rgb(masks_ind, slots):
        colors = plt.cm.get_cmap("hsv", slots + 1)
        cmap_rgb = colors(np.linspace(0, 1, slots + 1))[:, 0:3]
        masks_RGB = cmap_rgb[masks_ind].squeeze(1)
        masks_RGB_tensor = torch.from_numpy(masks_RGB)
        return masks_RGB_tensor

    def plot_reconstructions_and_decompositions(self, images, num_slots):
        monet_output = self.forward(images, num_slots)
        batch_size, img_channels = images.shape[0:2]
        
        colors = plt.cm.get_cmap("hsv", num_slots + 1)
        cmap = colors(np.linspace(0, 1, num_slots + 1))
        
        # get mask indices using argmax [batch_size, 1, 64, 64]
        masks_ind = monet_output["log_m_k"].exp().argmax(1).detach().cpu()
        # convert into RGB values  [batch_size, 64, 64, 3]
        masks_RGB = MONet.convert_masks_indices_to_mask_rgb(masks_ind, num_slots)              
        fig = plt.figure(figsize=(14, 10))
        for counter in range(batch_size):
            orig_img = images[counter]
            # data
            plt.subplot(3 + num_slots, batch_size + 1, counter + 2)
            plt.imshow(transforms.ToPILImage()(orig_img))
            plt.axis('off')
            # reconstruction mixture
            x_tilde = monet_output["x_tilde"][counter].clamp(0, 1)
            plt.subplot(3 + num_slots, batch_size + 1, counter + 2 + (batch_size + 1))
            plt.imshow(transforms.ToPILImage()(x_tilde))
            plt.axis('off')
            # segmentation (binary) from attention network
            plt.subplot(3 + num_slots, batch_size + 1, counter + 2 + (batch_size + 1)*2)
            plt.imshow(masks_RGB[counter])
            plt.axis('off')
            # unmasked component reconstructions
            x_r_k = monet_output["x_r_k"][counter].clamp(0, 1)
            for slot in range(num_slots):
                x_rec = x_r_k[slot]
                plot_idx =  counter + 2 + (batch_size + 1)*(slot+3)
                plt.subplot(3 + num_slots, batch_size + 1, plot_idx)
                plt.imshow(transforms.ToPILImage()(x_rec))
                plt.axis('off')
        # annotation plots
        ax = plt.subplot(3 + num_slots, batch_size + 1, 1)
        ax.annotate('Data', xy=(1, 0.5), xycoords='axes fraction',
                    fontsize=14, va='center', ha='right')
        ax.set_aspect('equal')
        ax.axis('off')
        ax = plt.subplot(3 + num_slots, batch_size + 1, batch_size + 2)
        ax.annotate('Reconstruction\nmixture', xy=(1, 0.5), xycoords='axes fraction',
                    fontsize=14, va='center', ha='right')
        ax.set_aspect('equal')
        ax.axis('off')
        ax = plt.subplot(3 + num_slots, batch_size + 1, 2*batch_size + 3)
        ax.annotate('Segmentation', xy=(1, 0.5), xycoords='axes fraction',
                    fontsize=14, va='center', ha='right')
        ax.set_aspect('equal')
        ax.axis('off')
        for slot in range(num_slots):
            ax = plt.subplot(3 + num_slots, batch_size + 1, 
                             1 + (batch_size + 1)*(slot+3))
            ax.annotate(f'S{slot+1}', xy=(1, 0.5), xycoords='axes fraction',
                        fontsize=14, va='center', ha='right', weight='bold',
                        color=cmap[slot])
            ax.set_aspect('equal')
            ax.axis('off')
        return fig
    
    def plot_ComponentVAE_results(self, images, num_slots):
        monet_output = self.forward(images, num_slots)
        x_r_k = monet_output["x_r_k"]
        masks = monet_output["log_m_k"].exp()
        # get mask indices using argmax [batch_size, 1, 64, 64]
        masks_ind = masks.argmax(1).detach().cpu()
        # convert into RGB values  [batch_size, 64, 64, 3]
        masks_RGB = MONet.convert_masks_indices_to_mask_rgb(masks_ind, num_slots) 
        
        colors = plt.cm.get_cmap('hsv', num_slots + 1)
        cmap = colors(np.linspace(0, 1, num_slots + 1))
        n_samples, img_channels = images.shape[0:2]
        fig = plt.figure(constrained_layout=False, figsize=(14, 14))
        grid_spec = fig.add_gridspec(2, n_samples, hspace=0.1)
        
        for counter in range(n_samples):
            orig_img = images[counter]
            x_tilde = monet_output["x_tilde"][counter].clamp(0, 1)
            segmentation_mask = masks_RGB[counter]
            # upper plot: Data, Reconstruction Mixture, Segmentation
            upper_grid = grid_spec[0, counter].subgridspec(3, 1)
            for upper_plot_index in range(3):
                ax = fig.add_subplot(upper_grid[upper_plot_index])
                if upper_plot_index == 0:
                    plt.imshow(transforms.ToPILImage()(orig_img))
                elif upper_plot_index == 1:
                    plt.imshow(transforms.ToPILImage()(x_tilde))   
                else:
                    plt.imshow(segmentation_mask)
                plt.axis('off')
                if counter == 0:  # annotations
                    if upper_plot_index == 0:  # Data
                        ax.annotate('Data', xy=(-0.1, 0.5), 
                                    xycoords='axes fraction', ha='right',
                                    fontsize=14, va='center',)
                    elif upper_plot_index == 1:  # Reconstruction mixture
                        ax.annotate('Reconstruction\nmixture', xy=(-0.1, 0.5), 
                                     va='center',
                                     xycoords='axes fraction', fontsize=14, ha='right')
                    else:  # Segmentation
                        ax.annotate('Segmentation', xy=(-0.1, 0.5), va='center',
                                     xycoords='axes fraction', fontsize=14, ha='right')
            # lower plot: Component VAE reconstructions
            lower_grid = grid_spec[1, counter].subgridspec(num_slots, 2, 
                                                           wspace=0.1, hspace=0.1)
            for row_index in range(num_slots):
                x_slot_r = x_r_k[counter][row_index]
                m_slot_r = masks[counter][row_index]
                for col_index in range(2):
                    ax = fig.add_subplot(lower_grid[row_index, col_index])
                    if col_index == 0:  # unmasked
                        plt.imshow(transforms.ToPILImage()(x_slot_r.clamp(0, 1)))
                        if row_index == 0:
                            plt.title('Unmasked', fontsize=14)
                        plt.axis('off')
                    else:  # masked
                        masked = ((1 - m_slot_r)*torch.ones_like(x_slot_r) 
                                  + m_slot_r*x_slot_r)
                        #masked = m_slot_r*x_slot_r
                        plt.imshow(transforms.ToPILImage()(masked.clamp(0, 1)))
                        if row_index == 0:
                            plt.title('Masked', fontsize=14)
                        plt.axis('off')
                    ax.set_aspect('equal')
                    if counter == 0 and col_index == 0:  # annotations
                        ax.annotate(f'S{row_index+1}', xy=(-0.1, 0.5), 
                                    xycoords='axes fraction', ha='right',
                                    fontsize=14, va='center', weight='bold',
                                    color=cmap[row_index])
        return
                                   
    ########################################
    ########## DATA RELATED HOOKS ##########
    ########################################

    def prepare_data(self) -> None:
        n_samples = self.n_samples
        self.dataset = generate_dataset(n_samples=n_samples)
        return

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, 
                          num_workers=12, shuffle=True){% endraw %}{% endcapture %}
{% include code.html code=code lang="python" %}


[^7]: Note that concatenation of masks leads to a three dimensional
    tensor.


* **Training Procedure**: [Burgess et al.
  (2019)](https://arxiv.org/abs/1901.11390) chose `RMSProp` for the
  optimization with a learning rate of `0.0001` and a batch size of
  `64`, see Appendix B.3. Thanks to the [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
  framework, these paramters are already defined in the model and we can easily integrate 
  tensorboard into our training procedure:

{% capture code %}{% raw %}from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger


def train(n_samples, num_epochs, SEED=1):
    seed_everything(SEED)

    monet = MONet(n_samples)
    logger = TensorBoardLogger('./results', name="SimplifiedMultiSprites")
    # initialize pytorch lightning trainer
    num_gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(
        deterministic=True,
        gpus=num_gpus,
        track_grad_norm=2,
        gradient_clip_val=2,  # don't clip
        max_epochs=num_epochs,
        progress_bar_refresh_rate=20,
        logger=logger,
    )
     # train model
    trainer.fit(monet)
    trained_monet = monet
    return trained_monet

trained_monet = train(n_samples=50000, num_epochs=10){% endraw %}{% endcapture %}
{% include code.html code=code lang="python" %}

![Training](/assets/img/04_MONet/MONET_train.png "Training")


### Results

The following visualization are inspired by Figure 3 and 7 of 
[Burgess et al. (2019)](https://arxiv.org/abs/1901.11390) and mainly serve to evaluate
the representation quality of the trained model.


* **MONet Reconstructions and Decompositions**: The most intuitive
  visualization is to show some (arbitrarly chosen) fully
  reconstructed images (i.e, `Reconstruction mixture` $\widetilde{\textbf{x}} = \sum_{k=1}^K
  \textbf{m}_k \odot \widetilde{\textbf{x}}_k$) compared to the
  original input $\textbf{x}$ (`Data`) together with the learned segmentation
  masks (i.e., `Segmentation` $\\{ \textbf{m}_k \\}$) of the attention network. Note
  that in order to visualize the segmentations
  in one plot, we cast the attenion masks into binary
  attention masks by applying `arg max` pixel-wise over all $K$
  attention masks. In addition, all
  umasked component VAE reconstructions (i.e., `S(k)`
  $\widetilde{\textbf{x}}_k$) are shown, see figure below.

  | ![MONet Reconstruction and Decompositions](/assets/img/04_MONet/reconstruction_and_decompositions.png "MONet Reconstructions and Decompositions") |
  | :--         |
  | **Figure 7 of** [Burgess et al. (2019)](https://arxiv.org/abs/1901.11390): Each example shows the image fed as input data to the model, with corresponding outputs from the model. Reconstruction mixtures show sum of components from all slots, weighted by the learned masks from the attention network. Colour-coded segmentation maps summarize the attention masks $\\{\textbf{m}_k \\}$. Rows labeld S1-5 show the reconstruction components of each slot. |


{% capture code %}{% raw %}dataloader = trained_monet.train_dataloader()
random_batch = next(iter(dataloader))[0]
fig = trained_monet.plot_reconstructions_and_decompositions(batch[0: 4], 3){% endraw %}{% endcapture %}
{% include code.html code=code lang="python" %}

![MONet Reconstructions and Decompositions after Train](/assets/img/04_MONet/MONET_rec.png "Reconstructions and Decompositions")

* **Component VAE Results**: In order to evaluate the perfomance of
  the component VAE, we are interested in the unmasked
  slot-wise reconstructions (i.e., `unmasked` refers to
  $\widetilde{\textbf{x}}_k$ for each slot $k$) and the slot-wise
  reconstructions masked by the VAE's reconstructed masks (i.e.,
  `masked` refers to $\widetilde{\textbf{m}}_k \odot
  \widetilde{\textbf{x}}_k$). Ideally, masked versions capture either
  a single object, the background or nothing at all (representing no
  object), see figure below. In addition, we are going to plot the
  ground truth masked reconstructions (i.e., `gt masked` refers to
  $\textbf{m}_k \odot \widetilde{\textbf{x}}_k$) such that the
  difference between `gt masked` and `masked` indicates the
  reconstruction error of the attention masks.

  | ![Component VAE Results](/assets/img/04_MONet/component_VAE_results.png "Component VAE Results") |
  | :--         |
  | **Figure 3 of** [Burgess et al. (2019)](https://arxiv.org/abs/1901.11390): Each example shows the image fet as input data to the model, with corresponding outputs from the model. Reconstruction mixtures show sum of components from all slots, weighted by the learned masks from the attention network. Color-coded segmentation maps summarise the attention masks $\\{\textbf{m}_k\\}$. Rows labeled S1-7 show the reconstruction components of each slot. Unmasked version are shown side-by-side with corresponding versions that are masked with the VAE's reconstructed masks $\widetilde{\textbf{m}}_k$. |

{% capture code %}{% raw %}dataloader = trained_monet.train_dataloader()
random_batch = next(iter(dataloader))[0]
fig = trained_monet.plot_ComponentVAE_results(batch[0: 4], 3){% endraw %}{% endcapture %}
{% include code.html code=code lang="python" %}

![MONet Component VAE](/assets/img/04_MONet/MONET_CompVAE.png "MONet Component VAE")


## Drawbacks of Paper

* deterministic attention mechanism implying that objective function is not a
  valid lower bound on the marginal likelihood (as mentioned by [Engelcke et al. (2020)](https://arxiv.org/abs/1907.13052))
* image generation suffers from discrepancy between inferred and reconstructed masks
<!-- * only works on simple images in which multiple objects of the same class occur -->
<!-- * even simple images require high training times -->
* lots of hyperparameters (network architectures, $\beta$, $\gamma$, optimization)

## Acknowledgment

There are a lot of implementations out there that helped me very much in
understanding the paper:

* [Darwin Bautista's implementation](https://github.com/baudm/MONet-pytorch)
  includes derivation of the NLL (which in the end, I did not use for simplicity).
* [Karl Stelzner's implementation](https://github.com/stelzner/monet/) is kept
  more simplisitic and is therefore easier to understand.
* [Martin Engelcke, Claas Voelcker and Max
  Morrison](https://github.com/applied-ai-lab/genesis) included an
  implementation of MONet in the Genesis repository.



[^6]: In practice, the reconstruction error for a fixed component
    (fixed $k$) becomes incalculable for binary attention masks, since $\log 0$ is
    undefined. However, the reconstruction error is effectively
    unconstrained outside of masked regions, since for $\lim \log
    m\_{k,i} \rightarrow -\infty$ the reconstruction error for the
    corresponding pixel and slot approaches 0.


--------------------------------------------------------------------------------------------
