---
title: "Spatial Broadcast Decoder: A Simple Architecture for Learning Disentangled Representations in VAEs"
permalink: "/paper_summaries/spatial_broadcast_decoder"
author: "Markus Borea"
tags: [variational autoencoder, disentanglement, generalization]
toc: true
toc_sticky: true
toc_label: "Table of Contents"
published: true 
nextjournal_link: "https://nextjournal.com/borea17/spatial-broadcast-decoder/"
github_link: "https://github.com/borea17/Notebooks/blob/master/02_Spatial_Broadcast_Decoder.ipynb"
type: "paper summary"
---

[Watters et al. (2019)](https://arxiv.org/abs/1901.07017) introduce
the *Spatial Broadcast Decoder (SBD)* as an architecture for the
decoder in Variational Auto-Encoders
[(VAEs)](https://borea17.github.io/blog/auto-encoding_variational_bayes) 
to improve 
disentanglement in the latent
space[^1], reconstruction accuracy and
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

[^1]: As outlined by [Watters et al. (2019)](https://arxiv.org/abs/1901.07017), there is "yet no consensus on the definition of a disentangled representation". However, in their paper they focus on *feature compositionality* (i.e., composing a scene in terms of independent features such as color and object) and refer to it as *disentangled representation*.  

## Model Description

As stated in the title, the model architecture of the Spatial Broadcast decoder
is very simple: Take a standard VAE decoder and replace all upsampling
deconvolutional layers by tiling the latent code $\textbf{z}$ across the original
image space, appending fixed coordinate channels and applying an convolutional
network with $1 \times 1$ stride, see the figure below. 

| ![Schematic of the Spatial Broadcast VAE](/assets/img/03_SBD/sbd.png "Schematic of the Spatial Broadcast VAE") |
| :--         |
| Schematic of the Spatial Broadcast VAE. In the decoder, the latent code $\textbf{z}\in\mathbb{R}^{k}$ is broadcasted (*tiled*) to the image width $w$ and height $h$. Additionally, two "coordinate" channels are appended. The result is fed to an unstrided convolutional decoder. (right) Pseudo-code of the spatial operation. Taken from [Watters et al. (2019)](https://arxiv.org/abs/1901.07017).|

<!-- | (left) Schematic of the Spatial Broadcast VAE. In the decoder, we broadcast (tile) the latent code $\textbf{z}$ of size $k$ to the image width $w$ and height $h$, and concatenate two "coordinate" channels. This is then fed to an unstrided convolutional decoder. (right) Pseudo-code of the spatial operation. Taken from [Watters et al. (2019)](https://arxiv.org/abs/1901.07017).| -->

**Motivation**: The presented architecture is mainly motivated by two reasons:
* **Deconvolution layers cause optimization difficulties**: [Watters
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
  
* **Appended coordinate channels improve positional generalization and
  optimization**: Previous work by [Liu et al.
  (2018)](https://arxiv.org/abs/1807.03247) showed that standard
  convolution/deconvolution networks (CNNs) perform badly when trying to learn trivial
  coordinate transformations (e.g., learning a mapping from Cartesian space
  into one-hot pixel space or vice versa). This behavior may seem
  counterintuitive (easy task, small dataset), however the feature of translational
  equivariance (i.e., shifting an object in the input equally shifts its
  representation in the output) in CNNs[^2]
  hinders learning this task: The filters have by design no 
  information about their position. Thus, coordinate transformations
  result in complicated functions which makes optimization difficult.
  E.g., changing the input coordinate slighlty might push the
  resulting function in a completelty different direction. 

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
  Spatial Broadcast decoder architecture puts a prior on dissociating 
  positional from non-positional features in the latent distribution.  
  Datasets without positional variation in turn seem unlikely to benefit from this
  architecture. However, [Watters et al.
  (2019)](https://arxiv.org/abs/1901.07017) showed that the Spatial 
  Broadcast decoder could still help in these datasets and attribute this to the
  replacement of deconvolutional layers. 
  
[^2]: In typical image classification problems, translational equivariance is highly valued since it ensures that if a filter detects an object (e.g., edges), it will detect it irrespective of its position.  

<!-- ## Learning the Model -->

<!-- Basically, the Spatial Broadcast decoder is a function approximator for -->
<!-- probabilistic decoder in a VAE. Thus, learning the model works exactly -->
<!-- as in VAEs (see my -->
<!-- [post](https://borea17.github.io/blog/auto-encoding_variational_bayes)): -->
<!-- The optimal parameters are learned jointly  -->
<!-- by training the VAE using the AEVB algorithm ([Kingma and Welling, -->
<!-- 2013](https://arxiv.org/abs/1312.6114)). The remaining  -->
<!-- part of this post aims to reproduce some of results -->
<!-- by [Watters et al. (2019)](https://arxiv.org/abs/1901.07017), i.e., -->
<!-- comparing the Spatial Broadcast decoder with a standard -->
<!-- deconvolutional decoder.  -->

## Implementation

[Watters et al. (2019)](https://arxiv.org/abs/1901.07017) conducted
experiments with several datasets and could show that
incorporating the Spatial Broadcast decoder into state-of-the-art VAE
architectures consistently increased their perfomance. While this is
impressive, it is always frustrating to not being able to reproduce
results due to missing implementation details, less computing
resources or simply not having enough time to work on a
reimplementation. 

The following reimplementation intends to eliminate that frustration
by reproducing some of their experiments on much smaller datasets with
similar characteristics such that training will take less time
(less than 30 minutes with a NVIDIA Tesla K80 GPU). 

### Data Generation

A dataset that is similar in spirit to the *colored sprites dataset*
will be generated, i.e., procedurally generated objects from known
factors of variation. [Watters et al.
(2019)](https://arxiv.org/abs/1901.07017) use a binary [dsprites
dataset](https://github.com/deepmind/dsprites-dataset) consisting of
737,280 images and transform these during training into colored images
by uniformly sampling from a predefined HSV space (see Appendix A.3).
As a result, the dataset has 8 factors of variation ($x$-position,
$y$-position, size, shape, angle, 3D-color) with infinite samples (due
to sampling of color). They used $1.5 \cdot 10^6$ training steps.

To reduce training time, we are going to generate a much simpler
dataset consisting of $3675$ images with a circle
(fixed size) inside generated from a predefined set of possible colors
and positions such that there are only 3
factors of variation ($x$-position, $y$-position, discretized color).
In this case $3.4 \cdot 10^2$ training steps suffice for approximate convergence.


| ![Examples of Dataset](/assets/img/03_SBD/dataset.png "Examples of Dataset") |
| :--         |
| Visualization of self-written dataset. |

The code below creates the dataset. Note that it is kept more generic
than necessary to allow the creation of several variations of this
dataset, i.e., more dedicated experiments can be conducted. 

<!-- Two datasets will be generated that are similar in spirit to  -->
<!-- * the *colored sprites dataset*, i.e., procedurally generated objects from -->
<!--   known factors of variation. -->
<!-- * a *dataset with small objects*. Note that [Watters et al. -->
<!--   (2019)](https://arxiv.org/abs/1901.07017) stated that in this case -->
<!--   the use of the Spatial Broadcast decoder `provides a particularly -->
<!--   dramatic benefit`. -->

```python
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import numpy as np
import torch
from torch.utils.data import TensorDataset


def generate_img(x_position, y_position, shape, color, img_size, size=20):
    """Generate an RGB image from the provided latent factors
    
    Args:
        x_position (float): normalized x position
        y_position (float): normalized y position
        shape (string): can only be 'circle' or 'square'
        color (string): color name or rgb string
        img_size (int): describing the image size (img_size, img_size)
        size (int): size of shape
        
    Returns:
        torch tensor [3, img_size, img_size] (dtype=torch.float32)
    """
    # creation of image
    img = Image.new('RGB', (img_size, img_size), color='black')
    # map (x, y) position to pixel coordinates
    x_position = (img_size - 2 - size) * x_position 
    y_position = (img_size - 2 - size) * y_position
    # define coordinates
    x_0, y_0 = x_position, y_position
    x_1, y_1 = x_position + size, y_position + size
    # draw shapes
    img1 = ImageDraw.Draw(img)
    if shape == 'square':
        img1.rectangle([(x_0, y_0), (x_1, y_1)], fill=color)
    elif shape == 'circle':       
        img1.ellipse([(x_0, y_0), (x_1, y_1)], fill=color)
    return transforms.ToTensor()(img).type(torch.float32)


def generate_dataset(img_size, shape_sizes, num_pos, shapes, colors):
    """procedurally generated from 4 ground truth independent latent factors, 
       these factors are/can be 
           Position X: num_pos values in [0, 1]
           Poistion Y: num_pos values in [0, 1]
           Shape: square, circle
           Color: standard HTML color name or 'rgb(x, y, z)'
    
    Args:
           img_size (int): describing the image size (img_size, img_size)  
           shape_sizes (list): sizes of shapes
           num_pos (int): discretized positions
           shapes (list): shapes (can only be 'circle', 'square')
           colors (list): colors
    
    Returns:
           data: torch tensor [n_samples, 3, img_size, img_size]
           latents: each entry describes the latents of corresp. data entry
    """
    num_shapes, num_colors, sizes = len(shapes), len(colors), len(shape_sizes)
    
    n_samples = num_pos*num_pos*num_shapes*num_colors*sizes
    data = torch.empty([n_samples, 3, img_size, img_size])
    latents = np.empty([n_samples], dtype=object)
    
    index = 0
    for x_pos in np.linspace(0, 1, num_pos):
        for y_pos in np.linspace(0, 1, num_pos):
            for shape in shapes:
                for size in shape_sizes:
                    for color in colors:
                        img = generate_img(x_pos, y_pos, shape, color, 
                                           img_size, size)
                        data[index] = img
                        latents[index] = [x_pos, y_pos, shape, color]
                    
                        index += 1
    return data, latents


circles_data, latents = generate_dataset(img_size=64, shape_sizes=[16],
                                         num_pos=35,
                                         shapes=['circle'],
                                         colors=['red', 'green', 'blue'])
sprites_dataset = TensorDataset(circles_data)
```

### Model Implementation

Although in principle implementing a VAE is fairly simple (see [my
post](https://borea17.github.io/blog/auto-encoding_variational_bayes)
for details), in practice one must choose lots of
hyperparmeters. These can be divided into three broader categories:

* **Encoder/Decoder and Prior Distribution**: As suggested by [Watters et al.
    (2019)](https://arxiv.org/abs/1901.07017) in Appendix A, we use a
    Gaussian decoder distribution with fixed diagonal covariance structure
    $p_{\boldsymbol{\theta}} \left(\textbf{x}^\prime | \textbf{z}^{(i)}\right) = \mathcal{N}\left( \textbf{x}^\prime |
    \boldsymbol{\mu}_D^{(i)}, \sigma^2 \textbf{I} \right)$, hence the 
    reconstruction accuracy can be calculated as follows[^3] 

    $$
    \text{Reconstruction Acc.} = \log p_{\boldsymbol{\theta}} \left(
    \textbf{x}^{(i)} | \textbf{z}^{(i)} \right) = - \frac {1}{2 \sigma^2}
    \sum_{k=1}^{D} \left(x_k^{(i)} - \mu_{D_k}^{(i)} \right)^2 + \text{const}.
    $$
    
    For the encoder distribution a Gaussian with diagonal covariance
    $q_{\boldsymbol{\phi}} \sim
    \mathcal{N} \left( \textbf{z} | \boldsymbol{\mu}_E,
    \boldsymbol{\sigma}_D^2 \textbf{I} \right)$ and as prior a
    centered multivariate Gaussian $p\_{\boldsymbol{\theta}}
    (\textbf{z}) = \mathcal{N}\left( \textbf{z} | \textbf{0}, \textbf{I} \right)$
     are chosen (both typical choices).
    
* **Network Architecture for Encoder/Decoder**: The network
  architectures for the standard encoder and decoder consist of
  convolutional and deconvolutional layers (since these perform
  typically much better on image data). The Spatial Broadcast decoder
  defines a different kind of architecture, see [Model
  Description](https://borea17.github.io/blog/spatial_broadcast_decoder#data-generation). 
  The exact architectures are taken from Appendix A.1 of [Watters et
  al.](https://arxiv.org/abs/1901.07017), see code below[^4]:
  
  ```python
  from torch import nn


  class Encoder(nn.Module):
      """"Encoder class for use in convolutional VAE
      
      Args:
          latent_dim: dimensionality of latent distribution

      Attributes:
          encoder_conv: convolution layers of encoder
          fc_mu: fully connected layer for mean in latent space
          fc_log_var: fully connceted layers for log variance in latent space
      """

      def __init__(self, latent_dim=6):
          super().__init__()
          self.latent_dim = latent_dim

          self.encoder_conv = nn.Sequential(
              # shape: [batch_size, 3, 64, 64]
              nn.Conv2d(3,  64, kernel_size=4, stride=2, padding=1),
              nn.ReLU(),
              # shape: [batch_size, 64, 32, 32]
              nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
              nn.ReLU(),
              nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
              nn.ReLU(),
              nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
              nn.ReLU(),
              # shape: [batch_size, 64, 4, 4],
              nn.Flatten(),
              # shape: [batch_size, 1024]
              nn.Linear(1024, 256),
              nn.ReLU(),
              # shape: [batch_size, 256]
          )
          self.fc_mu = nn.Sequential(
              nn.Linear(in_features=256, out_features=self.latent_dim),
          )
          self.fc_log_var = nn.Sequential(
              nn.Linear(in_features=256, out_features=self.latent_dim),
          )
          return

      def forward(self, inp):
          out = self.encoder_conv(inp)
          mu = self.fc_mu(out)
          log_var = self.fc_log_var(out)
          return [mu, log_var]


  class Decoder(nn.Module):
      """(standard) Decoder class for use in convolutional VAE,
      a Gaussian distribution with fixed variance (identity times fixed variance
      as covariance matrix) used as the decoder distribution
      
      Args:
          latent_dim: dimensionality of latent distribution
          fixed_variance: variance of distribution

      Attributes:
          decoder_upsampling: linear upsampling layer(s)
          decoder_deconv: deconvolution layers of decoder (also upsampling)
      """

      def __init__(self, latent_dim, fixed_variance):
          super().__init__()
          self.latent_dim = latent_dim
          self.coder_type = 'Gaussian with fixed variance'
          self.fixed_variance = fixed_variance

          self.decoder_upsampling = nn.Sequential(
              nn.Linear(self.latent_dim, 256),
              nn.ReLU(),
              # reshaped into [batch_size, 64, 2, 2]
          )
          self.decoder_deconv = nn.Sequential(
              # shape: [batch_size, 64, 2, 2]
              nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
              nn.ReLU(),
              # shape: [batch_size, 64, 4, 4]
              nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
              nn.ReLU(),
              nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
              nn.ReLU(),
              nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
              nn.ReLU(),
              nn.ConvTranspose2d(64,  3, kernel_size=4, stride=2, padding=1),
              # shape: [batch_size, 3, 64, 64]
          )
          return

      def forward(self, inp):
          ups_inp = self.decoder_upsampling(inp)
          ups_inp = ups_inp.view(-1, 64, 2, 2)
          mu = self.decoder_deconv(ups_inp)
          return mu
          
          
  class SpatialBroadcastDecoder(nn.Module):
      """SBD class for use in convolutional VAE,
        a Gaussian distribution with fixed variance (identity times fixed 
        variance as covariance matrix) used as the decoder distribution

      Args:
          latent_dim: dimensionality of latent distribution
          fixed_variance: variance of distribution

      Attributes:
          img_size: image size (necessary for tiling)
          decoder_convs: convolution layers of decoder (also upsampling)
      """

      def __init__(self, latent_dim, fixed_variance):
          super().__init__()
          self.img_size = 64
          self.coder_type = 'Gaussian with fixed variance'
          self.latent_dim = latent_dim
          self.fixed_variance = fixed_variance

          x = torch.linspace(-1, 1, self.img_size)
          y = torch.linspace(-1, 1, self.img_size)
          x_grid, y_grid = torch.meshgrid(x, y)
          # reshape into [1, 1, img_size, img_size] and save in state_dict
          self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
          self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))

          self.decoder_convs = nn.Sequential(
              # shape [batch_size, latent_dim + 2, 64, 64]
              nn.Conv2d(in_channels=self.latent_dim+2, out_channels=64,
                        stride=(1, 1), kernel_size=(3,3), padding=1),           
              nn.ReLU(),
              # shape [batch_size, 64, 64, 64]
              nn.Conv2d(in_channels=64, out_channels=64, stride=(1,1), 
                        kernel_size=(3, 3), padding=1),
              nn.ReLU(),
              # shape [batch_size, 64, 64, 64]
              nn.Conv2d(in_channels=64, out_channels=3, stride=(1,1), 
                        kernel_size=(3, 3), padding=1),
              # shape [batch_size, 3, 64, 64]         
          )
          return

      def forward(self, z):
          batch_size = z.shape[0]
          # reshape z into [batch_size, latent_dim, 1, 1]
          z = z.view(z.shape + (1, 1))
          # tile across image [batch_size, latent_im, img_size, img_size]
          z_b = z.expand(-1, -1, self.img_size, self.img_size)
          # upsample x_grid and y_grid to [batch_size, 1, img_size, img_size]
          x_b = self.x_grid.expand(batch_size, -1, -1, -1)
          y_b = self.y_grid.expand(batch_size, -1, -1, -1)
          # concatenate vectors [batch_size, latent_dim+2, img_size, img_size]
          z_sb = torch.cat((z_b, x_b, y_b), dim=1)
          # apply convolutional layers
          mu_D = self.decoder_convs(z_sb)
          return mu_D
  ```
  
  The VAE implementation below combines the encoder and decoder
  architectures (slightly modified version of my last [VAE
  implementation](https://borea17.github.io/blog/auto-encoding_variational_bayes#vae-implementation)).
  
  ```python
  from torch.distributions.multivariate_normal import MultivariateNormal


  class VAE(nn.Module):
      """A simple VAE class

      Args:
          vae_tpe: type of VAE either 'Standard' or 'SBD'
          latent_dim: dimensionality of latent distribution
          fixed_var: fixed variance of decoder distribution
      """

      def __init__(self, vae_type, latent_dim, fixed_var):
          super().__init__()
          self.vae_type = vae_type

          if self.vae_type == 'Standard':
              self.decoder = Decoder(latent_dim=latent_dim, 
                                    fixed_variance=fixed_var)
          else:
              self.decoder = SpatialBroadcastDecoder(latent_dim=latent_dim,
                                                     fixed_variance=fixed_var)

          self.encoder = Encoder(latent_dim=latent_dim)
          self.normal_dist = MultivariateNormal(torch.zeros(latent_dim), 
                                                torch.eye(latent_dim))
          return

      def forward(self, x):      
          z, mu_E, log_var_E = self.encode(x)
          # regularization term per batch, i.e., size: (batch_size)
          regularization_term = 0.5 * (1 + log_var_E - mu_E**2
                                        - torch.exp(log_var_E)).sum(axis=1)

          batch_size = x.shape[0]
          if self.decoder.coder_type == 'Gaussian with fixed variance':
              # x_rec has shape (batch_size, 3, 64, 64)
              x_rec = self.decode(z)
              # reconstruction accuracy per batch, i.e., size: (batch_size)
              factor = 0.5 * (1/self.decoder.fixed_variance)
              recons_acc = - factor * ((x.view(batch_size, -1) - 
                                      x_rec.view(batch_size, -1))**2
                                    ).sum(axis=1)
          return -regularization_term.mean(), -recons_acc.mean()

      def reconstruct(self, x):
          mu_E, log_var_E = self.encoder(x)
          x_rec = self.decoder(mu_E)
          return x_rec

      def encode(self, x):
          # get encoder distribution parameters
          mu_E, log_var_E = self.encoder(x)
          # sample noise variable for each batch
          batch_size = x.shape[0]
          epsilon = self.normal_dist.sample(sample_shape=(batch_size, )
                                            ).to(x.device)
          # get latent variable by reparametrization trick
          z = mu_E + torch.exp(0.5*log_var_E) * epsilon
          return z, mu_E, log_var_E

      def decode(self, z):
          # get decoder distribution parameters
          mu_D = self.decoder(z)
          return mu_D
  ```

* **Training Parameters**: Lastly, training neural networks itself
  consists of several hyperparmeters. Again, we are using the same
  setup as defined in Appendix A.1 of [Watters et
  al. (2019)](https://arxiv.org/abs/1901.07017), see code below.

  ```python
  from livelossplot import PlotLosses
  from torch.utils.data import DataLoader

  
  def train(dataset, epochs, VAE):
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
      print('Device: {}'.format(device))

      data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

      VAE.to(device)
      optimizer = torch.optim.Adam(VAE.parameters(), lr=3e-4)

      losses_plot = PlotLosses(groups={'avg log loss': 
                                      ['kl loss', 'reconstruction loss']})
      print('Start training with {} decoder\n'.format(VAE.vae_type))
      for epoch in range(1, epochs +1):
          avg_kl = 0 
          avg_recons_err = 0
          for counter, mini_batch_data in enumerate(data_loader):
              VAE.zero_grad()

              kl_div, recons_err = VAE(mini_batch_data[0].to(device))
              loss = kl_div + recons_err
              loss.backward()
              optimizer.step()

              avg_kl += kl_div.item() / len(dataset)
              avg_recons_err += recons_err.item() / len(dataset)

          losses_plot.update({'kl loss': np.log(avg_kl), 
                              'reconstruction loss': np.log(avg_recons_err)})
          losses_plot.send()
      trained_VAE = VAE
      return trained_VAE
  ```

### Visualization Functions

Evaluating the representation quality of trained models is a difficult
task, since we are not only interested in the reconstruction accuracy
but also in the latent space and its properties. Ideally the latent
space offers a disentangled representation such that each latent
variable represents a factor of variation with perfect reconstruction
accuracy (i.e., for evaluation it is very helpful to know in advance
how many and what factors of variation exist). Although there are some
metrics to quantify disentanglement, `many of them have serious
shortcomings and there is yet no consensus in the literature which to
use` ([Watters et al., 2019](https://arxiv.org/abs/1901.07017)). 
Instead of focusing on some metric, we are going to visualize the
results by using two approaches:

* **Reconstructions and Latent Traversals**: A very popular and
  helpful plot is to show some (arbitrarly chosen) reconstructions
  compared to the original input together with a series of latent
  space traversals. I.e., taking some encoded input and looking at the
  reconstructions when sweeping each coordinate in the latent space in
  a predefined interval (here from -2 to +2) while keeping all other
  coordinates constant. Ideally, each sweep can be associated with
  a factor of variation. The code below will be used to generate these
  plots. Note that the reconstructions are clamped into $[0, 1]$ as
  this is the allowed image range.
  
  ```python
  import matplotlib.pyplot as plt
  %matplotlib inline
  

  def reconstructions_and_latent_traversals(STD_VAE, SBD_VAE, dataset, SEED=1):
      np.random.seed(SEED)
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      latent_dims = STD_VAE.encoder.latent_dim

      n_samples = 7
      i_samples = np.random.choice(range(len(dataset)), n_samples, replace=False)

      # preperation for latent traversal
      i_latent = i_samples[n_samples//2]
      lat_image = dataset[i_latent][0]
      sweep = np.linspace(-2, 2, n_samples)

      fig = plt.figure(constrained_layout=False, figsize=(2*n_samples, 2+latent_dims))
      grid = plt.GridSpec(latent_dims + 5, n_samples*2 + 3, 
                          hspace=0.2, wspace=0.02, figure=fig)
      # standard VAE
      for counter, i_sample in enumerate(i_samples):
          orig_image = dataset[i_sample][0]
          # original
          main_ax = fig.add_subplot(grid[1, counter + 1])
          main_ax.imshow(transforms.ToPILImage()(orig_image))
          main_ax.axis('off')
          main_ax.set_aspect('equal')

          # reconstruction
          x_rec = STD_VAE.reconstruct(orig_image.unsqueeze(0).to(device))
          # clamp output into [0, 1] and prepare for plotting
          recons_image =  torch.clamp(x_rec, 0, 1).squeeze(0).cpu()

          main_ax = fig.add_subplot(grid[2, counter + 1])
          main_ax.imshow(transforms.ToPILImage()(recons_image))
          main_ax.axis('off')
          main_ax.set_aspect('equal')
      # latent dimension traversal
      z, mu_E, log_var_E = STD_VAE.encode(lat_image.unsqueeze(0).to(device))
      for latent_dim in range(latent_dims):
          for counter, z_replaced in enumerate(sweep):
              z_new = z.detach().clone()
              z_new[0][latent_dim] = z_replaced

              # clamp output into [0, 1] and prepare for plotting
              img_rec = torch.clamp(STD_VAE.decode(z_new), 0, 1).squeeze(0).cpu()

              main_ax = fig.add_subplot(grid[4 + latent_dim, counter + 1])
              main_ax.imshow(transforms.ToPILImage()(img_rec))
              main_ax.axis('off')
      # SBD VAE
      for counter, i_sample in enumerate(i_samples):
          orig_image = dataset[i_sample][0]
          # original
          main_ax = fig.add_subplot(grid[1, counter + n_samples + 2])
          main_ax.imshow(transforms.ToPILImage()(orig_image))
          main_ax.axis('off')
          main_ax.set_aspect('equal')
          # reconstruction
          x_rec = SBD_VAE.reconstruct(orig_image.unsqueeze(0).to(device))
          # clamp output into [0, 1] and prepare for plotting
          recons_image = torch.clamp(x_rec, 0, 1).squeeze(0).cpu()

          main_ax = fig.add_subplot(grid[2, counter + n_samples + 2])
          main_ax.imshow(transforms.ToPILImage()(recons_image))
          main_ax.axis('off')
          main_ax.set_aspect('equal')
      # latent dimension traversal
      z, mu_E, log_var_E = SBD_VAE.encode(lat_image.unsqueeze(0).to(device))
      for latent_dim in range(latent_dims):
          for counter, z_replaced in enumerate(sweep):
              z_new = z.detach().clone()
              z_new[0][latent_dim] = z_replaced
              # clamp output into [0, 1] and prepare for plotting
              img_rec = torch.clamp(SBD_VAE.decode(z_new), 0, 1).squeeze(0).cpu()

              main_ax = fig.add_subplot(grid[4+latent_dim, counter+n_samples+2])
              main_ax.imshow(transforms.ToPILImage()(img_rec))
              main_ax.axis('off')
      # prettify by adding annotation texts
      fig = prettify_with_annotation_texts(fig, grid, n_samples, latent_dims)
      return fig

  def prettify_with_annotation_texts(fig, grid, n_samples, latent_dims):
      # figure titles
      titles = ['Deconv Reconstructions', 'Spatial Broadcast Reconstructions',
                'Deconv Traversals', 'Spatial Broadcast Traversals']
      idx_title_pos = [[0, 1, n_samples+1], [0, n_samples+2, n_samples*2+2],
                      [3, 1, n_samples+1], [3, n_samples+2, n_samples*2+2]]
      for title, idx_pos in zip(titles, idx_title_pos):
          fig_ax = fig.add_subplot(grid[idx_pos[0], idx_pos[1]:idx_pos[2]])
          fig_ax.annotate(title, xy=(0.5, 0), xycoords='axes fraction', 
                          fontsize=14, va='bottom', ha='center')
          fig_ax.axis('off')
      # left annotations
      fig_ax = fig.add_subplot(grid[1, 0])
      fig_ax.annotate('input', xy=(1, 0.5), xycoords='axes fraction', 
                      fontsize=12,  va='center', ha='right')
      fig_ax.axis('off')
      fig_ax = fig.add_subplot(grid[2, 0])
      fig_ax.annotate('recons', xy=(1, 0.5), xycoords='axes fraction', 
                      fontsize=12, va='center', ha='right')
      fig_ax.axis('off')
      fig_ax = fig.add_subplot(grid[4:latent_dims + 4, 0])
      fig_ax.annotate('latent coordinate traversed', xy=(0.9, 0.5), 
                      xycoords='axes fraction', fontsize=12,
                      va='center', ha='center', rotation=90)
      fig_ax.axis('off')
      # pertubation magnitude
      for i_y_grid in [[1, n_samples+1], [n_samples+2, n_samples*2+2]]:
          fig_ax = fig.add_subplot(grid[latent_dims + 4, i_y_grid[0]:i_y_grid[1]])
          fig_ax.annotate('pertubation magnitude', xy=(0.5, 0), 
                          xycoords='axes fraction', fontsize=12,
                          va='bottom', ha='center')
          fig_ax.set_frame_on(False)
          fig_ax.axes.set_xlim([-2.5, 2.5])
          fig_ax.xaxis.set_ticks([-2, 0, 2])
          fig_ax.xaxis.set_ticks_position('top')
          fig_ax.xaxis.set_tick_params(direction='inout', pad=-16)
          fig_ax.get_yaxis().set_ticks([])
      # latent dim
      for latent_dim in range(latent_dims):
          fig_ax = fig.add_subplot(grid[4 + latent_dim, n_samples*2 + 2])
          fig_ax.annotate('lat dim ' + str(latent_dim + 1), xy=(0, 0.5), 
                          xycoords='axes fraction', 
                          fontsize=12, va='center', ha='left')
          fig_ax.axis('off')
      return 
  ```
  
* **Latent Space Geometry**: While latent traversals may be helpful, [Watters et al.
(2019)](https://arxiv.org/abs/1901.07017) note that this techniques
suffers from two shortcommings:
   1. Latent space entanglement might be difficult to perceive by eye.
   2. Traversals are only taken at some point in space. It could be
      that traversals at some points are more disentangled than at
      other positions. Thus, judging disentanglement by the
      aforementioned method might be ultimately dependent to randomness.
   
   To overcome these limitations, they propose a new method which they
   term *latent space geometry*. The main idea is to visualize a 
   transformation from a 2-dimensional generative factor space
   (subspace of all generative factors) into the 2-dimensional latent
   subspace (choosing the two latent components that correspond to the
   factors of variation). Latent space geometry that preserves the
   chosen geometry of the generative factor space (while scaling and
   rotation might be allowed depending on the chosen generative factor
   space) indicates disentanglement. 
   
   To put this into practice, the code below creates circle images 
   by varying $x$ and $y$ positions uniformly and keeping the other
   generative factors (*here* only color) constant. Accordingly, the
   geometry of the generative factor space is a uniform grid (which
   will be plotted). These images will be encoded into mean and
   variance of the latent distribution. In order to find the latent components that
   correspond to the $x$ and $y$ position, we choose the components
   with smallest mean variance across all reconstructions, i.e., the
   most informative components[^5]. Then, we can plot the latent space
   geometry by using the latent components of the mean (encoder
   distribution), see code below.
   
   ```python
def latent_space_geometry(STD_VAE, SBD_VAE):
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      plt.figure(figsize=(18, 6))
      # x,y position grid in [0.2, 0.8] (generative factors)
      equi = np.linspace(0.2, 0.8, 31)
      equi_without_vert = np.setdiff1d(equi, np.linspace(0.2, 0.8, 6))

      x_pos = np.append(np.repeat(np.linspace(0.2, 0.8, 6), len(equi)),
                        np.tile(equi_without_vert, 6))
      y_pos = np.append(np.tile(equi, 6),
                        np.repeat(np.linspace(0.8, 0.2, 6), len(equi_without_vert)))
      labels = np.append(np.repeat(np.arange(6), 31),
                        np.repeat(np.arange(6)+10, 25))
      # plot generative factor geometry
      plt.subplot(1, 3, 1)
      plt.scatter(x_pos, y_pos, c=labels, cmap=plt.cm.get_cmap('rainbow', 10))
      plt.gca().set_title('Ground Truth Factors', fontsize=16)
      plt.xlabel('X-Position')
      plt.ylabel('Y-Position')

      # generate images
      img_size = 64
      shape_size = 16
      images = torch.empty([len(x_pos), 3, img_size, img_size]).to(device)
      for counter, (x, y) in enumerate(zip(x_pos, y_pos)):
          images[counter] = generate_img(x, y, 'circle', 'red', 
                                        img_size, shape_size)

      # STD VAE
      [all_mu, all_log_var] = STD_VAE.encoder(images)
      # most informative latent variable
      lat_1, lat_2 = all_log_var.mean(axis=0).sort()[1][:2]
      # latent coordinates
      x_lat = all_mu[:, lat_1].detach().cpu().numpy()
      y_lat = all_mu[:, lat_2].detach().cpu().numpy()
      # plot latent space geometry
      plt.subplot(1, 3, 2)
      plt.scatter(x_lat, y_lat, c=labels, cmap=plt.cm.get_cmap('rainbow', 10))
      plt.gca().set_title('DeConv', fontsize=16)
      plt.xlabel('latent 1 value')
      plt.ylabel('latent 2 value')

      # SBD VAE
      [all_mu, all_log_var] = SBD_VAE.encoder(images)
      # most informative latent variable
      lat_1, lat_2 = all_log_var.mean(axis=0).sort()[1][:2]
      # latent coordinates
      x_lat = all_mu[:, lat_1].detach().cpu().numpy()
      y_lat = all_mu[:, lat_2].detach().cpu().numpy()
      # plot latent space geometry
      plt.subplot(1, 3, 3)
      plt.scatter(x_lat, y_lat, c=labels, cmap=plt.cm.get_cmap('rainbow', 10))
      plt.gca().set_title('Spatial Broadcast', fontsize=16)
      plt.xlabel('latent 1 value')
      plt.ylabel('latent 2 value')
      return   
   ```
   
 [^5]: An intuitve way to understand why latent compontents with smaller variance within the encoder distribution are more informative than others is to think about the sampled noise and the loss function: If the variance is high, the latent code $\textbf{z}$ will vary a lot which in turn makes the task for the decoder more difficult. However, the regularization term (KL-divergence) pushes the variances towards 1. Thus, the network will only reduce the variance of its components if it helps to increase the reconstruction accuracy.

### Results

Lastly, let's train our models and look at the results:

```python
epochs = 150
latent_dims = 5 # x position, y position, color, extra slots
fixed_variance = 0.3

standard_VAE = VAE(vae_type='Standard', latent_dim=latent_dims, 
                   fixed_var=fixed_variance)
SBD_VAE = VAE(vae_type='SBD', latent_dim=latent_dims, 
              fixed_var=fixed_variance)
```


```python
trained_standard_VAE  = train(sprites_dataset, epochs, standard_VAE)
```

![training_STD_VAE_results](/assets/img/03_SBD/log_loss_STD.png "training STD results")


```python
trained_SBD_VAE = train(sprites_dataset, epochs, SBD_VAE)
```

![training_SBD_VAE_results](/assets/img/03_SBD/log_loss_SBD.png "training SBD results")

At the log-losses plots, we can already see that using the Spatial
Broadcast  decoder results in an improved reconstruction accuracy and
regularization term. Now let's compare both models visually by their

* **Reconstructions and Latent Traversals**:

  ```python 
  reconstructions_and_latent_traversals(trained_standard_VAE, 
                                        trained_SBD_VAE, sprites_dataset)
  ```

  ![reconstruction_and_latent_traversal](/assets/img/03_SBD/latent_traversal.png "Reconstruction and Latent Traversal")
  
  While the reconstructions within both models look pretty good, the
  latent space traversal shows an entangled representation in the
  standard (DeConv) VAE whereas the Spatial Broadcast model seems quite
  disentangled.  
  
* **Latent Space Geometry**:

  ```python
  latent_space_geometry(trained_standard_VAE, trained_SBD_VAE)
  ```
  
  ![latent_space_geometry](/assets/img/03_SBD/latent_space_geometry.png "Latent Space Geometry")
  
  The latent space geometry verifies our previous findings: The
  DeConv decoder has an entangled latent space (transformation
  is highly non linear) whereas in the Spatial Broadcast decoder the
  latent space geometry highly resembles the generating factors
  geometry (affine transformation). The transformation of the Spatial
  Broadcast decoder indicates very similar behavior in the $X-Y$
  position subspace (of generative factors) as in the corresponding
  latent subspace. 

[^4]: The Spatial Broadcast decoder architecture is slightly modified: Kernel size of 3 instead of 4 to get the desired output shapes.

<!-- [^4]: Note that in the Spatial Broadcast decoder the height and width -->
<!--     of the CNN input needs to be both 6 larger than the target -->
<!--     output (image) size to accommodate for the lack of padding. This -->
<!--     is not stated in the paper, however described in the appendix B.1 -->
<!--     of the follow up paper by [Burgess et al. (2019)](https://arxiv.org/abs/1901.11390). -->
    
## Drawbacks of Paper


* although there are fewer parameters in the Spatial Broadcast
  decoder, it does require more memory (in the implementation about
  50% more)
* longer training times compared to standard DeConv VAE
* appended coordinate channels do not help when there is no positional
variation
<!-- * mostly applicable in the context of static images with positional -->
<!--   variation  -->
<!--   => temporal correlations -->


## Acknowledgement

[Daniel Daza's](https://dfdazac.github.io/) blog was really helpful
and the presented code is highly inspired by his [VAE-SBD implementation](https://github.com/dfdazac/vaesbd).

-----------------------------------------------------------

[^3]: For simplicity, we are setting the number of (noise variable)
    samples $L$ per datapoint to 1 (see equation for
    $\displaystyle \widetilde{\mathcal{L}}$ in [*Reparametrization
    Trick*](https://borea17.github.io/paper_summaries/auto-encoding_variational_bayes#model-description)
    paragraph). 
    Note that [Kingma and Welling
    (2013)](https://arxiv.org/abs/1312.6114) stated that in their
    experiments setting $L=1$ sufficed as long as the minibatch size
    was large enough. 

