---
title: "Spatial Transformer Networks"
permalink: "/paper_summaries/spatial_transformer"
author: "Markus Borea"
tags: ["generalization"]
published: true
toc: true
toc_sticky: true
toc_label: "Table of Contents"
type: "paper summary"
nextjournal_link: "https://nextjournal.com/borea17/spatial_transformer_networks/"
github_link: "https://github.com/borea17/Notebooks/blob/master/04_Spatial_Transformer_Networks.ipynb"
---

[Jaderberg et al. (2015)](https://arxiv.org/abs/1506.02025) introduced
the learnable **Spatial Transformer (ST)** module that can be used to
empower standard neural networks to actively spatially
transform feature maps or input data. In essence, the ST can
be understood as a black box that applies some spatial transformation
(e.g., crop, scale, rotate) to a given input (or part of it)
conditioned on the particular input during a single forward path. In
general, STs can also be seen as a learnable attention mechanism
(including spatial transformation on the region of interest). Notably,
STs can be easily integrated in existing neural network architectures
without any supervision or modification to the optimization, i.e., STs
are differentiable plug-in modules. The authors could show that STs
help the models to learn invariances to translation, scale, rotation
and more generic warping which resulted in state-of-the-art
performance on several benchmarks, see image below. 

| ![Spatial Transformer in Practice](/assets/img/09_spatial_transformer/ST_inpractice.gif "Spatial Transformer in Practice") |
| :--  |
|  **ST Example**: Results (after training) of using a ST as the first layer of a fully-connected network (`ST-FCN Affine`, left) or a convolutional neural network (`ST-CNN Affine`, right) trained for cluttered MNIST digit recognition are shown. Clearly, the output of the ST exhibits much less translation variance and attends to the digit. Taken from [Jaderberg et al. (2015)](https://arxiv.org/abs/1506.02025) linked [video](https://goo.gl/qdEhUu).|

## Model Description 

The aim of STs is to provide neural networks with spatial
transformation and attention capabilities in a reasonable and
efficient way. Note that standard neural network architectures (e.g.,
CNNs) are limited in this regard[^1]. Therefore, the ST constitutes
parametrized transformations $\mathcal{T}_{\boldsymbol{\theta}}$ that
transform the regular input grid to a new sampling grid, see image
below. Then, some form of interpolation is used to compute the pixel
values in the new sampling grid (i.e., interpolation between values of
the old grid). 

| ![Parametrized Sampling Grids](/assets/img/09_spatial_transformer/parametrised_sampling_grid.png "Parametrized Sampling Grids") |
| :--  |
| Two examples of applying the parametrised sampling grid to an image $\textbf{U}$ producing the output $\textbf{V}$. The green dots represent the new sampling grid which is obtained by transforming the regular grid $\textbf{G}$ (defined on $\textbf{V}$) using the transformation $\mathcal{T}$. <br> (a) The sampling grid is the regular grid $\textbf{G} = \mathcal{T}\_{\textbf{I}} (\textbf{G})$, where $\textbf{I}$ is the identity transformation matrix. <br> (b) The sampling grid is the result of warping the regular grid with an affine transformation $\mathcal{T}\_{\boldsymbol{\theta}} (\textbf{G})$. <br> Taken from [Jaderberg et al. (2015)](https://arxiv.org/abs/1506.02025). |

To this end, the ST is divided into three consecutive parts:

* **Localisation Network**: Its purpose is to retrieve the parameters
  $\boldsymbol{\theta}$ of the spatial transformation
  $\mathcal{T}\_{\boldsymbol{\theta}}$ taking the current feature map
  $\textbf{U}$ as input, i.e., $\boldsymbol{\theta} = f\_{\text{loc}}
  \left(\textbf{U} \right)$. Thereby, the spatial transformation is
  conditioned on the input. Note that dimensionality of
  $\boldsymbol{\theta}$ depends on the transformation type which needs
  to be defined beforehand, see some examples below. Furthermore, the
  localisation network can take any differentiable form, e.g., a CNN
  or FCN.
  
  <ins>*Examples of Spatial Transformations*</ins>  
  The following examples highlight how a regular grid 
  
  $$
  \textbf{G} = \left\{ \begin{bmatrix} x_i^t \\ y_i^t \end{bmatrix}
  \right\}_{i=1}^{H^t \cdot W^t}
  $$

  defined on the output/target map $\textbf{V}$ (i.e., $H^t$ and $W^t$ denote
  height and width of $\textbf{V}$) can be transformed into a new sampling grid 
  
  $$
  \widetilde{\textbf{G}} = \left\{ \begin{bmatrix} x_i^s \\ y_i^s \end{bmatrix}
  \right\}_{i=1}^{H^s \cdot W^s}
  $$
  
  defined on the input/source feature map $\textbf{U}$ using a parametrized
  transformation $\mathcal{T}\_{\boldsymbol{\theta}}$, i.e., $\widetilde{G} =
  T\_{\boldsymbol{\theta}} (G)$. Visualizations have bee created by
  me, interactive versions can be found [here](https://github.com/borea17/InteractiveTransformations). 
  
  <!-- * <ins>Affine Transformations</ins> -->
  
    | <img width="800" height="512" src='/assets/img/09_spatial_transformer/affine_transform.gif'> |
    | :--  |
    |  This transformation allows cropping, translation, rotation, scale and skew to be applied to the input feature map. It has 6 degrees of freedom (DoF). |

  <!-- * <ins>(Standard) Attention</ins> -->

    | <img width="800" height="512" src='/assets/img/09_spatial_transformer/attention_transform.gif'> |
    | :--  |
    |  This transformation is more constrained with only 3-DoF. Therefore it only allows cropping, translation and isotropic scaling to be applied to the input feature map.|
  

    | <img width="800" height="512" src='/assets/img/09_spatial_transformer/projective_transform.gif'> |
    | :--  |
    |  This transformation has 8-DoF and can be seen as an extension to the affine transformation. The main difference is that affine transformations are constrained to preserve parallelism.  |

  <!-- * <ins>Thin Plate Spline (TPS) Transformations</ins> -->

* **Grid Generator**: Its purpose to create the new sampling grid
  $\widetilde{\textbf{G}}$ on the input feature map $\textbf{U}$ by applying
  the predefined parametrized transformation using the parameters
  $\boldsymbol{\theta}$ obtained from the localisation network, see
  examples above. 
  <!-- Note that [Jaderberg et al. -->
  <!-- (2015)](https://arxiv.org/abs/1506.02025) define normalized -->
  <!-- coordinates for the target feature map, i.e., $-1 \le x_i^t, y_i^t \le 1$.  -->

* **Sampler**: Its purpose is to compute the warped version of the
  input feature map $\textbf{U}$ by computing the pixel values in the
  new sampling grid $\widetilde{\textbf{G}}$ obtained from the grid
  generator. Note that the new sampling grid does not necessarily
  align with the input feature map grid, therefore some kind of
  interpolation is needed. [Jaderberg et al.
  (2015)](https://arxiv.org/abs/1506.02025) formulate this
  interpolation as the application of a sampling kernel centered at a
  particular location in the input feature map, i.e., 
  
  $$
    V_i^c = \sum_{n=1}^{H^s} \sum_{m=1}^{W^s} U_{n,m}^c \cdot \underbrace{k(x_i^s - x_m^t;
    \boldsymbol{\Phi}_x)}_{k_{\boldsymbol{\Phi}_x}} \cdot \underbrace{k(y_i^s - y_n^t; \boldsymbol{\Phi}_y)}_{k_{\boldsymbol{\Phi}_y}}, 
  $$
  
  where $V_i^c \in \mathbb{R}^{W^t \times H^t}$ denotes the new pixel
  value of the $c$-th channel at the $i$-th position of the new
  sampling grid coordinates[^2] $\begin{bmatrix} x_i^s &
  y_i^s\end{bmatrix}^{T}$ and $\boldsymbol{\Phi}_x,
  \boldsymbol{\Phi}_y$ are the parameters of a generic sampling kernel
  $k()$ which defines the image interpolation. As the sampling grid
  coordinates are not channel-dependent, each channel is transformed
  in the same way resulting in spatial consistency between channels.
  Note that although in theory we need to sum over all input
  locations, in practice we can ignore this sum by just looking at the
  kernel support region for each $V_i^c$ (similar to CNNs). 
  
  The sampling kernel can be chosen freely as long as (sub-)gradients
  can be defined with respect to $x_i^s$ and $y_i^s$. Some possible
  choices are shown below.
  
  $$
    \begin{array}{lcc}
    \hline
      \textbf{Interpolation Method} & k_{\boldsymbol{\Phi}_x} &
    k_{\boldsymbol{\Phi}_x} \\ \hline
      \text{Nearest Neightbor} &  \delta( \lfloor x_i^s + 0.5\rfloor -
    x_m^t) &  \delta( \lfloor y_i^s + 0.5\rfloor - y_n^t) \\ 
      \text{Bilinear} &  \max \left(0, 1 -  \mid x_i^s - x_m^t \mid
    \right) &  \max (0, 1 - \mid y_i^s - y_m^t\mid ) \\ \hline
    \end{array}
  $$

The figure below summarizes the ST architecture and shows how the
individual parts interact with each other.

| ![Architecture of Spatial Transformer](/assets/img/09_spatial_transformer/spatial_transformer.png "Architecture of Spatial Transformer") |
| :--  |
|  Taken from [Jaderberg et al. (2015)](https://arxiv.org/abs/1506.02025)  |


[^1]: Clearly, convolutional layers are not rotation or scale
    invariant. Even the translation-equivariance property does not
    necessarily make CNNs translation-invariant as typically some
    fully connected layers are added at the end. Max-pooling layers
    can introduce some translation invariance, however are limited by
    their size such that often large translation are not captured. 
    
    
[^2]: [Jaderberg et al. (2015)](https://arxiv.org/abs/1506.02025)
    define the transformation with normalized coordinates, i.e., $-1
    \le x_i^s, y_i^s \le 1$. However, in the sampling kernel equations
    it seems more likely that they assume unnormalized/absolute coordinates, e.g.,
    in equation 4 of the paper normalized coordinates would be nonsensical. 


**Motivation**: With the introduction of GPUs, convolutional layers
enabled computationally efficient training of feature detectors on
patches due to their weight sharing and local connectivity concepts.
Since then, CNNs have proven to be the most powerful framework when it
comes to computer vision tasks such as image classification or
segmentation.  

Despite their success, [Jaderberg et al.
(2015)](https://arxiv.org/abs/1506.02025) note that CNNs are still
lacking mechanisms to be spatially invariant to the input data in a
computationally and parameter efficient manner. While convolutional
layers are translation-equivariant to the input data and the use of
max-pooling layers has helped to allow the network to be somewhat
spatially invariant to the position of features, this invariance is
limited to the (typically) small spatial support of max-pooling (e.g.,
$2\times 2$). As a result, CNNs are typically not invariant to larger
transformations, thus need to learn complicated functions to
approximate these invariances. 

<!-- Data augmentation is a standard trick -->
<!-- to increase the performance of CNNs by  -->

What if we could enable the network to learn transformations of the
input data? This is the main idea of STs! Learning spatial invariances
is much easier when you have spatial transformation capabilities. The
second aim of STs is to be computationally and parameter efficient.
This is done by using structured, parameterized transformations which
can be seen as a weight sharing scheme.

## Implementation

[Jaderberg et al. (2015)](https://arxiv.org/abs/1506.02025) performed
several supervised learning tasks (distorted MNIST, Street View House
Numbers, fine-grained bird classification) to test the performance of
a standard architecture (FCN or CNN) against an architecture that
includes one or several ST modules. They could emperically validate
that including STs results in performance gains, i.e., higher
accuracies across multiple tasks.

The following reimplementation aims to reproduce a subset of the
distored MNIST experiment (RTS distorted MNIST) comparing a standard
CNN with a ST-CNN architecture. A starting point for the
implementation was [this pytorch tutorial by Ghassen
Hamrouni](https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html). 

### RTS Distorted MNIST

While [Jaderberg et al. (2015)](https://arxiv.org/abs/1506.02025)
explored multiple distortions on the MNIST handwriting dataset, this
reimplementation focuses on the rotation-translation-scale (RTS)
distorted MNIST, see image below. As described in appendix A.4 of
[Jaderberg et al. (2015)](https://arxiv.org/abs/1506.02025) this
dataset can easily be generated by augmenting the standard MNIST
dataset as follows:
* randomly rotate by sampling the angle uniformly in
  $[+45^{\circ}, 45^{\circ}]$,
* randomly scale by sampling the factor uniformly in $[0.7, 1.2]$,
* translate by picking a random location on a $42\times 42$ image
  (MNIST digits are $28 \times 28$).

| ![RTS Distorted MNIST Examples](/assets/img/09_spatial_transformer/distortedMNIST.png "RTS Distorted MNIST Examples") |
| :--  |
| **RTS Distorted MNIST Examples** |

Note that this transformation could also be used as a data
augmentation technique, as the resulting images remain (mostly) valid
digit representations (humans could still assign correct labels).

The code below can be used to create this dataset:

```python
import torch
from torchvision import datasets, transforms


def load_data():
    """loads MNIST datasets with 'RTS' (rotation, translation, scale)
    transformation

    Returns:
        train_dataset (torch dataset): training dataset
        test_dataset (torch dataset): test dataset
    """
    def place_digit_randomly(img):
        new_img = torch.zeros([42, 42])
        x_pos, y_pos = torch.randint(0, 42-28, (2,))
        new_img[y_pos:y_pos+28, x_pos:x_pos+28] = img
        return new_img
        
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=(-45, 45), 
                                scale=(0.7, 1.2)),
        transforms.ToTensor(),
        transforms.Lambda(lambda img: place_digit_randomly(img)),
        transforms.Lambda(lambda img: img.unsqueeze(0))
    ])
    train_dataset = datasets.MNIST('./data', transform=transform, 
                                   train=True, download=True)
    test_dataset = datasets.MNIST('./data', transform=transform, 
                                   train=True, download=True)
    return train_dataset, test_dataset


train_dataset, test_dataset = load_data()
```

### Model Implementation

The model implementation can be divided into three tasks:

* **Network Architectures**: The network architectures are based upon
  the description in appendix A.4 of [Jaderberg et al.
  (2015)](https://arxiv.org/abs/1506.02025). Note that there is only
  one ST at the beginning of the network such that the resulting
  transformation is only applied over one channel (input channel). For
  the sake of simplicity, we only implement an affine transformation
  matrix. Clearly, including an ST increases the networks capacity
  due to the number of added trainable parameters. To allow for a fair
  comparison, we therefore increase the capacity of the convolutional
  and linear layers in the standard CNN.
  
  The code below creates both architectures and counts their trainable
  parameters.
  
  ```python
  import torch.nn as nn
  import numpy as np
  import torch.nn.functional as F


  def get_number_of_trainable_parameters(model):
    """taken from
    discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


  class CNN(nn.Module):

    def __init__(self, img_size=42, include_ST=False):
        super(CNN, self).__init__()
        self.ST = include_ST
        self.name = 'ST-CNN Affine' if include_ST else 'CNN'
        c_dim = 32 if include_ST else 36
        self.convs = nn.Sequential(
            nn.Conv2d(1, c_dim, kernel_size=9, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.ReLU(True),
            nn.Conv2d(c_dim, c_dim, kernel_size=7, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.ReLU(True),
        )
        out_conv = int((int((img_size - 8)/2) - 6)/2)
        self.classification = nn.Sequential(
            nn.Linear(out_conv**2*c_dim, 50),
            nn.ReLU(True),
            nn.Linear(50, 10),
            nn.LogSoftmax(dim=1),            
        )
        if include_ST:
            loc_conv_out_dim = int((int(img_size/2) - 4)/2) - 4
            loc_regression_layer = nn.Linear(20, 6)
            # initalize final regression layer to identity transform
            loc_regression_layer.weight.data.fill_(0)
            loc_regression_layer.bias = nn.Parameter(
                torch.tensor([1., 0., 0., 0., 1., 0.]))
            self.localisation_net = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=(2,2), stride=2),
                nn.ReLU(True),
                nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=0),
                nn.ReLU(True),
                nn.Flatten(),
                nn.Linear(loc_conv_out_dim**2*20, 20),
                nn.ReLU(True),
                loc_regression_layer
            ) 
        return

    def forward(self, img):
        batch_size = img.shape[0]
        if self.ST:
            out_ST = self.ST_module(img)
            img = out_ST
        out_conv = self.convs(img)
        out_classification = self.classification(out_conv.view(batch_size, -1))
        return out_classification

    def ST_module(self, inp):
        # act on twice downsampled inp
        down_inp = F.interpolate(inp, scale_factor=0.5, mode='bilinear',
                                  recompute_scale_factor=False, align_corners=False)
        theta_vector = self.localisation_net(down_inp)
        # affine transformation
        theta_matrix = theta_vector.view(-1, 2, 3)
        # grid generator
        grid = F.affine_grid(theta_matrix, inp.size(), align_corners=False)
        # sampler
        out = F.grid_sample(inp, grid, align_corners=False)
        return out

    def get_attention_rectangle(self, inp):
        assert inp.shape[0] == 1, 'batch size has to be one'
        # act on twice downsampled inp
        down_inp = F.interpolate(inp, scale_factor=0.5, mode='bilinear',
                                 recompute_scale_factor=False, align_corners=False)
        theta_vector = self.localisation_net(down_inp)
        # affine transformation matrix
        theta_matrix = theta_vector.view(2, 3).detach()
        # create normalized target rectangle input image
        target_rectangle = torch.tensor([
            [-1., -1., 1., 1., -1.], 
            [-1., 1., 1., -1, -1.], 
            [1., 1., 1., 1., 1.]]
        ).to(inp.device)
        # get source rectangle by transformation
        source_rectangle = torch.matmul(theta_matrix, target_rectangle)
        return source_rectangle


  # instantiate models
  cnn = CNN(img_size=42, include_ST=False)
  st_cnn = CNN(img_size=42, include_ST=True)
  # print trainable parameters
  for model in [cnn, st_cnn]:
    num_trainable_params = get_number_of_trainable_parameters(model)
    print(f'{model.name} has {num_trainable_params} trainable parameters')
  ```

  ![Trainable Parameters](/assets/img/09_spatial_transformer/trainable_params.png "Trainable Paramas")

* **Training Procedure**: As described in appendix A.4 of [Jaderberg et al.
  (2015)](https://arxiv.org/abs/1506.02025), the networks are trained
  with standard SGD, batch size of $256$ and base learning rate of
  $0.01$. To reduce computation time, the number of epochs is limited
  to $50$.
  
  The loss function is the multinomial cross entropy loss, i.e., 
  
  $$
    \text{Loss} = - \sum_{i=1}^N \sum_{k=1}^C p_i^{(k)} \cdot \log
    \left( \widehat{p}_i^{(k)} \right),
  $$
  
  where $k$ enumerates the number of classes, $i$ enumerates the
  number of images, $p_i^{k} \in \\{0, 1\\}$ denotes the true probability of image
  $i$ and class $k$ and $\widehat{p}_i^{k} \in [0, 1]$ is the
  probability predicted by the network. Note that the true probability
  distribution is categorical (hard labels), i.e.,
  
  $$
    p_i^{(k)} = 1_{k = y_i} = \begin{cases}1 & \text{if } k = y_i \\ 0
    & \text{else}\end{cases}
  $$
  
  where $y_i \in \\{0, 1, \cdots, 9 \\}$ is the label assigned to the
  $i$-th image $\textbf{x}_i$. Thus, we can rewrite the loss as follows  
  
  $$
    \text{Loss} = - \sum_{i=1}^N \log \left( \widehat{p}_{i, y_i}
    \right),
  $$
  
  which is the definition of the negative log likelihood loss
  ([NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html))
  in Pytorch, when the logarithmized predictions $\log \left(
  \widehat{p}_{i, y_i} \right)$ (matrix of size $N\times C$) and
  class labels $y_i$ (vector of size $N$) are given as input. 
  
  The code below summarizes the whole training procedure.
  
  ```python
  from livelossplot import PlotLosses
  from torch.utils.data import DataLoader


  def train(model, dataset):
      # fix hyperparameters
      epochs = 50
      learning_rate = 0.01
      batch_size = 256
      step_size_scheduler = 50000
      gamma_scheduler = 0.1
      # set device    
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      print(f'Device: {device}')

      data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4)

      model.to(device)
      optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=gamma_scheduler,
                                                  step_size=step_size_scheduler)

      losses_plot = PlotLosses()
      print(f'Start training with {model.name}')
      for epoch in range(1, epochs+1):
          avg_loss = 0
          for data, label in data_loader:
              model.zero_grad()

              log_prop_pred = model(data.to(device))
              # multinomial cross entropy loss
              loss = F.nll_loss(log_prop_pred, label.to(device))

              loss.backward()
              optimizer.step()
              scheduler.step()

              avg_loss += loss.item() / len(data_loader)

          losses_plot.update({'log loss': np.log(avg_loss)})
          losses_plot.send()
      trained_model = model
      return trained_model 
  ```

* **Test Procedure**: A very simple test procedure to evaluate both
  models is shown below. It is basically the same as in [the pytorch
  tutorial](https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html). 
  
  ```python
  def test(trained_model, test_dataset):
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True,
                              num_workers=4)
      with torch.no_grad():
          trained_model.eval()
          test_loss = 0
          correct = 0
          for data, label in test_loader:
              data, label = data.to(device), label.to(device)

              log_prop_pred = trained_model(data)
              class_pred = log_prop_pred.max(1, keepdim=True)[1]

              test_loss += F.nll_loss(log_prop_pred, label).item()/len(test_loader)
              correct += class_pred.eq(label.view_as(class_pred)).sum().item()

          print(f'{trained_model.name}: avg loss: {np.round(test_loss, 2)},  ' +
                f'avg acc {np.round(100*correct/len(test_dataset), 2)}%')
      return
   ```

### Results

Lastly, the results can also divided into three sections:

* **Training Results**: Firstly, we train our models on the training dataset and compare the logarithmized losses:
  
  ```python
  trained_cnn = train(cnn, train_dataset)
  ```

  ![Training Results CNN](/assets/img/09_spatial_transformer/train_cnn_results.png "Training Results CNN")
  
  ```python
  trained_st_cnn = train(st_cnn, train_dataset)
  ``` 
  
  ![Training Results ST-CNN](/assets/img/09_spatial_transformer/train_st_cnn_results.png "Training Results ST-CNN")
  
  The logarithmized losses already indicate that the ST-CNN performs
  better than the standard CNN (at least, it decreases the loss
  faster). However, it can also be noted that training the ST-CNN
  seems less stable. 

* **Test Performance**: While the performance on the training dataset
  may be a good indicator, test set performance is much more
  meaningful. Let's compare the losses and accuracies between both
  trained models:
  
  ```python
  for trained_model in [trained_cnn, trained_st_cnn]:
      test(trained_model, test_dataset)
  ```

  ![Test Results](/assets/img/09_spatial_transformer/test_results.png "Test Results")
  
  Clearly, the ST-CNN performs much better than the standard CNN. Note
  that training for more epochs would probably result in even better
  accuracies in both models. 

* **Visualization of Learned Transformations**: Lastly, it might be
  interesting to see what the ST module actually does after training:
  
  ```python
  import matplotlib.pyplot as plt
  from matplotlib.patches import ConnectionPatch


  def visualize_learned_transformations(trained_st_cnn, test_dataset, digit_class=8):
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      trained_st_cnn.to(device)
      n_samples = 5

      data_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
      batch_img, batch_label = next(iter(data_loader))
      i_samples = np.where(batch_label.numpy() == digit_class)[0][0:n_samples]

      fig = plt.figure(figsize=(n_samples*2.5, 2.5*4))
      for counter, i_sample in enumerate(i_samples):
          img = batch_img[i_sample]
          label = batch_label[i_sample]

          # input image
          ax1 = plt.subplot(4, n_samples, 1 + counter)
          plt.imshow(transforms.ToPILImage()(img), cmap='gray')
          plt.axis('off')
          if counter == 0:
              ax1.annotate('Input', xy=(-0.3, 0.5), xycoords='axes fraction',
                          fontsize=14, va='center', ha='right')

          # image including border of affine transformation
          img_inp = img.unsqueeze(0).to(device)
          source_normalized = trained_st_cnn.get_attention_rectangle(img_inp)
          # remap into absolute values
          source_absolute = 0 + 20.5*(source_normalized.cpu() + 1)
          ax2 = plt.subplot(4, n_samples, 1 + counter + n_samples)
          x = np.arange(42)
          y = np.arange(42)
          X, Y = np.meshgrid(x, y)
          plt.pcolor(X, Y, img.squeeze(0), cmap='gray')
          plt.plot(source_absolute[0], source_absolute[1], color='red')
          plt.axis('off')
          ax2.axes.set_aspect('equal')
          ax2.set_ylim(41, 0)
          ax2.set_xlim(0, 41)
          if counter == 0:
              ax2.annotate('ST', xy=(-0.3, 0.5), xycoords='axes fraction',
                          fontsize=14, va='center', ha='right')
          # add arrow between
          con = ConnectionPatch(xyA=(21, 41), xyB=(21, 0), coordsA='data', 
                                coordsB='data', axesA=ax1, axesB=ax2, 
                                arrowstyle="-|>", shrinkB=5)
          ax2.add_artist(con)

          # ST module output
          st_img = trained_st_cnn.ST_module(img.unsqueeze(0).to(device))

          ax3 = plt.subplot(4, n_samples, 1 + counter + 2*n_samples)
          plt.imshow(transforms.ToPILImage()(st_img.squeeze(0).cpu()), cmap='gray')
          plt.axis('off')
          if counter == 0:
              ax3.annotate('ST Output', xy=(-0.3, 0.5), xycoords='axes fraction',
                          fontsize=14, va='center', ha='right')
          # add arrow between
          con = ConnectionPatch(xyA=(21, 41), xyB=(21, 0), coordsA='data', 
                                coordsB='data', axesA=ax2, axesB=ax3,
                                arrowstyle="-|>", shrinkB=5)
          ax3.add_artist(con)

          # predicted label
          log_pred = trained_st_cnn(img.unsqueeze(0).to(device))
          pred_label = log_pred.max(1)[1].item()

          ax4 = plt.subplot(4, n_samples, 1 + counter + 3*n_samples)
          plt.text(0.45, 0.43, str(pred_label), fontsize=22)
          plt.axis('off')
          #plt.title(f'Ground Truth {label.item()}', y=-0.1, fontsize=14)
          if counter == 0:
              ax4.annotate('Prediction', xy=(-0.3, 0.5), xycoords='axes fraction',
                          fontsize=14, va='center', ha='right')
          # add arrow between
          con = ConnectionPatch(xyA=(21, 41), xyB=(0.5, 0.65), coordsA='data', 
                                coordsB='data', axesA=ax3, axesB=ax4, 
                                arrowstyle="-|>", shrinkB=5)
          ax4.add_artist(con)
      return


  visualize_learned_transformations(st_cnn, test_dataset, 2)
   ```

  ![Transformation Visualization](/assets/img/09_spatial_transformer/transformation_visualization.png "Transformation Visualization")
  
  Clearly, the ST module attends to the digits such that the
  ST output has much less variation in terms of rotation, translation
  and scale making the classification task for the follow up CNN easier.
  
  Pretty cool, hugh?
   
   
  




---------------------------------------------------------------------------
