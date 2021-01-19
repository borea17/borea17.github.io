---
title: "Importance Weighted Autoencoders"
permalink: "/paper_summaries/iwae"
author: "Markus Borea"
tags: ["generative model"]
published: false
toc: true
toc_sticky: true
toc_label: "Table of Contents"
type: "paper summary"
---

[Burda et al. (2016)](https://arxiv.org/abs/1509.00519) introduce the **Importance
Weighted Autoencoder (IWAE)** as a simple modification in the training
of [variational
autoencoders](https://borea17.github.io/paper_summaries/auto-encoding_variational_bayes)
**(VAEs)**. Notably, they proved that this modification leads to a **strictly
tighter lower bound on the data log-likelihood**. Furthermore, the standard VAE
formulation is contained within the IWAE framework as a special case. In
essence, the modification consists of using multiple samples from the
*recognition network* / *encoder* and adapting the loss function with
*importance-weighted sample losses*. In their experiments, they could emprically
validate that employing IWAEs leads to richer latent space representations
compared to VAEs.

## Model Description

An IWAE can be understood as a standard VAE in which multiple samples are drawn
from the encoder distribution $q_{\boldsymbol{\phi}} \Big( \textbf{z} |
\textbf{x} \Big)$ and then fed through the decoder $p_{\boldsymbol{\theta}}
\Big( \textbf{z} | \textbf{x} \Big)$. In principle, this modification has been
already proposed in the original VAE paper by [Kingma and Welling
(2013)](https://arxiv.org/abs/1312.6114). However, [Burda et al.
(2016)](https://arxiv.org/abs/1509.00519) additionally proposed to use a
different objective function. This (empirical) objective function is basically the data
log-likelihood $\log p_{\boldsymbol{\theta}} (\textbf{x})$ where the sampling distribution is
exchanged to $q_{\boldsymbol{\phi}} \Big( \textbf{z} |
\textbf{x} \Big)$ via the method of *importance sampling*.

### High-Level Overview

The IWAE framework builds upon a standard VAE architecture (see image below).
There are two neural networks as approximations for the encoder
$q_{\boldsymbol{\phi}} \left(\textbf{z} | \textbf{x} \right)$ and the decoder
distribution $p_{\boldsymbol{\theta}} \left( \textbf{x} | \textbf{z} \right)$.
More precisely, the networks estimate the parameters that parametrize
these distributions. Typically, the latent distribution is assumed to be a
Gaussian such that the encoder network estimates the mean and variance
of $q_{\boldsymbol{\phi}} \left(\textbf{z} | \textbf{x} \right) \sim
\mathcal{N}\left( \boldsymbol{\mu}\_{\text{E}}, \text{diag} \left(
\boldsymbol{\sigma}^2\_{\text{E}}\right) \right)$. To allow for backpropagation,
we apply the reparametrization trick to the latent distribution which
essentially consist of transforming samples from some (fixed) random
distribution, e.g. $\boldsymbol{\epsilon} \sim \mathcal{N} \left(\textbf{0},
\textbf{I} \right)$, into the desired distribution using a deterministic mapping.

The main difference between a VAE and IWAE lies in the objective function which
is explained in more detail in the next section.


| ![VAE/IWAE Architecture](/assets/img/02_AEVB/schematic_VAE.png "VAE/IWAE Architecture") |
| :--:        |
| **VAE/IWAE Architecture** |

### Derivation

Let $\textbf{X} = \\{\textbf{x}^{(i)}\\}_{i=1}^N$ denote a dataset of $N$ i.i.d.
samples where each observed datapoint $\textbf{x}^{(i)}$ is obtained by first
sampling a latent vector $\textbf{z}$ from the prior
$p\_{\boldsymbol{\theta}}(\textbf{z})$ and then sampling $\textbf{x}^{(i)}$
itself from the scene model $p\_{\boldsymbol{\theta}} \Big( \textbf{x} |
\textbf{z} \Big)$. Now we introduce an auxiliary distribution
$q\_{\boldsymbol{\phi}} \Big( \textbf{z} | \textbf{x} \Big)$ (with its own
paramaters) as an approximation to the true, but unknown posterior
$p\_{\boldsymbol{\theta}} \left( \textbf{z} | \textbf{x} \right)$. Accordingly,
the data likelihood of a one sample $\textbf{x}^{(i)}$ can be stated as
follows

$$
p_{\boldsymbol{\theta}} (\textbf{x}^{(i)}) = \mathbb{E}_{\textbf{z} \sim
p_{\boldsymbol{\theta}} \Big(\textbf{z} \Big)} \Big[
p_{\boldsymbol{\theta}} \left( \textbf{z} | \textbf{x}^{(i)} \right) \Big] =
\int p_{\boldsymbol{\theta}} (\textbf{z}) p_{\boldsymbol{\theta}} \left(
\textbf{z} | \textbf{x}^{(i)} \right) d\textbf{z} =
\int p_{\boldsymbol{\theta}} \left(\textbf{x}^{(i)} , \textbf{z} \right) d\textbf{z}
$$

Now, we use the simple trick of *importance sampling* to change the sampling
distribution into the approximated posterior, i.e.,

$$
p_{\boldsymbol{\theta}} \left( \textbf{x}^{(i)}\right)
= \int \frac {p_{\boldsymbol{\theta}} \left(\textbf{x}^{(i)} , \textbf{z}
\right)} {q_{\boldsymbol{\phi}} \left( \textbf{z} | \textbf{x}^{(i)} \right)}
q_{\boldsymbol{\phi}} \left( \textbf{z} | \textbf{x}^{(i)} \right) d\textbf{z}
= \mathbb{E}_{\textbf{z} \sim q_{\boldsymbol{\phi}} \left( \textbf{z} |
\textbf{x}^{(i)} \right)}
\left[ \frac {p_{\boldsymbol{\theta}} \left(\textbf{x}^{(i)} , \textbf{z}
\right)} {q_{\boldsymbol{\phi}} \left( \textbf{z} | \textbf{x}^{(i)} \right)}
\right]
$$

#### VAE Formulation

In the standard VAE approach, we use the **evidence lower bound**
[(ELBO)](https://borea17.github.io/ML_101/probability_theory/evidence_lower_bound)
on $\log p\_{\boldsymbol{\theta}} \Big( \textbf{x}\Big)$ as the objective
function. This can be derived by applying Jensen's Inequality on the data
log-likelihood:

$$
\log p_{\boldsymbol{\theta}} \left( \textbf{x}^{(i)} \right)
= \log \mathbb{E}_{\textbf{z} \sim q_{\boldsymbol{\phi}} \left( \textbf{z} |
\textbf{x}^{(i)} \right)}
\left[ \frac {p_{\boldsymbol{\theta}} \left(\textbf{x}^{(i)} , \textbf{z}
\right)} {q_{\boldsymbol{\phi}} \left( \textbf{z} | \textbf{x}^{(i)} \right)}
\right] \ge \mathbb{E}_{\textbf{z} \sim q_{\boldsymbol{\phi}} \left( \textbf{z} |
\textbf{x}^{(i)} \right)}
\left[ \log \frac {p_{\boldsymbol{\theta}} \left(\textbf{x}^{(i)} , \textbf{z}
\right)} {q_{\boldsymbol{\phi}} \left( \textbf{z} | \textbf{x}^{(i)} \right)}
\right] = \mathcal{L}^{\text{ELBO}}
$$

Using simple algebra, this can be rearranged into

$$
\mathcal{L}^{\text{ELBO}}\left(\boldsymbol{\theta}, \boldsymbol{\phi}; \textbf{x}^{(i)} \right) =
\underbrace{
\mathbb{E}_{\textbf{z} \sim q_{\phi} \left( \textbf{z}| \textbf{x}^{(i)} \right)}
\left[ \log p_{\boldsymbol{\theta}} \Big(\textbf{x}^{(i)} | \textbf{z} \Big)
\right]}_{
\text{Reconstruction Accuracy}}
-
\underbrace{
D_{KL} \left( q_{\phi} \Big( \textbf{z} |
\textbf{x}^{(i)} \Big) || p_{\boldsymbol{\theta}} \Big( \textbf{z} \Big) \right)
}_{\text{Regularization}}
$$

While the regularization term can usually be solved analytically, the
reconstruction accuracy in its current formulation poses a problem for
backpropagation: Gradients cannot backpropagate through a sampling operation. To
circumvent this problem, the standard VAE formulation includes the
reparametrization trick:
<center>Substitute sampling $\textbf{z} \sim q_{\boldsymbol{\phi}}$ by using a
deterministic mapping $\textbf{z} = g_{\boldsymbol{\phi}}
(\boldsymbol{\epsilon},
\textbf{x})$ with the differential transformation
$g_{\boldsymbol{\phi}}$ of an auxiliary noise variable
$\boldsymbol{\epsilon}$ with $\boldsymbol{\epsilon}\sim p(\boldsymbol{\epsilon})$.
</center>
<br>
As a result, we can rewrite the EBLO as follows

$$
\mathcal{L}^{\text{ELBO}}\left(\boldsymbol{\theta}, \boldsymbol{\phi};
\textbf{x}^{(i)} \right) =
\mathbb{E}_{\boldsymbol{\epsilon} \sim p \left( \boldsymbol{\epsilon} \right)}
\left[ \log p_{\boldsymbol{\theta}} \Big(\textbf{x}^{(i)} |
g_{\boldsymbol{\phi}} \left( \boldsymbol{\epsilon}, \textbf{x}^{(i)} \right) \Big)
\right] -
D_{KL} \left( q_{\phi} \Big( \textbf{z} |
\textbf{x}^{(i)} \Big) || p_{\boldsymbol{\theta}} \Big( \textbf{z} \Big) \right)
$$

Lastly, the expectation is approximated using Monte-Carlo integration, leading
to the standard VAE objective

$$
\begin{align}
  \widetilde{\mathcal{L}}^{\text{VAE}}_k \left(\boldsymbol{\theta},
  \boldsymbol{\phi};
  \textbf{x}^{(i)}\right) &=
  \frac {1}{k} \sum_{l=1}^{k} \log p_{\boldsymbol{\theta}}\left(
  \textbf{x}^{(i)}| g_{\boldsymbol{\phi}}
 \left( \boldsymbol{\epsilon}^{(l)}, \textbf{x}^{(i)} \right)\right)
  -D_{KL} \left(  q_{\boldsymbol{\phi}} \left( \textbf{z} | \textbf{x}^{(i)} \right),
  p_{\boldsymbol{\theta}} (\textbf{z}) \right)\\
  &\text{with} \quad \boldsymbol{\epsilon}^{(l)} \sim p(\boldsymbol{\epsilon})
\end{align}
$$

Note that commonly $k=1$ in VAEs as long as the minibatch size is large enough.
As stated by [Kingma and Welling (2013)](https://arxiv.org/abs/1312.6114):

> We found that the number of samples per datapoint can be set to 1 as long as
> the minibatch size was large enough.


#### IWAE Formulation

Before we introduce the IWAE estimator, remind that the Monte-Carlo estimator of
the data likelihood (when the sampling distribution is changed via importance
sampling, see
[Derivation](https://borea17.github.io/paper_summaries/iwae#derivation)) is
given by

$$
p_{\boldsymbol{\theta}} (\textbf{x} ) =
\mathbb{E}_{\textbf{z} \sim q_{\boldsymbol{\phi}} \left( \textbf{z} |
\textbf{x}^{(i)} \right)}
\left[ \frac {p_{\boldsymbol{\theta}} \left(\textbf{x} , \textbf{z}
\right)} {q_{\boldsymbol{\phi}} \left( \textbf{z} | \textbf{x} \right)}
\right] \approx \frac {1}{k} \sum_{l=1}^{k}
\frac {p_{\boldsymbol{\theta}} \left(\textbf{x} , \textbf{z}^{(l)}
\right)} {q_{\boldsymbol{\phi}} \left( \textbf{z}^{(l)} | \textbf{x}
\right)} \quad \text{with} \quad \textbf{z}^{(l)} \sim q_{\boldsymbol{\phi}} \left( \textbf{z} |
\textbf{x}^{(i)} \right)
$$

As a result, the data log-likelihood estimator for one sample $\textbf{x}^{(i)}$
can be stated as follows

$$
\begin{align}
\log p_{\boldsymbol{\theta}} (\textbf{x}^{(i)} ) &\approx \log \left[ \frac {1}{k} \sum_{l=1}^{k}
\frac {p_{\boldsymbol{\theta}} \left(\textbf{x}^{(i)} , \textbf{z}^{(l)}
\right)} {q_{\boldsymbol{\phi}} \left( \textbf{z}^{(l)} | \textbf{x}^{(i)}
\right)}\right] = \widetilde{\mathcal{L}}^{\text{IWAE}}_k \left( \boldsymbol{\theta},
\boldsymbol{\phi}; \textbf{x}^{(i)} \right) \\
&\text{with} \quad \textbf{z}^{(l)} \sim q_{\boldsymbol{\phi}} \left( \textbf{z} |
\textbf{x}^{(i)} \right)
\end{align}
$$

which leads to an empirical estimate of the IWAE objective. However, [Burda et
al. (2016)](https://arxiv.org/abs/1509.00519) do not use the data log-likelihood
in its plain form as the true IWAE objective. Instead they introduce the IWAE
objective as follows

$$
\mathcal{L}^{\text{IWAE}}_k \left(\boldsymbol{\theta}, \boldsymbol{\phi};
\textbf{x}^{(i)}\right)
=  \mathbb{E}_{\textbf{z}^{(1)}, \dots,  \textbf{z}^{(k)} \sim q_{\phi} \left( \textbf{z}|
\textbf{x}^{(i)} \right)}
\left[
\log \frac {1}{k}
\sum_{l=1}^k
\frac {p_{\boldsymbol{\theta}} \left(\textbf{x}^{(i)}, \textbf{z}^{(l)}\right)}
{q_{\phi} \left( \textbf{z}^{(l)} | \textbf{x}^{(i)} \right)}
\right]
$$

For notation purposes, they denote

$$
\text{(unnormalized) importance weights:} \quad
{w}^{(i, l)} = \frac {p_{\boldsymbol{\theta}} \left(\textbf{x}^{(i)}, \textbf{z}^{(l)}\right)}
{q_{\phi} \left( \textbf{z}^{(l)} | \textbf{x}^{(i)} \right)}
$$

By applying Jensen's Inequality, we can see that in fact the (true) IWAE estimator is
merely a lower-bound on the plain data log-likelihood

$$
\mathcal{L}^{\text{IWAE}}_k \left( \boldsymbol{\theta}, \boldsymbol{\phi};
\textbf{x}^{(i)} \right)
= \mathbb{E} \left[ \log \frac {1}{k} \sum_{l=1}^{k} {w}^{(i,
l)}\right] \le \log \mathbb{E} \left[ \frac {1}{k} \sum_{l=1}^{k}
{w}^{(i,l)} \right] = \log p_{\boldsymbol{\theta}} \left( \textbf{x}^{(i)} \right)
$$

They could prove that with increasing $k$ the lower bound gets strictly tighter
and approaches the true data log-likelihood in the limit of $k \rightarrow
\infty$. Note that since the empirical IWAE estimator
$\widetilde{\mathcal{L}}_k^{\text{IWAE}}$ can be understood as a Monte-Carlo
estimator on the true data log-likelihood, this property can also simply be
argued by the properties of Monte-Carlo integration. IMHO, the only
reason why their formulation may be superior to estimating the true data
log-likelihood using Monte-Carlo integration is that it contains the ELBO
as a special case ($k=1$).

Let us take a closer look on how to compute gradients (fast) for the empirical
estimate of the IWAE objective:


$$
\begin{align}
\nabla_{\boldsymbol{\phi}, \boldsymbol{\theta}}
\widetilde{\mathcal{L}}_k^{\text{IWAE}} \left( \boldsymbol{\theta}, \boldsymbol{\phi};
\textbf{x}^{(i)} \right) &= \nabla_{\boldsymbol{\phi}, \boldsymbol{\theta}}
\log \frac {1}{k} \sum_{l=1}^k w^{(i,l)} \left( \textbf{x}^{(i)},
\textbf{z}^{(i, l)}_{\boldsymbol{\phi}}, \boldsymbol{\theta} \right) \quad
\text{with} \quad
\textbf{z}^{(i, l)} \sim q_{\boldsymbol{\phi}} \left(\textbf{z} |
\textbf{x}^{(i)} \right)\\
&\stackrel{\text{(*)}}{=}
\sum_{l=1}^{k} \frac {w^{(i, l)}}{\sum_{m=1}^{k} w^{(i,
m)}} \nabla_{\boldsymbol{\phi}, \boldsymbol{\theta}} \log w^{(i,l)} =
\sum_{l=1}^{k} \widetilde{w}^{(i, l)} \nabla_{\boldsymbol{\phi}, \boldsymbol{\theta}} \log w^{(i,l)},
\end{align}
$$

where we introduced the following notation

$$
\text{(normalized) importance weights:} \quad
\widetilde{w}^{(i, l)} = \frac {w^{(i,l)}}{\sum_{m=1}^k w^{(i, m)}}
$$

-----------------------------

> $(*)$: **Gradient Derivation**:
>
>$$
>\begin{align}
>\frac {\partial \left[ \log \frac {1}{k} \sum_i^{k} w_i \left( \boldsymbol{\theta}
>\right) \right]}{\partial \boldsymbol{\theta}} &\stackrel{\text{chain rule}}{=}  \frac {\partial
>\log a}{\partial a} \sum_{i}^{k} \frac {\partial a}{\partial w_i} \frac
>{\partial w_i}{\partial \boldsymbol{\theta}} \quad \text{with}
>\quad a = \frac {1}{k} \sum_{i}^k w_i (\boldsymbol{\theta})\\
>&= \frac {k}{\sum_l^k w_l} \sum_{i}^{k}\frac {1}{k} \frac {\partial
>w_i (\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = \frac {1}{\sum_l^k
>w_l} \sum_{i}^{k} \frac {\partial
>w_i (\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}
>\end{align}
>$$
>
>Lastly, we use the following identity
>
>$$
>\frac {\partial w_i (\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = w_i
>(\boldsymbol{\theta}) \cdot
>\frac {\partial \log w_i (\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}
>\stackrel{\text{chain rule}}{=} w_i (\boldsymbol{\theta}) \cdot \frac {1}{w_i
>(\boldsymbol{\theta})} \cdot
>\frac {\partial w_i (\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} =
>\frac {\partial w_i (\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}
>$$

-----------------------------

Similar to VAEs, this formulation poses a problem for backpropagation due to the
sampling operation. We use the same reparametrization trick to circumvent this
problem and obtain a low variance update rule:

$$
\begin{align}
\nabla_{\boldsymbol{\phi},
\boldsymbol{\theta}}
\widetilde{\mathcal{L}}_k^{\text{IWAE}} &=
\sum_{l=1}^{k} \widetilde{w}^{(i, l)} \nabla_{\boldsymbol{\phi},
\boldsymbol{\theta}} \log w^{(i,l)} \left( \textbf{x}^{(i)},
\textbf{z}_{\boldsymbol{\phi}}^{(i,l)}, \boldsymbol{\theta} \right)
\quad \text{with} \quad
\textbf{z}^{(i,l)} \sim q_{\boldsymbol{\phi}} \left(\textbf{z} | \textbf{x}^{(i)} \right)\\
&= \sum_{l=1}^k \widetilde{w}^{(i,l)} \nabla_{\boldsymbol{\phi},
\boldsymbol{\theta}} \log w^{(i,l)} \left(\textbf{x}^{(i)},
g_{\boldsymbol{\phi}} \left( \textbf{x}^{(i)},
\boldsymbol{\epsilon}^{(l)}\right), \textbf{x}^{(i)} \right), \quad \quad
\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})
\end{align}
$$

Before, we are going to implement this formulation, let us look whether we can
separate out the KL divergence for the true IWAE objective of [Burda et al.
(2016)](https://arxiv.org/abs/1509.00519). Therefore, we state the update
for the true objective:

$$
\begin{align}
\nabla_{\boldsymbol{\phi},
\boldsymbol{\theta}}
\mathcal{L}_k^{\text{IWAE}} &=
\nabla_{\boldsymbol{\phi},
\boldsymbol{\theta}}
\mathbb{E}_{\textbf{z}^{(1)}, \dots, \textbf{z}^{(l)}} \left[ \log \frac {1}{k}
\sum_{l=1}^{k} w^{(l)} \left( \textbf{x},
\textbf{z}^{(l)}_{\boldsymbol{\phi}}, \boldsymbol{\theta} \right) \right]\\
&=
\mathbb{E}_{\textbf{z}^{(1)}, \dots, \textbf{z}^{(l)}} \left[
\sum_{l=1}^{k} \widetilde{w}_i
\nabla_{\boldsymbol{\phi},
\boldsymbol{\theta}}
\log w^{(l)} \left( \textbf{x}, \textbf{z}_{\boldsymbol{\phi}}^{(l)}, \boldsymbol{\theta} \right) \right]\\
&=
\mathbb{E}_{\boldsymbol{\epsilon}^{(1)}, \dots, \boldsymbol{\epsilon}^{(l)}} \left[
\sum_{l=1}^{k} \widetilde{w}_i
\nabla_{\boldsymbol{\phi},
\boldsymbol{\theta}}
\log w^{(l)} \left( \textbf{x}, g_{\boldsymbol{\phi}}
\left(\textbf{x}, \boldsymbol{\epsilon}^{(l)} \right), \boldsymbol{\theta} \right) \right]
\end{align}
$$

we can still do some
simplifications

$$
\log w^{(i,l)}  =
\log p_{\boldsymbol{\theta}} \left( \textbf{x}^{(i)}| g_{\boldsymbol{\phi}} \left( \textbf{x}^{(i)},
\boldsymbol{\epsilon}^{(l)}\right) \right) + \log p_{\boldsymbol{\theta}} \left( \textbf{z}
\right) - \log q_{\boldsymbol{\phi}} \left(\textbf{x}^{(i)} | \textbf{z} \right)
$$




## Implementation

{% capture code %}{% raw %}class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(WINDOW_SIZE**2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 2*LATENT_DIM)
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, WINDOW_SIZE**2)
        )
        return

    def compute_loss(self, x):
        [x_tilde, z, mu_z, log_var_z] = self.forward(x)
        NLL = (1/(2*VAR_FIXED))*(x - x_tilde).pow(2).sum((1,2,3)).mean()
        KL_Div = -0.5*(1 + log_var_z - mu_z.pow(2) - log_var_z.exp()).sum(1).mean()
        return NLL, KL_Div

    def forward(self, x):
        z, mu_z, log_var_z = self.encode(x)
        x_tilde = self.decode(z)
        return [x_tilde, z, mu_z, log_var_z]

    def encode(self, x):
        # get encoder distribution parameters
        out_encoder = self.encoder(x)
        mu_E, log_var_E = torch.chunk(out_encoder, 2, dim=1)
        # sample noise variable for each batch
        epsilon = torch.randn_like(log_var_E)
        # get latent variable by reparametrization trick
        z = mu_E + torch.exp(0.5*log_var_E) * epsilon
        return z, mu_E, log_var_E

    def decode(self, z):
        # get decoder distribution parameters
        x_tilde = self.decoder(z)
        # reshape into [batch, 1, WINDOW_SIZE, WINDOW_SIZE] (input shape)
        x_tilde = x_tilde.view(-1, 1, WINDOW_SIZE, WINDOW_SIZE)
        return x_tilde{% endraw %}{% endcapture %}
{% include code.html code=code lang="python" %}


{% capture code %}{% raw %}class IWAE(VAE):

    def __init__(self, k):
        super(IWAE, self).__init__()
        self.k = k
        return

    def compute_loss(self, x):
        [x_tilde, z, mu_z, log_var_z] = self.forward(x)
        # make x size as x_tilde [batch, samples, 1, WINDOW_SIZE, WINDOW_SIZE]
        x = x.unsqueeze(1).repeat(1, self.k, 1, 1, 1)
        # compute NLL [batch, samples]
        NLL = (1/(2*VAR_FIXED))*(x - x_tilde).pow(2).sum(axis=(2, 3, 4))
        # compute KL div [batch, samples]
        KL_Div =  -0.5*(1 + log_var_z - mu_z.pow(2) - log_var_z.exp()).sum(2)
        # get importance log_weights [batch, samples]
        log_weights = (NLL + KL_Div)
        # get tilde{w} (sum_k \log {\tilde{w}} = 1)
        w_tilde = F.softmax(log_weights, dim=1).detach()
        # compute loss
        loss = (w_tilde*log_weights).sum(1).mean()
        NLL_VAE = NLL[:, 0].mean()
        KL_Div_VAE =  KL_Div[:,0].mean()
        return loss, NLL_VAE, KL_Div_VAE

    def encode(self, x):
        # get encoder distribution parameters
        out_encoder = self.encoder(x)
        mu_E, log_var_E = torch.chunk(out_encoder, 2, dim=1)
        # increase shape for sampling [batch, samples, latent_dim]
        mu_E = mu_E.view(x.shape[0], 1, -1).repeat(1, self.k, 1)
        log_var_E = log_var_E.view(x.shape[0], 1, -1).repeat(1, self.k, 1)
        # sample noise variable for each batch and sample
        epsilon = torch.randn_like(log_var_E)
        # get latent variable by reparametrization trick
        z = mu_E + torch.exp(0.5*log_var_E) * epsilon
        return z, mu_E, log_var_E

    def decode(self, z):
        batch_size = z.shape[0]
        # parallelize computation by stacking samples along batch dim
        z = z.view(-1, z.shape[2])  # [batch*samples, latent_dim]
        # get decoder distribution parameters
        x_tilde = self.decoder(z)  # [batch*samples, WINDOW_SIZE**2]
        # reshape into [batch, samples, 1, WINDOW_SIZE, WINDOW_SIZE]
        x_tilde = x_tilde.view(batch_size, self.k, 1, WINDOW_SIZE, WINDOW_SIZE)
        return x_tilde{% endraw %}{% endcapture %}
{% include code.html code=code lang="python" %}
