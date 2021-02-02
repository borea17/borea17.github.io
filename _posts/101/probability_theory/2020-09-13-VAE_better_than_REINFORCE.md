---
title: "Why does the NLL gradient in VAEs exhibit less variance than the REINFORCE estimator?"
permalink: "/ML_101/probability_theory/VAE_NLL_gradient_better_than_REINFORCE_estimator"
author: "Markus Borea"
published: true
type: "101 probability"
---

The **standard VAE gradient** and **REINFORCE estimator** (for NLL) are both **unbiased
estimators** of the same gradient, however the `VAE update tends to have lower
variance in practice because it makes use of the log-likelihood gradients with
respect to the latent variables` [(Burda et al. ,2016)](https://arxiv.org/abs/1509.00519).

## Explanation

Remind that the **REINFORCE / [score-function
estimator](https://borea17.github.io/ML_101/probability_theory/score_function_estimator)**
of the **negative log-likelihood (NLL)** in a **variational autoencoder (VAE)**
is given by

$$
\begin{align}
&\nabla_{\boldsymbol{\phi}}
\mathbb{E}_{q_\boldsymbol{\phi}\left(\textbf{z} | \textbf{x} \right)}
\Big[
\log p_{\boldsymbol{\theta}} \left( \textbf{x} | \textbf{z} \right)
\Big] =
\mathbb{E}_{q_\boldsymbol{\phi}\left(\textbf{z} | \textbf{x} \right)}
\Big[
\log p_{\boldsymbol{\theta}} \left( \textbf{x} | \textbf{z} \right)
\nabla_{\boldsymbol{\phi}} q_\boldsymbol{\phi}\left(\textbf{z} | \textbf{x} \right)
\Big]\\
&\approx \frac {1}{N} \sum_{i=1}^N \log p_{\boldsymbol{\theta}} \left(
\textbf{x} | \textbf{z}^{(i)} \right)
\nabla_{\boldsymbol{\phi}} q_\boldsymbol{\phi}\left(\textbf{z}^{(i)} |
\textbf{x} \right)
\quad \text{with} \quad \textbf{z}^{(i)} \sim q_\boldsymbol{\phi}\left(\textbf{z} | \textbf{x} \right).
\end{align}
$$

The strength of the REINFORCE estimator is also its limitation: We do not need
to compute a derivative w.r.t. to
$\log p_{\boldsymbol{\theta}} \left(\textbf{x} | \textbf{z}^{(i)}\right)$. While
this is a nice feature, it leads to high variances, because it is missing
information.

<!-- It is actually a **random search in disguise**.  -->

[Kingma and Welling (2013)](https://arxiv.org/abs/1312.6114), [Rezende et
al. (2014)](https://arxiv.org/abs/1401.4082), [Titsias and Lázaro-Gredilla
(2014)](http://proceedings.mlr.press/v32/titsias14.html) (independently)
introduced the *reparameterization trick* which essentially consists of
using auxiliary variables  with fixed distributions $\boldsymbol{\epsilon} \sim
p(\boldsymbol{\epsilon})$, such that sampling
$\textbf{z} \sim q_\boldsymbol{\phi}\left(\textbf{z} | \textbf{x} \right)$ can be written
as a deterministic mapping $\textbf{z} = g_{\boldsymbol{\phi}} (\textbf{x},
\boldsymbol{\epsilon})$. Therefore, the **standard VAE gradient** of the **NLL**
looks as follows

$$
\begin{align}
& \nabla_{\boldsymbol{\phi}} \mathbb{E}_{p(\boldsymbol{\epsilon})} \Big[ \log
p_{\boldsymbol{\theta}} \left( \textbf{x} | g_{\boldsymbol{\phi}} (\textbf{x},
\boldsymbol{\epsilon}) \right) \Big] =
\mathbb{E}_{p(\boldsymbol{\epsilon})} \Big[ \nabla_{\boldsymbol{\phi}} \log
p_{\boldsymbol{\theta}} \left( \textbf{x} | g_{\boldsymbol{\phi}} (\textbf{x},
\boldsymbol{\epsilon}) \right) \Big]\\
&\approx \frac {1}{N} \sum_{i=1}^{N} \nabla_{\boldsymbol{\phi}} \log p_{\boldsymbol{\theta}}
\left(\textbf{x} | \textbf{z}^{(i)}_{\boldsymbol{\phi}} \right) \quad \text{with} \quad
\textbf{z}^{(i)}_{\boldsymbol{\phi}} = g_{\boldsymbol{\phi}} \left(\textbf{x},
\boldsymbol{\epsilon}^{(i)}\right) \quad \text{and} \quad
\boldsymbol{\epsilon}^{(i)}\sim p(\boldsymbol{\epsilon})
\end{align}
$$

Clearly, during **backpropagation** the VAE makes use of the log-likelihood
gradients w.r.t the latent variables $\log
p_{\boldsymbol{\theta}}\left(\textbf{x}|\textbf{z} \right)$ (i.e.,
propagates from the encoder to the end of the decoder) whereas the
score-function estimator takes only the gradients from the encoder
$q_{\boldsymbol{\phi}}\left( \textbf{x} | \textbf{z} \right)$ and uses the
output of the decoder $\log p_{\boldsymbol{\theta}}\left(\textbf{x}|\textbf{z}
\right)$ as a mere scalar.
