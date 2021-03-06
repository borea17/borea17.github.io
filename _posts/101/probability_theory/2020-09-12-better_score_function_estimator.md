---
title: "How can the variance of the score function estimator be decreased?"
permalink: "/ML_101/probability_theory/better_score_function_estimator"
author: "Markus Borea"
published: true
type: "101 probability"
---

Remind that the score function estimator moves the gradient of an
expectation inside the expecation to allow for Monte Carlo integration, i.e.,

$$
\begin{align}
\nabla_{\boldsymbol{\theta}} \mathbb{E}_{\textbf{z} \sim
  p_{\boldsymbol{\theta}}\left(\textbf{z} \right)} \big[ f
  (\textbf{z})\big] &= \mathbb{E}_{\textbf{z} \sim
  p_{\boldsymbol{\theta}}\left(\textbf{z} \right)} \left[
  f(\textbf{z}) \nabla_{\boldsymbol{\theta}} \log
  p_{\boldsymbol{\theta}} (\textbf{z}) \right] \\
  &\approx \frac {1}{N}
  \sum_{i=1}^N f\left(\textbf{z}^{(i)}\right) \nabla_{\boldsymbol{\theta}} \log
  p_{\boldsymbol{\theta}} \left( \textbf{z}^{(i)}\right) \quad
  \text{with} \quad \textbf{z}^{(i)} \sim p_{\boldsymbol{\theta}}(\textbf{z}),
\end{align}
$$

where the term $\nabla\_{\boldsymbol{\theta}}\log p\_{\boldsymbol{\theta}} (\textbf{z})$ is called
the score function. A nice property of the score function is that its
expectated value is zero, i.e.,

$$
\mathbb{E}_{\textbf{z} \sim
  p_{\boldsymbol{\theta}}\left(\textbf{z} \right)} \left[
  \nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}
  (\textbf{z}) \right] = \textbf{0}
$$

Using this property and the linearity of the expectation, we can add
an arbitrary term with zero expectation:

$$
\begin{align}
\nabla_{\boldsymbol{\theta}} \mathbb{E}_{\textbf{z} \sim
  p_{\boldsymbol{\theta}}\left(\textbf{z} \right)} \big[ f
  (\textbf{z})\big] &= \mathbb{E}_{\textbf{z} \sim
  p_{\boldsymbol{\theta}}\left(\textbf{z} \right)} \left[
  \Big(f(\textbf{z}) - \lambda \Big)\nabla_{\boldsymbol{\theta}} \log
  p_{\boldsymbol{\theta}} (\textbf{z}) \right] \\
  &\approx \frac {1}{N}
  \sum_{i=1}^N \Big(f\left(\textbf{z}^{(i)}\right) - \lambda \Big)\nabla_{\boldsymbol{\theta}} \log
  p_{\boldsymbol{\theta}} \left( \textbf{z}^{(i)}\right) \quad
  \text{with} \quad \textbf{z}^{(i)} \sim p_{\boldsymbol{\theta}}(\textbf{z}),
\end{align}
$$

where $\lambda$ is called **control variate** or **baseline** which
allows us to decrease the variance. Note that the choice of $\lambda$
is non-trivial and an arbitrary function could also increase the variance.

## Derivation

The expectation of the score function is zero:

$$
\begin{align}
\mathbb{E}_{\textbf{z} \sim
  p_{\boldsymbol{\theta}}\left(\textbf{z} \right)} \left[
  \nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}
  (\textbf{z}) \right] &= \int \nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}
  (\textbf{z}) p_{\boldsymbol{\theta}} (\textbf{z}) d\textbf{z} \\
  &=\int \frac {\nabla_{\boldsymbol{\theta}} p_{\boldsymbol{\theta}}
  (\textbf{z})} {p_{\boldsymbol{\theta}}
  (\textbf{z})}  p_{\boldsymbol{\theta}}
  (\textbf{z}) d\textbf{z} &&\text{(log derivative trick)}\\
  &= \nabla_{\boldsymbol{\theta}} \int p_{\boldsymbol{\theta}}
  (\textbf{z}) d\textbf{z} &&\text{(Leibniz integral rule)}\\
  &= \nabla_{\boldsymbol{\theta}} \textbf{1} = \textbf{0}
\end{align}
$$

The log derivative trick comes from the application of the chain rule,
see [this
post](https://borea17.github.io/ML_101/probability_theory/score_function_estimator#derivation).

The variance of the score function estimator (including the baseline $\lambda$)

$$
\text{Var} \Big[ \mathbb{E}_{\textbf{z} \sim
  p_{\boldsymbol{\theta}}\left(\textbf{z} \right)} \left[
  \Big(f(\textbf{z}) - \lambda \Big)\nabla_{\boldsymbol{\theta}} \log
  p_{\boldsymbol{\theta}} (\textbf{z}) \right] \Big] =
$$
