---
title: "What is a score function estimator (REINFORCE estimator)?"
permalink: "/ML_101/probability_theory/score_function_estimator"
author: "Markus Borea"
published: true
type: "101 probability"
---

There are several use cases in which we are interested in estimating
the gradient of an expectation for samples drawn from a parametric
distribution $\textbf{z} \sim
p\_{\boldsymbol{\theta}}\left(\textbf{z}\right)$ and evaluated at $f(\textbf{z})$, i.e., 

$$
  \nabla_{\boldsymbol{\theta}} \mathbb{E}_{\textbf{z} \sim
  p_{\boldsymbol{\theta}}\left(\textbf{z} \right)} \big[ f
  (\textbf{z})\big] = \nabla_{\boldsymbol{\theta}} \int f(\textbf{z})
  p_{\boldsymbol{\theta}} (\textbf{z}) d\textbf{z}
$$

The **score function estimator**, **REINFORCE estimator** or **usual
(naïve) Monte Carlo estimator** approximates this gradient as follows 

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
the score function. This gradient estimator is unbiased, however
it (typically) exhibits very high variance.

Note that the main advantage of the estimator is that we moved the
gradient inside the expectation such that we can use Monte Carlo
sampling to approximate it. Furthermore, it does not make any
restriction on the function $f$, i.e., $f$ could even be
non-differentiable and we could still compute the gradient of its
expected value. 

<!-- [^1]: Note that the term score function is only used when -->
<!--     $p\_{\boldsymbol{\theta}} (\textbf{z})$ is valid likelihood -->
<!--     function, i.e., the outputs are probabilities parameterized by -->
<!--     $\boldsymbol{\theta}$.  -->

## Derivation

$$
\begin{align}
\nabla_{\boldsymbol{\theta}} \mathbb{E}_{\textbf{z} \sim
  p_{\boldsymbol{\theta}}\left(\textbf{z} \right)} \big[ f
  (\textbf{z})\big] 
  &= \nabla_{\boldsymbol{\theta}} \int f(\textbf{z})
  p_{\boldsymbol{\theta}} (\textbf{z}) d\textbf{z}\\
  &= \int f(\textbf{z}) \nabla_{\boldsymbol{\theta}} p_{\boldsymbol{\theta}} (\textbf{z})
  d\textbf{z} && \text{(Leibniz integral rule)} \\
  &= \int f(\textbf{z})
  \nabla_{\boldsymbol{\theta}}
  \log p_{\boldsymbol{\theta}} (\textbf{z}) p_{\boldsymbol{\theta}} (\textbf{z})
  d\textbf{z} && \text{(log derivative trick)}\\ 
  &= \mathbb{E}_{\textbf{z} \sim p_{\boldsymbol{\theta}}} \Big[ f
  (\textbf{z}) \nabla_{\boldsymbol{\theta}} \log
  p_{\boldsymbol{\theta}} (\textbf{z}) \Big]
\end{align}
$$

Lastly, we can approximate the expectation using Monte Carlo sampling,
i.e.,

$$
\begin{align}
\mathbb{E}_{\textbf{z} \sim p_{\boldsymbol{\theta}}} \Big[ f
  (\textbf{z}) \nabla_{\boldsymbol{\theta}} \log
  p_{\boldsymbol{\theta}} (\textbf{z}) \Big] \approx \frac {1}{N}
  \sum_{i=1}^N f\left(\textbf{z}^{(i)}\right) \nabla_{\boldsymbol{\theta}} \log
  p_{\boldsymbol{\theta}} \left( \textbf{z}^{(i)}\right) \quad
  \text{with} \quad \textbf{z}^{(i)} \sim p_{\boldsymbol{\theta}}(\textbf{z})
\end{align}
$$

The log derivative trick comes from the chain rule:

$$
\begin{align}
\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}} (\textbf{z}) = \frac
{\partial \log p (\textbf{z}; \boldsymbol{\theta})} {\partial
\boldsymbol{\theta}} &= \frac {\partial h}{\partial g} \frac {\partial
g}{\partial \boldsymbol{\theta}} && \text{with} \quad h(g) = \log g\\
&= \frac {1}{g} \frac {\partial g}{\partial \boldsymbol{\theta}} &&
\text{with} \quad g(\boldsymbol{\theta}) = p(\textbf{z}; \boldsymbol{\theta})\\
&= \frac {1}{p(\textbf{z}; \boldsymbol{\theta})} \frac{\partial
p(\textbf{z}; \boldsymbol{\theta})} {\partial \boldsymbol{\theta}} \\
&= \frac {\nabla_{\boldsymbol{\theta}} p_{\boldsymbol{\theta}} (\textbf{z})}{p_{\boldsymbol{\theta}}(\textbf{z})}
\end{align}
$$


## Examples

The following examples shall illustrate the use of the **score
function estimator** in its **plain form**. Note that due to its high
variance, in practice we would typically modify this estimator.

1. **Reinforcement Learning**: The idea of policy
   gradients is to iteratively update an initial random policy 
   towards maximizing the expected return, i.e., given a parameterized
   policy function $\pi_{\boldsymbol{\theta}} \left(a | s \right)$ the
   goal can be stated as follows 
   
   $$
     \max_{\boldsymbol{\theta}} \mathbb{E}_{a \sim \pi_{\boldsymbol{\theta}}
     \left( a|s\right)} \Big[ G_t \Big] = \max_{\boldsymbol{\theta}}
     \mathbb{E}_{a \sim \pi_{\boldsymbol{\theta}}
     \left( a|s\right)} \left[ \sum_{k=1}^{\infty} \gamma^{k}
     R_{t+k+1} (s, a) \right],
   $$
   
   where $G_t$ denotes the total discounted reward from time step $t$
   on-wards, $\gamma$ is the discount factor and $R\_{t+k+1}$ the
   immediately received reward after time step $t+k$. In order to
   maximize, we need to compute the gradient of that expectation. The
   **REINFORCE** or **Monte-Carlo policy gradient** method computes the
   gradient by using the **score function estimator**, i.e.,  
   
   $$
   \begin{align}
    \nabla_{\boldsymbol{\theta}} \mathbb{E}_{a \sim \pi_{\boldsymbol{\theta}}
    \left( a|s\right)} \Big[ G_t \Big] &= \mathbb{E}_{a\sim \pi_{\boldsymbol{\theta}}
    \left( a|s\right)} \left[ G_t \cdot \nabla_{\boldsymbol{\theta}}\log \pi_{\boldsymbol{\theta}} \left(a|s\right) \right]\\
    &\approx \frac {1}{N} \sum_{i=1}^N \frac {1}{T_i} \sum_{t=0}^{T_i-1} G_{i, t} \cdot \nabla_{\boldsymbol{\theta}} \log \pi^{(i,
   t)}_{\boldsymbol{\theta}} \quad \text{with} \quad \pi^{(i,
   t)}_{\boldsymbol{\theta}} \sim \pi_{\boldsymbol{\theta}} \left(A_t | S_t \right),
   \end{align}
   $$
   
   where $T_i$ denotes the number of steps in the $i-$th trajectory. 
2. **Discrete Variables as Latent Representation in VAEs**: In cases,
   where the encoder distribution $q\_{\boldsymbol{\phi}} \left(
   \textbf{z} | \textbf{x} \right)$ shall encode a discrete
   distribution (e.g., Bernoulli distribution), the reparameterization
   trick cannot be applied (since any reparameterization includes
   discontinuous operations). The **score function estimator** may be
   used instead such that gradient of the reconstruction accuracy is computed as follows 
   
   $$
   \begin{align}
     \nabla_{\boldsymbol{\phi}} \mathbb{E}_{\textbf{z} \sim q_{\boldsymbol{\phi}}
     \left(\textbf{z} | \textbf{x}^{(i)} \right)} \left[ \log
     p_{\boldsymbol{\theta}} \left(\textbf{x}^{(i)} | \textbf{z}
     \right) \right] &= \mathbb{E}_{\textbf{z} \sim q_{\boldsymbol{\phi}}
     \left(\textbf{z} | \textbf{x}^{(i)} \right)} \left[ \log
     p_{\boldsymbol{\theta}} \left(\textbf{x}^{(i)} | \textbf{z}
     \right) \nabla_{\boldsymbol{\phi}} \log q_{\boldsymbol{\phi}}
     \left(\textbf{z} | \textbf{x}^{(i)} \right)\right] \\
     &\approx \frac {1}{N} \sum_{k=1}^N \log
     p_{\boldsymbol{\theta}} \left(\textbf{x}^{(i)} | \textbf{z}^{(k)}
     \right) \cdot \nabla_{\boldsymbol{\phi}} \log q_{\boldsymbol{\phi}}
     \left(\textbf{z}^{(k)} | \textbf{x}^{(i)} \right),\\
     & \text{with} \quad \textbf{z}^{(k)} \sim q_{\boldsymbol{\phi}}
     \left(\textbf{z} | \textbf{x}^{(i)} \right)
   \end{align}
   $$

## Acknowledgements

Syed Javed's [blog
post](https://stillbreeze.github.io/REINFORCE-vs-Reparameterization-trick/)
was very helpful to get things clear.

----------------------------------------------------------------------------------
