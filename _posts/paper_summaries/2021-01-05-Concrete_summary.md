---
title: "The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables"
permalink: "/paper_summaries/concrete_distribution"
author: "Markus Borea"
published: false
toc: true
toc_sticky: true
toc_label: "Table of Contents"
type: "paper summary"
---

[Maddison et al. (2016)](https://arxiv.org/abs/1611.00712) introduce
**CON**tinuous relaxations of dis**CRETE** (**concrete**) random variables as an
approximation to discrete variables. The **Concrete distribution** is mo tivated
by the fact that backpropagation through discrete random variables is not
directly possible. While for continuous random variables, the
**reparametrization trick** is applicable to allow gradients to flow through a
sampling operation, this does not work for discrete variables due to the
discontinuous operations associated to their sampling. The **concrete
distribution** allows for a simple reparametrization through which gradients can
propagate such that a low-variance biased gradient estimator of the discrete
path can be obtained.

## Model Description

The **Concrete distribution** builds upon the (very old) **Gumbel-Max trick**
that allows for a reparametrization of a categorical distribution into a
deterministic function over the distribution parameters and an auxiliary noise
distribution. The problem within this reparameterization is that it relies on an
$\text{argmax}$-operation such that backpropagation remains out of reach. Therefore,
[Maddison et al. (2016)](https://arxiv.org/abs/1611.00712) propose to use
$\text{softmax}$-operation as a continuous relaxation of the $\text{argmax}$.
This idea has been concurrently developed at the same time by [Jang et al.
(2016)](https://arxiv.org/abs/1611.01144) who called it the **Gumbel-Softmax
trick**.

### Gumbel-Max Trick

The Gumbel-Max trick basically refactors sampling of a deterministic random
variable into a component-wise addition of the discrete distribution parameters
and an auxiliary noise followed by $\text{argmax}$, i.e.,

$$
\begin{align}
\text{Sampling }&z \sim \text{Cat} \left(\alpha_1, \dots, \alpha_N \right)
\text{ can equally expressed as } z = \arg\max_{k} \Big(\log\alpha_k + G_k\Big)\\
&\text{with } G_k \sim \text{Gumbel Distribution}\left(\mu=0, \beta=1 \right)
\end{align}
$$

| <img width="900" height="400" src='/assets/img/concrete/Gumble_Max.png'> |
| :--:        |
| **Computational Graph Gumbel-Max Trick**. Taken from [Maddison et al. (2016)](https://arxiv.org/abs/1611.00712). |

**Derivation**: Let's take a closer look on how and why that works. Firstly, we
show that samples from $\text{Cat} \left(\alpha_1, \dots, \alpha_N \right)$ are
equally distributed to

$$
z = \arg \min_{k} \frac {\epsilon_k}{\alpha_k} \quad \text{with} \quad
\epsilon_k \sim \text{Exp}\left( 1 \right)
$$

Therefore, we observe that each term inside the $\text{argmin}$ is independent
exponentially distributed with ([easy proof](https://math.stackexchange.com/a/85578))

$$
\frac {\epsilon_k}{\alpha_k} \sim  \text{Exp} \Big( \alpha_k \Big)
$$

The next step is to show that the index of the variable which achieves the
minimum is distributed according to the categorical distribution ([easy
proof](https://en.wikipedia.org/wiki/Exponential_distribution#Distribution_of_the_minimum_of_exponential_random_variables))

$$
\arg \min_{k} \frac {\epsilon_k}{\alpha_k} = P \left( k | z_k = \min \{ z_1, \dots, z_N \} \right) = \frac
{\alpha_k}{\sum_{i=1}^N \alpha_i}
$$

A nice feature of this formulation is that the categorical distribution
parameters $\\{\alpha_i\\}_{i=1}^N$ do not need to be normalized before
reparameterization as normalization is ensured by the factorization itself.
Lastly, we simply reformulate this mapping by applying the log and multiplying
by minus 1

$$
z = \arg \min_{k} \frac {\epsilon_k}{\alpha_k} =\arg \min_k \Big(\log \epsilon_k -
\log \alpha_k \Big) = \arg \max_k \Big(\log \alpha_k  - \log \epsilon_k\Big)
$$

This looks already very close to the **Gumbel-Max trick** defined above. Remind
that to generate exponential distributed random variables, we can simply
transform uniformly distributed samples of the unit interval as follows

$$
\epsilon_k = -\log u_k \quad \text{with} \quad u_k \sim
\text{Uniform Distribution} \Big(0, 1\Big)
$$

Thus, we get that

$$
- \log \epsilon_k = - \log \Big( - \log u_k \Big) = G_k \sim
\text{Gumbel Distribution} \Big( \mu=0, \beta=1 \Big)
$$


### Gumbel-Softmax Trick

The problem in the **Gumbel-Max trick** is the $\text{argmax}$-operation as the
derivative of $\text{argmax}$ is 0 everywhere except at the boundary of state
changes, where it is undefined. Thus, [Maddison et al.
(2016)](https://arxiv.org/abs/1611.00712) use the temperature-valued
$\text{Softmax}$ as a continuous relaxation of the $\text{argmax}$ computation
such that

$$
\begin{align}
\text{Sampling }&z \sim \text{Cat} \left(\alpha_1, \dots, \alpha_N \right)
\text{ is relaxed cont. to } z_k = \frac {\exp \left( \frac {\log \alpha_k + G_k}{\lambda} \right)}
{\sum_{i=1}^N \exp \left( \frac {\log \alpha_i + G_i}{\lambda} \right)}\\
&\text{with } G_k \sim \text{Gumbel Distribution}\left(\mu=0, \beta=1 \right)
\end{align}
$$

where $\lambda \in [0, \infty[$ is the temperature and $\alpha_k \in [0,
\infty[$ are the categorical distribution parameters. The temperature can be
understood as a hyperparameter that controls the *sharpness* of the
$\text{softmax}$, i.e., how much the *winner-takes-all* dynamics of the
softmax is taken:

* $\lambda \rightarrow 0$: $\text{softmax}$ smoothly approaches discrete $\text{argmax}$
  computation
* $\lambda \rightarrow \infty$: $\text{softmax}$ leads to uniform distribution.

Note that the samples $z_k$ obtained by this reparameterization follow a new
family of distributions, the **Concrete distribution**. Thus, $z_k$ are called
**Concrete random variables**.

| <img width="900" height="400" src='/assets/img/concrete/Gumble_Softmax.png'> |
| :--:        |
| **Computational Graph Gumbel-Softmax Trick**. Taken from [Maddison et al. (2016)](https://arxiv.org/abs/1611.00712). |

**Intuition**: To better understand the relationship between the *Concrete*
distribution and the *discrete categorical* distribution, let's look on an
exemplary result. Remind that the $\text{argmax}$ operation for a
$n$-dimensional categorical distribution returns states on the vertices of the simplex

$$
\boldsymbol{\Delta}^{n-1} = \left\{ \textbf{x}\in \{0, 1\}^n \mid \sum_{k=1}^n x_k = 1 \right\}
$$

Concrete random variables are relaxed to return states in the interior of the
simplex

$$
\widetilde{\boldsymbol{\Delta}}^{n-1} = \left\{ \textbf{x}\in [0, 1]^n \mid \sum_{k=1}^n x_k = 1 \right\}
$$

The image below shows how the distribution of concrete random variables changes
for an exemplary discrete categorical distribution $(\alpha_1, \alpha_2,
\alpha_3) = (2, 0.5, 1)$ and different temperatures $\lambda$.

| <img width="900" height="400" src='/assets/img/concrete/simplex.png'> |
| :---        |
| **Relationship between Concrete and Discrete Variables**: A discrete distribution with unnormalized probabilities $(\alpha_1, \alpha_2, \alpha_3) = (2, 0.5, 1)$ and three corresponding **Concrete densities** at increasing temperatures $\lambda$.<br> Taken from [Maddison et al. (2016)](https://arxiv.org/abs/1611.00712). |

### Concrete Distribution

While the Gumbel-Softmax trick defines how to obtain samples from a **Concrete
distribution**, [Maddison et al. (2016)](https://arxiv.org/abs/1611.00712)
provide a definition of its density and prove some nice properties:

**Definition**: The **Concrete distribution** $\text{X} \sim
\text{Concrete}(\boldsymbol{\alpha}, \lambda)$ with temperature $\lambda \in
[0, \infty[$ and location $\boldsymbol{\alpha} =
\begin{bmatrix} \alpha_1 & \dots & \alpha_n \end{bmatrix} \in [0, \infty]^{n}$
has a density

$$
  p_{\boldsymbol{\alpha}, \lambda} (x) = (n-1)! \lambda^{n-1} \prod_{k=1}^n
  \left( \frac {\alpha_k x_k^{-\lambda - 1}} {\sum_{i=1}^n \alpha_i x_i^{-\lambda}} \right)
$$

**Nice Properties and their Implications**:

1. *Reparametrization*: Instead of sampling directly from the Concrete
   distribution, one can obtain samples by the following deterministic ($d$) reparametrization

   $$
   X_k \stackrel{d}{=} \frac {\exp \left( \frac {\log \alpha_k + G_k}{\lambda} \right)}
{\sum_{i=1}^N \exp \left( \frac {\log \alpha_i + G_i}{\lambda} \right)} \quad
   \text{with} \quad G_k \sim \text{Gumbel}(0, 1)
   $$

   This property ensures that we can easily compute unbiased low-variance
   gradients w.r.t. the location parameters $\boldsymbol{\alpha}$ of the
   Concrete distribution.

2. *Rounding*: Rounding a Concrete random variable results in the discrete
   random variable whose distribution is described by the logits $\log \alpha_k$

   $$
   P (\text{X}_k > \text{X}_i \text{ for } i\neq k) = \frac {\alpha_k}{\sum_{i=1}^n \alpha_i}
   $$

   This property again indicates the close relationship between concrete and discrete distributions.

3. *Convex eventually*:

   $$
   \text{If } \lambda \le \frac {1}{n-1}, \text{ then } p_{\boldsymbol{\alpha},
   \lambda} \text{ is log-convex in } x
   $$

   This property basically tells us if $\lambda$ is small enough, there are no modes in the
   interior of the probability simplex.



### Discrete VAE



## Implementation


## Acknowledgements

The lecture on [discrete latent
variables](https://www.youtube.com/watch?v=-KzvHc16HlM) by Artem Sobolev as well
as the [NIPS presentation](https://www.youtube.com/watch?v=JFgXEbgcT7g) by Eric
Jang were really helpful resources.
