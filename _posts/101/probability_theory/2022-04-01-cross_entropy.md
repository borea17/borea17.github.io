---
title: "Why do we use the cross entropy loss in classification?"
permalink: "/ML_101/probability_theory/cross_entropy_loss"
author: "Markus Borea"
published: true
type: "101 probability"
---

The cross entropy loss is commonly used during classification, since the **minimization of the cross
entropy corresponds to the maximum likelihood estimator**.

## Derivation - Classification Task

We assume that we are given a classification task with a data set of $N$ observations $
\Big(\boldsymbol{x}^{(i)}, y^{(i)} \Big)$, where $\textbf{x}^{(i)}\in\mathbb{R}^d$
denotes some $d$-dimensional feature vector and $y^{(i)} \in \\{0, 1, \dots, C\\}$ its corresponding
label. 

### Goal

The goal of classification is to predict the probability of each class given a feature
vector. Thus, we actually want to infer the conditional probability distribution
$p(y|\boldsymbol{x})$. Note that we do not have access to this distribution, but only to samples $y^{(i)}$ from
it. 

### Generative Process

We assume that each tuple $\big( \textbf{x}^{(i)}, y^{(i)} \big)$ is generated in the following procedure:

1. **Prior Distribution** (Features): $\boldsymbol{x} \sim p(\boldsymbol{x})$ with $\boldsymbol{x}
   \in \mathbb{R}^d$  
   A feature vector $\boldsymbol{x}^{(i)}$ is sampled from some prior distribution $p(\boldsymbol{x})$.
2. **Conditional Distribution** (Label): $y \sim p(y|\boldsymbol{x}) = \text{Cat} \Big(
   \boldsymbol{\beta}(\textbf{x}) \Big)$ with $y \in \\{0, 1, \dots, C\\}$  
   The associated label $y^{(i)}$ is obtained via sampling from the class distribution
   $p(y|\boldsymbol{x}^{(i)})$. 
   
### Function Approximator

For the following section, we assume that we have some function approximator
$q (y|\boldsymbol{x}; \boldsymbol{\theta})$ that aims to be as close as possible to the true class
distribution $p(y|\boldsymbol{x})$ and in which $\boldsymbol{\theta}$ denote the parameters of our model (e.g., weights in a neural network). 

### Maximum Likelihood Estimator

The idea of the *maximum likelihood estimator* is to find the optimal model parameters
$\hat{\boldsymbol{\theta}}$ by maximizing a *likelihood function* $\mathcal{L}(\boldsymbol{\theta} |
\textbf{D})$ of the model parameters over the data set

$$
\hat{\boldsymbol{\theta}} = \arg \max_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta} | \textbf{D}),
$$

where $\hat{\boldsymbol{\theta}}$ is commonly referred to as *maxmimum likelihood estimator*. The
likelihood function estimates for each specific $\boldsymbol{\theta}$ how likely it is that our data
set $\mathcal{D}$ has been generated with these model parameters. As a result, the maximum
likelihood estimator denotes the model parameters under which the observed data is most probable
given our assumed statistical model.

Note that the likelihood is not a probability of the model parameters, i.e, 

$$
\int \mathcal{L} (\boldsymbol{\theta} | \mathcal{D}) d
\boldsymbol{\theta} \neq 1
$$

and thus needs to be distinguished from the posterior of the model parameters
$p(\boldsymbol{\theta} | \mathcal{D})$.

We divide the likelihood function into a product of likelihoods using the i.i.d. assumption, i.e., 

$$
\hat{\boldsymbol{\theta}} = \arg \max_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta} |
\textbf{D}) \stackrel{i.i.d.}{=} \arg \max_{\boldsymbol{\theta}} \prod_{i=1}^N \mathcal{l} \left(\boldsymbol{\theta} | \big(\boldsymbol{x}^{(i)}, y^{(i)}\big) \right),
$$

where i.i.d. denotes that our observations are independently and identically drawn from the same
joint distribution $p(\textbf{x}, y)$.

### Likelihood as the Probability Mass Function

Now the question arises, how do we compute each term $\mathcal{l} \left(\boldsymbol{\theta} |
\big(\boldsymbol{x}^{(i)}, y^{(i)}\big) \right)$ if we are given a specific model parameter
$\boldsymbol{\theta}$? 

Since we assume that the true class distribution $p(y|\boldsymbol{x})$ follows a Categorical
distribution which is a discrete distribution, we can simply use its probability mass function.

>The **probability mass function** of a **Categorical distribution** can be stated as follows 
>
>$$
>f(y | \boldsymbol{p} ) = \prod_{i=0}^C p_i^{1_{y==i}},
>$$
>
>where $\boldsymbol{p}= \begin{bmatrix} p_0 & \dots & p_C \end{bmatrix}$, $p_i$ represents the
>probability of seeing $i$ and $1_{x==i}$ denotes the indicator function.

As a result, each term can be written as follows

$$
\mathcal{l} \left(\boldsymbol{\theta} | \big(\boldsymbol{x}^{(i)}, y^{(i)}\big) \right) =
\prod_{k=1}^C \left(q (y=k|\boldsymbol{x}^{(i)}; \boldsymbol{\theta}) \right)^{1_{y^{(i)} == k}}
$$


### Negative Log Likelihood 

We can convert the product into a sum by applying the logarithm

$$
\log \mathcal{L}(\boldsymbol{\theta} | \textbf{D}) = \log \left( \prod_{i=1}^N  \mathcal{l}
\left(\boldsymbol{\theta} | \big(\boldsymbol{x}^{(i)}, y^{(i)}\big) \right)  \right) = \sum_{i=1}^N
\log \mathcal{l} \left(\boldsymbol{\theta} | \big(\boldsymbol{x}^{(i)}, y^{(i)}\big) \right),
$$

where $\log \mathcal{l} (\cdot)$ is commonly referred to as *log likelihood*. Note that *log* is a
strictly concave function and thus does not change the maximum likelihood estimator.

Additionally, we turn the maximization problem into a minimzation problem 

$$
\hat{\boldsymbol{\theta}} = \arg \max_{\boldsymbol{\theta}} \sum_{i=1}^N \log \mathcal{l}
\left(\boldsymbol{\theta} | \big(\boldsymbol{x}^{(i)}, y^{(i)}\big) \right) = 
\arg \min_{\boldsymbol{\theta}} \sum_{i=1}^N - \log \mathcal{l} \left(\boldsymbol{\theta} | \big(\boldsymbol{x}^{(i)}, y^{(i)}\big) \right)
,
$$

where $-\log \mathcal{l} (\cdot)$ is commonly referred to as *negative log likelihood*.

### Cross Entropy as Negative Log Likelihood

Lastly, we show that the negeative log likelihood equals the cross entropy

$$
- \log \mathcal{l} \left(\boldsymbol{\theta} | \big(\boldsymbol{x}^{(i)}, y^{(i)}\big) \right) 
= \sum_{k=1}^C  1_{y^{(i)} == k} \cdot \log q (y=k|\boldsymbol{x}^{(i)}; \boldsymbol{\theta})
$$

We can interpret/convert the true labels as one-hot vectors such that we can associate a label
distribution probability $\boldsymbol{h}^{(i)}$ per label $y^{(i)}$,

$$
h^{(i)}_k = \begin{cases}1 & \text{if } i\text{th sample belongs to }k \text{th class} \\ 0
& \text{else.} \end{cases} 
$$

Then, the negative log likelihood can be rewritten into the cross entropy $H$ between the label
distribution $\boldsymbol{h}$ and the estimated (from the model) probability distribution $\boldsymbol{q}$

$$
- \log \mathcal{l} \left(\boldsymbol{\theta} | \big(\boldsymbol{x}^{(i)}, y^{(i)}\big) \right) 
= - \sum_{k=1}^C h^{(i)}_k \cdot \log q_k (\boldsymbol{x}^{(i)} ; \boldsymbol{\theta}) =
H(\boldsymbol{h}^{(i)}, \boldsymbol{q}^{(i)}_\boldsymbol{\theta} ) = H^{(i)}
$$

The likelihood estimator for the whole dataset can be concisely written as follows

$$
\hat{\boldsymbol{\theta}} = \arg \max_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta} |
\textbf{D}) = \arg \min_{\boldsymbol{\theta}} \sum_{i=1}^N H^{(i)}
.
$$




