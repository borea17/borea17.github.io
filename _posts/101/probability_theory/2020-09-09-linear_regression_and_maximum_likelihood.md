---
title: "How is linear regression connected to maximum likelihood?"
permalink: "/ML_101/probability_theory/linear_regression_and_maximum_likelihood"
author: "Markus Borea"
published: false
type: "101 probability"
---

Under the assumption that the data is generated from a linear function
where each observation exhibts i.i.d. Gaussian noise (i.e., each noise variable $\nu_i$ is
sampled independently from the same/identical Gaussian distribution), 
the **negative log-likelihood** is proportional to the squared sum of
the residuals. Thus, **minimizing the negative log-likelihood
(maximizing the log-likelihood) leads to the same solution as the
least squares method**. 


<!-- with i.i.d. Gaussian noise (i.e., each noise variable $\mu_i$ is -->
<!-- sampled independently from the same/identical  $\mu \sim \mathcal{N}(0, \sigma^2)$) -->
<!-- independent and identical ($\sigma$ constant) Gaussian noise ($\nu \sim \mathcal{N} (0, \sigma^2)$) -->
<!-- the **negative log-likelihood** is proportional to the squared sum of the residuals, i.e., -->
<!-- **minimizing the negative log-likelihood (maximizing the log-likelihood) leads to the same solution -->
<!-- as the least squares method**. -->


**Derivation:**  
Given a dataset $\mathcal{D} = \{(\textbf{x}_1, y_1), \dots,
(\textbf{x}_N, y_N)\}$ and the **model assumption**:
                                 
$$
    y_i = \textbf{w}^{\text{T}} \textbf{x}_i + \beta + \nu_i \quad \quad \nu \sim \mathcal{N}(0, \sigma^2)
$$

the parameters of the model are $\boldsymbol{\theta} = (\textbf{w}, \beta,
\sigma^{2})$. In terms of **maximum likelihood**, we want to maximize:

$$
\begin{align}
p(\mathcal{D} | \boldsymbol{\theta}) &=
p (\textbf{x}_1) \cdot p(y_1| \textbf{x}_1; \boldsymbol{\theta}) \cdot ... \cdot p(\textbf{x}_N| \textbf{x}_1, \dots,
\textbf{x}_{N-1}) \cdot p(y_N| \textbf{x}_N; \boldsymbol{\theta})\\&=
p_X (\textbf{x}_1, \dots, \textbf{x}_N) \prod_{i=1}^N p(y_i | \textbf{x}_i; \boldsymbol{\theta})
\end{align}
$$

where the density $p_X$ is unknown and independent of $\boldsymbol{\theta}$. Given our model assumption, we can write

$$
p(y_i|\textbf{x}_i, \boldsymbol{\theta})  \sim
\mathcal{N} \left(\textbf{w}^{\text{T}} \textbf{x}_i + \beta, \sigma^2  \right) = \frac {1}{\sqrt{2\pi \sigma^2}}
\exp\left( - \frac {(y_i - \textbf{w}^{\text{T}} \textbf{x}_i - \beta)^2} {\sigma^2} \right)
$$

Thus, we get for the **negative log-likelihood**

$$
-\ln p(\mathcal{D}|\theta) = \sum_{i=1}^N \frac {1}{2} \log \big(2 \pi \sigma^2 \big) + \frac {1}{2\sigma^2}
\big(y_i - \textbf{w}^{\text{T}} \textbf{x}_i - \beta \big)^2 \propto
\sum_{i=1}^N \big(y_i - \textbf{w}^{\text{T}} \textbf{x}_i - \beta \big)^2
$$

As stated above, minimizing the negative log-likelihood (left side)
leads to the same solution as minimizing the sum of the squares of the
residuals (right side), i.e., the method of least squares. For a more detailed derivation, see
[d2l.ai](http://www.d2l.ai/chapter_linear-networks/linear-regression.html).
Note that in linear regression, we do not intent to estimate
$\sigma^2$. However, it may be helpful since a large variance
indicates that our linear regression model will produce highly biased
estimates. 
