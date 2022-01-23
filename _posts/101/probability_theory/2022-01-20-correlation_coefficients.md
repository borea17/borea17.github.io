---
title: "What is the difference between Pearson and Spearman correlation?"
permalink: "/ML_101/probability_theory/pearson_vs_spearman_correlation"
author: "Markus Borea"
published: false
type: "101 probability"
---

**Pearson's correlation coefficient** measures the **linear relationship** whereas **Spearman's
(rank) correleation coeffient** assesses the **mononotic relationship** between two
variables. **Each coefficient has its legitmation** depending on the use-case.

<!-- E.g., if there is a clear directional non-linear relationship, we would often prefer Spearman over Pearson. On the other hand, if the data at hand rather looks like a noisy two-dimensional line, Pearson's coefficient is more adequate.  -->

$$
\fbox{
$\textbf{Pearson's correlation coefficient } \hspace{0.5cm}
r \left( \textbf{X}, \textbf{Y} \right)
= \frac {\text{Cov} \left( \textbf{X}, \textbf{Y}\right)} {\sqrt{\text{Var}
\left(\textbf{X}\right)} \sqrt{\text{Var}\left(\textbf{Y}\right)}}
$}
$$

$$
\fbox{
$\textbf{Spearman's correlation coefficient } \hspace{0.5cm}
r_s = 
$
}
$$

## Explanation

### Pearson Correlation Coefficient

**Covariance** is a **measure of the joint variability** within two variables, i.e., it tells us how
strongly two variables vary together. Mathmatically, it is defined as follows

$$
\begin{align}
\text{Cov} \left( \textbf{X}, \textbf{Y} \right) 
&= 
\mathbb{E} \Big[ 
\left(\textbf{X} - \mathbb{E}\big[\textbf{X}\big] \right)
\left(\textbf{Y} - \mathbb{E}\big[\textbf{Y}\big] \right)
\Big]\\
& \stackrel{\text{sample}}{=} \frac {1} {N}
\sum_{i=1}^N \left( x_i - {\mu}_{\textbf{X}} \right) 
\left( y_i - {\mu}_{\textbf{Y}} \right)
\end{align}
$$

So *what does that mean in plain words*? Suppose that $\text{Cov}\left(\textbf{X},
\textbf{Y}\right) > 0$: Then, it simply means that a greater than average value for one variable
($x_i > \mu_{\textbf{X}})$ is expected to be associated with a greater than average 
value for the other variable ($y_i > \mu_{\textbf{Y}})$. Thus, **covariance encodes information
about the direction of the joint distribution**. 

There is no upper or lower bound for the covariance (which we can directly see by scaling one
variable). The magnitude of the covariance informs us about the (expected) shape of the
distribution.

Expectation is a linear operation, thus it follows that the **covariance gives some estimate about
the linear dependence between two random variables**.

**Correlation**, also known as **Pearson's correlation coeffecient**, is the **normalized version of
the covariance**, therefore measures by its magnitude the strength of the linear correlation between
$\textbf{X}$ and $\textbf{Y}$.


| ![Correlation Examples](/assets/img/ml_101/Correlation_examples.png "Correlation Examples") |
| :--- |
| **Pearson Correlation Coefficient Examples**. Taken from [wikipedia](https://en.wikipedia.org/wiki/Correlation#/media/File:Correlation_examples2.svg).<br> Note `the figure in the center has a slope of 0 but in that case the correlation coefficient is undefined because the variance of Y is zero`.|


>
> **Useful Properties**:
>
> * covariance is **independent under shifting of the variables**, i.e.,
>
>$$
>\text{Cov} \left( \textbf{X}, \textbf{Y} \right) 
>= 
>\text{Cov} \left( \textbf{X} + a, \textbf{Y} + b\right) 
>$$
>
> * **linear combinations of covariances can be facored out**, i.e., 
>
>$$
>\text{Cov} \left( a\textbf{X} + \textbf{Y}, \textbf{Z} \right) 
>= a\text{Cov} \left( \textbf{X}, \textbf{Z} \right) +
>\text{Cov} \left( \textbf{Y}, \textbf{Z} \right) 
>$$
>


### Spearman Correlation Coefficient



