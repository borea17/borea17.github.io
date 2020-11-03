---
title: "How can the average loss be calculated with batch means?"
permalink: "/ML_101/probability_theory/average_loss_calculation"
author: "Markus Borea"
published: true
type: "101 engineering"
---

Consider the following case, where we have a dataset $\textbf{X} =
\\{ \textbf{x}_i \\}\_{i=1}^N$ consisting of $N$ samples
$\textbf{x}_i$. Instead of iterating over each sample independently,
we decide stream minibatches of size $b$ (as this often leads to a
stabilized training). Now the question arises, what is the average
loss? E.g., can we simply take the average of the batch losses to
compute the average of the epoch ? 

Let's dive into it:  

Just imagine, we wouldn't take minibatches, but take each sample
separately and compute corresponding losses $l_i$. In this case, it is
fairly obvious that the average loss per sample can be computed as
follows:

$$
 \text{Average Loss} = \frac {1}{N} \sum_{i=1}^{N} l_i 
$$

Batch losses $\tilde{l}_k$ are just means over the current batch $k$, i.e., 

$$
\tilde{l}_k = \frac {1}{b} \sum_{j=1}^{b} l_{k, j}
$$

Let's assume that each batch is written in a different column $k$ and
that all batch sizes are equal, in other words, that $N$ is divisible
by $b$.


So the question comes down to, how do we have to sum all $\tilde{l}_k$ to
compute the average loss? And the answer is simple:

$$
\frac {1}{N} \sum_{i=1}^N l_i =\frac {1}{N} \sum_{k=1}^{N/b} b \cdot
\tilde{l}_k = \frac{b}{N} \sum_{k=1}^{N/b} \tilde{l}_k = 
\sum_{k=1}^{N/b}  \frac {\tilde{l}_k}{\text{num batches}},
$$

where $\frac {b}{N}$ is the inverse of the number of batches
$\text{num batches}$. In Pytorch, the number of batches can easily
retrieved as the length of the dataloader ;).
