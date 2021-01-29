---
title: "What is the idea behind importance sampling?"
permalink: "/ML_101/probability_theory/importance_sampling"
author: "Markus Borea"
published: false
type: "101 probability"
---


## Derivation

$$
\mathbb{E}_{p(\textbf{x})} \Big[ f(\textbf{x}) \Big] = \int f(\textbf{x}) p(\textbf{x})
d\textbf{x} \stackrel{\forall q(\textbf{x})=0: p(\textbf{x}) = 0}{=}
\int f(\textbf{x}) \frac {p(\textbf{x})}{q(\textbf{x})} q(\textbf{x})
d\textbf{x}
= \mathbb{E}_{q(\textbf{x})} \Big[ f(\textbf{x}) \frac {p(\textbf{x})}{q(\textbf{x})} \Big]
$$

q: proposal distribution
p: target distribution
