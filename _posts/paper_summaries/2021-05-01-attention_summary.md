---
title: "Attention Is All You Need"
permalink: "/paper_summaries/attention_is_all_you_need"
author: "Markus Borea"
tags: [attention]
toc: true
toc_sticky: true
toc_label: "Table of Contents"
published: false
type: "paper summary"
---

NOTE: THIS IS CURRENTLY WIP!

[Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)

RNN problems:
. long range dependencies, information flow (path length), backpropagation


## Model Description

### Attention Mechanism

The **attention mechanism** can be intuitively understood as a means to 
assign importance to each entity in a collection of entities (e.g., words in a sentence or pixels in an image). 

what part should we focus on?
what is relevant?

$$
\text{Attention}(\textbf{Q}, \textbf{K}, \textbf{V}) = \text{softmax} \left( 
\frac {\textbf{Q}, \textbf{K}^{\text{T}}} {\sqrt{d_k}} 
\right) \textbf{V}
$$

output: query
input: key
releance: dot product

compatibility score/ relevance score

https://www.youtube.com/watch?v=TQQlZhbC5ps
https://www.tensorflow.org/text/tutorials/transformer

## Implementation
