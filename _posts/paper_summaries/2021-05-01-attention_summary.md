---
title: "Attention Is All You Need"
permalink: "/paper_summaries/attention_is_all_you_need"
author: "Markus Borea"
tags: [attention, transformer]
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

In essence, an **attention mechanism** can be intuitively understood as a means to assign individual
importance (or rather *attention*) to each entity in a collection of entities (e.g., words in a
sentence or pixels in an image) using some cues as input. Mathmatically, this translates into
**computing a weighted average over all entities in which the attention weights are obtained from
the attention cues**. 

More abstractly, the attention mechanism can be used to answer the following questions

* What entities (e.g., pixels or words) should we attend to or focus on?
* What entities (e.g., pixels or words) are relevant for the task at hand?

[Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) call their particular attention mechanism
**Scaled Dot-Product Attention**. Therein, the collection of entities is termed **values** and the
attention cues are termed **queries** and **keys**. Attention to particular values (entities) is
obtained by computing the weighted average over all values (entities) in which the **attention
weights** are obtained by combining the attention cues.

The attention cues (queries and keys) are vectors of length $d_k$ defined per value and can
be seen as two different answers to the same question: *How much attention should we put to this
entity?* The **alignment** between the attention cues is computed via the dot-product (hence the
name), additionally the **alignment scores** are passed through a *Softmax*-layer to obtain
normalized **attention weights**. Finally, these attention weights are used to compute the weighted
average. 

To speed things up, queries, keys and values are packed into matrices $\textbf{Q}, \textbf{K}
\in \mathbb{R}^{N_v \times d_k}$ and $\textbf{V} \in \mathbb{R}^{N_v \times d_v}$,
respectively. As a result, the concise formulation of Scaled Dot-Product Attention is given by 

$$
\text{Attention}(\textbf{Q}, \textbf{K}, \textbf{V}) = 
\underbrace{\text{softmax} 
	\left(
	%\overbrace{
	\frac {\textbf{Q} \textbf{K}^{\text{T}}} {\sqrt{d_k}}
	%}^{\text{attention alignment }  \textbf{L} \in }
	\right)
}_{\text{attention weight }\textbf{W} \in \mathbb{R}^{N_v \times N_v}} \textbf{V}
$$

Note that 

| ![Matrix Packing for Scaled Dot-Product Attention](/assets/paper_summaries/010_attention/img/matrix_packing.png "Matrix Packing for Dot-Product Attention") |
| :--         |
| **Matrix Packing for Scaled Dot-Product Attention**:<br> Given an example sentence, the associated values $v_i$ may be obtained by some *word2vec* model. The attention cue vectors $k_i$ and $q_i$ |

> Example IMAGE


## Implementation
