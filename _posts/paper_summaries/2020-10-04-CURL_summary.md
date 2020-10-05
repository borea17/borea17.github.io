---
title: "CURL: Contrastive Unsupervised Representations for Reinforcement Learning"
permalink: "/paper_summaries/CURL"
author: "Markus Borea"
tags: ["unsupervised learning", "contrastive learning", "model-free RL"]
published: false
toc: true
toc_sticky: true
toc_label: "Table of Contents"
type: "paper summary"
---

NOTE: THIS IS CURRENTLY WIP

[Srinivas et al. (2020)](https://arxiv.org/abs/2004.04136) present a
novel framework of unsupervised (self-supervised) contrastive learning within
reinforcement learning (RL) algorithms that achieves state-of-the-art
results in terms of data-efficiency on pixel-based RL tasks. Motivated
by the recent successes of contrastive pre-training in computer
vision, the authors aim at using contrastive learning to build 
powerful representations in a latent space over which the agent can
learn a policy in less time. This approach is supported by their
hypothesis that control algorithms built on top of useful
semantic representations should be significantly more data-efficient.
Note that the implicit assumption is that useful state information
(e.g., physical state based features) can be extracted from raw pixel
data. Notably, their framework "**C**ontrastive **U**nsupervised
Represenations for **R**einforcement **L**earning" (**CURL**) adds
lightweight overhead in terms of architecture and model learning
compared to other approaches in sample-efficient pixel-based RL (e.g.,
[Dreamer](https://arxiv.org/abs/1912.01603)). As a proof of concept, 
they show that CURL enables model-free RL algorithms to outperform
state-of-the-art pixel-based (model-free and model-based) methods on
complex continuous control tasks (DeepMind Control Suite) and discrete
control tasks (Atari). 

<!-- Note that since no pixel-based -->
<!-- reconstruction losses are included, the bullet problem and the -->
<!-- noisy --> 
<!-- tv problem are circumvented. -->

## Model Description

The main idea of CURL is to combine contrastive learning with standard
model-free RL algorithms by adding a contrastive learning objective as
an auxiliary task to the training. Note that the quality of the
learned representations is highly dependent on the exact auxiliary loss
definition. However, a major advantage of CURL is that the
implementation overhead is quite small and that is does not
require customized hyperparameters (which often occur in
reconstruction-based methods). 

The figure below summarizes the basic design of the CURL architecture:
First, a batch of transitions is sampled from the replay buffer in
which each observation is a stack of temporally sequential frames
(typically 3 in DMControl, 4 in Atari). Secondly, the batch of observations
is data-augmented[^1] with two different augmentations into *query
observation* $o_q$ and *key observations* $o_k$. Thirdly, the
data-augmented versions are encoded into lower-dimensional embeddings 
query $q$ and key $k$ using the query and key encoder,
respectively. Lastly, the *queries* are passed into the RL algorithm as
lower dimensional representations of the high dimensional original
observations, while the *query-key* pairs are passed into the compuation
of the contrastive learning objective. Note that the key encoder
weights are the moving average of the query weights, thus these
parameters are not backpropagated. 
<!-- which essentially ensures matching of the -->
<!-- embeddings compared to the original observations.  -->

[^1]: Data augmentation alone might be a key reason for the success of
    CURL, see the [RAD paper](https://arxiv.org/pdf/2004.14990.pdf)
    that appeared shortly after CURL.

| ![CURL Architecture](/assets/img/06_CURL/CURL_architecture.png "CURL Architecture") |
| :--         |
| **CURL Architecture**.  Taken from [Srinivas et al. (2020)](https://arxiv.org/abs/2004.04136). |

<!-- - minimal changes to the architecture and training pipeline -->
<!-- - does not require any custom architectural choices of hyperparameters -->
<!-- => crucial for reproducible end-to-end training -->

### Contrastive Learning 

Momentum contrastive learning



- best understood as performing a dictionary lookup task wherin the
  postive and negatives represent a set of keys with respect to a
  query (or an anchor)

### Employed RL Algorithms

**Soft Actor Critic (SAC)**

**Rainbow DQN**




<!-- Motivation: -->
<!-- - data-efficiency => practicability -->
<!-- - sample-efficiency -->
<!-- - improving sample efficiency important both in simulation and the -->
<!--   real world (especially real world) -->

## Implementation


  
Two ideas to improve sample efficiency:
 1. auxiliary tasks on agents sensory observations
 2. world models that that predict future
 
CURL: use auxiliary tasks to improve sample efficiency

Hypothesis:
"If an agent learns a useful semantic representation from high
dimensional observations, control algorithms build on top of those
representations should be significantly more efficient"

- contrastive pre-training success in computer vision


------------------------------------------------------------------------
