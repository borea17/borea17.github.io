---
title: "Improving Sample Efficiency in Model-Free Reinforcement Learning from Images"
permalink: "/paper_summaries/SAC+AE"
author: "Markus Borea"
tags: ["model-free RL"]
published: false
toc: true
toc_sticky: true
toc_label: "Table of Contents"
type: "paper summary"
---

NOTE: THIS IS CURRENTLY WIP!

[Yarats et al. (2020)](https://arxiv.org/abs/1910.01741) 

successful variant termed **SAC+AE**

- challenge is to efficiently learn a mapping from pixels to an
  appropriate representation for control using only a sparse reward signal
- incorporating reconstruction loss into an off-pilicy learning
  algorithm often leads to training instability
- model-free methods on Atari and DeepMind Control (DMC) take tens of
  millions of steps
- natural solitions to improve sample efficiency are 
   - to use off-policy methods
   - add an auxiliary task with an unsupervised objective
      - simplest version: autoencoder with pixel reconstruction objective
      - 
      
      
- recommend a simple and effective autoencoder-based off-policy method
  that can be trained end-to-end
- first model-free off-policy approach to train latent state
  representation jointly with policy
- performance matches state-of-the-art model-based methods

## Model Description

* what is a $\rho_{\pi}$ ? known as occupancies

* see Ziebar et al. 2008

## Implementation
