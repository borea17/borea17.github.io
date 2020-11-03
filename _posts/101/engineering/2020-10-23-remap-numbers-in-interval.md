---
title: "How to map numbers in an interval $[a, b]$ onto another interval $[c, d]$ ?"
permalink: "/ML_101/probability_theory/remap-numbers-in-interval"
author: "Markus Borea"
published: true
type: "101 engineering"
---

**Short Answer**:

Use the following function for each value $x \in [a, b]$

$$
  f(x) = c + \frac {d - c} {b -a} \left(x - a\right).
$$

Note that $f(x)$ is a bijektiv function, i.e., there is a one-to-one
correspondence between both intervals (even if the ranges are
different).

**Long Answer**:

Let's derive the above formula which can be done in two steps:

1. *Shift the interval* $[a, b]$ *into the unit interval* $[0, 1]$.  

   Therefore, we start by shifting the lower bound $a$ to $0$ using
   the following formula: 
   
    $$
    f_{a\rightarrow 0} (x) = x - a
    $$

    which results in the new interval $[0, b-a]$. Now we can
    simply scale each value by the inverse of $b-a$ to shift the
    interval $[0, b-a]$ into the unit interval $[0, 1]$, i.e.,

    $$
    f_{\text{unit}} = \frac {1}{b-a} \left(x-a\right)
    $$
 
2. *Shit the unit interval* $[0, 1]$ *into the target interval* $[c, d]$.

   Actually, we are now just doing the reverse process by firstly
   adding rescaling each value by the desired length $d-c$, i.e.,
   
   $$
    f_{\text{new length}} = \frac {d- c}{b-a} \left(x - a \right)
   $$
   
   which results in the new interval $[0, d-c]$. Finally, we shift the
   interval towards the desired lower by simply adding $c$ to get the
   final equation 
   
   $$
      f(x) = c + \frac {d - c} {b -a} \left(x - a\right).
   $$


## Acknowledgement

Based on this [answer](https://math.stackexchange.com/a/914843/612615).
