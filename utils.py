#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np


Tensor = torch.Tensor

def rel_entropy(p: Tensor, q: Tensor) -> Tensor:
    """Computes the relative entropy between probability tensors p and q."""
    return torch.where(p == torch.tensor(0.), torch.tensor(0.), p*p.log()-p*q.log())

    
def jensenshannon(p: Tensor, q: Tensor, base=None, *, dim=0) -> float:
    """
    Compute the Jensen-Shannon distance (metric) between
    two probability tensors. This is the square root
    of the Jensen-Shannon divergence.
    
    The Jensen-Shannon distance between two probability
    vectors `p` and `q` is defined as,
    .. math::
       \\sqrt{\\frac{D(p \\parallel m) + D(q \\parallel m)}{2}}
    where :math:`m` is the pointwise mean of :math:`p` and :math:`q`
    and :math:`D` is the Kullback-Leibler divergence.

    This routine will normalize `p` and `q` if they don't sum to 1.0.
    """
    p /= torch.sum(p, dim=dim)
    q /= torch.sum(q, dim=dim)
    m = (p + q) / 2.0
    left = rel_entropy(p.double(), m)
    right = rel_entropy(q.double(), m)
    left_sum = torch.sum(left, dim=dim)
    right_sum = torch.sum(right, dim=dim)
    js = left_sum + right_sum
    if base is not None:
        js /= base.log()
    return torch.sqrt(js / 2.0)