# gauss-newton-mlp
MATLAB implementation of Newton’s (Gauss–Newton) algorithm for training a simple multilayer perceptron with sigmoid activation.
# Gauss–Newton MLP (MATLAB)

This repository demonstrates how to apply **Newton’s method (Gauss–Newton approximation with Levenberg–Marquardt damping)** to train a simple **multilayer perceptron (MLP)** with sigmoid activation functions in MATLAB.

---

## Overview

- Single-hidden-layer MLP with **3 hidden units** and **sigmoid activation**  
- Direct input-to-output connections included  
- Implements:
  - Forward pass
  - Jacobian computation
  - Gradient calculation
  - Gauss–Newton Hessian approximation
  - Damped Newton update (Levenberg–Marquardt style)

The goal is **educational**: to show how second-order optimization can be applied to neural networks, compared to traditional gradient descent.

---

## Mathematical Background

Cost function:

\[
E = \tfrac{1}{2} (t - y)^2
\]

Gradient:

\[
\nabla E = -J^\top (t - y)
\]

Gauss–Newton Hessian approximation:

\[
H \approx J^\top J
\]

Damped Newton update:

\[
\Delta w = - (H + \lambda I)^{-1} \nabla E
\]

where:
- \( J \) = Jacobian of network output wrt weights  
- \( \lambda \) = damping factor for stability  

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/rcaposova/gauss-newton-mlp.git
   cd gauss-newton-mlp
