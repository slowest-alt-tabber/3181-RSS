COPYRIGHT WARNING: Copyright in these original lectures is owned by Monash University. You may transcribe, take notes, download or stream lectures for the purpose of your research and study only. If used for any other purpose, (excluding exceptions in the Copyright Act 1969 (Cth)) the University may take legal action for infringement of copyright.

Do not share, redistribute, or upload the lecture to a third party without a written permission!

### FIT3181/5215 Deep Learning

Week 04: Back-Propagation and Optimization for Deep Learning

CE/Lecturer: Dr Trung Le (Clayton)

Lecturer: Dr Binh Nguyen (Clayton) Lecturer: Dr Ting Fung Fung (Malaysia) Lecturer: Dr Arghya Pal (Malaysia) Lecturer: Dr Sailaja Rajanala (Malaysia)

Email: trunglm@monash.edu

Department of Data Science and AI Faculty of Information Technology, Monash University, Australia

2

# Outline

###  Revision of calculus.

###  Gradient descent and stochastic gradient descent.

###  Backpropagation in feed-forward neural networks.

###  Optimizers for deep learning.

###  Further reading recommendations

o Deep Learning – Chapter 8 o Dive into Deep Learning – Chapter 11 (https://d2l.ai/chapter_optimization/index.html) o Ruder’s blog: https://ruder.io/optimizing-gradient-descent/index.html

3 Declare loss and optimizer

# Our Story

Train the model Back-propagation Optimizers

Since 1986 Declare the model

Small detour to calculus loss.backward() optimizer.step() Loss landscape of DL models

DL models (FFN, CNN, RNN, ViT,...) Back-propagation Backprop Optimizers for DL

𝜕𝑙 Update Compute gradients 𝑊 = 𝑊 − 𝜂

𝜕𝑊 𝜕𝑙 𝜕𝑙

𝜕𝑙,

𝑏 = 𝑏 − 𝜂 𝜕𝑊 𝜕𝑏

𝜕𝑏

## Small Detour to Calculus

## A small detour to calculus

###  Calculus = mathematics of change (very important for deep learning)

###  Properties of derivative:

𝑓 𝑥+ℎ −𝑓(𝑥) ′ o 𝑓 𝑥 = ∇𝑓(𝑥) = lim

ℎ→0 ℎ ′ ′ o 𝑢𝑣 = 𝑢 𝑣 + 𝑢𝑣′

′ ′ 𝑢 𝑢 𝑣−𝑢𝑣′ o =

2 𝑣 𝑣 𝑢 𝑢 o 𝑒 ′ = 𝑢′𝑒

′ 𝑢 ′ o log 𝑢 =

𝑢

𝑛

###  Multi-variate function 𝑓: ℝ → ℝ with 𝑦 = 𝑓 𝑥 = 𝑓(𝑥, … , 𝑥).

1 𝑛 𝜕𝑓 o Gradient/derivative: (𝑎) = ∇𝑥𝑓(𝑎) = [∇𝑥 𝑓 𝑎 , ∇𝑥 𝑓 𝑎 , … , ∇𝑥 𝑓 𝑎 ].

𝜕𝑥 1 2 𝑛

###  Chain rule :

𝜕𝑢 𝜕𝑢 𝜕𝑣 o = ×

𝜕𝑥 𝜕𝑣 𝜕𝑥

5

6

## Derivative for multi-variate functions

3 2 ℝ ℝ

𝑥 = 1 1 1

𝑚 𝜕𝑓1 𝜕𝑓1 𝜕𝑓1 𝑚 𝑛 (𝑎) … … (𝑎) … (𝑎)

###  Given a function 𝑓: ℝ → ℝ

𝜕𝑥1 𝜕𝑥𝑗 𝜕𝑥𝑚

… … … … … … 𝑚 o 𝑓 𝑥 = (𝑓 𝑥 , … , 𝑓 (𝑥)) where 𝑓, … , 𝑓: ℝ →

1 𝑛 1 𝑛

… … … … … … ℝ and 𝑥 = (𝑥, … , 𝑥). Let denote 𝑦 = 𝑓(𝑥). 𝜕𝑦

1 𝑚 𝜕𝑓 𝜕𝑓 𝜕𝑓

(𝑎) = ∇𝑓(𝑎) = 𝑖 𝑖 𝑖

(𝑎) … … (𝑎) … (𝑎) 𝑛 𝒎 𝜕𝑥 o The derivative of 𝑓 at the point 𝒂 ∈ ℝ, denoted by 𝜕𝑥 𝜕𝑥 𝜕𝑥

1 𝑗 𝑚 𝜕𝑦 … … … … … … ∇𝑓(𝑎) (function related notion) or (𝑎) (variable

𝜕𝑥 𝜕𝑓𝑛 𝜕𝑓𝑛 𝜕𝑓𝑛

(𝑎) … … (𝑎) … (𝑎) related notion) is a matrix 𝑛 by 𝑚 (i.e., the Jacobian 𝜕𝑥 𝜕𝑥 𝜕𝑥

1 𝑗 𝑚 matrix).

Jacobian matrix

###  Chain rule

𝒎 𝒏 𝒏 𝒑 𝒎 𝒑 o Given a function 𝒇: ℝ → ℝ, 𝐠: ℝ → ℝ, denote 𝒉 = 𝒈 ∘ 𝒇: ℝ → ℝ, meaning that 𝒉 𝒙 =

𝒈 𝒇 𝒙 . We further define 𝑦 = 𝑓(𝑥) and 𝑧 = 𝑔 𝑦 = 𝑔 𝑓 𝑥 = ℎ(𝑥).

𝜕𝑧 𝜕𝑧 𝜕𝑦 𝑚 o For x ∈ ℝ,∇ℎ 𝑥 = ∇𝑔 𝑓 𝑥 × ∇𝑓 𝑥 or equivalently = . .

𝜕𝑥 𝜕𝑦 𝜕𝑥 𝑝 × 𝑚 𝑝 × 𝑛 𝑛 × 𝑚

𝑥 𝑦 = 𝑓(𝑥) 𝑧 = 𝑔 𝑦 = 𝑔 𝑓 𝑥 = ℎ(𝑥)

𝑓 𝑔 𝑚 𝑛 𝑝 ℝ ℝ ℝ

ℎ

7

## Example

2 2 2 2

###  𝑦 = 𝑓 𝑥 = 𝑓 𝑥, 𝑥, 𝑥 = 𝑥 + 𝑥, 𝑥 + 𝑥 𝑥

1 2 3 1 2 2 3 2 3 2 o 𝑓: ℝ → ℝ

2 2 o 𝑓1 𝑥 = 𝑓1 𝑥1, 𝑥2, 𝑥3 = 𝑥 + 𝑥

1 2 2 2 o 𝑓2 𝑥 = 𝑓2 𝑥1, 𝑥2, 𝑥3 = 𝑥 + 𝑥 𝑥2

2 3 𝜕𝑦

2×3 o = ∇𝑓 ∈ ℝ

𝜕𝑥

𝜕𝑓1 𝜕𝑓1 𝜕𝑓1 𝜕𝑦 𝜕𝑥 𝜕𝑥 𝜕𝑥 2𝑥1 2𝑥2 0

1 2 3 = ∇𝑥𝑓 = =

2 𝜕𝑥 𝜕𝑓2 𝜕𝑓2 𝜕𝑓2 0 2𝑥 + 𝑥 2𝑥 𝑥

2 3 2 3 𝜕𝑥1 𝜕𝑥2 𝜕𝑥3

𝒉𝟏

## Example 𝒆

Penultimate Output Prediction Probabilities 𝒑 =

𝟏 𝒉 𝒉 𝒉 𝒉

𝒆 𝟏 + 𝒆 𝟐 + 𝒆 𝟑 + 𝒆 𝟒 Layer Layer 𝐿

𝒑 𝒙 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) Output layer

𝒉𝟐 𝒉𝟏 𝒑𝟏 𝒆

𝒑𝟐 = 𝒉 𝒉𝟏 𝒉𝟐 𝒉𝟑 𝒉𝟒 𝟐 𝒑 𝒆 + 𝒆 + 𝒆 + 𝒆

𝟐 softmax loss 𝑙 = 𝑙𝑜𝑠𝑠 = − log 𝑝2 1×3 𝒉𝟑

𝒑𝟑 ℎ 𝒆 𝑥 = 𝑥 𝑥 𝑥 ∈ ℝ 𝑒 2

1 2 3 𝒉

𝟑 𝒑 =

=− 𝑙𝑜𝑔 𝟑 𝒉 𝒉 𝒉 𝒉

ℎ ℎ ℎ ℎ 𝒆 𝟏 + 𝒆 𝟐 + 𝒆 𝟑 + 𝒆 𝟒 𝑦 = 2 𝒑 𝑒 1+𝑒 2+𝑒 3+𝑒 4

𝒉𝟒 𝟒

𝒉𝟒 𝒆 Logit ℎ = ℎ ℎ ℎ ℎ Prob p = 𝑝 𝑝 𝑝 𝑝 𝒑𝟒 =

1 2 3 4 1 2 3 4 𝒉 𝒉 𝒉 𝒉

𝒆 𝟏 + 𝒆 𝟐 + 𝒆 𝟑 + 𝒆 𝟒 𝜕𝑙 Compute ?

𝜕𝒉 𝑢 ℎ2 𝑒

ℎ1 ℎ2 ℎ3 ℎ4 ❑ 𝑙 = − 𝑙𝑜𝑔 = log 𝑒 + 𝑒 + 𝑒 + 𝑒 − ℎ2

ℎ1 ℎ2 ℎ3 ℎ4 𝑒 +𝑒 +𝑒 +𝑒

ℎ1 𝜕𝑙 ∇ℎ 𝑢 𝑒

1 ❑ = = = 𝑝1

ℎ1 ℎ2 ℎ3 ℎ4 𝜕ℎ1 𝑢 𝑒 +𝑒 +𝑒 +𝑒

ℎ2 𝜕𝑙 ∇ℎ 𝑢 𝑒

2 ❑ = − 1 = − 1 = 𝑝2 − 1

ℎ1 ℎ2 ℎ3 ℎ4 𝜕ℎ2 𝑢 𝑒 +𝑒 +𝑒 +𝑒

ℎ3 𝜕𝑙 ∇ℎ 𝑢 𝑒

3 ❑ = = = 𝑝3

ℎ1 ℎ2 ℎ3 ℎ4 𝜕ℎ3 𝑢 𝑒 +𝑒 +𝑒 +𝑒

ℎ4 𝜕𝑙 ∇ℎ 𝑢 𝑒

4 ❑ = = = 𝑝4

ℎ1 ℎ2 ℎ3 ℎ4 𝜕ℎ4 𝑢 𝑒 +𝑒 +𝑒 +𝑒 𝜕𝑙 ❑ = 𝑝1, 𝑝2 − 1, 𝑝3, 𝑝4 = 𝑝1, 𝑝2, 𝑝3, 𝑝4 − 0,1,0,0 = 𝒑 − 𝟏𝟐 = 𝒑 − 𝟏𝒚

𝜕𝒉

8

9

# Example

### Intermediate layer

ത ത  ℎ = 𝑥𝑊 + 𝑏 and ℎ = 𝜎(ℎ)

ℎ ℎ ℎ3ℎ4 o ℎ = 𝜎(𝑥𝑊 + 𝑏) 1 2

ℎ = 𝜎(ℎത) o 𝜎 is the activation function 𝜕ℎ 𝜕ℎ 𝜕ℎഥ

′ ത 𝑇 4×3  = × = 𝑑𝑖𝑎𝑔 𝜎 ℎ 𝑊 ∈ ℝ

ഥ 𝜎 𝜎 𝜎 𝜎 𝜕𝑥 𝜕ℎ 𝜕𝑥

ℎത ℎത ℎത ℎത 𝜕ℎ1 𝜕ℎ1 𝜕ℎ1 𝜕ℎ1 1 2 3 4

ℎത = 𝑥𝑊 + 𝑏 𝜕 ℎത 𝜕 ℎത 𝜕 ℎത 𝜕 ℎത

1 2 3 4

𝜎′(ℎത) 0 0 0 𝜕ℎ 𝜕ℎ 𝜕ℎ 𝜕ℎ 1

2 2 2 2 ത ത ത ത 0 𝜎′(ℎത) 0 0 𝜕ℎ 𝜕 ℎ1 𝜕 ℎ2 𝜕 ℎ3 𝜕 ℎ4 2 3×4

ത 𝑊 ∈ ℝ  = = = 𝑑𝑖𝑎𝑔(𝜎′(ℎ))

𝜕 ℎത 𝜕ℎ 𝜕ℎ 𝜕ℎ 𝜕ℎ ത

3 3 3 3 1×4

0 0 𝜎′(ℎ3) 0 𝑏 ∈

ℝ 𝜕 ℎത 𝜕 ℎത 𝜕 ℎത 𝜕 ℎത

1 2 3 4

0 0 0 𝜎′(ℎത) 𝜕ℎ 𝜕ℎ 𝜕ℎ 𝜕ℎ 4

4 4 4 4 ത ത ത ത 1×3 𝜕 ℎ1 𝜕 ℎ2 𝜕 ℎ3 𝜕 ℎ4 𝑥 ∈ ℝ

𝜕 ℎത

𝑇  = 𝑊

𝜕𝑥

10

# How to code with PyTorch

ത ത  ℎ = 𝑥𝑊 + 𝑏 and ℎ = 𝑠𝑖𝑔𝑚𝑜𝑖𝑑(ℎ)

o ℎ = 𝑠𝑖𝑔𝑚𝑜𝑖𝑑(𝑥𝑊 + 𝑏) o 𝜎 = 𝑠𝑖𝑔𝑚𝑜𝑖𝑑 is the activation function 𝜕ℎ 𝜕ℎ 𝜕 ℎത

′ ത 𝑇 ത ത 𝑇 𝑇  = × = 𝑑𝑖𝑎𝑔 𝜎 ℎ 𝑊 = 𝑑𝑖𝑎𝑔 𝜎 ℎ ⊗ [1 − 𝜎 ℎ ] 𝑊 = 𝑑𝑖𝑎𝑔 ℎ ⊗ 1 − ℎ 𝑊

𝜕𝑥 𝜕 ℎത 𝜕𝑥 o ⊗ is element-wise product

Declare 𝑾, 𝒙, 𝒃 Forward propagation Backward propagation

## Loss Landscape of DL models

## Recall optimization problem in deep learning

Hidden layer Hidden layer

1 𝐿−1 ℎ (𝒙) ℎ (𝒙)

Output layer Input layer 𝐿 𝐿

ℎ (𝒙) 𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) 0 ℎ (𝒙) = 𝒙

𝑦ො = argmax1≤𝑚≤𝑀𝑝𝑚(𝑥) Ground- truth label

𝑙 𝑦, 𝑦ො = 𝐶𝐸 1𝑦, 𝑝(𝑥) = − log 𝑝𝑦(𝑥) 𝒚

SoftMax

Training set Loss function

𝑁 𝑁 𝐷 = { 𝑥1, 𝑦1 , … (𝑥𝑁, 𝑦𝑁)} 1 1

𝐿 𝐷; 𝜃 ≔ ෍ 𝐶𝐸 1𝑦, 𝑝 𝑥𝑖 = − ෍ log 𝑝𝑦 (𝑥𝑖)

𝑖 𝑖 𝑁 𝑁

𝑖=1 𝑖=1

𝐿 𝑙 𝑙  How to solve the optimization problem efficiently (𝜃 ≔ 𝑊, 𝑏)?

𝑙=1 𝐿 1 1 exp{ℎ𝑦 (𝑥𝑖)}

𝑁 𝑁 𝑖 o min 𝐿 𝐷; 𝜃 ≔ − σ log 𝑝 𝑥 = − σ log

𝑖=1 𝑦𝑖 𝑖 𝑖=1 𝑀 𝐿 𝜃 𝑁 𝑁 σ exp{ℎ (𝑥)}

𝑚=1 𝑚 𝑖 1 𝑁 o Generalize: min 𝐽(𝜃) ≔ σ 𝑙(𝑓 𝑥; 𝜃 , 𝑦)

𝑖=1 𝑖 𝑖 𝜃 𝑁

13

## Optimization problem in ML and DL

 Most of optimization problems (OP) in machine learning (deep learning) has the following form:

𝑵 1 min 𝐽 𝜃 = 𝛺 𝜃 + ෍ 𝑙 𝑦, 𝑓 𝑥; 𝜃 - Occam's Razor

𝜃 𝑖 𝑖

𝑁 principle: prefer

𝒊=𝟏

simplest model that can well predict data. Regularization term

2 2 𝐤 𝐤

- Ω θ = 𝜆 σ σ 𝑊 = 𝜆 σ 𝑾 𝑘 𝑖,𝑗 𝑖,𝑗 𝑘 𝐹 Empirical loss
- Encourage simple models

- Work well on training set
- Avoid overfitting

How to efficiently solve this optimization problem?

6 N is the training size and might be very big (e.g., 𝑁 ≈ 10)

First-order iterative methods (gradient Second-order iterative methods (Newton and descent, steepest descent) quasi Newton methods)

Use the gradient (first derivative) g = 𝛻 𝐽(𝛉) to Use the Hessian matrix (second derivative) H =

𝛉

2 update parameters 𝛻 𝐽(𝛉) to update parameters

𝛉

14

15

# Gradient and Hessian matrix

 Given an objective function 𝐽(𝜃) with 𝜃 = [𝜃, 𝜃, … , 𝜃 ]

1 2 𝑃 o For DL models

▪ 𝜃 includes weight matrices, filters, and biases which are trainable model parameters.

6 ▪ 𝑃 is the number of trainable parameters (𝑃 could be 20 × 10). ▪ 𝐽(𝜃) is the loss function over a training set.

 Gradient g = ∇𝐽(𝜃) is the first order derivative and defined as

𝜕𝐽

(𝜃) 𝜕𝜃1 o ∇𝐽 𝜃 = 𝑔 = … … … .

𝜕𝐽

(𝜃) 𝜕𝜃𝑃

2  Hessian matrix 𝐻(𝜃) is the second order derivative ∇ 𝐽(𝜃) and defined as

2 2 2 𝜕 𝐽 𝜕 𝐽 𝜕 𝐽

(𝜃) … … (𝜃) … (𝜃) 𝜕𝜃1𝜕𝜃1 𝜕𝜃1𝜕𝜃𝑗 𝜕𝜃1𝜕𝜃𝑃

… … … … … … … … … … … … 2 2 2 2 o ∇ 𝐽 𝜃 = 𝐻(𝜃) = 𝜕 𝐽 𝜕 𝐽 𝜕 𝐽

(𝜃) … … (𝜃) … (𝜃) 𝜕𝜃𝑖𝜕𝜃1 𝜕𝜃𝑖𝜕𝜃𝑗 𝜕𝜃𝑖𝜕𝜃𝑃

… … … … … … 2 2 2 𝜕 𝐽 𝜕 𝐽 𝜕 𝐽

(𝜃) … … (𝜃) … (𝜃) 𝜕𝜃𝑃𝜕𝜃1 𝜕𝜃𝑃𝜕𝜃𝑗 𝜕𝜃𝑃𝜕𝜃𝑃

16

# Local minima-maxima and saddle point

 Given an objective function 𝐽(𝜃) with 𝜃 =

[𝜃1, 𝜃2, … , 𝜃𝑃] o 𝜃 is said to be a critical point if ∇𝐽 𝜃 = 𝟎 (vector 𝟎) B  Let us denote the set of eigenvalues of Hessian

2 D matrix ∇ 𝐽 𝜃 = 𝐻(𝜃) by

C o 𝜆1 ≤ 𝜆2 ≤ ⋯ ≤ 𝜆𝑃  Local minima

2 o ∇𝐽 𝜃 = 𝟎 and ∇ 𝐽 𝜃 = 𝐻(𝜃) ≻ 0 (positive semi-

definite matrix)

A E o 0 ≤ 𝜆1 ≤ 𝜆2 ≤ ⋯ ≤ 𝜆𝑃  Local maxima

2 o ∇𝐽 𝜃 = 𝟎 and ∇ 𝐽 𝜃 = 𝐻(𝜃) ≺ 0 (negative semi-

definite matrix) o 𝜆1 ≤ 𝜆2 ≤ ⋯ ≤ 𝜆𝑃 ≤ 0  Saddle point

2 o ∇𝐽 𝜃 = 𝟎 and ∇ 𝐽 𝜃 = 𝐻(𝜃) ≺ 0 (indefinite matrix) o 𝜆1 ≤ 𝜆2 ≤ ⋯ < 0 < ⋯ ≤ 𝜆𝑃

17

# More on saddle point

Saddle point

[Source: Internet]

2 2 𝑓 𝜃 = 𝑓 𝜃1, 𝜃2 = 𝜃 − 𝜃

1 2 𝜕𝑓 𝜕𝜃 2𝜃 0 0

1 1 Gradient 𝑔 = = = → a critical point .

𝜕𝑓 −2𝜃

2 0 0 𝜕𝜃2

2 2 𝜕 𝑓 𝜕 𝑓

2 𝜕𝜃 𝜕𝜃1𝜕𝜃2 2 0

1 Hessian matrix is 𝐻 = =

2 2 𝜕 𝑓 𝜕 𝑓 0 −2

2 𝜕𝜃2𝜕𝜃1 𝜕𝜃

2

0 Two eigenvalues 𝜆1 = −2 < 0 < 2 = 𝜆2 → is a saddle point.

0

## Numbers of local minima vs saddle points

###  We assume to pick randomly a training set

o The Hessian matrix 𝐻(𝜃) is a random matrix with random eigenvalues

𝜆1, 𝜆2, … , 𝜆𝑃 o We assume that ℙ 𝜆1 ≥ 0 = ℙ 𝜆2 ≥ 0 = ⋯ = ℙ 𝜆𝑃 ≥ 0 = 0.5

###  Therefore, we have

𝑃 o ℙ 𝑚𝑖𝑛𝑖𝑚𝑎 = ℙ 𝜆 ≥ 0 ℙ 𝜆 ≥ 0 … ℙ 𝜆 ≥ 0 = 0.5

1 2 𝑃

𝑃 o ℙ 𝑚𝑎𝑥𝑖𝑚𝑎 = ℙ 𝜆 ≤ 0 ℙ 𝜆 ≤ 0 … ℙ 𝜆 ≤ 0 = 0.5

1 2 𝑃

𝑃−1 o ℙ 𝑠𝑎𝑑𝑑𝑙𝑒 𝑝𝑜𝑖𝑛𝑡 = 1 − ℙ 𝑚𝑖𝑛𝑖𝑚𝑎 − ℙ 𝑚𝑎𝑥𝑖𝑚𝑎 = 1 − 0.5

###  The ratio of #local minima/maxima against #saddle points

𝑃 o #local-minima:#local-maxima:#saddle-point=1: 1: 2 − 2 o Number of saddle points is even exponentially much more than that of local

minima/maxima

18

19

# The loss surface of DL optimization problem

Loss surface of a ResNet without skip connection [Hao Li et al., NeurIPS 2017]

##  The optimization problem in deep learning:

𝐿 1 1 exp{ℎ𝑦 (𝑥𝑖)}

𝑁 𝑁 𝑖

### o min 𝐽 𝜃 ≔ 𝐿 𝐷; 𝜃 ≔ σ 𝑙 𝑓 𝑥; 𝜃 , 𝑦 = − σ log

𝑖=1 𝑖 𝑖 𝑛=1 𝑀 𝐿 𝜃 𝑁 𝑁 σ exp{ℎ (𝑥)}

𝑚=1 𝑚 𝑖

###  A very complex and complicated objective function

o Highly non-linear and non-convex function o The loss surface is very complex o Many local minima points, but the number of saddle points is even exponentially much more

## Gradient descent and stochastic

## gradient descent

## Gradient descend

Convex case

−𝜂𝛻 𝐽 𝜃 −𝜂𝛻 𝐽 𝜃 −𝜂𝛻 𝐽 𝜃 −𝜂𝛻𝜃 𝐽 𝜃𝑡 = 0

𝜃 𝑡 𝜃 𝑡 𝜃 𝑡

### ❑ We need to solve

(Source: www.cs.ubc.ca) o min 𝐽(𝜃)

𝜃

Non-convex case

### ❑ Follow to the opposite side of the current gradient

o 𝜃𝑡+1 = 𝜃𝑡 − 𝜂𝛻𝜃𝐽 𝜃𝑡 where 𝜼 > 𝟎 is the learning rate.

### ❑ Guarantee to converge to a global minima if 𝐽(. ) is

### convex.

### ❑ Get stuck in a local minima or saddle points if 𝐽(. )

### is non-convex. DL case: easy to get stuck

in saddle points

21

## Gradient descend

### Algorithm

### ❑ Input: objective function 𝑱(𝜽)

𝜽𝟏 ∗

### ❑ Output: optimal solution 𝜽

2 𝜽𝟑

### 1\. Initialize parameters 𝜃0 randomly ~𝑁(0, 𝜎). 𝜽

𝟐

### 2\. for t=1 to T

𝜕𝐽

### 3\. Compute gradients ∇𝜃𝐽 𝜃𝑡 = (𝜃𝑡)

𝜕𝜃

### 4\. Update 𝜃𝑡+1 = 𝜃𝑡 − 𝜂𝑡∇𝜃𝐽 𝜃𝑡

∗

### 5\. Return 𝜃 = 𝜃𝑇+1

22

## Example of Gradient Descend

### ❑ Consider the optimization problem:

2 2

### min 𝑓 𝑤 = 𝑤 − 1 + 𝑤 − 3

𝑤

2 2

### ❑ Currently, we are at 𝑤𝑡 = 1 with 𝑓 𝑤𝑡 = 𝑓 1 = 0 + 2 =

### 4\. What is 𝑤𝑡+1 if using learning rate 𝜂 = 0.1?

′ o 𝑓 𝑤 = 2 𝑤 − 1 + 2 𝑤 − 3 = 4𝑤 − 8

′ ′ o 𝑓 𝑤𝑡 = 𝑓 1 = −4

′ o 𝑤𝑡+1 = 𝑤𝑡 − 𝜂𝑓 𝑤𝑡 = 1 − 0.1 −4 = 1.4 o 𝑓 𝑤𝑡+1 = 2.72 < 𝑓 𝑤𝑡 = 4 𝑤 𝑤

𝑡 𝑡+1

23

24

# Gradient descent for deep learning

###  For training deep nets, we need to solve

1 𝑁 1 𝑁 o min 𝐿 𝐷; 𝜃 ≔ σ 𝑙 𝑥, 𝑦; 𝜃 = σ 𝑙 𝑦, 𝑓(𝑥; 𝜃)

𝑖=1 𝑖 𝑖 𝑖=1 𝑖 𝑖 𝜃 𝑁 𝑁

𝐿 exp{ℎ𝑦 (𝑥𝑖)}

𝑖 where 𝑙 𝑥𝑖, 𝑦𝑖; 𝜃 = − log 𝑝(𝑦 = 𝑦𝑖|𝑥𝑖) = − log is the loss incurred by

𝑀 𝐿 σ exp{ℎ (𝑥)} 𝑚=1 𝑚 𝑖 (𝑥𝑖, 𝑦𝑖).

###  Gradient descent update

𝜂 𝑁 o 𝜃 = 𝜃 − 𝜂∇ 𝐿 𝐷; 𝜃 = 𝜃 − σ ∇ 𝑙(𝑥, 𝑦; 𝜃) where 𝜼 > 𝟎 is a learning rate.

𝑡+1 𝑡 𝜃 𝑡 𝑡 𝑖=1 𝜃 𝑖 𝑖 𝑡

𝑁

1 𝑁 o To compute the gradient ∇ 𝐿 𝐷; 𝜃 = σ ∇ 𝑙(𝑥, 𝑦; 𝜃), we need to go through all

𝜃 𝑡 𝑖=1 𝜃 𝑖 𝑖 𝑡

𝑁 data points in 𝐷 → the computational cost is 𝑂(𝑁).

6

###  This is very computationally expensive for big datasets (𝑁 ≈ 10).

###  How to estimate the gradient ∇ 𝐿 𝐷; 𝜃 more efficiently?

𝜃 𝑡

## Stochastic gradient descent

###  The optimization problem in deep learning has the form

1 𝑁 1 𝑁 o min 𝐿 𝐷; 𝜃 ≔ σ 𝑙 𝑥, 𝑦; 𝜃 = σ 𝑙 𝑦, 𝑓(𝑥; 𝜃)

𝑖=1 𝑖 𝑖 𝑖=1 𝑖 𝑖 𝜃 𝑁 𝑁

###  Evaluation of the full gradient is expensive. We want to just estimate this gradient

o Sample a mini-batch 𝒊𝟏, 𝒊𝟐, … , 𝒊𝒃~𝑼𝒏𝒊({𝟏, 𝟐, … , 𝑵}) where 𝑏 is the mini-batch (batch) size.

▪ The batch size is usually 32,64,128,256, and so on.

1 𝑏 o Construct 𝐿෨ 𝜃 : = σ 𝑙(𝑥, 𝑦; 𝜃) as the average loss of those in the current batch.

𝑘=1 𝑖𝑘 𝑖𝑘 𝑏 o 𝐸 𝛻 𝐿෨ 𝜃 = 𝛻 𝐿 𝐷; 𝜃

𝑖1,…,𝑖𝑏 𝜃 𝑡 𝜃 𝑡

1 𝑏 ▪ 𝛻 𝐿෨ 𝜃 = σ ∇ 𝑙(𝑥, 𝑦; 𝜃) is unbiased estimation of 𝛻 𝐿 𝐷; 𝜃

𝜃 𝑡 𝑘=1 𝜃 𝑖𝑘 𝑖𝑘 𝑡 𝜃 𝑡

𝑏 ▪ 𝑂(𝑏) compares to 𝑂(𝑁).

###  The update rule of SGD

𝟏 o 𝜃 = 𝜃 − 𝜂 𝛻 𝐿෨ 𝜃 with learning rate 𝜼 ∝ 𝑶()

𝑡+1 𝑡 𝑡 𝜃 𝑡 𝒕

𝒕 o We use 𝛻 𝐿෨ 𝜃 as an unbiased estimate of the full gradient 𝛻 𝐿(𝐷; 𝜃)

𝜃 𝑡 𝜃 o How to compute 𝛻 𝐿෨ 𝜃 efficiently for deep networks?

𝜃 𝑡

25

## Example of Stochastic Gradient Descent

1 1000

2

###  Given the function 𝑓 𝑤 = σ 𝑤 − 𝑖 , we need to solve

𝑖=1 1000

### min 𝑓(𝑤)

𝑤

### using SGD with the learning rate 𝜂 = 0.1.

###  Assume we sample a batch 𝑖 = 1, 𝑖 = 2, 𝑖 = 3, 𝑖 = 4 of indices. At the iteration

1 2 3 4

### 𝑡, 𝑤𝑡 = 10. What is the value of 𝑤𝑡+1 at the next iteration?

1

### ሚ 2 2 2 2

### ❑ 𝑓 𝑤 = [ 𝑤 − 1 + 𝑤 − 2 + 𝑤 − 3 + 𝑤 − 4 ]

4

### ሚ′

### ❑ 𝑓 𝑤 = 2𝑤 − 5

### ሚ′ ሚ′

### ❑ 𝑓 𝑤 = 𝑓 10 = 2 × 10 − 5 = 15

t

### ሚ′

### ❑ 𝑤 = 𝑤 − 𝜂 𝑓 𝑤 = 10 − 0.1 × 15 = 8.5.

𝑡+1 𝑡 𝑡

26

## Back propagation in feed-

## forward neural networks

## Back propagation

𝑙 loss

###  Given a data point and label

3 3 3

- ℎ, 𝑝 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ)

### ℎ pair (𝑥, 𝑦)

3 𝑛2×𝑛3 𝑊 ∈ ℝ

- 3 o 𝑙 𝑥, 𝑦; 𝜃 = 𝑏 3 1×𝑛3 𝑏 ∈ ℝ 3 exp{ℎ𝑦(𝑥)} − log 𝑀 3 σ exp{ℎ (𝑥)}

2 3 2 2 2 𝑚=1 𝑚

𝜎 𝑊 ത ത ℎ ℎ, ℎ = 𝜎(ℎ)

### 2 𝑛1×𝑛2  What are the derivatives?

𝑊 ∈ ℝ ത2 + ℎ 2 1×𝑛2

𝑏 ∈ ℝ

o ∇ 𝑘 𝑙(𝑥, 𝑦; 𝜃) and ∇ 𝑘 𝑙(𝑥, 𝑦; 𝜃)

𝑊 𝑏 2

- 𝑏 ത1 1 ത1 for 𝑘 = 1,2,3? ℎ, ℎ = 𝜎(ℎ)

1 2

𝜎 𝑊 1 𝑛 ×𝑛 o Using back propagation to ℎ 0 1

𝑊 ∈ ℝ

1 1×𝑛1 𝑏 ∈ ℝ compute these derivatives ത1 + ℎ conveniently.

0 ℎ = 𝑥

- 1

### 𝑏  Update the model using

### 0 1 SGD with the derivatives.

ℎ = 𝑥 𝑊

29

## Back propagation 𝑓 𝑥, 𝑦, 𝑧 = log(exp(𝑥) + exp(𝑦) + exp(𝑧))

3 ′ ′ ′ From loss to ℎ ∇𝑓 = 𝑓𝑥, 𝑓𝑦, 𝑓𝑧 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥([𝑥, 𝑦, 𝑧])

3 exp ℎ𝑦 𝑥 loss 3 𝒍  𝑙 𝑥, 𝑦; 𝜃 = − log = −ℎ 𝑥 + log

𝑀 3 𝑦 σ exp ℎ 𝑥 𝑀 3 𝑚=1 𝑚 log σ exp ℎ 3 𝑚=1 𝑚

\+ ℎ

### where 𝟏 = 𝟏 if 𝒎 = 𝒚 and 𝟎 otherwise.

- 3 𝒎=𝒚 𝑏 𝟑 𝝏𝒍 𝒆𝒙𝒑{𝒉𝒎}

2 3

### 𝜎 𝑊  = −𝟏 + , 𝒎 =

ℎ 𝟑 𝒎=𝒚 𝑴 𝟑

𝝏𝒉𝒎 σ 𝒆𝒙𝒑 𝒉

𝒌=𝟏 𝒌 ത2 +

### ℎ 𝟏, . . , 𝑴

2 𝝏𝒍

- 𝑏 𝟑 𝟑

###  𝒈 = = −𝟏 + 𝒔𝒐𝒇𝒕𝒎𝒂𝒙 𝒉 = 𝒑 −

𝟑 𝒚 𝝏𝒉 1 2

𝜎 𝑊

### ℎ 𝟏𝒚

### ത1 + where 𝟏 is the corresponding one-hot vector.

ℎ 𝒚

3

- 1 ❑ 𝑔 has a shape [1 × 𝑛 ]. 𝑏 3

0 1 ℎ = 𝑥 𝑊

30

31

# Back propagation

3 3

### From loss to 𝑊, 𝑏

3 2 3 3

### 𝒍 loss  ℎ = ℎ 𝑊 + 𝑏

3 + ℎ

𝟑

- 3 𝑻 𝑏 𝝏𝒍 𝝏𝒍 𝝏𝒉 𝟐 𝟑

###  = . = 𝒉 𝒈

𝟑 𝟑 𝟑 𝝏𝑾 𝝏𝒉 𝝏𝑾 2 3

𝜎 𝑊 ℎ

o 𝑛2 × 1 × 1 × 𝑛3 → [𝑛2 × 𝑛3] ത2 + ℎ

2

- 𝑏 𝟑 𝝏𝒍 𝝏𝒍 𝝏𝒉 𝟑

### 1 2  = . = 𝒈

𝜎 𝑊 𝟑 𝟑 𝟑 ℎ 𝝏𝒃 𝝏𝒉 𝝏𝒃

o 1 × 𝑛3 ത1 + ℎ

- 1 𝑏

0 1 ℎ = 𝑥 𝑊

## Back propagation

2 ത2 From loss to ℎ and ℎ

3 2 3 3

###  ℎ = ℎ 𝑊 + 𝑏

𝒍 loss

𝟑 𝝏𝒍 𝝏𝒍 𝝏𝒉 𝑻 𝟐 𝟑 𝟑

### 3  𝒈 = = . = 𝒈 𝑾

ℎ + 𝟐 𝟑 𝟐

𝝏𝒉 𝝏𝒉 𝝏𝒉 3 o 1 × 𝑛3 × 𝑛3 × 𝑛2 → [1 × 𝑛2]

- 𝑏

### 𝟐 ഥ𝟐

###  𝒉 = 𝝈(𝒉) (element-wise activation)

2 3

𝜎 𝑊 ℎ

𝟐 𝝏𝒉 ′ 𝟐

### ഥ

### ത2 +  = 𝒅𝒊𝒂𝒈(𝝈 (𝒉))

ℎ ഥ𝟐

𝝏𝒉 2 ′ ത2

- 𝑏 o 𝜎 (ℎ) is element-wise derivative and 𝑑𝑖𝑎𝑔(𝑢) is the diagonal matrix corresponding to the vector 𝒖 (the diagnose

1 2

𝜎 𝑊 ℎ

is 𝑢 and others are zeros). 1 𝟐 ℎത + 𝝏𝒍 𝝏𝒍 𝝏𝒉

### 𝟐 𝟐 ′ ഥ𝟐

###  𝒈ഥ = = . = 𝒈 𝒅𝒊𝒂𝒈(𝝈 (𝒉))

ഥ𝟐 𝟐 ഥ𝟐 𝝏𝒉 𝝏𝒉 𝝏𝒉

- 1 𝑏 o 1 × 𝑛2 × 𝑛2 × 𝑛2 → [1 × 𝑛2]

0 1 ℎ = 𝑥 𝑊

32

## Back propagation

2 2 From loss to 𝑊 and 𝑏

𝒍 loss

### ത2 1 2 2

###  ℎ = ℎ 𝑊 + 𝑏

3 + ℎ

- 3 𝑏 ഥ𝟐 𝝏𝒍 𝝏𝒍 𝝏𝒉 𝑻 𝟏 𝟐

### 2 3  = . = 𝒉 𝒈ഥ

𝜎 𝑊 ℎ 𝟐 ഥ𝟐 𝟐

𝝏𝑾 𝝏𝒉 𝝏𝑾 ത2 + o 𝑛 × 1 × 1 × 𝑛 → [𝑛 × 𝑛 ] ℎ 1 2 1 2

2

- 𝑏

1 2 ഥ𝟐 ℎ 𝜎 𝑊 𝝏𝒍 𝝏𝒍 𝝏𝒉

𝟐

###  = . = 𝒈ഥ

𝟐 ഥ𝟐 𝟐 𝝏𝒃 𝝏𝒉 𝝏𝒃 ത1 ℎ +

o 1 × 𝑛2

- 1 𝑏

0 1 ℎ = 𝑥 𝑊

33

## Back propagation

1 ത1 From loss to ℎ and ℎ

### ത2 1 2 2

###  ℎ = ℎ 𝑊 + 𝑏

𝒍 loss

ഥ𝟐 𝝏𝒍 𝝏𝒍 𝝏𝒉 𝑻 𝟏 𝟐 𝟐

### 3  𝒈 = = . = 𝒈ഥ 𝑾

ℎ + 𝟏 𝟐 𝟏

𝝏𝒉 𝝏𝒉ഥ 𝝏𝒉

- 3 o 1 × 𝑛2 × 𝑛2 × 𝑛1 → [1 × 𝑛1] 𝑏

### 𝟏 ഥ𝟏

###  𝒉 = 𝝈(𝒉) (element-wise activation)

3 2 𝜎 𝑊 ℎ

𝟏 𝝏𝒉 ′ 𝟏

### ഥ

### ത2 +  = 𝒅𝒊𝒂𝒈(𝝈 (𝒉))

ℎ ഥ𝟏

𝝏𝒉 2 ′ ത1

- 𝑏 o 𝜎 (ℎ) is element-wise derivative and 𝑑𝑖𝑎𝑔(𝑢) is the diagonal matrix corresponding to the vector 𝒖 (the diagnose is 2

1 𝜎 𝑊 ℎ

𝑢 and others are zeros). ത1 𝟏 ℎ + 𝝏𝒍 𝝏𝒍 𝝏𝒉

### 𝟏 𝟏 ′ ഥ𝟏

###  𝒈ഥ = = . = 𝒈 𝒅𝒊𝒂𝒈(𝝈 (𝒉))

ഥ𝟏 𝟏 ഥ𝟏 𝝏𝒉 𝝏𝒉 𝝏𝒉

- 1 𝑏 o 1 × 𝑛1 × 𝑛1 × 𝑛1 → [1 × 𝑛1]

0 1 ℎ = 𝑥 𝑊

34

## Back propagation

1 1 From loss to 𝑊 and 𝑏

### 𝒍 loss ത1 0 1 1 0

###  ℎ = ℎ 𝑊 + 𝑏 (ℎ = 𝑥)

3 + ℎ

ഥ𝟏

- 3 𝝏𝒍 𝝏𝒍 𝝏𝒉 𝑻 𝑏 𝟎 𝟏

###  = . = 𝒉 𝒈ഥ

𝟏 ഥ𝟏 𝟏 𝝏𝑾 𝝏𝒉 𝝏𝑾 2 3

𝜎 𝑊 ℎ o 𝑛 × 1 × 1 × 𝑛 → [𝑑 = 𝑛 × 𝑛 ]

0 1 0 1 ത2 + ℎ

ഥ𝟏 2

- 𝑏 𝝏𝒍 𝝏𝒍 𝝏𝒉 𝟏

###  = . = 𝒈ഥ

𝟏 ഥ𝟏 𝟏 𝝏𝒃 𝝏𝒉 𝝏𝒃 1 2

𝜎 𝑊 ℎ

o 1 × 𝑛1 ത1 + ℎ

- 1 𝑏 𝜕𝑙 Exercise: How to compute ? 𝜕𝑥

0 1 ℎ = 𝑥 𝑊

35

36

# SGD for deep learning

b = 32 //batch size iter_per_epoch = N/b //epoch means one round going

through all data points n_epoch = 50 //number of epochs

3 3 ℎ, 𝑝 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ) for epoch=1 to n_epoch do

3 3×2 𝑊 ∈ ℝ

3 1×2 𝑏 ∈ ℝ for i=1 to iter_per_epoch do

𝑏 2 2 2

ℎത, ℎ = 𝜎(ℎത) Sample a minibatch 𝐵 = 𝑥𝑖, 𝑦𝑖 from the training set

𝑗 𝑗

𝑗=1 2 4×3

𝑊 ∈ ℝ

2 1×3 𝑏 ∈ ℝ Do forward propagation for B

𝐿 ത1 1 ത1 𝜕𝑙 𝜕𝑙 ℎ, ℎ = 𝜎(ℎ) Do back propagation to compute ,

𝑘 𝑘 𝜕𝑊 𝜕𝑏 𝑘=1

1 3×4 𝑊 ∈ ℝ

1 1×4 𝑏 ∈ ℝ for k=1 to L do

𝜕𝑙 0 𝑊 = 𝑊 − 𝜂 ℎ = 𝑥

𝑘 𝑘 𝑘

𝜕𝑊 𝜕𝑙 𝑏𝑘 = 𝑏𝑘 − 𝜂

𝑘 𝜕𝑏

37

# Forward – Backward Propagations

𝒍 CE loss

Backward Forward

𝒑

softmax

𝝏𝒍

𝟏×𝒏𝟑 3 𝒈 = = 𝒑 − 𝟏 ∈ ℝ ℎ + 𝟑 𝒚

𝟑 𝝏𝒉

𝟑 𝝏𝒍 3 𝝏𝒍 𝝏𝒍 𝝏𝒉 𝟑 𝟑

𝟑 𝟏×𝒏𝟑 𝑏 = . = 𝒈 ∈ ℝ 𝒃 = 𝒃 − 𝜼

- 𝟑 𝟑 𝟑 𝟑 𝝏𝒃 𝝏𝒉 𝝏𝒃 𝝏𝒃 𝝏𝒍 𝝏𝒍 𝝏𝒍 𝟑 𝟑 𝟑 3 𝝏𝒉 𝑻 𝟐 𝟑 𝒏 ×𝒏 𝑾 = 𝑾 − 𝜼 𝑊 = . = 𝒉 𝒈 ∈ ℝ 𝟐 𝟑 𝟑 𝟑 𝟑 𝟑 𝝏𝑾 𝝏𝑾 𝝏𝒉 𝝏𝑾 𝟑

2 𝝏𝒍 𝝏𝒍 𝝏𝒉 ℎ 𝟐 𝟑 𝟑 𝟏×𝒏𝟐

𝝈 𝒈 = = . = 𝒈 𝑾 ∈ ℝ

𝟐 𝟑 𝟐 𝝏𝒉 𝝏𝒉 𝝏𝒉

𝟐 𝝏𝒍 𝝏𝒍 𝝏𝒉 ത2 𝟐 𝟐 ′ 𝟐 𝟏×𝒏 ℎ + 𝒈ഥ = = . = 𝒈 𝒅𝒊𝒂𝒈(𝝈 (𝒉ഥ)) ∈ ℝ 𝟐

ഥ𝟐 𝟐 ഥ𝟐 𝝏𝒉 𝝏𝒉 𝝏𝒉

𝝏𝒍 𝝏𝒍 𝝏𝒍 ഥ𝟐 𝟐 𝟐

𝝏𝒉 𝟏×𝒏

- 2 = . 𝟐 𝒃 = 𝒃 − 𝜼 𝑏 =∈ ℝ 𝟐 𝟐 ഥ𝟐 𝟐 𝝏𝒃 𝝏𝒃 𝝏𝒉 𝝏𝒃 ഥ𝟐

1 𝝏𝒍 𝝏𝒍 𝝏𝒉 ഥ𝟐

𝟏 𝟐 𝟐 𝟏×𝒏𝟏 𝝏𝒍 𝝏𝒍 𝝏𝒉 𝑻 ℎ 𝝈 𝒈 = = . = 𝒈ഥ 𝑾 ∈ ℝ 2 𝟏 𝟐 𝒏 ×𝒏

𝟏 𝟐 𝝏𝒍 𝟏 ഥ𝟐 𝟏 𝑊 = . = 𝒉 𝒈ഥ ∈ ℝ 𝝏𝒉 𝝏𝒉 𝝏𝒉 𝟐 ഥ𝟐 𝟐 𝟐 𝟐

𝝏𝑾 𝝏𝒉 𝝏𝑾 𝑾 = 𝑾 − 𝜼

𝟐 𝝏𝑾 𝟏 1 𝝏𝒍 𝝏𝒍 𝝏𝒉 ത 𝟏 𝟏 ′ ഥ𝟏 𝟏×𝒏𝟏 ℎ + 𝒈ഥ = = . = 𝒈 𝒅𝒊𝒂𝒈 𝝈 𝒉 ∈ ℝ

ഥ𝟏 𝟏 ഥ𝟏 𝝏𝒉 𝝏𝒉 𝝏𝒉

ഥ𝟏 𝝏𝒍 𝝏𝒍 𝝏𝒍 𝝏𝒉

𝟏 𝟏×𝒏𝟏 𝟏 𝟏

- 1 = . = 𝒈ഥ ∈ ℝ 𝒃 = 𝒃 − 𝜼 𝑏 𝟏 ഥ𝟏 𝟏 𝟏 𝝏𝒃 𝝏𝒉 𝝏𝒃 𝝏𝒃 𝝏𝒍 ഥ𝟏 𝟏 𝟏 𝝏𝒍 𝝏𝒍 𝝏𝒉 𝑻 1 𝟎 𝟏 𝒏 ×𝒏 𝑾 = 𝑾 − 𝜼 𝟎 𝟏 𝟏 𝑊 = . = 𝒉 𝒈ഥ ∈ ℝ 𝝏𝑾

0 𝟏 ഥ𝟏 𝟏 ℎ = 𝑥 𝝏𝑾 𝝏𝒉 𝝏𝑾

𝑏 1 𝑏𝑎𝑡𝑐ℎ_𝑙𝑜𝑠𝑠 = ෍ 𝐶𝐸(1𝑦, 𝑝𝑖) Prediction

𝑖 𝑏

𝑖=1

## Mini-batch feed-forward probabilities

𝑛3 = 𝑀 = 4

1 𝑝1 𝑦1  Input

𝑝2 1𝑦

2 𝑏 = 32 𝑏 = 32 o Batch 𝑋: [𝑏, 𝑛0 = 𝑑 = 5] (b is Labels 𝑌 𝑃

the batch size)

𝑝𝑏 1𝑦

𝑏 One-hot labels 𝑛3 = 𝑀 = 4  Hidden layer 1

𝑙𝑜𝑔𝑖𝑡1 𝑦1 = 3, 𝑀 = 4 𝟏 𝟏 𝟏 o 𝒉 = 𝝈(𝑿𝑾 + 𝒃) 𝑙𝑜𝑔𝑖𝑡2

1 = [0,0,1,0] Output layer 𝑦1

3 3 𝑏 = 32 o Tensor [𝑏, 𝑛 = 7] ℎ (𝑥) ℎ

1

Backward Forward

3 5×4 𝑙𝑜𝑔𝑖𝑡𝑏 𝑊 ∈ ℝ,  Hidden layer 2

propagation propagation

3 1×4 𝑏 ∈ ℝ 𝑛 = 5 𝟐 𝟏 𝟐 𝟐 2 o 𝒉 = 𝝈(𝒉 𝑾 + 𝒃)

Hidden layer o Tensor [𝑏, 𝑛2 = 5]

2 2 ℎ (𝑥) ℎ

𝑏 = 32  Output layer 2 7×5

𝑊 ∈ ℝ, 𝟑 𝟐 𝟑 𝟑 2 1×5 o 𝒉 = 𝒉 𝑾 + 𝒃 𝑏 ∈ ℝ

𝑛1 = 7 3 o 𝑃 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ, 𝑑𝑖𝑚 = 1) Hidden layer 1

ℎ 1 ℎ (𝑥)

𝑏 = 32 o Tensor P: [𝑏, 𝑛𝐿 = 𝑀 = 4]

1 5×7 Training set 𝑊 ∈ ℝ,  The loss of the batch 1 1×7

𝑏 ∈ ℝ

𝑑 = 𝑛0 = 5 1 𝑏 Mini-batches o σ 𝐶𝐸(1, 𝑝) = Input layer

𝑖=1 𝑦𝑖 𝑖 𝑥 𝑏 1

0 1 ℎ (𝑥) = 𝑥

𝑏 𝑖 − σ log 𝑝 𝑥

𝑖=1 𝑦𝑖 𝑥, 𝑦, … , (𝑥, 𝑦), 2 𝑏 = 32 𝑏 1 1 𝑏 𝑏

Form a mini-batch 𝑥, 𝑦, … , (𝑥, 𝑦) (batch size) o Update weight matrices and 𝑏+1 𝑏+1 2𝑏 2𝑏

…………………………. 𝑥𝑏 biases to minimize the batch loss.

𝑥, 𝑦, … , (𝑥, 𝑦) Data 𝑋 ∗ ∗ 𝑁 𝑁

38

## Why does deep learning need GPU and TPU?

###  Let consider

𝒍 loss 𝟑 𝟐 ഥ𝟐 𝟏 ഥ𝟏

𝝏𝒍 𝝏𝒍 𝝏𝒉 𝝏𝒉 𝝏𝒉 𝝏𝒉 𝝏𝒉

= . . . . . 𝟏 𝟑 𝟐 ഥ𝟐 𝟏 ഥ𝟏 𝟏 𝝏𝑾 𝝏𝒉 𝝏𝒉 𝝏𝒉 𝝏𝒉 𝝏𝒉 𝝏𝑾 3 + ℎ 𝑻 𝑻 𝑻

𝟎 𝟑 ′ ഥ𝟐 𝟐 ′ ഥ𝟏 = 𝒉 𝒑 − 𝟏𝒚 𝑾 𝒅𝒊𝒂𝒈 𝝈 𝒉 𝑾 𝒅𝒊𝒂𝒈 𝝈 𝒉

- 3 𝑏

###  For a deep net, this back propagation requires many

2 3

𝜎 𝑊

### ℎ matrix multiplications

ത2 + o We need specific hardware that can parallel and significantly speed up ℎ

matrix multiplication operation 2

- 𝑏 o GPU (Graphic Processing Unit) and TPU (Tensor Processing Unit)

1 2

𝜎 𝑊 ℎ

ത1 + ℎ

- 1 𝑏

0 1 ℎ = 𝑥 𝑊

GPU (Source: HelloTech) TPU (Source: Wikepedia)

39

## Deep learning pipeline

Tuning hyper-parameters

###  We want to train our DL model on a training set such that the trained model can

### predict well unseen data in a separate testing set.

Original dataset

Training set Validation set Testing set

(80%) (10%) (10%)

DL model 𝜂 = 0.1 DL model 𝜂 = 0.01 DL model 𝜂 = 0.001

𝐿 𝑖 𝑖 Model parameters: 𝑊, 𝑏

𝑖=1

- To be learnt using optimizers

DL method

Hyper-parameters: learning rate 𝜂, #layers, \#neurons per layer

- To be tuned using validation dataset

40

## Optimizers for deep learning

## Challenges of optimization for Deep Learning

###  The optimization problem in deep learning:

𝐿 1 1 exp{ℎ𝑦 (𝑥𝑖)}

𝑁 𝑁 𝑖 o min 𝐽 𝜃 ≔ 𝐿 𝐷; 𝜃 ≔ σ 𝑙 𝑥, 𝑦; 𝜃 = − σ log

𝑖=1 𝑖 𝑖 𝑛=1 𝑀 𝐿 𝜃 𝑁 𝑁 σ exp{ℎ (𝑥)}

𝑚=1 𝑚 𝑖

###  A very complex and complicated objective function

o Highly non-linear and non-convex function o The loss surface is very complex o Many local minima points, but the number of saddle points is even exponentially much

more

###  Need efficient optimizers to solve

o SGD with momentum, Adagrad, Adadelta, RMSProp, Adam, and Nadam o They are built-in optimizers of PyTorch.

(Source: Jan Jakubik)

43

## SGD and SGD with momentum

SGD with momentum SGD

o Input: 𝜂 > 0, 𝛼 ∈ [0,1] and initial model 𝜃 o Input: 𝜂 > 0 and initial model 𝜃

while stopping criterion not met do while stopping criterion not met do

𝟏 𝟏 𝒃 𝒃 Sample a mini-batch { 𝒙, 𝒚, … , (𝒙, 𝒚)} 𝟏 𝟏 𝒃 𝒃 Sample a mini-batch { 𝒙, 𝒚, … , (𝒙, 𝒚)}

𝟏 𝒃

𝒊 𝒊 𝟏 Compute 𝒈 = σ 𝛁 𝒍(𝒇(𝒙, 𝜽), 𝒚)

𝒃 𝒊 𝒊 𝒊=𝟏 𝜽 Compute 𝒈 = σ 𝛁 𝒍(𝒇(𝒙, 𝜽), 𝒚) 𝒃

𝒊=𝟏 𝜽 𝒃

Compute 𝒗 = 𝜶𝒈 + 𝟏 − 𝜶 𝒗 //velocity 𝑣 Update 𝜽 = 𝜽 − 𝜼𝒈

Update 𝜽 = 𝜽 − 𝜼𝒗 end while

end while

 SGD with momentum uses a velocity vector 𝒗 which

stores the past gradients together with the current  SGD uses only the gradient of the mini-batch to update the

gradient to speed up SGD model

o 𝛼 is a hyper-parameter that indicates how quickly the contributions of previous

gradients. In practice, this is usually set to 0.5, 0.9, and 0.99.  It is fast at first several epochs and becomes much slower later.

o The momentum primarily solves 2 problems: poor conditioning of the Hessian

matrix and variance in the stochastic gradient.

(Source: Sebastian Ruder)

44

## AdaGrad

AdaGrad

−6 o Input: 𝜂 > 0, 𝜖 > 0 (10), and initial model 𝜃

while stopping criterion not met do

𝟏 𝟏 𝒃 𝒃 Sample a mini-batch { 𝒙, 𝒚, … , (𝒙, 𝒚)}

𝟏 𝒃

𝒊 𝒊 Compute 𝒈 = σ 𝛁 𝒍(𝒇(𝒙, 𝜽), 𝒚)

𝒊=𝟏 𝜽 𝒃 Accumulate the square gradient: 𝜸 = 𝜸 + 𝒈⨀𝒈

𝜼 Update 𝜽 = 𝜽 − ⨀𝒈

𝝐+𝜸 end while Note: ⨀ means element-wise product

 Learning rates are scaled by the square root of the cumulative sum

of squared gradients  Direction with large partial derivatives

o Thus, rapid decrease in their learning rates  Direction with small partial derivatives

o Hence relatively small decrease in their learning rates  Weakness: always decrease the learning rate!

o Excellent for convex problem, but not so good for DL (with non-convex problems)

45

## RMSProp

RMSProp

−6 o Input: 𝜂 > 0, 𝜖 > 0 (10), 𝛽 ∈ [0,1] and initial model 𝜃

while stopping criterion not met do

𝟏 𝟏 𝒃 𝒃 Sample a mini-batch { 𝒙, 𝒚, … , (𝒙, 𝒚)}

𝟏 𝒃

𝒊 𝒊 Compute 𝒈 = σ 𝛁 𝒍(𝒇(𝒙, 𝜽), 𝒚)

𝒊=𝟏 𝜽 𝒃 Accumulate the square gradient: 𝜸 = 𝜷𝜸 + (𝟏 − 𝜷)𝒈⨀𝒈

𝜼 Update 𝜽 = 𝜽 − ⨀𝒈

𝝐+𝜸 end while Note: ⨀ means element-wise product

 A modification of AdaGrad to work better for non-convex setting.  Instead of cumulative sum, use exponential moving/smoothing average.  RMSProp has been shown to be an effective and practical optimization algorithm for DNN.

o Currently one of the go-to optimization methods being employed routinely by DL applications.

46

## Adam

Adam

−6 o Input: 𝜂 > 0, 𝜖 > 0 10, 𝛽1, 𝛽2 ∈ [0,1] and initial model 𝜃

###  The best variant that

### essentially combines

𝒕 = 𝟏

### RMSProp with momentum while stopping criterion not met do

𝟏 𝟏 𝒃 𝒃 Sample a mini-batch { 𝒙, 𝒚, … , (𝒙, 𝒚)}

𝟏 𝒃

𝒊 𝒊 Compute 𝒈 = σ 𝛁 𝒍(𝒇(𝒙, 𝜽), 𝒚)

𝒊=𝟏 𝜽 𝒃

###  Suggested default values: 𝜂 = Accumulate the gradient: 𝐬 = 𝜷 𝒔 + 𝟏 − 𝜷 𝒈

𝟏 𝟏 Accumulate the square gradient: 𝜸 = 𝜷𝟐𝜸 + 𝟏 − 𝜷𝟐 𝒈⨀𝒈

### 0.001, 𝛼1 = 0.9, 𝛼2 =

𝒔 Correct 𝑠: 𝒔ො = # 𝑡 is the current iteration −6 𝒕

𝟏−𝜷

### 0.999 and 𝜖 = 10. 𝟏

𝜸 Correct 𝛾: 𝜸ෝ =

𝒕 𝟏−𝜷

𝟐 𝜼 Update 𝜽 = 𝜽 − ⨀𝒔ො

𝝐+𝜸ෝ 𝒕 = 𝒕 + 𝟏 end while Note: ⨀ means element-wise product

47

48

# Visual comparison of all optimizers

[Source: Sebastian Ruder]

49

## Optimizers in deep learning

Optimizers

Second order methods (Newton method)

−1 First order methods 𝜽 ← 𝜽 − 𝜂𝐻 𝒈

- Impossible for DL

Adaptive learning rate Standard Mini- based on gradients batched SGD

SGD Adagrad RMSProp Adam SGD with momentum

𝒈 ← 𝛻𝐽 𝜽 𝒈 ← 𝛻𝐽 𝜽 𝑡 ← 𝑡 + 1

𝒗 ← 𝛼𝒗 + (1 − 𝛼)𝛻𝐽 𝜽 𝛾 ← 𝛾 + 𝒈 ⊙ 𝒈 𝜸 ← 𝛽𝜸 + 1 − 𝛽 𝒈 ⊙ 𝒈 𝒈 ← 𝛻𝐽 𝜽 𝜽 ← 𝜽 − 𝜂𝛻𝐽 𝜽

𝜂 𝜂 𝜽 ← 𝜽 − 𝜂𝒗 𝜽 ← 𝜽 − ⊙ 𝒈 𝜽 ← 𝜽 − ⊙ 𝒈

𝜖 + 𝛾 𝜖 + 𝛾 𝒔 ← 𝛽1𝒔 + 1 − 𝛽1 𝒈

𝜸 ← 𝛽2 𝜸 + 1 − 𝛽2 𝒈 ⊙ 𝒈

𝒔 𝑠Ƹ ←

𝑡 1 − 𝛽

1 𝜸 𝛾ො ←

𝑡 1 − 𝛽

2

𝒔ො 𝜽 ← 𝜽 − 𝜂

𝜖 + 𝛾ො

50

# Summary

###  Optimization problem in DL and ML

o Regularization term + Empirical loss term

###  Gradient descent

###  Stochastic gradient descent

###  Backward propagation

###  Other optimizers in DL

o SGD with momentum, Adagrad, RMSProp, and Adam

###  First order methods and second order methods

## Thanks for your attention!