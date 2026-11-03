COPYRIGHT WARNING: Copyright in these original lectures is owned by Monash University. You may transcribe, take notes, download or stream lectures for the purpose of your research and study only. If used for any other purpose, (excluding exceptions in the Copyright Act 1969 (Cth)) the University may take legal action for infringement of copyright.

Do not share, redistribute, or upload the lecture to a third party without a written permission!

### FIT3181/5215 Deep Learning

Week 04: Stochastic Gradient Descent and Optimization

Lecturer: Dr Trung Le Trung Le and Tutor team Email: trunglm@monash.edu

Department of Data Science and AI Faculty of Information Technology, Monash University, Australia

## Introduction

##  The purpose of this tutorial is to practise:

### 1\. Logistic Regression with GD and SGD

### 2\. AutoGrad in PyTorch

### 3\. Use AutoGrad to implement Logistic Regression

## Logistic Regression

𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒 1

𝑖 𝑏𝑎𝑡𝑐ℎ_𝑙𝑜𝑠𝑠 = ෍ 𝐶𝐸(1 𝑖 , 𝑝)

𝑦 𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒

𝑖=1 𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒

###  A shallow feed-forward neural network

1

𝑖 = − ෍ log 𝑝 𝑖

𝑦 o Binary classifier with one layer and softmax function 𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒

𝑖=1

###  Forward propagation 𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒×2

𝑝 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 ℎ ∈ ℝ 𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒×2 𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒×𝑑 o ℎ = 𝑋𝑊 + 𝑏 ∈ ℝ with X ∈ ℝ

𝑑×2 1×2 and 𝑊 ∈ ℝ and 𝑏 ∈ ℝ

𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒×2 ℎ = 𝑋𝑊 + 𝑏 ∈ ℝ ℎ1 𝑒 o 𝑝 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 ℎ, dim = 1 = ൤𝑝1 = , 𝑝2 =

ℎ1 ℎ2 𝑒 +𝑒 ℎ2 𝑒

𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒×2 ൨ ∈ ℝ ℎ1 ℎ2 𝑒 +𝑒

1 1 1 𝑥 … 𝑥 𝑦 1 𝑑

###  𝑏𝑎𝑡𝑐ℎ_𝑙𝑜𝑠𝑠 = … 2 2 2

𝑥 … 𝑥 𝑦 1 𝑑 𝑥 𝑥2 𝑥 𝑥 1 𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒 1 𝑑−1 𝑑

𝑖 … … … …

### σ 𝐶𝐸(1 𝑖 , 𝑝) = 𝑋 = 𝑦 =

𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒 𝑖=1 𝑦

… … … … 1 𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒 𝑖

… … … …

### − σ log 𝑝 𝑖

𝑖=1 𝑦 𝑏𝑠 𝑏𝑠 𝑏𝑠 𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒 𝑥 … … 𝑥 𝑦

1 𝑑

## Logistic

## Regression

2D dataset

Forward propagation Backward propagation

1 1 𝑦 𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒×2

- ℎ = 𝑋𝑊 + 𝑏 ∈ ℝ with X ∈ 𝜕𝑙 𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒×2

- = 𝑝 − … ∈ ℝ

𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒×𝑑 𝑑×2 𝜕ℎ ℝ and 𝑊 ∈ ℝ and 𝑏 ∈ 1

𝑏𝑠 𝑦 1×2 ℝ

𝜕𝑙 1 𝜕𝑙

𝑇 𝑑×2

- = 𝑋 ∈ ℝ 𝜕𝑊 𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒 𝜕ℎ
- 𝑝 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 ℎ, 𝑑𝑖𝑚 = 1 = ൤𝑝1 = 𝜕𝑙 1 𝜕𝑙
- = 𝑠𝑢𝑚(, 𝑑𝑖𝑚 = 0)

ℎ1 ℎ2 𝑒 𝑒 𝜕𝑏 𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒 𝜕ℎ

𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒×2, 𝑝2 = ൨ ∈ ℝ ℎ1 ℎ2 ℎ1 ℎ2 𝑒 +𝑒 𝑒 +𝑒

1 𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒 𝑖

- 𝑏𝑎𝑡𝑐ℎ_𝑙𝑜𝑠𝑠 = σ 𝐶𝐸(1 𝑖 , 𝑝) = 𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒 𝑖=1 𝑦 SGD update 1 𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒 𝑖 − σ log 𝑝 𝑖 𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒 𝑖=1 𝑦 𝜕𝑙

- 𝑊 = 𝑊 − 𝜂 𝜕𝑊 𝜕𝑙
- 𝑏 = 𝑏 − 𝜂 𝜕𝑏

## AutoGrad in PyTorch

###  Declare a computational graph

###  Do backpropagation and compute the gradients

## Implement

Declare W with requires_grad = True Declare b with requires_grad = True

## Logistic

## Regression

## using AutoGrad

Compute the batch loss Do backpropagation

Use SGD to update W and b

## Thanks for your attention!

## Enjoy the tutorial 4!