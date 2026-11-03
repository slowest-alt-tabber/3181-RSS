COPYRIGHT WARNING: Copyright in these original lectures is owned by Monash University. You may transcribe, take notes, download or stream lectures for the purpose of your research and study only. If used for any other purpose, (excluding exceptions in the Copyright Act 1969 (Cth)) the University may take legal action for infringement of copyright.

Do not share, redistribute, or upload the lecture to a third party without a written permission!

### FIT3181/5215 Deep Learning

Week 02: Prelude to Deep Learning and

Feed-forward Neural Networks

CE/Lecturer: Dr Trung Le (Clayton)

Lecturer: Dr Binh Nguyen (Clayton) Lecturer: Dr Ting Fung Fung (Malaysia) Lecturer: Dr Arghya Pal (Malaysia) Lecturer: Dr Sailaja Rajanala (Malaysia)

Email: trunglm@monash.edu

Department of Data Science and AI Faculty of Information Technology, Monash University, Australia

2

# Outline

###  Prelude to Deep Learning

###  Multi-layered feed-forward neural networks (NNs)

o Evolution of feed-forward NNs o Parameterization of feed-forward NNs o Modern activation function o Forward propagation

###  Computational graph of feed-forward neural networks

###  Train feed-forward NNs

###  Further reading recommendations

o Dive into Deep Learning – Chapter 4: https://d2l.ai/chapter_multilayer-perceptrons/index.html o Deep Learning - Chapter 6 o Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow – Chapter 10

3

# Applications of Deep Learning

## Prelude to Deep Learning

5

# A Prelude to Deep Learning

###  A young and fast moving field, but with history

### dated back to 40s (connectionism)!

###  Rooted in machine learning and early pattern

### recognition theory.

###  Drawn heavily on knowledge of human brain,

### statistics and applied maths

[Image courtesy: @DeepLearningIT]

###  Central technology to build Artificial Intelligence

### systems.

6

# A prelude to deep learning

1990 Dark age of DL 2012

.. and TODAY, 152 – 1000 layers (Resnet) 4 nodes, 3 layers

GPT-5.6: (estimated) 2.2 trillion params

Truck, Advancements in AlexNet won Train, hardware: GPU, Boat,

ImageNet 2012 TPU Uber,

UAV, …… Massive datasets: ImageNet, MSCoco

[Machine Learning, Tom Mitchell, 1997] Early day of NN application

7

# Early history of DL

Ackley, Hinton and Sejnowsky Frank Rosenblatt (Cornell psychologist)

| probabilistic Hopfield networks| higher- | put weights to McCulloch-Pitts model | Coined

energy states << than lower-energy states | 2006 ‘Perceptron’ | guaranteed to work if separable mid-2000s

Coined Boltzmann machine

mid-1990s Marvin Minsky and Seymour Papert | published book “Perceptrons” | expressive neuron layers but didn’t know how to learn| Machine learning = neural networks

Hinton, Osindero, and Teh | published “A fast learning algorithm The heat for NN was over 1986 for deep belief nets” |

| learning with bigger and more 1949 1982

hidden layers was infeasible | 1985 1950 1969 1943 Resurgent of NN and connectionism

John Hopfield (Caltech physicist) | re-invented Autoencorder (AE) | | notice the analogy between Stacked Sparse AE: Google learns neurons and spin glasses ‘cat’ from 10 million youtube videos Donald Hebb (Canadian psychologist) | Hebb’s rule::“Neurons that fire together wire together” | Cornerstone of connectionism: attempt to explain how neurons learn |

Rumerlhart, Hinton, and Williams Warren McCulloch and Walter Pitts

|invented Backprop | solved XOR problem | | first formal model of a neuron | aka McCulloch-

trained multiplayer perceptron (NETtalk) | Pitts model | AND and OR gate, switch on when number of active inputs > T | Doesn’t learn! |

8

Yann LeCun Yoshua Bengio Geoff Hinton Director of AI Research, Facebook Head, Montreal Institute for Google Brain Toronto, University of

Learning Algorithms Torronto

9

# When DL works?

deep learning models

Computational infrastructure, researcher, environment and policy as

the launch pad

Big data as energy

resource

Andrew Ng’s Analogy

©Copyright Dinh Phung, 2019

10

# Breakthrough in object recognition

Increase in depth curve

Human performance

5.1% error

ResNet 2015 beats human performance in object recognition task!

©Dinh Phung, 2017 VIASM 2017, Data Science Workshop, FIRST 10

11

# DL applications

### Computer vision/NLP/Speech Analysis and big data

Drug Large Speech

discovery & Language Recognition & Recommender

Medical Models & Computer System

Image Generative AI Vision

Analysis

12

# DL applications

Nature, 2016

### Reinforcement learning

LeeLee  SedolSedol

13

# DL applications

### Reinforcement Learning

1997

2016 from search heuristics to ...

### AUTOMATED LEARNING!

## Other applications

Generative AI DALL.E/Mid Journey Chatbot/LLMs

Machine translation Video Surveillance and Analysis Tesla self-driving cars with seq2seq

14

15

# Where DL works and doesn’t work?

Cryptocurrency credit scoring high

social media Telco data

transaction data fraud data

s

entertainments

NLP movies

photo, images, faces

Current DL speech, audio data autonomous poetry works well Human law vehicles

arts Airplane engine data

manufacturing data low CERN

low Data and problems

high governed by physics

and natural laws

16

# Where does DL sit within ML field?

Approaches Problem Solved Solution Origin Solution

Symbolists Knowledge acquisition Logic, philosophy Inverse deduction

Steve Muggleton Ross Quinlan Tom Mitchell

Structure discovery Evolutionary Evolutionary Genetic programming

(nature) biology

John Koza John Holland Hod Lipson

Credit assignment Backpropagation and Connectionists Neuroscience

(nurture) deep learning

Geoff Hinton Yann LeCun Yoshua Bengio

Bayesian Uncertainty Statistics Probabilistic Inference

David Heckerman Judea Pearl Michael Jordan

Analogizers Similarity Psychology Kernel machines

Vladimir Vapnik Peter Hart Douglas Hofstadter

Image credit: Wikipedia Source: “The Master Algorithm”, Domingo

## Deep Learning Models in This Unit

19

# Discriminative machine learning

Prediction probabilities ML model 𝒉

𝑝 𝑥 = softmax(ℎ(𝑥)) ℎ 𝑥 𝑝1 𝑥 1 feed ℎ 𝑥 softmax 𝑝2 𝑥 predict

2 p(𝑥) Data point 𝑥 ℎ(𝑥) 𝑓 𝑥 = 𝑦ො ≔ argmax1≤𝑚≤10𝑝𝑚 (𝑥)

………… ………… Ground-truth label

ℎ 𝑥 𝑝10 𝑥 10 𝑦 = 8 ∈ {1, . . , 10}

Suffer loss 𝑙 𝑦, 𝑦ො = 𝐶𝐸 1𝑦, 𝑝 𝑥 = − log 𝑝𝑦

(1)  ℎ = −1, ℎ = ℎ = ℎ = ℎ = ℎ = 1, ℎ =

1 2 3 4 5 6 7

(2) 5, ℎ8 = 3, ℎ9 = ℎ10 = 0

(3) o Possibilities to assign data example 𝑥 to classes.

(4) (5)

(6)  𝑝 = 0.002, 𝑝 = 𝑝 = 𝑝 = 𝑝 = 𝑝 =

1 2 3 4 5 6

(7) 0.015, 𝑝7 = 0.8, 𝑝8 = 0.11, 𝑝9 = 𝑝10 = 0.005

(8) o 𝑦ො =7≠𝑦=8 → incorrect prediction. (9) o 𝑙 𝑦, 𝑦ො = CE 1, 𝑝 = −log p = −log𝑝 = (10)

𝑦 𝑦 8 − log 0.11 CIFAR 10 dataset

# Deep learning

Prediction probabilities ML model 𝒉

𝑝 𝑥 = softmax(ℎ(𝑥)) ℎ 𝑥 𝑝1 𝑥 1 feed ℎ 𝑥 softmax 𝑝2 𝑥 predict

2 p(𝑥) Data point 𝑥 ℎ(𝑥) 𝑓 𝑥 = 𝑦ො ≔ argmax1≤𝑚≤𝑀𝑝𝑚 (𝑥)

………… ………… Ground-truth label

ℎ 𝑥 𝑝𝑀 𝑥 𝑀 𝑦 ∈ {1, . . , 𝑀}

Suffer loss 𝑙 𝑦, 𝑦ො = 𝐶𝐸(1𝑦, 𝑝(𝑥))

Deep neural networks with multiple neural layers and neurons

❑ Several SOTA deep nets will be covered in this

unit: Vision Transformers Multilayered feedforward nets ❑ Multilayered feedforward neural nets

❑ For working with vectors, 1D tensors ❑ Convolutional neural nets/ViT

❑ For working with images, 2D/3D tensors Convolutional neural nets

❑ Recurrent neural nets/Transformers

❑ For working with sequences, sentences, texts

Recurrent neural nets Transformers 20

# Deep learning

Prediction probabilities ML model 𝒉

𝑝 𝑥 = softmax(ℎ(𝑥)) ℎ 𝑥 𝑝1 𝑥 1 feed ℎ 𝑥 softmax 𝑝2 𝑥 predict

2 p(𝑥) Data point 𝑥 ℎ(𝑥) 𝑓 𝑥 = 𝑦ො ≔ argmax1≤𝑚≤𝑀𝑝𝑚 (𝑥)

………… ………… Ground-truth label

ℎ 𝑥 𝑝𝑀 𝑥 𝑀 𝑦 ∈ {1, . . , 𝑀}

Suffer loss 𝑙 𝑦, 𝑦ො = 𝐶𝐸(1𝑦, 𝑝(𝑥))

Deep neural networks with multiple neural layers and neurons

Our main character today

❑ Several SOTA deep nets will be covered in this

unit: Vision Transformers Multilayered feedforward nets ❑ Multilayered feedforward neural nets

❑ For working with vectors, 1D tensors ❑ Convolutional neural nets/ViT

❑ For working with images, 2D/3D tensors Convolutional neural nets

❑ Recurrent neural nets/Transformers

❑ For working with sequences, sentences, texts

Recurrent neural nets Transformers 21

## Feed-forward Neural Networks

## Evolution of Feed-forward Neural Networks

𝑦ො

Output layer

𝐿 ℎ (𝑥) 𝐿 𝐿 𝑊, 𝑏

Hidden layer

𝐿−1 ℎ (𝑥) 𝐿−1 𝐿−1 1 𝑊, 𝑏

step Σ

.5 .5

2 2 𝑊, 𝑏 𝑤0

∑ step step

Hidden layer Σ Σ

1 ℎ (𝑥).5 .5 .5.5 𝑤1 𝑤2 𝑤 1 1

𝑀 𝑊, 𝑏

3 4 Input layer

### 𝑥 𝑥 … 𝑥 1 -2

1 2 𝑀

0 ℎ (𝑥) = 𝑥

Single Perceptron Multi-layered Perceptron Multi-layered Feedforward NN

23

24

# Deep NNs: parameterisation

Hidden layer Hidden layer

1 𝐿−1 Output layer ℎ (𝑥) … ℎ (𝑥) Input layer is the 𝑑- 𝐿

ℎ (𝑥) has 𝑀 nodes dimensional vector of

providing 0 ℎ (𝑥) = 𝑥

discriminative values

### 1 𝑑 1×𝑑 𝑦ො

𝑥 = 𝑥, … , 𝑥 ∈ ℝ

softmax layer for classification 2 2 … 𝑊, 𝑏

 Weight matrices and biases  Layer 𝑘 has 𝑛 neurons

𝑘

𝑘 𝑛𝑘−1×𝑛𝑘 𝑘 1×𝑛𝑘

- 𝑊 ∈ ℝ and 𝑏 ∈ ℝ for 𝑘 = 1,2, … , 𝐿

o 𝑛0 = 𝑑 and 𝑛𝐿 = 𝑀

25

# DNNs: making prediction via FP

𝑪𝑬 𝒙, 𝒚 = − log 𝑝𝑦(𝑥)

### Forward propagation- Classification

𝑦ො

𝐿 𝒑 𝒙 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥))

softmax 0 ℎ (𝑥) = 𝑥 with the ground-truth label 𝑦

Output layer for 𝑘 = 1 to 𝐿 − 1 do

𝑳 𝐿−1 𝐿 𝐿 𝑘 𝑘−1 𝑘 𝑘 𝒉 𝒙 = ℎ 𝑥 𝑊 + 𝑏 ℎത (𝑥) = ℎ (𝑥)𝑊 + 𝑏 //linear operation

𝐿 𝐿 𝑘 ത𝑘 𝑊, 𝑏 ℎ (𝑥) = 𝜎 ℎ (𝑥) //activation to bring non-linearity 𝐿 𝐿−1 𝐿 𝐿 Hidden layer ℎ (𝑥) = ℎ (𝑥)𝑊 + 𝑏

𝑳−𝟏 ത𝐿−1 𝒉 𝒙 = 𝜎(ℎ (𝑥)) 𝐿 𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) //prediction probabilities ഥ𝑳−𝟏 𝐿−2 𝐿−1 𝐿−1

𝒉 𝒙 = ℎ 𝑥 𝑊 + 𝑏 𝐿−1 𝐿−1 𝑊, 𝑏

## …

Forward 𝐿 𝑘 𝑘 Hidden layer  Parameter: 𝜃 = 𝑊, 𝑏 is given Propagation

𝑘=1 𝟐 ത2

𝒉 𝒙 = 𝜎(ℎ (𝑥)) ഥ𝟐 1 2 2 𝒉 𝒙 = ℎ (𝑥)𝑊 + 𝑏  𝜎: ℝ → ℝ is the activation function

2 2 𝑊, 𝑏

- Sigmoid, tanh, ReLU, and etc. 𝐿

 Output: 𝑝(𝑥) = softmax ℎ (𝑥)

Hidden layer 𝐿 exp{ℎ 𝑥 } 𝟏 ത1

𝑘 𝒉 𝒙 = 𝜎(ℎ (𝑥))

- Pr 𝑦 = 𝑘 𝑥 = 𝑝𝑘(𝑥) = 𝑀 𝐿 𝟏 1 1 ∑ exp{ℎ 𝑥 } 𝒉ഥ 𝒙 = 𝑥𝑊 + 𝑏 𝑗=1 𝑗

1 1

- 𝑦ො = argmax 𝑝𝑘(𝑥) 𝑊, 𝑏 1≤𝑘≤𝑀 Input layer 0 ℎ (𝑥) = 𝑥

## Modern Activation Functions 1

𝜎 𝑧 = 𝑠 𝑧 = ′

1 + exp{−𝑧} 𝜎 𝑧 = 𝜎(𝑧)(1 − 𝜎(𝑧))  The sigmoid function sigmoid 𝟏

o 𝒔 𝒛 = Σ −𝒛

𝟏+𝒆 o Logistic S-shaped, continuous, and differentiable o Output range: 0 to 1

exp{𝑧} − exp{−𝑧} 𝜎 𝑧 = 𝑡𝑎𝑛ℎ 𝑧 = ′ 2

𝜎 𝑧 = 1 − 𝜎 (𝑧) exp 𝑧 + exp{−𝑧} tanh  The hyperbolic tangent function Σ 𝒛 −𝒛

𝒆 −𝒆 o 𝐭𝐚𝐧𝐡 𝒛 =

𝒛 −𝒛 𝒆 +𝒆 o Logistic S-shaped, continuous, and differentiable o Output range: −1 to 1, hence help speed up

convergence.

ReLU

1 𝑖𝑓 𝑧 ≥ 0 ′ Σ 𝜎 𝑧 = ቊ

 The rectified linear function (ReLU) 𝜎 𝑧 = 𝑅𝑒𝐿𝑈 𝑧 = max{0, 𝑧}

0 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒 o ReLU 𝒛 = 𝐦𝐚𝐱 𝟎, 𝒛 o Continuous, but not differentiable at 0 o Hence, can make GD bounce around o Fast to compute, works very well in practice! o No cap on output value, hence help with gradient

vanishing problem.

26

## Importance of Activation Functions

Aims to bring non-linearities to neural networks

Our data Networks without activations Networks with activations

27

## Linear and Softmax Layers

Input Hidden Hidden

Hidden Output Layer Layer Prediction Probabilities

Layer 1 Layer … Layer 0 𝐿 ℎ 𝑥 = 𝑥 1 … 𝐿−1 𝐿 𝒑 𝒙 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥))

ℎ 𝑥 ℎ 𝑥 ℎ 𝑥

softmax CE loss 𝑥 = 1.2 −1 2

𝑙𝑜𝑠𝑠 = 𝐶𝐸 1𝑦, 𝑝 𝑥 = − log 𝑝𝑦(𝑥) 𝑦 ∈ {1,2, … , 𝑀}

ഥ𝟏 𝑳 𝒉 act 𝟏 1 𝒍𝒐𝒈𝒊𝒕𝒔 = 𝒉 𝒙

𝒉 = 𝜎(ℎത) 1 𝐿−1 𝐿 𝐿 = 𝑥𝑊 + 𝑏1 𝜎 = ℎ 𝑥 𝑊 + 𝑏

non-linearity

h (logits) p (pred prob)

Penultimate Output Prediction Probabilities 𝒉

𝟏 𝒆 Layer Layer 𝐿

𝒑 𝒙 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) 𝒑𝟏 = softmax 𝒉𝟏 𝒉𝟐 𝒉𝟑 𝒉𝟒

𝒆 + 𝒆 + 𝒆 + 𝒆 𝒉𝟏 𝒑𝟏

𝒉𝟐 𝒉 𝒆 𝟐 𝒑

𝟐 𝒑 = softmax 𝟐

𝒉𝟏 𝒉𝟐 𝒉𝟑 𝒉𝟒 𝒆 + 𝒆 + 𝒆 + 𝒆 𝒑𝟑 𝒉𝟑

𝒉𝟑 𝒆 𝒉 𝒑𝟒 𝒑 = 𝟒 𝟑 𝒉 𝒉 𝒉 𝒉 real-valued scores probabilities 𝒆 𝟏 + 𝒆 𝟐 + 𝒆 𝟑 + 𝒆 𝟒

Logit ℎ = ℎ ℎ ℎ ℎ Prob p = 𝑝 𝑝 𝑝 𝑝 𝒉𝟒

1 2 3 4 1 2 3 4 𝒆 transforms real-valued discriminative scores 𝒑 =

𝟒 𝒉 𝒉 𝒉 𝒉

𝒆 𝟏 + 𝒆 𝟐 + 𝒆 𝟑 + 𝒆 𝟒 (ranged in (−∞, +∞)) to probability values (ranged in [0, 1]) but preserving the order.

28

29

# What is a softmax layer?

Output layer

𝐿 ℎ (𝒙)

 Softmax function transforms real-valued

predicted class 𝑦ො discriminative scores ℎ to discrete probabilities 𝑝.

𝐿 𝑝 = softmax ℎ  Input: vector of dimension 𝑀

𝑦ො = argmax 𝑝𝑚 o e.g., ℎ = ℎ1, ℎ2, ℎ3 = 1, 2, 3 , M = 3 1≤𝑚≤𝑀  Output: a discrete distribution of dimension 𝑀

o e.g., 𝑝 = 𝑝, 𝑝, 𝑝 = [0.09, 0.24, 0.67] h (logits) p (pred prob)

1 2 3  How to calculate 𝑝?

𝑚

softmax 𝑀 𝑀 exp{ℎ𝑚} o ℎ = ℎ𝑚 → 𝑝 = softmax ℎ ≔

𝑚=1 𝑀

∑ exp{ℎ } 𝑖=1 𝑖

𝑚=1 𝑀 o 𝑝 = 𝑝𝑚 becomes a distribution over classes {1, … , 𝑀}

𝑚=1

𝑀 ▪ 𝑝 ≥ 0 (1 ≤ 𝑚 ≤ 𝑀) and ∑ 𝑝 = 1 . real-valued scores probabilities

𝑚 𝑚=1 𝑚

transforms real-valued discriminative scores  Prediction step: return the prediction for the class (ranged in (−∞, +∞)) to probability values (ranged

in [0, 1]) but preserving the order. that has the highest probability

𝑦ො = argmax 𝑝𝑚

1≤𝑚≤𝑀

- i.e., return 3 in the above example (why?)

30

# DNNs: making prediction via FP

### Example of spam email detection

3  From emails, assume that we extract three features, hence data points 𝑥 ∈ ℝ

3 o 𝑥 = 𝑥1, 𝑥2, 𝑥3 ∈ ℝ o There are two classes and labels: spam (𝑦 = 1) and non-spam (𝑦 = 2)

Input Hidden Hidden Output o Network architecture:

Layer Layer 1 Layer 2 Layer o 3 → 4 𝑠𝑖𝑔𝑚𝑜𝑖𝑑 → 3(𝑠𝑖𝑔𝑚𝑜𝑖𝑑) → 2 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 1 2 3

0 ℎ 𝑥 ℎ 𝑥 = 𝑥 ℎ 𝑥 ℎ 𝑥 0 1×3 3

𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) o ℎ 𝑥 = 𝑥 = 1.2, −1,2.2 ∈ ℝ

spam o Hidden layer 1

𝑥 = 1.2 −1 2

non-spam ത1 0 1 1 1×4 o ℎ 𝑥 = ℎ 𝑥 𝑊 + 𝑏 ∈ ℝ (linear operation)

𝑦 = 1 4 1 ത1 ത1 1×4 2 4×3 3 3×2 o ℎ 𝑥 = 𝑠𝑖𝑔𝑚𝑜𝑖𝑑 ℎ 𝑥 = 𝑠𝑖𝑔𝑚𝑜𝑖𝑑 ℎ 𝑥 ∈ ℝ (activation) 1 3×4

𝑖 𝑊 ∈ ℝ 𝑊 ∈ ℝ 𝑊 ∈ ℝ

𝑖=1

1 1×4 2 1×3 3 1×2 𝑏 ∈ ℝ 𝑏 ∈ ℝ 𝑏 ∈ ℝ o Hidden layer 2

ത2 1 2 2 1×3 o ℎ 𝑥 = ℎ 𝑥 𝑊 + 𝑏 ∈ ℝ (linear operation)

2 ത2 1×3 o ℎ 𝑥 = 𝑠𝑖𝑔𝑚𝑜𝑖𝑑 ℎ 𝑥 ∈ ℝ (element-wise activation)

0 ℎ (𝑥) = 𝑥 o Output layer

for 𝑘 = 1 to 2 do 3 2 3 3 1×2 o ℎ 𝑥 = ℎ 𝑥 𝑊 + 𝑏 ∈ ℝ (linear operation) 𝑘 𝑘−1 𝑘 𝑘

ℎത = ℎ (𝑥)𝑊 + 𝑏 //linear operation 3 1×2 o 𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 ℎ 𝑥 ∈ ℝ (softmax activation) 𝑘 ത𝑘

ℎ (𝑥) = 𝜎 ℎ (𝑥) //activation 3 2 3 2 o Assume that 𝑝 𝑥 = 0.3,0.7 ℎ (𝑥) = ℎ (𝑥)𝑊 + 𝑏

3 𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) //prediction probabilities o Prediction 𝑦ො = 2 ≠ 𝑦 = 1→ incorrect prediction

Exercise: check all the dimensions for matrix multiplication consistency! What is the cross-entropy loss in this case?

31

# DNNs: making prediction via FP

### Forward propagation- Regression

Output layer

𝐿 0 ℎ (𝑥) ℎ (𝑥) = 𝑥

𝐿 𝐿 𝑊, 𝑏 for 𝑘 = 1 to 𝐿 − 1 do

ത𝑘 𝑘−1 𝑘 𝑘 ℎ (𝑥) = ℎ (𝑥)𝑊 + 𝑏 //linear operation Hidden layer 𝑘 𝑘 𝐿−1 ℎ (𝑥) = 𝜎 ℎത (𝑥) //activation ℎ (𝑥) 𝐿 𝐿−1 𝐿 𝐿 𝐿−1 𝐿−1 ℎ (𝑥) = ℎ (𝑥)𝑊 + 𝑏 𝑊, 𝑏

## …

2 2 𝑊, 𝑏 𝐿 𝑘 𝑘  Parameter: 𝜃 = 𝑊, 𝑏

𝑘=1

Hidden layer

1  𝜎: ℝ → ℝ is the activation function ℎ (𝑥)

o Sigmoid, tanh, ReLU

1 1 𝑊, 𝑏  Output:

Input layer 𝐿 o 𝑦ො = ℎ (𝑥) 0

ℎ (𝑥) = 𝒙

𝑊 𝑊

2 𝑊 1,, 𝑏 𝑏 3 Forward propagation

2 1 ,

𝑏 3

ℎ Input Hidden Output

Hidden 0 (ℎ ℎ 𝑥 ℎ

2 1 (3) ((layer 𝑥 𝑥 = layer) layer 𝑥

)) layer 𝑥

Declare FFN

Making prediction

Declare one data example

## Layer View versus Neural View

34

# Neuron and Layer views

 Neuron (micro) view  Layer (macro) view

𝜎 0.5 Σ 1 1

2 2 ℎ ℎ 2 1 -0.5 𝜎 𝜎 0.5

Σ Σ

-1 2 ത2 ത2 ℎ ℎ 2 1

-1 1 1 -2 1 1 1 ℎ ℎ ℎ 1 2 3 -1.5 𝜎 𝜎 -.5 𝜎 0.5

3 Σ ത1 Σ Σ 1 ℎ -2 1 ത 2 ത ℎ ℎ 1 1 2 3

1 -1 1 2

1 𝑥1 𝑥

𝑥2 3

35

# Neuron and Layer views

 Neuron (micro) view  Layer (macro) view

1 1 −2

𝜎 0.5 1 1 𝑊 = −1 1 2, 𝑏 = [−1.5 − 0.5 0.5] Σ

3 2 1 1 1

2 2 ℎ ℎ 2 ഥ𝟏 ത1 ത1 ത1 1 𝒉 = ℎ ℎ ℎ 𝜎

1 2 3 -0.5 𝜎 0.5 1 1 −2 Σ Σ

-1 2 2 ത2 ത = 𝑥 𝑥 𝑥 + [−1.5 − 0,5 0.5] ℎ ℎ

1 2 3 −1 1 2 1 2

-1 1 1 -2 3 2 1

1 1 1 1 1 ℎ ℎ = 𝑊 𝑥 + 𝑏 1 2 ℎ

3 -1.5 𝜎 𝜎 -.5 𝜎

0.5 3 Σ ത1 Σ Σ 𝟏 1 1 ℎ -2

ത ത 2 1 𝒉 = 𝜎(ℎ) ℎ ത

1 1 2 ℎ

1 3 -1 1 2

1 ∗ ∗ 2 2 𝑥1 𝑥 𝑊 = ∗ ∗, 𝑏 = [∗∗] 𝑥2 3

∗ ∗

36

# Neuron and Layer views

 Neuron (micro) view  Layer (macro) view

1 1 −2

𝜎 0.5 1 1 𝑊 = −1 1 2, 𝑏 = [−1.5 − 0.5 0.5] Σ

3 2 1 1 1

2 2 ℎ ℎ 2 ഥ𝟏 ത1 ത1 ത1 1 𝒉 = ℎ ℎ ℎ 𝜎

1 2 3 -0.5 𝜎 0.5 1 1 −2 Σ Σ

-1 2 2 ത2 ത = 𝑥 𝑥 𝑥 + [−1.5 − 0,5 0.5] ℎ ℎ

1 2 3 −1 1 2 1 2

-1 1 1 -2 3 2 1 1 1 1 1 1 = 𝑊 𝑥 + 𝑏 ℎ ℎ ℎ

1 2 3 -1.5 𝜎 𝜎 -.5 𝜎 𝟏 ത1 0.5 𝒉 = 𝜎(ℎ) 3

Σ ത1 Σ Σ 1 ℎ -2 ത 2 1 ℎ ത 1 1 2 ℎ −1 2 1 3

-1 2 2 1 2 𝑊 = 1 1, 𝑏 = [−0.5 0.5] 1

𝑥1 𝑥 −1 −2 𝑥2 3

ഥ𝟐 ത2 ത2 1 2 2 𝒉 = ℎ ℎ = ℎ 𝑊 + 𝑏

1 2

37

# Neuron and Layer views

 Neuron (micro) view  Layer (macro) view

1 1 −2 𝑥 = 𝑥1 𝑥2 𝑥3 1 1 𝜎 0.5 𝑊 = −1 1 2, 𝑏 = [−1.5 − 0.5 0.5]

Σ 1 3 2 1 1 1 1 1

𝑊 = −1, 𝑏 = −1.5 2 1 1 𝟏 1 1 1 2 ℎ ഥ ത ത ത ℎ 2 𝒉 = ℎ ℎ ℎ 1 3

1 2 3

-0.5 𝜎 𝜎 0.5 1 1 −2

Σ Σ ഥ𝟏 1 1

-1 2 2 𝒉 = 𝑥𝑊 + 𝑏 = 𝑥 𝑥 𝑥 −1 1 2 + [−1.5 − 0,5 0.5] ത2 ത 𝟏 1 1

1 2 3 ℎ ℎ

1 2 𝟏 1 -1 1 1 -2 ത

𝒉 = 𝜎(ℎ) 3 2 1 𝟏 1 1 1 1 1 = 𝑊 𝑥 + 𝑏 1

ℎ ℎ ℎ 1 1 2 3 -1.5 𝜎 𝜎 -.5 𝜎 1 1

0.5 𝑊 = 1 𝑏 = −0.5 𝟏 ത1 2 2 𝒉 = 𝜎(ℎ) 3

Σ ത1 Σ Σ 1 ℎ -2 ത 2 1 2 ℎ ത 1 1 2 ℎ −1 2 1 3 𝟏 1 1

-1 𝒉ഥ = 𝑥𝑊 + 𝑏 2 2 1 2 𝟐 2 2 𝑊 = 1 1, 𝑏 = [−0.5 0.5] 1

𝟏 ത1 𝒉 = 𝜎(ℎ) 𝑥 𝟐 2 −1 −2 1 𝑥 𝑥

2 3

−2 1 1 ഥ𝟐 ത2 ത2 1 2 2 𝑊 = 2 𝑏 = 0.5 𝒉 = ℎ ℎ = ℎ 𝑊 + 𝑏 3 3

1 2

1 ഥ𝟏 1 1 𝒉 = 𝑥𝑊 + 𝑏 𝟑 3 3 𝟏 ത1 𝒉 = 𝜎(ℎ) 𝟑 3

## Computational graph

40

# Computational graph (used in PyTorch)

Problem: 𝑓 𝑥, 𝑦, 𝑧 = 𝑥 + 𝑦 𝑧

𝒇 𝒙, 𝒚, 𝒛 = 𝒙 + 𝒚 𝒛 What is the output evaluated at 𝑥 = 2, 𝑦 = 3, 𝑧 = −2 ?

𝑓 = 𝑢𝑧 = −10 Step 1:

\* a) construct computational graph b) forward propagation 𝑢

𝑧 c) record value at each node

\+

𝑥 𝑦

## Computational graph of feedforward nets

3 3 ℎ, 𝑝 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ) 𝑝 softmax

3 3×2 𝑊 ∈ ℝ 3 3 1×2 ℎ + 𝑏 ∈ ℝ

ത2 2 ത2 ℎ, ℎ = 𝜎(ℎ) 3

- 𝑏 2 4×3 𝑊 ∈ ℝ 3 2 1×3

2 𝜎 𝑊 𝑏 ∈ ℝ ℎ

ത1 1 ത1 2 ℎ, ℎ = 𝜎(ℎ) ℎത +

1 3×4 𝑊 ∈ ℝ 2

- 𝑏 1 1×4 𝑏 ∈ ℝ

1 2 ℎ 𝜎 𝑊 0

ℎ = 𝑥

0 1 ℎ (𝑥) = 𝑥 ℎത +

for 𝑘 = 1 to 2 do

ത𝑘 𝑘−1 𝑘 𝑘 ℎ = ℎ (𝑥)𝑊 + 𝑏 //linear operation

- 1 𝑏 𝑘 ത𝑘 ℎ (𝑥) = 𝜎 ℎ (𝑥) //activation 3 2 3 2 ℎ (𝑥) = ℎ (𝑥)𝑊 + 𝑏

0 1 ℎ = 𝑥 𝑊

3 𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) //prediction probabilities

41

## Training deep neural networks

44

# Training a deep NN

###  We have learned how to make prediction if the model had already trained.

###  Machine learning is all about how to train these models from data!

o Training set 𝐷 = { 𝑥1, 𝑦1 , … , (𝑥𝑁, 𝑦𝑁)}

###  Back Propagation is currently the best approach to train a DNN.

45

# Training deep nets

### 𝑦ො

## 𝑥

suffer loss 𝑙 𝑦, 𝑦ො = CE 1𝑦, 𝑝 𝑥

= − log 𝑝𝑦(𝑥) 2 2 … 𝑊, 𝑏 Training set Loss function

𝑁 𝑁 𝐷 = { 𝑥1, 𝑦1 , … (𝑥𝑁, 𝑦𝑁)} 1 1

𝐿 𝐷; 𝜃 ≔ ෍ 𝐶𝐸 1𝑦, 𝑝 𝑥𝑖 = − ෍ log 𝑝𝑦 (𝑥𝑖)

𝑖 𝑖 𝑁 𝑁

𝑖=1 𝑖=1

(negative log likelihood) 𝐿 𝑙 𝑙  DNN parameters: 𝜃 ≔ 𝑊, 𝑏

𝑙=1  Find model parameters (weight matrices and biases) so that the

model predictions fit the training set as much as possible:

min 𝐿(𝐷; 𝜃) 𝜃  Use optimizers SGD, Adagrad, Adam, RMSProp to update the

model parameters (lectures 4)

𝑏 1 𝑏𝑎𝑡𝑐ℎ_𝑙𝑜𝑠𝑠 = ෍ 𝐶𝐸(1𝑦, 𝑝𝑖) Prediction

𝑖 𝑏

𝑖=1

## Mini-batch feed-forward probabilities

𝑛3 = 𝑀 = 4

1 𝑝1 𝑦1  Input

𝑝2 1𝑦 2 𝑏 = 32 𝑏 = 32 o Batch 𝑋: [𝑏, 𝑛0 = 𝑑 = 5] (b is Labels 𝑌 𝑃

the batch size)

𝑝𝑏 1𝑦

𝑏 One-hot labels 𝑛3 = 𝑀 = 4  Hidden layer 1

𝑙𝑜𝑔𝑖𝑡1 𝑦1 = 3, 𝑀 = 4 𝟏 𝟏 𝟏 o 𝒉 = 𝝈(𝑿𝑾 + 𝒃) 𝑙𝑜𝑔𝑖𝑡2

Output layer 1𝑦 = [0,0,1,0]

1 3 3 𝑏 = 32 o Tensor [𝑏, 𝑛 = 7] ℎ (𝑥) ℎ

1

3 5×4 𝑙𝑜𝑔𝑖𝑡𝑏 𝑊 ∈ ℝ,  Hidden layer 2

3 1×4 𝑏 ∈ ℝ 𝑛 = 5 𝟐 𝟏 𝟐 𝟐 2 o 𝒉 = 𝝈(𝒉 𝑾 + 𝒃)

Hidden layer o Tensor [𝑏, 𝑛 = 7] 2

2 2

ℎ (𝑥) ℎ

𝑏 = 32  Output layer 2 7×5

𝑊 ∈ ℝ, 𝟑 𝟐 𝟑 𝟑 2 1×5 o 𝒉 = 𝒉 𝑾 + 𝒃 𝑏 ∈ ℝ

𝑛1 = 7 3 o 𝑃 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ, 𝑑𝑖𝑚 = 1) Hidden layer 1

ℎ 1 ℎ (𝑥)

𝑏 = 32 o Tensor P: [𝑏, 𝑛𝐿 = 𝑀 = 4]

1 5×7 Training set 𝑊 ∈ ℝ,  The loss of the batch 1 1×7

𝑏 ∈ ℝ Mini-batches 𝑑 = 𝑛0 = 5 1 𝑏 o ∑ 𝐶𝐸(1, 𝑝) = Input layer

𝑖=1 𝑦𝑖 𝑖 𝑥1 𝑏

0 1 ℎ (𝑥) = 𝑥

𝑏 𝑖 − ∑ log 𝑝 𝑥

𝑖=1 𝑦𝑖 𝑥, 𝑦, … , (𝑥, 𝑦), 2 𝑏 = 32 𝑏 1 1 𝑏 𝑏

Form a mini-batch 𝑥, 𝑦, … , (𝑥, 𝑦) (batch size) 𝑏+1 𝑏+1 2𝑏 2𝑏 o Update weight matrices and

…………………………. 𝑥𝑏 biases to minimize the batch loss.

𝑥, 𝑦, … , (𝑥, 𝑦) Data 𝑋 ∗ ∗ 𝑁 𝑁

46

Forward propagation

Declare FFN

Making prediction

Declare a batch

## Training FFNs using Mini-batches

Training set 1 epoch

𝒚 = 𝒄𝒂𝒕 𝒚𝟐 = 𝒅𝒐𝒈 𝒚𝟔𝟒 = 𝒍𝒊𝒐𝒏 𝒚 = 𝒔𝒑𝒊𝒅𝒆𝒓 𝒚 = 𝒄𝒐𝒘 𝒚 = 𝒄𝒂𝒕 𝟏 ∗ ∗+𝟔𝟑 𝟔𝟎𝟎𝟎𝟎

𝒙 𝒙𝟐 … 𝒙 𝒙 𝟏 … 𝒙 … … ∗ … 𝒙 𝟔𝟎𝟎𝟎𝟎

… 𝟑𝟐 ∗+𝟑𝟏 … …

… … …

A mini-batch Another mini-batch

One vector: [1, 5] Our FFN

→ Batch: [32, 5]

32 1 Input Hidden Hidden Output 𝑏𝑎𝑡𝑐ℎ_𝑙𝑜𝑠𝑠(𝜃) = ෍ 𝐶𝐸(1, 𝑝 𝑥, 𝜃 )

𝑦𝑖 𝑖 … Layer 32 𝑖=1

Layer Layer 1 Layer 2

1 2 3 0 ℎ 𝑥 … ℎ 𝑥 = 𝑥 ℎ 𝑥 ℎ 𝑥

3 𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) 32 Optimizer

………

- Adam,

… - RMSProp,

- SGD

5 1 3×4 2 4×3 3 3×2

𝑊 ∈ ℝ 𝑊 ∈ ℝ 𝑊 ∈ ℝ

1 4×1 2 3×1 3 2×1 𝑏 ∈ ℝ 𝑏 ∈ ℝ 𝑏 ∈ ℝ

min 𝑏𝑎𝑡𝑐ℎ_𝑙𝑜𝑠𝑠(𝜃) 𝜃 𝑙 𝑙 𝐿 Testing set Model parameters 𝜃 = 𝑊, 𝑏

𝑙=1

… … … ……. … … …

Test accuracy 𝒙′ 𝒙′ 𝒙′𝟏𝟎𝟎𝟎

𝟏 𝟐

48

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

49

## Summary

### ❖ Prelude to deep learning

❖ Feed-forward NN (vector data), Convolutional NN (image, 2D-3D grid data), Recurrent NN (sequential data).

### ❖ Perceptron

❖ Stack perceptron and activation function → deep neural nets.

### ❖ Feed-forward neural networks

❖ Parameterization ❖ Forward propagation and compute the cross-entropy loss ❖ Computational graph of deep nets ❖ Train deep nets ❖ Representation learning of deep nets

50

## Thanks for your attention!