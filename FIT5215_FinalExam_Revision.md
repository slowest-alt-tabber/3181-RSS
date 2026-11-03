COPYRIGHT WARNING: Copyright in these original lectures is owned by Monash University. You may transcribe, take notes, download or stream lectures for the purpose of your research and study only. If used for any other purpose, (excluding exceptions in the Copyright Act 1969 (Cth)) the University may take legal action for infringement of copyright.

Do not share, redistribute, or upload the lecture to a third party without a written permission!

## FIT3181/5215 Deep Learning

Revision for Final Exam

Lecturer: Trung Le Email: trunglm@monash.edu

Department of Data Science and AI Faculty of Information Technology, Monash University, Australia

2

# The big picture …

Lectures 3,6,10: Deep Computer Vision Convolutions, CNN, architectures, robustness Week 03 DL for Vision (I): Convolutional Neural Networks Week 06 DL for Vision (II): Network Architectures, Visualization,

Interpretability, and Robustness Week 10 ViT and Model Fine-tuning Techniques

Lectures 1,2,4,5: fundamentals Lectures 7,9: Deep learning for sequential, time-series

RNNs, LSTM, GRU, Seq2Seq, attention mechanism, Deep Learning/ML Deep Neural Networks Transformers, Self-Attention etc. Computation graphs Week 07 Deep learning for time-series and temporal data: RNNs, LSTMs Optimization, BackProp

Week 09 Advanced sequential models

Lecture 8: Discrete Learning Representation Week 01 Machine Learning and Mathematical Background Revisit

Word2Vec, Skip-Gram, CBOW Week 02 Prelude to Deep Learning and Feed-forward Neural

Networks Week 08 Representation and deep learning for language: Word2Vec Week 04 Back-Propagation and Optimization for Deep Learning Week 05 Practical skills in deep learning

Lecture 11: Deep Generative Models Generative Adversarial Network (GAN), Diffusion Models Week 11 Deep generative models: GANs and Diffusion Models

3

# Knowledge graph of Deep Learning unit

Deep Learning models

Supervised models Self-Supervised models Generative models

Feed-forward Convolution Recurrent Advanced Generative

NN NN NN sequential Word2Vec Adversarial

modes Networks

Pretext: Vector data Vision data Sequential data predicts

Diffusion ViT target or Visualize LSTM, GRU

context words Models Attack/defense Attention mechanism Seq2Seq Transformer BERT

▪ Skip-Gram ▪ CBOW How to train supervised models?

Self-

Model Fine-tuning Define loss function Attention

𝑵 1 - Prompt-tuning, LoRA, Adapter

mechanism min𝜃 𝐽 𝜃 = 𝛺 𝜃 + ෍ 𝑙 𝑦, 𝑓 𝑥; 𝜃

𝑖 𝑖 𝑁

𝒊=𝟏 Forward propagation ❖ Input → Layers → Prediction Probabilities → Loss Backward propagation ❖ Loss → Layers → Input

## Feed-forward NNs

5

# Deep NNs: parameterisation

Hidden layer Hidden layer

1 𝐿−1 Output layer ℎ (𝑥) … ℎ (𝑥) Input layer is the 𝑑- 𝐿

ℎ (𝑥) has 𝑀 nodes dimensional vector of

providing 0 ℎ (𝑥) = 𝑥

discriminative values

## 1 𝑑 1×𝑑 𝑦ො

𝑥 = 𝑥, … , 𝑥 ∈ ℝ

softmax layer for classification 2 2 … 𝑊, 𝑏

 Weight matrices and biases  Layer 𝑘 has 𝑛 neurons

𝑘

𝑘 𝑛𝑘−1×𝑛𝑘 𝑘 1×𝑛𝑘

- 𝑊 ∈ ℝ and 𝑏 ∈ ℝ for 𝑘 = 1,2, … , 𝐿

o 𝑛0 = 𝑑 and 𝑛𝐿 = 𝑀

6

# DNNs: making prediction via FP

𝑪𝑬 𝒙, 𝒚 = − log 𝑝𝑦(𝑥)

### Forward propagation- Classification

𝑦ො

𝐿 𝒑 𝒙 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥))

softmax 0 ℎ (𝑥) = 𝑥 with the ground-truth label 𝑦

Output layer for 𝑘 = 1 to 𝐿 − 1 do

𝑳 𝐿−1 𝐿 𝐿 𝑘 𝑘−1 𝑘 𝑘 𝒉 𝒙 = ℎ 𝑥 𝑊 + 𝑏 ℎത (𝑥) = ℎ (𝑥)𝑊 + 𝑏 //linear operation

𝐿 𝐿 𝑘 ത𝑘 𝑊, 𝑏 ℎ (𝑥) = 𝜎 ℎ (𝑥) //activation to bring non-linearity 𝐿 𝐿−1 𝐿 𝐿 Hidden layer ℎ (𝑥) = ℎ (𝑥)𝑊 + 𝑏

𝑳−𝟏 ത𝐿−1 𝒉 𝒙 = 𝜎(ℎ (𝑥)) 𝐿 𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) //prediction probabilities 𝑳−𝟏 𝐿−2 𝐿−1 𝐿−1

𝒉ഥ 𝒙 = ℎ 𝑥 𝑊 + 𝑏

𝐿−1 𝐿−1 𝑊, 𝑏

## …

Forward 𝐿 𝑘 𝑘 Hidden layer  Parameter: 𝜃 = 𝑊, 𝑏 is given Propagation

𝑘=1 𝟐 ത2

𝒉 𝒙 = 𝜎(ℎ (𝑥)) ഥ𝟐 1 2 2 𝒉 𝒙 = ℎ (𝑥)𝑊 + 𝑏  𝜎: ℝ → ℝ is the activation function

2 2 𝑊, 𝑏

- Sigmoid, tanh, ReLU, and etc. 𝐿

 Output: 𝑝(𝑥) = softmax ℎ (𝑥)

Hidden layer 𝐿 exp{ℎ 𝑥 } 𝟏 ത1

𝑘 𝒉 𝒙 = 𝜎(ℎ (𝑥))

- Pr 𝑦 = 𝑘 𝑥 = 𝑝𝑘(𝑥) = 𝑀 𝐿 𝟏 1 1 σ exp{ℎ 𝑥 } 𝒉ഥ 𝒙 = 𝑥𝑊 + 𝑏 𝑗=1 𝑗
- 𝑦ො = argmax 𝑝 (𝑥) 1 1 𝑘 𝑊, 𝑏 1≤𝑘≤𝑀 Input layer 0 ℎ (𝑥) = 𝑥

# Modern Activation Functions 1

𝜎 𝑧 = 𝑠 𝑧 = ′

𝜎 𝑧 = 𝜎(𝑧)(1 − 𝜎(𝑧)) 1 + exp{−𝑧}  The sigmoid function sigmoid 𝟏

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

7

# Importance of Activation Functions

Aims to bring non-linearities to neural networks

Our data Networks without activations Networks with activations

8

# Linear and Softmax Layers

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

𝒆 + 𝒆 + 𝒆 + 𝒆 𝒉 𝒑𝟏 𝟏

𝒉𝟐 𝒉 𝒆 𝟐 𝒑

𝟐 softmax 𝒑𝟐 =

𝒉𝟏 𝒉𝟐 𝒉𝟑 𝒉𝟒 𝒆 + 𝒆 + 𝒆 + 𝒆 𝒑𝟑 𝒉𝟑

𝒉𝟑 𝒆 𝒉 𝒑𝟒 𝒑 = 𝟒 𝟑 𝒉 𝒉 𝒉 𝒉 real-valued scores probabilities 𝒆 𝟏 + 𝒆 𝟐 + 𝒆 𝟑 + 𝒆 𝟒

Logit ℎ = ℎ ℎ ℎ ℎ Prob p = 𝑝 𝑝 𝑝 𝑝 𝒉𝟒

1 2 3 4 1 2 3 4 𝒆 transforms real-valued discriminative scores

𝒑𝟒 =

𝒉𝟏 𝒉𝟐 𝒉𝟑 𝒉𝟒 (ranged in (−∞, +∞)) to probability values (ranged 𝒆 + 𝒆 + 𝒆 + 𝒆 in [0, 1]) but preserving the order.

9

10

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

σ exp{ℎ } 𝑖=1 𝑖

𝑚=1 𝑀 o 𝑝 = 𝑝𝑚 becomes a distribution over classes {1, … , 𝑀}

𝑚=1

𝑀 ▪ 𝑝 ≥ 0 (1 ≤ 𝑚 ≤ 𝑀) and σ 𝑝 = 1 . real-valued scores probabilities

𝑚 𝑚=1 𝑚

transforms real-valued discriminative scores  Prediction step: return the prediction for the class (ranged in (−∞, +∞)) to probability values (ranged

in [0, 1]) but preserving the order. that has the highest probability

𝑦ො = argmax 𝑝𝑚

1≤𝑚≤𝑀

- i.e., return 3 in the above example (why?)

11

# DNNs: making prediction via FP

### Example of spam email detection

3  From emails, assume that we extract three features, hence data points 𝑥 ∈ ℝ

3 o 𝑥 = 𝑥1, 𝑥2, 𝑥3 ∈ ℝ o There are two classes and labels: spam (𝑦 = 1) and non-spam (𝑦 = 2)

Input Hidden Hidden Output o Network architecture:

Layer Layer 1 Layer 2 Layer o 3 → 4 𝑠𝑖𝑔𝑚𝑜𝑖𝑑 → 3 𝑠𝑖𝑔𝑚𝑜𝑖𝑑 → 2 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 1 2 3

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

2 ത2 1×13 o ℎ 𝑥 = 𝑠𝑖𝑔𝑚𝑜𝑖𝑑 ℎ 𝑥 ∈ ℝ (element-wise activation)

0 ℎ (𝑥) = 𝑥 o Output layer

for 𝑘 = 1 to 2 do 3 2 3 3 1×2 o ℎ 𝑥 = ℎ 𝑥 𝑊 + 𝑏 ∈ ℝ (linear operation) 𝑘 𝑘−1 𝑘 𝑘

ℎത = ℎ (𝑥)𝑊 + 𝑏 //linear operation 3 1×2 o 𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 ℎ 𝑥 ∈ ℝ (softmax activation) 𝑘 ത𝑘

ℎ (𝑥) = 𝜎 ℎ (𝑥) //activation 3 2 3 2 o Assume that 𝑝 𝑥 = 0.3,0.7 ℎ (𝑥) = ℎ (𝑥)𝑊 + 𝑏

3 𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) //prediction probabilities o Prediction 𝑦ො = 2 ≠ 𝑦 = 1→ incorrect prediction

Exercise: check all the dimensions for matrix multiplication consistency! What is the cross-entropy loss in this case?

12

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

2 1 (3) ((layer 𝑥 𝑥 = ) layer 𝑥

) layer layer) 𝑥

Declare FFN

Making prediction

Declare one data example

14

# Training deep nets

## 𝑦ො

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

model parameters (Lecture 4)

𝑏 1 𝑏𝑎𝑡𝑐ℎ_𝑙𝑜𝑠𝑠 = ෍ 𝐶𝐸(1𝑦, 𝑝𝑖) Prediction

𝑖 𝑏

𝑖=1

# Mini-batch feed-forward probabilities

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

3 5×4 𝑙𝑜𝑔𝑖𝑡𝑏 𝑊 ∈ ℝ,  Hidden layer 2

3 1×4 𝑏 ∈ ℝ 𝑛 = 5 𝟐 𝟏 𝟐 𝟐 2 o 𝒉 = 𝝈(𝒉 𝑾 + 𝒃)

Hidden layer o Tensor [𝑏, 𝑛2 = 7]

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

15

Forward propagation

Declare FFN

Making prediction

Declare a batch

17

# Training FFNs using Mini-batches

Training set 1 epoch

𝒚 = 𝒄𝒂𝒕 𝒚𝟐 = 𝒅𝒐𝒈 𝒚𝟔𝟒 = 𝒍𝒊𝒐𝒏 𝒚 = 𝒔𝒑𝒊𝒅𝒆𝒓 𝒚 = 𝒄𝒐𝒘 𝒚 = 𝒄𝒂𝒕 𝟏 ∗ ∗+𝟔𝟑 𝟔𝟎𝟎𝟎𝟎

𝒙 𝒙𝟐 … 𝒙 𝒙 𝟏 … 𝒙 … … ∗ … 𝒙 𝟔𝟎𝟎𝟎𝟎

… 𝟑𝟐 ∗+𝟑𝟏 … …

… … …

A mini-batch Another mini-batch

One vector: [1, 5] Our FFN

→ Batch: [32, 5]

32 1 Input Hidden Hidden Output 𝑏𝑎𝑡𝑐ℎ_𝑙𝑜𝑠𝑠(𝜃) = ෍ 𝐶𝐸(1, 𝑝 𝑥, 𝜃 )

𝑦𝑖 𝑖 … 32

Layer 1 Layer 2 Layer 𝑖=1 Layer

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

## Optimization in Deep Learning

# A small detour to calculus

##  Calculus = mathematics of change (very important for deep learning)

##  Properties of derivative:

𝑓 𝑥+ℎ −𝑓(𝑥) ′ o 𝑓 𝑥 = ∇𝑓(𝑥) = lim

ℎ→0 ℎ ′ ′ o 𝑢𝑣 = 𝑢 𝑣 + 𝑢𝑣′

′ ′ 𝑢 𝑢 𝑣−𝑢𝑣′ o =

2 𝑣 𝑣 𝑢 𝑢 o 𝑒 ′ = 𝑢′𝑒

′ 𝑢 ′ o log 𝑢 =

𝑢

𝑛

##  Multi-variate function 𝑓: ℝ → ℝ with 𝑦 = 𝑓 𝑥 = 𝑓(𝑥, … , 𝑥).

1 𝑛 𝜕𝑓 o Gradient/derivative: (𝑎) = ∇𝑥𝑓(𝑎) = [∇𝑥 𝑓 𝑎 , ∇𝑥 𝑓 𝑎 , … , ∇𝑥 𝑓 𝑎 ].

𝜕𝑥 1 2 𝑛

##  Chain rule :

𝜕𝑢 𝜕𝑢 𝜕𝑣 o = ×

𝜕𝑥 𝜕𝑣 𝜕𝑥

19

20

# Derivative for multi-variate functions

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

𝜕𝑧 𝜕𝑧 𝜕𝑦 𝑚 o For x ∈ ℝ,∇ℎ 𝑥 = ∇𝑔 𝑓 𝑥 ×∇𝑓 𝑥 or equivalently = . .

𝜕𝑥 𝜕𝑦 𝜕𝑥 𝑝×𝑚 𝑝×𝑛 𝑛×𝑚

𝑥 𝑦 = 𝑓(𝑥) 𝑧 = 𝑔 𝑦 = 𝑔 𝑓 𝑥 = ℎ(𝑥)

𝑓 𝑔 𝑚 𝑛 𝑝 ℝ ℝ ℝ

ℎ

21

# Example

2 2 2 2

##  𝑦 = 𝑓 𝑥 = 𝑓 𝑥, 𝑥, 𝑥 = 𝑥 + 𝑥, 𝑥 + 𝑥 𝑥

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

# Example 𝒆

Penultimate Output Prediction Probabilities 𝒑 =

𝟏 𝒉 𝒉 𝒉 𝒉

𝒆 𝟏 + 𝒆 𝟐 + 𝒆 𝟑 + 𝒆 𝟒 Layer Layer 𝐿

𝒑 𝒙 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) Output layer

𝒉𝟐 𝒉𝟏 𝒑𝟏 𝒆

𝒑𝟐 = 𝒉 𝒉𝟏 𝒉𝟐 𝒉𝟑 𝒉𝟒 𝟐 𝒑 𝒆 + 𝒆 + 𝒆 + 𝒆

𝟐 softmax loss 𝑙 = 𝑙𝑜𝑠𝑠 = − log 𝑝2 1×3 𝒉𝟑

𝒑𝟑 ℎ 𝒆 𝑥 = 𝑥 𝑥 𝑥 ∈ ℝ 𝑒 2

1 2 2 𝒉

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

22

23

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

24

# How to code with PyTorch

ത ത  ℎ = 𝑥𝑊 + 𝑏 and ℎ = 𝑠𝑖𝑔𝑚𝑜𝑖𝑑(ℎ)

o ℎ = 𝑠𝑖𝑔𝑚𝑜𝑖𝑑(𝑥𝑊 + 𝑏) o 𝜎 = 𝑠𝑖𝑔𝑚𝑜𝑖𝑑 is the activation function 𝜕ℎ 𝜕ℎ 𝜕 ℎത

′ ത 𝑇 ത ത 𝑇 𝑇  = × = 𝑑𝑖𝑎𝑔 𝜎 ℎ 𝑊 = 𝑑𝑖𝑎𝑔 𝜎 ℎ ⊗ [1 − 𝜎 ℎ ] 𝑊 = 𝑑𝑖𝑎𝑔 ℎ ⊗ 1 − ℎ 𝑊

𝜕𝑥 𝜕 ℎത 𝜕𝑥 o ⊗ is element-wise product

Declare 𝑾, 𝒙, 𝒃 Forward propagation Backward propagation

# Recall optimization problem in deep learning

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

25

# Optimization problem in ML and DL

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

26

27

# Gradient and Hessian matrix

 Given an objective function 𝐽(𝜃) with 𝜃 = [𝜃, 𝜃, … , 𝜃 ]

1 2 𝑃 o For DL models

▪ 𝜃 includes weight matrices, filters, and biases which are trainable model parameters.

6 ▪ 𝑃 is the number of trainable parameters (𝑃 could be 20×10). ▪ 𝐽(𝜃) is the loss function over a training set.

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

28

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

29

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

# Numbers of local minima vs saddle points

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

30

31

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

# Gradient descend

Convex case

−𝛻 𝐽 𝜃 −𝛻 𝐽 𝜃 −𝛻 𝐽 𝜃 −𝛻𝜃 𝐽 𝜃𝑡 = 0

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

32

# Gradient descend

## Algorithm

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

33

# Example of Gradient Descend

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

34

35

# Gradient descent for deep learning

##  For training deep nets, we need to solve

1 𝑁 1 𝑁 o min 𝐿 𝐷; 𝜃 ≔ σ 𝑙 𝑥, 𝑦; 𝜃 = σ 𝑙 𝑦, 𝑓(𝑥; 𝜃)

𝑖=1 𝑖 𝑖 𝑖=1 𝑖 𝑖 𝜃 𝑁 𝑁

𝐿 exp{ℎ𝑦 (𝑥𝑖)}

𝑖 where 𝑙 𝑥𝑖, 𝑦𝑖; 𝜃 = − log 𝑝(𝑦 = 𝑦𝑖|𝑥𝑖) = − log is the loss incurred by

𝑀 𝐿 σ exp{ℎ (𝑥)} 𝑚=1 𝑚 𝑖 (𝑥𝑖, 𝑦𝑖).

##  Gradient descent update

𝜂 𝑁 o 𝜃 = 𝜃 − 𝜂∇ 𝐿 𝐷; 𝜃 = 𝜃 − σ ∇ 𝑙(𝑥, 𝑦; 𝜃) where 𝜼 > 𝟎 is a learning rate.

𝑡+1 𝑡 𝜃 𝑡 𝑡 𝑖=1 𝜃 𝑖 𝑖 𝑡

𝑁

1 𝑁 o To compute the gradient ∇ 𝐿 𝐷; 𝜃 = σ ∇ 𝑙(𝑥, 𝑦; 𝜃), we need to go through all

𝜃 𝑡 𝑖=1 𝜃 𝑖 𝑖 𝑡

𝑁 data points in 𝐷 → the computational cost is 𝑂(𝑁).

6

##  This is very computationally expensive for big datasets (𝑁 ≈ 10).

##  How to estimate the gradient ∇ 𝐿 𝐷; 𝜃 more efficiently?

𝜃 𝑡

# Stochastic gradient descent

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

36

# Example of Stochastic Gradient Descent

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

### ሚ 2 2 4 2

### ❑ 𝑓 𝑤 = [ 𝑤 − 1 + 𝑤 − 2 + 𝑤 − 3 + 𝑤 − 4 ]

4

### ሚ′

### ❑ 𝑓 𝑤 = 2𝑤 − 5

### ሚ′ ሚ′

### ❑ 𝑓 𝑤 = 𝑓 10 = 2×10 − 5 = 15

t

### ሚ′

### ❑ 𝑤 = 𝑤 − 𝜂 𝑓 𝑤 = 10 − 0.1×15 = 8.5.

𝑡+1 𝑡 𝑡

37

38

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

39

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

# Mini-batch feed-forward probabilities

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

Hidden layer o Tensor [𝑏, 𝑛2 = 7]

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

40

# SGD and SGD with momentum

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

41

# AdaGrad

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

42

## Convolutional Neural Network

# Convolution operation

### ❖𝑊 and 𝑥 are two tensors with the same shape, the convolutional operation between 𝑊 and 𝑥

### defined as

### 𝑊 ∗ 𝑥 = 𝑠𝑢𝑚(𝑊 ⊗ 𝑥)

### where 𝑊 ⊗ 𝑥 specifies the element-wise product and 𝑠𝑢𝑚 returns the sum of all elements in a

### tensor and multi-dimensional array.

𝑚×1 𝑊 𝑥1

1

###  𝟏𝑫 convolutional operation (𝑊, 𝑥 ∈ ℝ)

𝑊 = …, 𝑥 = … 𝑚 o 𝑊 ∗ 𝑥 = σ 𝑊 𝑥 𝑊 𝑥𝑚

𝑖=1 𝑖 𝑖 𝑚

𝑊 … 𝑊 𝑥11 … 𝑥1𝑛 𝑚×𝑛 11 1𝑛

###  𝟐𝑫 convolutional operation (𝑊, 𝑥 ∈ ℝ) … … …

𝑊 = … … … 𝑥 = 𝑚 𝑛 𝑊 … 𝑊 𝑥𝑚1 … 𝑥𝑚𝑛

𝑚1 𝑚𝑛 o 𝑊 ∗ 𝑥 = σ σ 𝑊 𝑥

𝑖=1 𝑗=1 𝑖𝑗 𝑖𝑗

𝑚×𝑛×𝑝

###  3𝑫 convolutional operation (𝑊, 𝑥 ∈ ℝ)

𝑚 𝑛 𝑝 o 𝑊 ∗ 𝑥 = σ σ σ 𝑊 𝑥

𝑖=1 𝑗=1 𝑘=1 𝑖𝑗𝑘 𝑖𝑗𝑘

44

45

# General architecture of CNNs

### 1\. Convolutional layer

### 2\. Pooling layer

### 3\. Fully connected layer

# Convolution layer with zero padding

𝒙 (input tensor (7,7)) strides = (2,2)

kernel size =(3,3) 0 0 0 0 0 0 0 0 0 feature map

𝑾 (filter or kernel) 0 1 -2 -1 5 3 2 1 0 1 1 1 3 8 19 7 0 1 3 -1 4 3 3 1 0 1 1 1 𝑓 = 3

ℎ 3 16 30 10

𝐻𝑜 = 4 0 1 -2 1 6 3 3 2 0 1 1 1

6 11 22 5 0 2 -2 2 5 2 1 0 0 𝑓 = 3

𝐻𝑖 = 7 𝑤

2 -2 8 3 0 0 3 -2 5 4 1 2 0

𝑊𝑜 = 4 0 1 2 -3 1 1 2 -1 0

 The sliding window moves from left → right, top → bottom with strides. 0 1 -2 -1 1 2 1 1 0

 We convolve the filter and the sliding windows to work out the neurons 0 0 0 0 0 0 0 0 0 on the feature map.

𝑊𝑖=7 𝑝 = 1 zero padding

46

# Convolution layer with zero padding

𝒙 (input tensor (7,7)) strides = (2,2)

kernel size =(3,3) 0 0 0 0 0 0 0 0 0 feature map

𝑾 (filter or kernel) 0 1 -2 -1 5 3 2 1 0 1 1 1 3 8 19 7 0 1 3 -1 4 3 3 1 0 1 1 1 𝑓 = 3

ℎ 3 15 30 10

𝐻𝑜 = 4 0 1 -2 1 6 3 3 2 0 1 1 1

6 11 22 5 0 2 -2 2 5 2 1 0 0 𝑓 = 3

𝐻𝑖 = 7 𝑤

2 -2 9 3 0 0 3 -2 5 4 1 2 0

𝑊𝑜 = 4 0 1 2 -3 1 1 2 -1 0

 The sliding window moves from left → right, top → bottom with strides. 0 1 -2 -1 1 2 1 1 0

 We convolve the filter and the sliding windows to work out the neurons 0 0 0 0 0 0 0 0 0 on the feature map.

𝑊𝑖=7 𝑝 = 1 zero padding

(Source: https://towardsdatascience.com/)

47

# Convolution layer

𝒙 (input tensor (7,7)) strides = (2,2)

kernel size =(3,3) 0 0 0 0 0 0 0 0 0 feature map

𝑾 (filter or kernel) 0 1 -2 -1 5 3 2 1 0 1 1 1 3 8 19 7 0 1 3 -1 4 3 3 1 0 1 1 1 𝑓 = 3

ℎ 3 15 30 10

𝐻𝑜 = 4 0 1 -2 1 6 3 3 2 0 1 1 1

6 11 22 5 0 2 -2 2 5 2 1 0 0 𝑓 = 3

𝐻𝑖 = 7 𝑤

2 -2 9 3 0 0 3 -2 5 4 1 2 0

𝑊𝑜 = 4 0 1 2 -3 1 1 2 -1 0

 The sliding window moves from left → right, top → bottom with strides. 0 1 -2 -1 1 2 1 1 0

 We convolve the filter and the sliding windows to work out the neurons 0 0 0 0 0 0 0 0 0 on the feature map.

𝑊𝑖=7 𝑝 = 1 zero padding

 𝑊, 𝐻: The width and height of the input image

𝑖 𝑖  𝑊, 𝐻: The width and height of the output image (feature map)

𝑜 𝑜

𝑊𝑖+2𝑝−𝑓𝑤 𝐻𝑖+2𝑝−𝑓ℎ 𝑊𝑜 = + 1 and 𝐻𝑜 = + 1

𝑠𝑤 𝑠ℎ 7+2×1−3 7+2×1−3 ❑ Our case: 𝑊 = + 1 = 4 and 𝐻 = + 1 = 4

𝑜 𝑜

2 2

48

# Convolution layer

𝒙 (input tensor (7,8)) strides = (2,2) 1 -2 -1 5 3 2 1 1

kernel size =(3,3) 1 3 -1 4 3 3 1 -1 feature map

𝑾 (filter or kernel) 1 -2 1 6 3 3 2 2 1 1 1 1 23 21

𝐻𝑖 = 7 2 -2 2 5 2 1 0 -1 𝑓 = 3 𝐻 = 3

1 1 1 ℎ 3 26 18 𝑜 0 3 -2 5 4 1 2 1 1 1 1

-1 8 13 1 2 -3 1 1 2 -1 2 𝑓 = 3

𝑤 𝑊 = 3

𝑜 1 -2 -1 1 2 1 1 3

𝑊𝑖 = 8 p=0 zero padding

49

# Convolution layer

𝒙 (input tensor (7,8)) strides = (2,2) 1 -2 -1 5 3 2 1 1

kernel size =(3,3) 1 3 -1 4 3 3 1 -1 feature map

𝑾 (filter or kernel) 1 -2 1 6 3 3 2 2 1 1 1 1 23 21

𝐻𝑖 = 7 2 -2 2 5 2 1 0 -1 𝑓 = 3 𝐻 = 3

1 1 1 ℎ 3 26 18 𝑜 0 3 -2 5 4 1 2 1 1 1 1

-1 8 13 1 2 -3 1 1 2 -1 2 𝑓 = 3

𝑤 𝑊 = 3

𝑜 1 -2 -1 1 2 1 1 3

𝑊𝑖 = 8 p=0 zero padding

 𝑊, 𝐻: The width and height of the input image

𝑖 𝑖  𝑊, 𝐻: The width and height of the output image (feature map)

𝑜 𝑜

𝑊𝑖+2𝑝−𝑓𝑤 𝐻𝑖+2𝑝−𝑓ℎ 𝑊𝑜 = + 1 and 𝐻𝑜 = + 1

𝑠𝑤 𝑠ℎ 8+0−3 7+0−3 ❑ Our case: 𝑊 = + 1 = 3 and 𝐻 = + 1 = 3

𝑜 𝑜

2 2

(Source: Internet)

50

# Convolution layer with multiple filters and feature maps

𝒙 (input tensor (4,7,8)) strides = (2,2)

𝑾𝟏(shape=(4,3,3)) feature map

1 1 -2 -1 5 3 2 1 1

1 1 1 1 23 21 1 3 -1 4 3 3 1 -1

1 1 1 3 26 18 feature volume (2,3,3) 1 -2 1 6 3 3 2 2 feature maps (2,3,3)

1 1 1

-1 8 13 𝐻𝑖 = 7 2 -2 2 5 2 1 0 -1

𝑾𝟐(shape=(4,3,3)) feature map

2 0 3 -2 5 4 1 2 1

-1 -1 -1 -1 -23 -21 1 2 -3 1 1 2 -1 2

-1 -1 -1 -3 -26 -18 1 -2 -1 1 2 1 1 3

-1 -1 -1 𝑊 = 8 1 -8 -13

𝑖

51

# Convolution layer with multiple filters and feature maps

𝒙 (input tensor (4,7,8)) strides = (2,2)

𝑾𝟏(shape=(4,3,3)) feature map

1 1 -2 -1 5 3 2 1 1

1 1 1 1 23 21 1 3 -1 4 3 3 1 -1

1 1 1 3 26 18 feature volume (2,3,3) 1 -2 1 6 3 3 2 2 feature maps (2,3,3)

1 1 1

-1 8 13 𝐻𝑖 = 7 2 -2 2 5 2 1 0 -1

𝑾𝟐(shape=(4,3,3)) feature map

2 0 3 -2 5 4 1 2 1

-1 -1 -1 -1 -23 -21 1 2 -3 1 1 2 -1 2

-1 -1 -1 -3 -26 -18 1 -2 -1 1 2 1 1 3

-1 -1 -1 𝑊 = 8 1 -8 -13

𝑖

Input images (batch_size,3,32,32)

Inputs (batch_size,4,7,8) Feature volume (batch_size,2,3,3)

CONV with filters (2,4,3,3) Feed through some layers

batch_size, in_channels , height, width,

out_channels, in_channels, fil_height, fil_width, 52

53

# Pooling operation

##  Makes the representations smaller

## and more manageable

##  Subsample the image

##  Operates over each feature map

## independently

(Source: FDL Ch. 5)

54

# Max/average pooling

❑ Max/average-pooling is locally invariant and applied

independently to each feature map/channel.

❑ In the SOTA CNNs (ResNet, GoogleNet, VGG), we

employ the pooling layer with ❑ Kernel/filter size = (2,2), strides = (2,2), padding= 0 ❑ Therefore, the output tensor has the size

224+0−2

### 𝑊𝑜 = + 1 = 112

2 224+0−2

### 𝐻𝑜 = + 1 = 112

2 ❑ With this setting, we down-sample the input size by 2

(Source: medium.com)

55

# Fully connected layer

Feature extractor Classifier

Fully connected layers [3,32,32]

flatten [5,5,10]

[1, 5x5x10]

###  The last tensor is flattened, and some fully connected layers are added to classify

### the input.

o The last 3D tensor [5,5,10] → 1 flat layer with 5x5x10= 250 neurons

Filters [4,3,4,4]

# CNN in

Filter 1

# Operation

𝑜𝑢𝑡_𝑠𝑖𝑧𝑒 𝑜𝑢𝑡_𝑠𝑖𝑧𝑒

15 + 2×1 − 2 32 + 0 − 4 softmax [3,4,4] =

= + 1 2

2 = 15 + 1 = 8 Filter 2

Feature volume Feature volume

Feature maps Feature maps

Pooling [3,4,4] layer

Filter 3

pool-size=(2,2) strides= (2,2) Input Layer padding= 1 [3,32,32]

[4,8,8] [4,15,15] [3,4,4] Latent Latent

representation representation Output Filter 4

layer 10 neurons for 10 classes MaxPool2d(kernel_size=2, stride=2, FC layer [3,4,4]

padding=1) 4x8x8 neurons Conv2D 1 padding= 0 Conv2d(3,4,kernel_size=4, stride=2, padding=0) strides = (2,2)

56

Filters [4,3,4,4]

4x8x8

# CNN in

Filter 1 2

10

# Operation

Feature volume 2 Feature volume Feature maps Feature maps

[2,4,8,8] [2,4,15,15] softmax [3,4,4]

[4,8,8] [4,15,15] Filter 2

Pooling layer [3,32,32] [3,4,4]

Filter 3

pool-size=(2,2) [4,15,15] [4,8,8]

strides= (2,2) Input Layer padding= 1

[3,4,4]

Output Filter 4 [3,32,32]

layer [2,3,32,32] 10 neurons

for 10 classes FC layer [3,4,4]

4x8x8 neurons Conv2D 1 padding= 0 strides = (2,2)

57

# Automatic feature extractor

Deep learning for visual data

receptive receptive

field field neuron

Neurons on higher layers have larger receptive fields (patches) on input images High-level feature/representation

- objects

Mid-level feature/representation

- circles, triangles, boxes

Low-level feature/representation

- edges, pixels, corners

58

59 Feature map 1

# How CNNs work?

Human face recognition ☺

Filter 1 (depth=1)

Filter 1 (depth=2) Filter 2 (depth=1)

Feature map

Feature map 2

 Higher level filters combine lower level filters to represent more abstract objects

o The context is locally expanded

60 Feature map 1

# How CNNs work?

Filter 1

Human face recognition ☺ Feature map 2

Filter 2

Flatten Prediction

Feature map 3

Filter 3

 CNN cannot capture spatial relationships

among objects Feature map 4 o How spatially related of two eyes, nose and

eye, eye and mouth

Filter 4

# Pooling layer

###  Max/average-pooling is locally invariant

### and applied independently to each

### feature map

###  Reduce the size of feature map to be

### more manageable

o Smaller size for flattening

###  Reduce the difference of output tensors

61

down-sampling and sharpening

# Why Do We Need Pooling Layer?

Less Difference Mouth feature map

Eyes

Less feature

Difference maps

Some

Nose Conv2D +

feature Pooling map Flatten

Mouth Prediction feature map

Less More

Difference Difference Eyes

feature maps

Less Nose Difference

Less feature Difference map

62

# Some comments from Hinton

Paper link: https://arxiv.org/pdf/1710.09829.pdf

(Source: medium.com)

 CNNs cannot capture the spatial

relationships among objects, hence it wrongly predicts every face with two eyes, one nose, and one mouth as a human face.

(Source: original paper of Capsule Net)

63

# Global pooling layer

##  Like pooling layer, but the kernel

## size is set to input size

o Each feature map becomes a single neuron o Global average pooling (GAP) o Global max pooling (GMP) o 4D input tensor [batch_size, in_channel, in_height, in_width]→ 2D output tensor [batch_size, in_channel]

64

Filters [4,3,4,4]

# Global AdaptiveAvgPool2d((1, 1))

Filter 1

Flatten(1)

# Pooling Layer

Feature volume

softmax Feature maps [3,4,4]

GAP Filter 2

Feature volume Feature maps

GAP Pooling [3,4,4] layer

GAP Filter 3

pool-size=(2,2)

GAP strides= (2,2) Input Layer padding= 1 [3,32,32]

[4,15,15]

[4,8,8] FC layer [3,4,4] Latent Output

4 neurons Latent representation layer Filter 4

representation

10 neurons for 10 classes MaxPool2d(kernel_size=2, stride=2, padding=1) [3,4,4]

Conv2d(3,4,kernel_size=4, stride=2, padding=0) Conv2D 1 padding= 0 strides = (2,2)

65

## SOTA CNNs

# ResNet

##  Won ImageNet competition in 2015

##  Many ResNet blocks

##  Each ResNet block contains many

## residual blocks

o Number of residual blocks o Number of channels for residual blocks

##  Each ResNet block

o From the second block, the first residual block

▪ Use 1x1 Conv skip connection o Otherwise

▪ Use standard skip connection

ResNet architecture

67

68

# ResNet

### Residual Block

(Source:Dive into DL)

Without skip connection With skip connection

##  Learn 𝑓 𝑥 = 𝑔 𝑥 + 𝑥 where 𝑔 𝑥 = 𝑓 𝑥 − 𝑥

o The model expressiveness is the same as previous o The gradient ∇𝑓 𝑥 = ∇𝑔 𝑥 + 𝟏 looks better where 𝟏 is the vector of all 1 with the

same shape as 𝑥

69

# ResNet

### Residual Block

Not use 1x1 Conv Use 1x1 Conv

 ResNet follows VGG’s full 3×3 convolutional layer design.

o The residual block has two 3×3 convolutional layers with the same number of output channels. o Each convolutional layer is followed by a batch normalization layer and a ReLU activation function. o Skip these two convolution operations and add the input directly before the final ReLU activation

function. o Requires that the output of the two convolutional layers must be of the same shape as the input, so

that they can be added together. ▪ If we want to change the number of channels, we need to introduce an additional 1×1 convolutional layer to

transform the input into the desired shape for the addition operation.

70

# ResNet

(8,64,64)

### Residual Block

padding=1 strides= (1,1) (8,64,64) \#filters=8

padding=1 strides= (1,1) (8,64,64) \#filters=8

(8,64,64)

Not use 1x1 Conv Use 1x1 Conv

 ResNet follows VGG’s full 3×3 convolutional layer design.

o The residual block has two 3×3 convolutional layers with the same number of output channels. o Each convolutional layer is followed by a batch normalization layer and a ReLU activation function. o Skip these two convolution operations and add the input directly before the final ReLU activation

function. o Requires that the output of the two convolutional layers must be of the same shape as the input, so

that they can be added together. ▪ If we want to change the number of channels, we need to introduce an additional 1×1 convolutional layer to

transform the input into the desired shape for the addition operation.

71

# ResNet

(16,32,32)

### Residual Block

padding=1 (16,32,32)

strides= (1,1) (16,32,32)

\#filters=16

padding=0 strides= (2,2) \#filters=16 padding=1

(16,32,32) strides= (2,2)

\#filters=16 (8,64,64) Not use 1x1 Conv Use 1x1 Conv

 ResNet follows VGG’s full 3×3 convolutional layer design.

o The residual block has two 3×3 convolutional layers with the same number of output channels. o Each convolutional layer is followed by a batch normalization layer and a ReLU activation function. o Skip these two convolution operations and add the input directly before the final ReLU activation

function. o Requires that the output of the two convolutional layers must be of the same shape as the input, so

that they can be added together. ▪ If we want to change the number of channels, we need to introduce an additional 1×1 convolutional layer to

transform the input into the desired shape for the addition operation.

72

# Implementation of Residual Block

(Source:Dive into DL)

(Source:Dive into DL)

# Implementation of ResNet

Residual block

ResNet block

Residual block

down sample Input size by 2

Residual block

ResNet block

Residual block

73

(Source:Dive into DL)

# Implementation of ResNet

Residual block

ResNet block

Residual block

down sample Input size by 2 (

1, 64,112,112) Residual block (1, 64,112,112)

(1, 64,112,112) (1, 64,56,56) ResNet block (1, 64,56,56)

(1, 128,28,28) (1, 256,14,14) (1, 256,1,1) Residual block (1,256)

(1,10)

74

## Practical Skills in Deep Learning

76

# Gradient vanishing

ത1 1 ത1 ℎ, ℎ = 𝜎(ℎ) ത2 2 ത2

ℎ, ℎ = 𝜎(ℎ) 3 3 0 ℎ, 𝑝 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ)

### Gradient Vanishing ℎ = 𝑥

o Gradients get smaller and smaller as the algorithm

progresses down to the lower layers. ▪ SGD update leaves the lower layer connection weights

virtually unchanged, and training never converges to a good solution. 3 2 ത2 1 ത1

𝜕𝑙 𝜕𝑙 𝜕ℎ 𝜕ℎ 𝜕 ℎ 𝜕ℎ 𝜕 ℎ

= . . . . . 1 3 2 ത2 1 ത1 1 𝜕𝑊 𝜕ℎ 𝜕ℎ 𝜕 ℎ 𝜕ℎ 𝜕 ℎ 𝜕𝑊

### Some activation functions are easy to get

𝑇 𝑇 3 ′ ത2 2 ′ ത1 0 𝑇 = 𝑝 − 1𝑦 𝑊 𝑑𝑖𝑎𝑔 𝜎 ℎ 𝑊 𝑑𝑖𝑎𝑔 𝜎 ℎ ℎ

### saturated

o Sigmoid or tanh

sigmoid sigmoid’s derivative

saturated saturated saturated

saturated 1 ′

𝜎 𝑧 = 𝜎(𝑧)(1 − 𝜎(𝑧)) 𝜎 𝑧 = 𝑠 𝑧 =

1 + exp{−𝑧}

77

# Gradient vanishing

ത1 1 ത1 2 2 2

### Gradient Vanishing ℎ, ℎ = 𝜎(ℎ) ℎത, ℎ = 𝜎(ℎത) 3 3

0 ℎ, 𝑝 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ) ℎ = 𝑥 o Gradients get smaller and smaller as the algorithm

progresses down to the lower layers. ▪ SGD update leaves the lower layer connection weights

virtually unchanged, and training never converges to a good solution.

3 2 ത2 1 ത1 𝜕𝑙 𝜕𝑙 𝜕ℎ 𝜕ℎ 𝜕 ℎ 𝜕ℎ 𝜕 ℎ

= . . . . .

### Some activation functions are easy to get 1 3 2 ത2 1 ത1 1

𝜕𝑊 𝜕ℎ 𝜕ℎ 𝜕 ℎ 𝜕ℎ 𝜕 ℎ 𝜕𝑊

𝑇 𝑇 3 ′ ത2 2 ′ ത1 0 𝑇 = 𝑝 − 1𝑦 𝑊 𝑑𝑖𝑎𝑔 𝜎 ℎ 𝑊 𝑑𝑖𝑎𝑔 𝜎 ℎ ℎ

### saturated

o Sigmoid or tanh

### Recipe

o Activation function plays an important role! Common

practice: ▪ Avoid sigmoid or saturated activation function ▪ ReLU is a common good choice o Good weight initialization is critical!

78

# Gradient exploding

Total loss

### Gradient Exploding

o The gradients can grow bigger and bigger, so many layers get insanely large weight updates, and the l𝒐𝒔𝒔 l𝒐𝒔𝒔 l𝒐𝒔𝒔𝒕 l𝒐𝒔𝒔

𝟎 𝟏 𝑻 training diverges. o Mostly being encountered in recurrent neural 𝑦0 𝑦1 𝑦 𝑦𝑇

𝑡 networks.

𝑉 𝑉 𝑉 𝑉 ℎ0 ℎ1 ℎ ℎ

𝑡 𝑇 𝑊 𝑊 𝑊

### Often happen for recursive models for example

𝑈 𝑈 𝑈 𝑈

### Recurrent Neural Network (RNN),

𝑥0 𝑥1 𝑥 𝑥

𝑡 𝑇

### Bidirectional RNN

𝜕𝑙𝑇 𝜕𝑙𝑇 𝜕ℎ𝑇 𝜕ℎ1

= × × ⋯× Tip: 𝜕ℎ 𝜕ℎ 𝜕ℎ 𝜕ℎ

0 𝑇 𝑇−1 0

- Let simplify 𝑊 as a scalar in the real-valued set 𝑚 𝑊
- 𝑊 → 0 if 𝑊 < 1. 𝑊 𝑚
- 𝑊 → ∞ if 𝑊 > 1. Multiplication of many matrices 𝑊

79

# He and Xavier weight initialization

### Initialisation in deep learning training is crucial:

o Some optimizers can be theoretically guaranteed to converge

regardless of initializations o Deep learning algorithms do not have these luxuries:

▪ It is iterative, but optimization deep neural networks is not yet well

understood ▪ Initial point is extremely important: it can determine if the algorithm

converges, or with some bad initialisation, it becomes unstable and fails together

### What is a good weight/filter initialization?

o Break the ‘symmetry’ of the network: two hidden nodes with the same Initialization is important for training DL models.

input should have different weights. o The gradient signal to flow well in both directions and don’t want the

signal to die out or to explode and saturate. o Large initial weights has better symmetry breaking effect, help

avoiding losing signals and redundant units, but could result in exploding values during backward and forward passes, especially in Recurrent Neural Networks.

80

# He and Xavier weight initialization

Gaussian version Uniform alternative

### Xavier initialization

o Try to ensure the variance of the 2

6 6 𝑤𝑋𝑎~𝑁 0, 𝑤 ~Uniform − ,

𝑋𝑎 outputs of each layer equal to the 𝑛 + 𝑛 𝑛 + 𝑛 𝑛 + 𝑛

𝑖𝑛 𝑜𝑢𝑡 𝑖𝑛 𝑜𝑢𝑡 𝑖𝑛 𝑜𝑢𝑡 variance of its inputs o Also need the gradients to have equal variance before and after

𝑛𝑖𝑛 = 3 flowing through a layer in the 2

Why ?

𝑛𝑖𝑛+𝑛𝑜𝑢𝑡 reverse direction

𝑛𝑜𝑢𝑡 = 2 o Good for sigmoid and tanh functions o Not good for ReLU

Paper link: https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

81

# He and Xavier weight initialization

2  Xavier initialization

𝑤𝑋𝑎~𝑁 0,

𝑛𝑖𝑛 + 𝑛𝑜𝑢𝑡 o Ensure the variance of the outputs of each layer equal to the variance of its inputs o Also need the gradients to have equal variance before and after flowing through a

𝑛𝑜𝑢𝑡 = 2 layer in the reverse direction

𝑛𝑖𝑛 = 3 o Good for sigmoid and tanh activation functions o Not good for ReLU

1 if sigmoid 2 𝑤𝐻𝑒~𝑁 0, 𝛼× 𝛼 = ቐ4 if tanh  He initialization 𝑛𝑖𝑛 + 𝑛𝑜𝑢𝑡

2 if ReLU o A variant of Xavier initialization where 𝛼 = 1 o Works better for ReLU.

Paper link: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf

82

# Deep Learning Pipeline

We want to train our DL model on a training set such that the trained model can predict well unseen data in a separate testing set.

Hyper-parameters to consider: learning rate, #layers, #neurons

83

# Regularization and Overfitting

### Three elements in ML and DL

o Data: training data, testing data, validation data o Model: a mathematical function 𝒇(𝒙; 𝜽) that maps an input instance 𝑥 to outcome 𝑦 o Evaluation: a performance metric to quantitively measure how well 𝑓(𝑥; 𝜃) is

### Machine learning as an optimization process. Learning from data by optimizing

### its loss

o Define a measure of loss via a loss function: 𝑙(𝑓 𝑥; 𝜃 , 𝑦)

−1 𝑁 o Compute its loss over all training data J 𝜃 = 𝑁 σ 𝑙(𝑓 𝑥, 𝜃 , 𝑦)

𝑖=1 𝑖 𝑖 ∗ ∗ o Learning = finding 𝜽 that minimizes the loss: 𝜃 = arg min 𝐽 𝜃

𝜃

### What might go wrong with this formulation?

o The choice of learning function 𝑓(𝑥; 𝜃) is too hard to learn o The choice of loss function 𝑙(𝑓 𝑥 , 𝑦) is inadequate o The model does well on training data, but perform poorly on unseen test data: overfitting problem!

84

# Overfitting & Underfitting

Ground-truth value

Predicted value

- Error

Overfitting with regression

### Underfitting

o The model is too simple to characterise a training set

▪ Use linear model to learn from non-linear data

### Overfitting

o The model performs very well on the training set, but cannot generalise to perform well on a separate

testing set o This is the most common problem in DL since deep networks are very powerful!

# Overfitting in Deep NNs

Overfitting

Overfit: tendency of the network to “memorize” all training samples, leading to poor generalization

Testing data

Overfitting with classification [Source: https://www.ibm.com/cloud/learn/overfitting]

 Overfitting occurs when your network models the training data too well and fails to generalize

to your test (validation) data. o Performance measured by errors on “unseen” data. o Minimize error alone on training data is not enough o Causes: too many layers, too many hidden nodes, and overtrained.

85

We can also experiment with model capacity itself in parallel.

[DL, ch11]

86

87

# Early stopping

Early stopping point

Underfitting Overfitting

Generalization gap

### At first, the train and valid losses gradually drop, but the model is not good enough to

### characterise data

o Underfitting is happening

### At a certain point, the train loss still decreases, while the valid loss starts increasing

o Overfitting starts happening and we need to do early stopping at this point

# Add Regularization

Reduce Overfitting

 Optimization problem

𝑵 1 min𝜽 𝐽 θ = 𝛺 θ + ෍ 𝑙 𝑦𝑖, 𝑓 xi; θ

𝑁

𝒊=𝟏  L2 regularization

2 2 𝐤 𝐤 Ω θ = 𝜆 ෍ ෍ 𝑊 = 𝜆 ෍ 𝑾

𝑖,𝑗

𝐹 𝑘 𝑖,𝑗 𝑘 o 𝜆 > 0 is regularization parameter

𝑘 o Gradient: 𝛻 𝑘 Ω 𝜽 = 2𝑾

𝑾 o Apply on weights (𝑾) only, not on biases (𝒃)  L1 regularization

𝐤 Ω 𝜽 = 𝜆 ෍ ෍ |𝑊 |

𝑖,𝑗 𝑘 𝑖,𝑗

𝜆 increases o 𝜆 > 0 is regularization parameter o Optimization is now much harder – subgradient. o Apply on weights (𝑾) only, not on biases (𝒃)

88

89

# Dropout

Reduce Overfitting

Without dropout With dropout

(Source: chatbotslife) (Source: Analytics Vidhya)

 This is a cheap technique to reduce model capacity

o Reduce overfitting  In each iteration, at each layer, randomly choose some neurons and drop all connections from

these neurons o dropout_rate= 1 – keep_prob

# Dropout

Reduce Overfitting

###  Computationally efficient

###  Can be considered a bagged ensemble of an

𝑁

### exponential number (2) of neural networks.

𝑙+1

##  Training 𝒉 × × ×

### 𝒓 ~ Bernoulli 𝜇

### ෩𝑙 𝑙

𝑙

## 𝒉 = 𝒉 ⨀ 𝒓 𝒉 × × × ×

⊤

### 𝑙+1 𝑙 ෩𝑙 𝑙

### 𝒉 = 𝜎 𝑊 𝒉 + 𝒃

𝑙−1 𝒉

###  Testing

o No dropout (dropout_rate =0).

90

91

# Internal covariate shift problem

Input batch Representation batch

𝟏 𝟏 𝟏 𝟏 𝟏 𝟏 𝟏 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟐 𝟐 𝟐 𝟐 𝟐 𝟐 𝟐 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟒 𝟒 𝟒 𝟒 𝟒 𝟒 𝟒 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑

𝑊 𝑊

2 1 , 𝑊, 𝑏 𝑊 𝑏

2 1 3 4

, , 𝑏 𝑏 3 4

Feature extractor Classifier

Paper link: http://proceedings.mlr.press/v37/ioffe15.pdf

92

# Internal covariate shift problem

Input batch 1 Representation batch 1

𝟏 𝟏 𝟏 𝟏 𝟏 𝟏 𝟏 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟐 𝟐 𝟐 𝟐 𝟐 𝟐 𝟐 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟒 𝟒 𝟒 𝟒 𝟒 𝟒 𝟒 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑

difference

###  Internal covariate shift

Input batch 2 Representation batch 2

### 𝟓 𝟓 𝟓 𝟓 𝟓 𝟓 𝟓 problem:

𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟔 𝟔 𝟔 𝟔 𝟔 𝟔 𝟔 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 o Significant difference in statistics 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟕 𝟕 𝟕 𝟕 𝟕 𝟕 𝟕 of input batches and also 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑

representation batches. 𝟖 𝟖 𝟖 𝟖 𝟖 𝟖 𝟖 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑

o Statistical differences among Input batch 3 difference Representation batch 3

mini-batches make the classifier 𝟗 𝟗 𝟗 𝟗 𝟗 𝟗 𝟗 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑

harder to learn from data. 𝟏𝟎 𝟏𝟎 𝟏𝟎 𝟏𝟎 𝟏𝟎 𝟏𝟎 𝟏𝟎 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟏𝟏 𝟏𝟏 𝟏𝟏 𝟏𝟏 𝟏𝟏 𝟏𝟏 𝟏𝟏 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟏𝟐 𝟏𝟐 𝟏𝟐 𝟏𝟐 𝟏𝟐 𝟏𝟐 𝟏𝟐 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑

𝑊 𝑊

𝑊 𝑊 1 2,, 3 𝑏 𝑏 4

,, 1 2 𝑏

𝑏 3 4

Feature extractor Classifier

93

# Batch Normalization

Input batch 1 Representation batch 1 Normalized representation 1

𝟏 𝟏 𝟏 𝟏 𝟏 𝟏 𝟏 𝟏 𝟏 𝟏 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝒛ො 𝒛ො 𝒛ො 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟏 𝟐 𝟑

Normalize to 𝑁(0, 𝐼) 𝟐 𝟐 𝟐 𝟐 𝟐 𝟐 𝟐 𝟐 𝟐 𝟐 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝒛ො 𝒛ො 𝒛ො 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟏 𝟐 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝒛ො 𝒛ො 𝒛ො 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟏 𝟐 𝟑 𝟒 𝟒 𝟒 𝟒 𝟒 𝟒 𝟒 𝟒 𝟒 𝟒 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝒛ො 𝒛ො 𝒛ො 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟏 𝟐 𝟑

difference Input batch 2 Representation batch 2 Normalized representation 2

𝟓 𝟓 𝟓 𝟓 𝟓 𝟓 𝟓 𝟓 𝟓 𝟓 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝒛ො 𝒛ො 𝒛ො 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟏 𝟐 𝟑

Normalize to 𝑁(0, 𝐼) 𝟔 𝟔 𝟔 𝟔 𝟔 𝟔 𝟔 𝟔 𝟔 𝟔 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝒛ො 𝒛ො 𝒛ො 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟏 𝟐 𝟑 𝟕 𝟕 𝟕 𝟕 𝟕 𝟕 𝟕 𝟕 𝟕 𝟕 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝒛ො 𝒛ො 𝒛ො 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟏 𝟐 𝟑 𝟖 𝟖 𝟖 𝟖 𝟖 𝟖 𝟖 𝟖 𝟖 𝟖 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝒛ො 𝒛ො 𝒛ො 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟏 𝟐 𝟑

Input batch 3 Representation batch 3 Normalized representation 3

difference 𝟗 𝟗 𝟗 𝟗 𝟗 𝟗 𝟗 𝟗 𝟗 𝟗 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝒛ො 𝒛ො 𝒛ො 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟏 𝟐 𝟑

Normalize to 𝑁(0, 𝐼) 𝟏𝟎 𝟏𝟎 𝟏𝟎 𝟏𝟎 𝟏𝟎 𝟏𝟎 𝟏𝟎 𝟏𝟎 𝟏𝟎 𝟏𝟎 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝒛ො 𝒛ො 𝒛ො 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟏 𝟐 𝟑 𝟏𝟏 𝟏𝟏 𝟏𝟏 𝟏𝟏 𝟏𝟏 𝟏𝟏 𝟏𝟏 𝟏𝟏 𝟏𝟏 𝟏𝟏 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝒛ො 𝒛ො 𝒛ො 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟏 𝟐 𝟑 𝟏𝟐 𝟏𝟐 𝟏𝟐 𝟏𝟐 𝟏𝟐 𝟏𝟐 𝟏𝟐 𝟏𝟐 𝟏𝟐 𝟏𝟐 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝒛ො 𝒛ො 𝒛ො 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟏 𝟐 𝟑

𝑊 𝑊

𝑊 𝑊 1 2,, 3 𝑏 𝑏 4

,, 1 2 𝑏

𝑏 3 4

Feature extractor Classifier

94

# Batch Normalization

Representation batch Normalized representation BN representation Input batch

𝟏 𝟏 𝟏 𝟏 𝟏 𝟏 𝟏 𝟏 𝟏 𝐳𝐁𝐍 𝐳𝐁𝐍 𝐳𝐁𝐍 𝟏 𝟏 𝟏 𝟏 𝐳 𝐳 𝐳 𝒛ො 𝒛ො 𝒛ො 𝟏 𝟐 𝟑 𝒙 𝒙 𝒙 𝒙 𝟏 𝟐 𝟑 𝟏 𝟐 𝟑 Scale by 𝜸 Activation 𝟏 𝟐 𝟑 𝟒 Normalize

𝟐 𝟐 𝟐 𝟐 𝟐 𝟐 and shift by 𝜷 𝟐 𝟐 𝟐 with 𝝈 𝐳 𝐳 𝐳 to 𝑁(0, 𝐼) 𝐳𝐁𝐍 𝐳𝐁𝐍 𝐳𝐁𝐍 𝟐 𝟐 𝟐 𝟐 𝟏 𝟐 𝟑 𝒛ො 𝒛ො 𝒛ො 𝟏 𝟐 𝟑 𝒙 𝒙 𝒙 𝒙 𝟏 𝟐 𝟑 𝟏 𝟐 𝟑 𝟒

𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝐳 𝐳 𝐳 𝒛ො 𝒛ො 𝒛ො 𝐳𝐁𝐍 𝐳𝐁𝐍 𝐳𝐁𝐍

𝟏 𝟐 𝟑 𝟏 𝟐 𝟑 𝟏 𝟐 𝟑 𝒙 𝒙 𝒙 𝒙 𝟏 𝟐 𝟑 𝟒

𝟒 𝟒 𝟒 𝟒 𝟒 𝟒 𝐳 𝐳 𝐳 𝟒 𝟒 𝟒 𝟏 𝟐 𝟑 𝒛ො 𝒛ො 𝒛ො 𝐳𝐁𝐍 𝐳𝐁𝐍 𝐳𝐁𝐍 𝟒 𝟒 𝟒 𝟒 𝟏 𝟐 𝟑 𝟏 𝟐 𝟑 𝒙 𝒙 𝒙 𝒙 𝟏 𝟐 𝟑 𝟒

𝜇1 𝜇2 𝜇3 𝜎1 𝜎2 𝜎3

𝑊 𝑊

2 1 , 𝑊 𝑊, 𝑏 𝑏

2 1 3 4

, , 𝑏 𝑏 3 4

Feature extractor Classifier

# Batch Normalization 𝑘 𝑘 𝑘

- Let 𝑧 = 𝑊 ℎ + 𝑏 be the mini-batch before activation. We compute the normalized  𝑧Ƹ as 𝑧−𝜇𝐵 −7
- 𝑧Ƹ = where 𝜖 is a small value such as 1𝑒

### 1\. Cope with internal covariate shift 2

𝜎 +𝜖 𝐵 1 𝑏

### 2\. Reduce gradient • 𝜇 = σ 𝑧 is the empirical mean

𝐵 𝑖=1 𝑖

𝑏 2 1 𝑏

### vanishing/exploding 2

- 𝜎 = σ 𝑧 − 𝜇 is the empirical variance 𝐵 𝑖=1 𝑖 𝐵 𝑏

### 3\. Reduce overfitting • We scale the normalized  𝑧Ƹ

- 𝑧𝐵𝑁 = 𝛾  𝑧Ƹ + 𝛽 where 𝛾, 𝛽 > 0 are two learnable parameters

### 4\. Make training more stable

(i.e., scale and shift parameters)

### 5\. Converge faster • We then apply the activation to obtain the next layer value

𝑘+1

1. Allow us to train with bigger learning rate • ℎ = 𝜎(𝑧𝐵𝑁)

𝑧1 𝑧2 0.1 0.2 ℎ1 ℎ2 … …

Mini-batch size

0.5 0.4

Standardize column-wise 95

96

# Batch Normalization

Testing Phase

### At testing time, let’s say we only want to

### test on a single input 𝑥

### Hence, we don’t have a set of mini-

### batch samples to compute mean and

### standard deviation.

### So, we replace the mini-batch 𝜇𝐵 and

### 𝜎𝐵 with running averages of 𝜇෤𝐵 and

### 𝜎෤𝐵 computed during the training

### process.

o 𝜇෤𝐵 = 𝜃 𝜇෤𝐵 + 1 − 𝜃 𝜇𝐵 o 𝜎෤𝐵 = 𝜃 𝜎෤𝐵 + 1 − 𝜃 𝜎𝐵 o 0 < 𝜃 < 1 is the momentum decay

# Data Augmentation

### Reality

o You have a tiny dataset of 10K images, and you need

to train a good deep net.

What are the criteria of a qualified training set?

Quality? Quantity?

- More diverge data?
- Collect more and more data?

- More qualified data?

Prof Dinh Phung Create many variants

97

98

# Data Augmentation

### Use simple transformations to augment

### data examples. Models will be challenged with

diverge data examples which might be encountered in the testing set o Rotation, Width Shifting, Height Shifting, Brightness, Shear

Intensity, Zoom, Channel Shift, Horizontal Flip, Vertical Flip

### This will reduce overfitting, making this a

### regularization technique. The trick is to

### generate realistic training instances

[Source: Stanford CS 230 Deep Learning]

# Data Augmentation in PyTorch

Without augmentation

With augmentation

99

# Label smoothing

Paper link: https://papers.nips.cc/paper/2019/file/f1748d6b0fd9d439f71450117eba2725-Paper.pdf

logits

𝐿 𝑳 𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ 𝑥 ) 𝒉 (𝒙) 𝑥 𝑦ො = 𝑎𝑟𝑔𝑚𝑎𝑥 𝑝 (𝑥)

1≤𝑚≤𝑀 𝑚 𝑦 ∈ {1, … , 𝑀}

Label smoothing No label smoothing 𝑙 𝑦, 𝑦ො = 𝐶𝐸(𝟏, 𝒑(𝒙)) 𝑙 𝑦, 𝑦ො = 𝐶𝐸(𝟏෩, 𝒑(𝒙))

𝑦 𝑦 2 2 𝛼 𝛼 𝛼 𝑊, 𝑏 …

𝟏𝑦 = [0, . . , 1, … , 0] 𝟏෩ = , … , 1 − 𝛼 + , … ,

𝑦

𝑀 𝑀 𝑀 𝒚-th position 𝒚-th position

Given data instance (𝒙, 𝒚) with the label 𝒚 ∈ {𝟏, … , 𝑴}, we compute the CE loss between the prediction probabilities 𝒑(𝒙) and the smooth label

𝛼 o 𝑙 𝑦, 𝑦ො = 𝐶𝐸(𝟏෩, 𝒑(𝒙)) with 𝟏෩ = 1 − 𝛼 ×𝟏 + ×𝟏 where 𝟏 is a vector of all 1 and 0 < 𝛼 < 1.

𝑦 𝑦 𝑦

𝑀

100

# Data mix-up

Paper link: https://openreview.net/pdf?id=r1Ddp1-Rb

[Source: https://medium.com/] (𝑥1, 𝟏𝑦)

1

𝜆~Beta(𝛼, 𝛼) Blended image 𝑥෤ = 𝜆×𝑥1 + 1 − 𝜆 ×𝑥2 Blended label 𝑦෤ = 𝜆×𝟏 + 1 − 𝜆 ×𝟏 (𝑥2, 𝟏𝑦)

𝑦1 𝑦2 2 𝐦𝐢𝐧 𝑪𝑬(𝒚෥, 𝒑(𝒙෥))

### ❑ for (𝑥1, 𝑦1), (𝑥2, 𝑦2) in zip(batch1, batch 2)

1. 𝜆~Beta 𝛼, 𝛼
2. 𝑥෤ = 𝜆×𝑥1 + 1 − 𝜆 ×𝑥2
3. 𝑦෤ = 𝜆×𝟏𝑦 + 1 − 𝜆 ×𝟏𝑦 1 2
4. Update optimizer to minimize 𝑪𝑬(𝒚෥, 𝒑(𝒙෥))

101

# Cut-mix

[Source: https://encord.com/blog/data-augmentation-guide/]

[Source: https://arxiv.org/pdf/1905.04899 ] 𝑴 𝐶×𝑊×𝐻 1 1 1 1 1 1 𝐶×𝑊×𝐻 𝑥, 𝟏, 𝑥 ∈ ℝ 𝐵 = [𝑟, 𝑟, 𝑟, 𝑟 ] 𝑥, 𝟏, 𝑥 ∈ ℝ 2 𝑦2 2 𝑥෤ 1 𝑥 𝑦 𝑤 ℎ 1 𝑦1 1 1

0 0 0 1 1 𝜆~Beta(𝛼, 𝛼) 0 0 0

𝐻×𝑊 1 1 𝑎𝑟𝑒𝑎(𝐵) 𝑊𝐻(1 − 𝜆) 𝑥෤ = 𝑀 ⊙ 𝑥1 + 1 − 𝑀 ⊙ 𝑥2, 𝑀 ∈ 0,1 = = 1 − 𝜆

1 0 0 0 1

𝑎𝑟𝑒𝑎(𝑖𝑚𝑎𝑔𝑒) 𝑊𝐻 𝑦෤ = 𝜆×𝟏𝑦 + 1 − 𝜆 ×𝟏𝑦

1 2 1 1

1 1 1 1 1 1 𝐦𝐢𝐧 𝑪𝑬(𝒚෥, 𝒑(𝒙෥)) ❑ for (𝑥1, 𝑦1), (𝑥2, 𝑦2) in zip(batch1, batch 2)

1. 𝜆~Beta 𝛼, 𝛼 ⊙ 𝑀 =
2. Sample a bounding box 𝐵 = [𝑟𝑥, 𝑟𝑦, 𝑟𝑤, 𝑟ℎ]

1. 𝑟𝑥 ∼ 𝑈𝑛𝑖[0, 𝑊], 𝑟𝑤 ∼ 𝑊 1 − 𝜆
2. 𝑟𝑦 ∼ 𝑈𝑛𝑖[0, 𝐻], 𝑟ℎ ∼ 𝐻 1 − 𝜆 𝑾×𝑯
3. 𝑴 ∈ 𝟎, 𝟏 by filling 1 outside B and 0 otherwise
4. 𝑥෤ = 𝑀 ⊙ 𝑥1 + 1 − 𝑀 ⊙ 𝑥2 ⊙ (1 − 𝑀) =
5. 𝑦෤ = 𝜆×𝟏𝑦 + 1 − 𝜆 ×𝟏𝑦 1 2
6. Update optimizer to minimize 𝑪𝑬(𝒚෥, 𝒑(𝒙෥)) 102

## Adversarial Attack and Defense

104

# Some notions

𝑓 𝑥; 𝜃 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ(𝑥; 𝜃)) (prediction probabilities) softmax

ℎ(𝑥; 𝜃) (logits)

𝑀 𝑓 𝑥; 𝜃 = 𝑓𝑚 𝑥; 𝜃

𝑚=1 𝒉 𝑝 𝑦 = 𝑚 𝑥 = 𝑓𝑚(𝑥; 𝜃) for 1 ≤ 𝑚 ≤ 𝑀 𝜽

𝑙 𝑓 𝑥; 𝜃 , 𝑦 = 𝐶𝐸 𝟏𝑦, 𝑓 𝑥; 𝜃 = − log 𝑓𝑦 (𝑥; 𝜃)

𝑥, 𝑦 ∈ {1,2, … , 𝑀}

105

# Distance between two images

Image 𝒙 Image 𝒙′ C C

H H

W

W  𝑥 (1 ≤ 𝑖 ≤ 𝐶 , 1 ≤ 𝑗 ≤ 𝐻, 1 ≤ 𝑘 ≤ 𝑊) is the pixel with coordinate (ℎ𝑒𝑖𝑔ℎ𝑡 = 𝑗, 𝑤𝑖𝑑𝑡ℎ = 𝑘) on

𝑖𝑗𝑘 the channel (or slice) 𝑖 of image 𝑥. o 0 ≤ 𝑥𝑖𝑗𝑘 ≤ 255 (original colour image) or 0 ≤ 𝑥𝑖𝑗𝑘 ≤ 1 (grey-scale image)  L distance

2

2 ′ 𝐶 𝐻 𝑊 ′ o 𝑥 − 𝑥 = σ σ σ 𝑥 − 𝑥

2 𝑖=1 𝑗=1 𝑘=1 𝑖𝑗𝑘 𝑖𝑗𝑘  L distance

1

′ 𝐶 𝐻 𝑊 ′ o 𝑥 − 𝑥 = σ σ σ |𝑥 − 𝑥 |

1 𝑖=1 𝑗=1 𝑘=1 𝑖𝑗𝑘 𝑖𝑗𝑘  L distance

∞

′ ′ o 𝑥 − 𝑥 ∞ = max |𝑥𝑖𝑗𝑘 − 𝑥 |

𝑖𝑗𝑘 1≤𝑖≤𝐶,1≤𝑗≤𝐻,1≤𝑘≤𝑊

106

# Adversarial examples

′ ′ 𝝐 ball 𝐵𝜖 𝑥 = {𝑥: 𝑥 − 𝑥 ≤ 𝜖} with a small 𝝐 > 0.

𝒉(𝒙; 𝜽)

𝒙𝒂𝒅𝒗

𝒉 crafted noise 𝜽 𝒇 𝒙; 𝜽 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ(𝑥; 𝜃))

softmax 𝒙

𝒙′

𝑎𝑟𝑔𝑚𝑎𝑥1≤𝑖≤𝑀 𝑓𝑖 𝑥; 𝜃 = 𝑦ො ≠ 𝑦ො𝑎𝑑𝑣 = 𝑎𝑟𝑔𝑚𝑎𝑥1≤𝑖≤𝑀 𝑓𝑖 𝑥𝑎𝑑𝑣; 𝜃

Good example which has same predicted label as 𝒙. Adversarial example which has different predicted label as 𝒙.

 The true sense of a robust model

o If 𝒙′ and 𝒙 are close, the predictions of 𝒙′ and 𝒙 must be the same.

′ ′ o If 𝒙 ∈ 𝑩𝝐(𝒙) (i.e., 𝒙 − 𝒙 ≤ 𝝐), 𝑎𝑟𝑔𝑚𝑎𝑥1≤𝑖≤𝑀 𝑓𝑖 𝑥; 𝜃 = 𝑦ො = 𝑦ො′ = 𝑎𝑟𝑔𝑚𝑎𝑥1≤𝑖≤𝑀 𝑓𝑖 𝑥′; 𝜃 .  Unfortunately, we can easily find adversarial examples 𝑥 ∈ 𝐵 (𝑥) that can fool the classifier

𝑎𝑑𝑣 𝜖 o 𝑎𝑟𝑔𝑚𝑎𝑥1≤𝑖≤𝑀 𝑓𝑖 𝑥; 𝜃 = 𝑦ො ≠ 𝑦ො𝑎𝑑𝑣 = 𝑎𝑟𝑔𝑚𝑎𝑥1≤𝑖≤𝑀 𝑓𝑖 𝑥𝑎𝑑𝑣 ; 𝜃

107

# Adversarial examples

(Source: OpenAI) (Source: Internet) 𝒙 𝒙𝒂𝒅𝒗 = 𝒙 + 𝒏𝒐𝒊𝒔𝒆

Small perturbation

Misleading autonomous car driving system

not difference difference

##  Adversarial example 𝒙 of a clean data example 𝒙 w.r.t a model

𝒂𝒅𝒗

## 𝑓(. ; 𝜃)

o 𝑑(𝑥𝑎𝑑𝑣, 𝑥) = ‖𝑥𝑎𝑑𝑣 − 𝑥‖ ≤ 𝜖 where . is a norm (e.g., . 1, . 2, . ∞) o 𝑓 predicts 𝑥 and 𝑥𝑎𝑑𝑣 with different labels, i.e., 𝑎𝑟𝑔𝑚𝑎𝑥𝑖𝑓𝑖 𝑥; 𝜃 ≠ 𝑎𝑟𝑔𝑚𝑎𝑥𝑖𝑓𝑖 𝑥𝑎𝑑𝑣; 𝜃

108

# Visualization of decision regions

𝑇 ⊥ 𝑔 𝑔 = 0

⊥ Each colour ResNet-50 𝑔

represents a class

ℎ(𝑥; 𝜃) and f 𝑥; 𝜃 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ(𝑥; 𝜃)

′ ⊥ 𝒙 = 𝒙 + 𝜶𝒈 + 𝜷𝒈 ′ ⊥

Predicts 𝒙 = 𝒙 + 𝜶𝒈 + 𝜷𝒈 𝜷

to decide the colour

𝒙(0,0) 𝜶 𝑔 = ∇ 𝑙(𝑓(𝑥; 𝜃), 𝑦)

𝑥

Following the gradient direction is easy to get out the blue region.

⊥ ❑ Start from 𝑥 = 𝑥 + 0𝑔 + 0𝑔 (coordinate (𝟎, 𝟎)), if we go along with the gradient direction 𝑔 =

⊥ ∇𝑥𝑙(𝑓(𝑥; 𝜃), 𝑦), we can reach 𝑥𝑎𝑑𝑣 = 𝑥 + 𝛼𝑔 + 0𝑔 such that ❑ 𝑥𝑎𝑑𝑣 is very close to 𝑥 → human cannot tell how they are different→ the same class from human

perceptron ❑ 𝑥𝑎𝑑𝑣 is predicted to different class from 𝑥 by the classifier ❑ 𝑥𝑎𝑑𝑣 is called adversarial example of 𝑥

109

# Decision regions of deep learning model

𝒙(0,0) 𝒙(0,0) 𝒙(0,0) 𝒙(0,0) 𝒙(0,0) ⊥ 𝑔

Each colour represents a class

𝑇 ⊥ 𝑔 𝑔 = 0

𝒙(0,0) 𝑔 = ∇ 𝑙(𝑓(𝑥; 𝜃), 𝑦)

𝑥

##  Deep learning models are fragile and easy to be attacked

⊥ o Start from 𝑥 = 𝑥 + 0𝑔 + 0𝑔 (coordinate (𝟎, 𝟎)), if we go along with the gradient

⊥ direction 𝑔 = ∇𝑥𝑙(𝑓(𝑥; 𝜃), 𝑦), we can reach 𝑥𝑎𝑑𝑣 = 𝑥 + 𝛼𝑔 + 0𝑔 such that ▪ 𝑥𝑎𝑑𝑣 is very close to 𝑥 → human cannot tell how they are different→ the same class from

human perceptron ▪ 𝑥𝑎𝑑𝑣 is predicted to different class from 𝑥 by the classifier ▪ 𝑥𝑎𝑑𝑣 is called adversarial example of 𝑥

Delving into Transferable Adversarial Examples and Black-box Attacks Yanpei L., et al., ICLR 2016

110

# Loss surface of deep learning model

𝒈 𝒈 ⊥ ⊥ 𝒈 𝒈

Zoom out Zoom in

⊥ The loss surface of DL model around 𝒙. This is plotted on 𝒙′ = 𝒙 + 𝝐𝟏𝒈 + 𝝐𝟐𝒈.

##  At the data point 𝒙, if we move along with the gradient direction

## 𝑔, the loss function is maximally locally increased.

##  Taylor expansion around 𝑥

𝑇 o 𝑙 𝑓(𝑥 + ℎ; 𝜃), 𝑦 ≈ 𝑙 𝑓(𝑥; 𝜃), 𝑦 + 𝑔 ℎ where 𝑔 = ∇𝑥𝑙(𝑓(𝑥; 𝜃), 𝑦). o The gradient direction is the steepest direction to increase the loss function locally.

Ensemble Adversarial Training Attacks and Defences Tramer et al., ICLR 2017

111

# Adversarial attack

### How to run untargeted attack? Untargeted attacks

Loss surface 𝒍(𝒇 𝒙′; 𝜽 , 𝒚)

′ ′ 𝐵 𝑥 = 𝑥: 𝑥 − 𝑥 ≤ 𝜖 𝒙 = 𝒂𝒓𝒈𝒎𝒂𝒙 ′ 𝒍(𝒇 𝒙′; 𝜽 , 𝒚) 𝜖 𝒂𝒅𝒗 𝒙 ∈𝑩 𝒙

𝝐 𝒙 has true label 𝒚 ∈ {𝟏, 𝟐, … , 𝑴}

𝒙𝑎𝑑𝑣 𝒙

 Fast Gradient Sign Method (FGSM)  Projected Gradient Descent (Ascent)

(PGD) o 𝑥𝑎𝑑𝑣 = 𝑥 + 𝜖𝑠𝑖𝑔𝑛(∇𝑥𝑙(𝑓 𝑥; 𝜃 , 𝑦))

o 𝑥0 = 𝑥 + 𝑈𝑛𝑖𝑓𝑜𝑟𝑚([−𝜖, 𝜖]) 1 𝑖𝑓 𝑡 > 0 o 𝑠𝑖𝑔𝑛 𝑡 = ቐ −1 𝑖𝑓 𝑡 < 0 o 𝒙෥ = 𝑥 + 𝜂𝑠𝑖𝑔𝑛(∇ 𝑙 𝑓 𝑥; 𝜃 , 𝑦 )

𝑡+1 𝑡 𝑥 𝑡 0 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒

o 𝑥𝑡+1 = 𝑃𝑟𝑜𝑗𝐵 𝑥 ( 𝑥෤𝑡+1)

𝜖 o One-step update

o Run in 𝑘 steps (𝑘 = 20), 𝜂 > 0 is the Explaining and Harnessing Adversarial Examples learning rate Goodfellow et al., ICLR 2015

o 𝑥𝑎𝑑𝑣 = 𝑥𝑘

Towards Deep Learning Models Resistant to Adversarial Attacks Madry et al., ICLR 2017

112

# Adversarial attack

### How to run targeted attack? Targeted attacks

′ Loss surface 𝒍(𝒇 𝒙; 𝜽 , 𝒚≠) 𝒚≠ is different from y

′ ′ ′ 𝐵 𝑥 = 𝑥: 𝑥 − 𝑥 ≤ 𝜖 𝒙 = 𝒂𝒓𝒈𝒎𝒊𝒏 ′ 𝒍(𝒇 𝒙; 𝜽 , 𝒚) 𝜖 𝒂𝒅𝒗 𝒙 ∈𝑩 𝒙 ≠

𝝐 𝒙 has true label 𝒚 ∈ {𝟏, 𝟐, … , 𝑴}

𝒙𝑎𝑑𝑣 𝒙

 Fast Gradient Sign Method (FGSM)

 Projected Gradient Descent (Ascent) o 𝑥𝑎𝑑𝑣 = 𝑥 − 𝜖𝑠𝑖𝑔𝑛(∇𝑥𝑙(𝑓 𝑥; 𝜃 , 𝑦#)) (PGD)

1 𝑖𝑓 𝑡 > 0 o 𝑥0 = 𝑥 + 𝑈𝑛𝑖𝑓𝑜𝑟𝑚([−𝜖, 𝜖]) o 𝑠𝑖𝑔𝑛 𝑡 = ቐ −1 𝑖𝑓 𝑡 < 0

o 𝑥෤𝑡+1 = 𝑥𝑡 − 𝜂𝑠𝑖𝑔𝑛(∇𝑥𝑙(𝑓 𝑥𝑡; 𝜃 , 𝑦#)) 0 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒

o 𝑥𝑡+1 = 𝑃𝑟𝑜𝑗𝐵 𝑥 ( 𝑥෤𝑡+1)

𝜖 o One-step update

o Run in 𝑘 steps (𝑘 = 20), 𝜂 > 0 is the

learning rate o 𝑥𝑎𝑑𝑣 = 𝑥𝑘

# Adversarial Examples and Perturbations

113

114

# Adversarial training

Loss surface 𝒍(𝒇 𝒙′; 𝜽 , 𝒚)

We wish a smooth (or flat) loss surface around 𝒙 𝒙

##  The ideal optimization problem for adversarial training

′ o min 𝔼 𝑥,𝑦 ~𝑝(𝑥,𝑦) max 𝑙(𝑓 𝑥; 𝜃 , 𝑦)

𝜃 ′

𝑥 ∈𝐵𝜖(𝑥) o 𝑝(𝑥, 𝑦) is the data distribution that generates the pair of data point 𝑥 and label 𝑦 o Minimize the loss of over most violated adversarial examples

##  PGD adversarial training

o for epoch in n_epochs

▪ for iter in range(n_iter_per_epoch)

o Sample mini-batch 𝑥1, 𝑦1 , … , (𝑥𝑏, 𝑦𝑏) from the training set

𝑎𝑑𝑣 𝑎𝑑𝑣 o Find PGD untargeted adversarial examples 𝑥, … , 𝑥 for 𝑥1, … , 𝑥𝑏 w.r.t labels 𝑦1, … , 𝑦𝑏

1 𝑏 1 𝑏 1 𝑏 𝑎𝑑𝑣 o 𝑏𝑎𝑡𝑐ℎ_𝑙𝑜𝑠𝑠 = σ 𝑙(𝑓 𝑥; 𝜃 , 𝑦) + σ 𝑙(𝑓 𝑥; 𝜃 , 𝑦)

𝑖=1 𝑖 𝑖 𝑖=1 𝑖 𝑖 𝑏 𝑏 𝜕𝑏𝑎𝑡𝑐ℎ_𝑙𝑜𝑠𝑠 o 𝜃 = 𝜃 − 𝜂

𝜕𝜃

## Recurrent Neural Networks

116

# Recurrent Neural Networks

Simplest RNN with two time-slices (no output)

𝑊

1×𝑖𝑛𝑝𝑢𝑡_𝑠𝑖𝑧𝑒 1×𝑖𝑛𝑝𝑢𝑡_𝑠𝑖𝑧𝑒

### ℎ ℎ  Input: 𝑥 ∈ ℝ, 𝑥 ∈ ℝ

0 1 0 1

1×ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒

### 𝑈 𝑈  ℎ = tanh 𝑥 𝑈 + 𝑏 ∈ ℝ

0 0

𝑖𝑛𝑝𝑢𝑡_𝑠𝑖𝑧𝑒×ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒 𝑥 𝑥 o 𝑈 ∈ ℝ 0 1

###  ℎ = some function of ℎ and 𝑥

1 0 1 values known from data 1×ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒

### = tanh ℎ0𝑊 + 𝑥1𝑈 + 𝑏 ∈ ℝ

ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒×ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒 hidden nodes, values to be o 𝑊 ∈ ℝ inferred from our models

hidden_size = 5

# Recurrent Neural Networks

batch_size = 4

𝑏𝑠×𝑖𝑛𝑝𝑢𝑡_𝑠𝑖𝑧𝑒  Input: 𝑋, 𝑋 ∈ ℝ

0 1 [batch_size, seq_len, hidden_size]

 ℎ = tanh 𝑋 𝑈 + 𝑏 ∈

0 0 𝑏𝑠×ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒 ℎ ℎ ℝ 0 1

𝑊 𝑖𝑛𝑝𝑢𝑡_𝑠𝑖𝑧𝑒×ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒

o 𝑈 ∈ ℝ 𝑈 𝑈

 ℎ = tanh(ℎ 𝑊 + 𝑋 𝑈 +

1 0 1

1×ℎ𝑖𝑑𝑑𝑒𝑛_𝑠𝑖𝑧𝑒 𝑋 𝑏 ) ∈ ℝ 𝑋0 1

seq_len = 2

batch_size = 4 input_size = 3

[batch_size, seq_len, input_size] input_size = 3 [seq_len, batch_size, input_size] torch.transpose(X, 0, 1)

117

118

# Recurrent Neural Networks

### with no output

𝑊 𝑊

𝑊 𝑊 ℎ ℎ𝑡−1 ℎ𝑡 ℎ𝑡+1 𝑡 𝑈 unrolling 𝑈 𝑈 𝑈

𝑥𝑡−1 𝑥𝑡 𝑥𝑡+1 𝑥𝑡

An unrolled RNN (no output)

###  Idea: sharing parameters for each data of the time index

###  Given a data sequence 𝑥, 𝑥, … , 𝑥

1 2 𝑇

###  RNN models a dynamic system driven by an external signal 𝑥

𝑡 ℎ𝑡 = 𝑓 ℎ𝑡−1 , 𝑥𝑡 = 𝑓 𝑓 ℎ𝑡−2, 𝑥𝑡−1 , 𝑥𝑡 = ⋯ = summary 𝑥1:𝑡, ℎ0

###  ℎ can be considered as a kind of lossy summary of the history 𝑥

𝑡 1:𝑡

119

# Recurrent Neural Networks

### Parameterization - 2 time slices

1×𝑖𝑛𝑝𝑢𝑡_𝑠𝑖𝑧𝑒

###  Input: 𝑥, 𝑥 ∈ ℝ, 𝑦, 𝑦 ∈ 𝑌

0 1 0 1

###  ℎ = tanh(𝑥 𝑈 + 𝑏)

0 0

ℎ0𝑉 + 𝑐 (regression) 𝑦ො 𝑦ො1 0  𝑦ො = ቊ

0

softmax (ℎ0𝑉 + 𝑐) (classification) 𝑉 𝑉

𝑊 o Suffer loss 𝑙(𝑦 ො0, 𝑦0) ℎ0 ℎ1

###  ℎ = some function of ℎ and 𝑥

𝑈 𝑈 1 0 1

### = tanh(ℎ0𝑊 + 𝑥1𝑈 + 𝑏)

𝑥0 𝑥1

ℎ1𝑉 + 𝑐 (regression)  𝑦ො = ቊ

1 Simplest RNN with two time-slices (with output) softmax (ℎ 𝑉 + 𝑐) (classification)

1 o Suffer loss 𝑙(𝑦 ො1, 𝑦1)

###  𝑇𝑜𝑡𝑎𝑙_𝑙𝑜𝑠𝑠 = 𝑙 𝑦ො, 𝑦 + 𝑙( 𝑦ො, 𝑦)

0 0 1 1

120

# Recurrent Neural Networks

### Parametrization over multiple time slices

𝑦ො 𝑦ො 𝑦ො 𝑦ො𝑡 𝑦ො 1×𝑖𝑛𝑝𝑢𝑡_𝑠𝑖𝑧𝑒 0 1 𝑡−1 𝑡+1

 Input:𝑥, 𝑥, . . , 𝑥, … ∈ ℝ, 𝑦, 𝑦, … ∈ 𝑌

0 1 𝑡 0 1 𝑉 𝑉 𝑉 𝑉 𝑉

 ℎ = tanh(𝑥 𝑈 + 𝑏)

0 0 𝑊 𝑊 𝑊 ℎ0 ℎ ℎ ℎ ℎ

1 𝑡−1 𝑡 𝑡+1

ℎ0𝑉 + 𝑐 (regression)  𝑦ො = ቊ

0 𝑈 𝑈 𝑈 𝑈 𝑈 softmax (ℎ 𝑉 + 𝑐) (classification)

0 𝑥 𝑥1 𝑥 𝑥 𝑥 0 𝑡−1 𝑡 𝑡+1 o Suffer loss 𝑙(𝑦 ො, 𝑦)

0 0  for t=1,2,..., T Simplest RNN with multiple time-slices (with output)

ℎ𝑡 = some function of ℎ𝑡−1 and 𝑥𝑡

= tanh(ℎ𝑡−1𝑊 + 𝑥𝑡𝑈 + 𝑏) 𝑦ො ℎ𝑡𝑉 + 𝑐 (regression) 𝑡

𝑦ො𝑡 = ቊ

softmax (ℎ𝑡𝑉 + 𝑐) (classification) 𝑉

o Suffer loss 𝑙(𝑦 ො𝑡, 𝑦𝑡 ) ℎ𝑡

𝑇  𝑻𝒐𝒕𝒂𝒍_𝒍𝒐𝒔𝒔 = σ 𝑙(𝑦 ො, 𝑦)

𝑡=0 𝑡 𝑡 𝑊 𝑈

𝑥𝑡

121

# Shape Transformation in RNNs

batch_size

batch_size 𝑦ො 𝑦ො2 𝑦ො 𝑦ො 𝑦ො𝐿 1 𝑡−1 𝑡

𝑉 𝑉 𝑉 𝑉 𝑉

… seq_len

batch_size

𝑉 𝑊 𝑊 𝑊 [batch_size, seq_len, hidden_size] ℎ1 ℎ ℎ ℎ ℎ

2 𝑡−1 𝑡 𝐿 real output

𝑈 𝑈 𝑈 𝑈 𝑈

batch_size

real input 𝑥 𝑥2 𝑥 𝑥 𝑥 1 𝑡−1 𝑡 𝐿

…

… batch_size seq_len

seq_len [seq_len, batch_size, input_size] [batch_size, seq_len, input_size]

torch.transpose(X, 0, 1)

122

# Deeper RNNs

𝑦ො 𝑦ො𝑡 𝑦ො 𝑡−1 𝑡+1

3 3 3 ℎ ℎ ℎ 𝑡−1 𝑡 𝑡+1

3 3 3 ℎ ℎ ℎ 𝑡−1 𝑡 𝑡+1

2 2 2 ℎ ℎ ℎ 𝑡−1 𝑡 𝑡+1

2 2 2 ℎ ℎ ℎ 𝑡−1 𝑡 𝑡+1

1 1 1 ℎ ℎ ℎ 𝑡−1 𝑡 𝑡+1

1 1 1 ℎ ℎ ℎ 𝑡−1 𝑡 𝑡+1

𝑥 𝑥𝑡 𝑥 𝑡−1 𝑡+1

𝑥 𝑥𝑡 𝑥 𝑡−1 𝑡+1

123

# RNNs Architecture Zoo

##  RNN’s architecture is very flexible for many real-world tasks

𝒚

𝒉 𝒉 𝒉

Sentiment analysis [https://www.twilio.com]

𝒙 𝒙 𝒙 many to one (sentiment analysis, image classification)

Image captioning 𝒚 𝒚 𝒚 [https://www.pcworld.com]

𝒉 𝒉 𝒉

𝒙 one to many (image captioning)

124

# RNNs Architecture Zoo

##  RNNs architecture is very flexible for many real-world tasks

𝒚 𝒚 𝒚 𝒚 𝒚 𝒚

𝒉 𝒉 𝒉 𝒉 𝒉 𝒉 𝒉 𝒉

𝒙 𝒙 𝒙 𝒙 𝒙 𝒙

many to many (1) many to many (2)

(video classification) (machine translation)

… more to come in our next lectures

# Problems of RNN

 RNNs don’t capture long-term dependency sky

adequately 𝒚

o A hidden state is computed based on only one previous

state → can only capture short-term dependency 𝒉 𝒉 𝒉 𝒉 𝒉 𝒉

 Modelling drawbacks

o Technical problem when training long sequences 𝒙 𝒙 𝒙 𝒙 𝒙 𝒙 ▪ vanishing gradient problem

o Many layers of nonlinear transformation prevent the data the clouds are in the

signals and gradient from flowing easily through the Vietnamese network.

𝒚

 How to address this? 𝒉 𝒉 𝒉 𝒉 𝒉 𝒉 𝒉 𝒉 𝒉 o Using gating mechanism: adding linear component from

previous layer! ▪ RNN → LSTM/GRU 𝒙 𝒙 𝒙 𝒙 𝒙 𝒙 𝒙 𝒙 𝒙

I was born in Vietnam I speak fluent

125

# Memory cells

Artificial memory cell

(Source: colah)

Biological memory cells in human brain (Source: news.feinberg.northwestern.edu)

###  Our RNN includes many simple RNN cells

o Input to a cell: ℎ (previous hidden state) and 𝑥 (current input 𝑐𝑡

𝑡−1 𝑡 𝑐

𝑡−1

❖ long-term ❖ long-term token)

memory memory o Output: ℎ𝑡 = tanh(𝑥𝑡𝑈 + ℎ𝑡−1𝑊 + 𝑏) o ℎ𝑡 can only capture short-term dependency → short-term memory

###  How to capture long-term memory more efficiently?

ℎ𝑡−1 o LSTM cell and GRU cell

❖ short-term

LSTM memory cell memory

126

# Long Short-Term Memory (LSTM)

𝑦ො𝑡

Long-term 𝑐

𝑡

ℎ𝑡 Short-term

𝑥𝑡

𝑦ො 𝑦ො 𝑦ො𝑇 𝑡 𝑡+1

𝑐 𝑐 𝑐𝑇 𝑡 𝑡+1 … 𝑐𝑡−1

\\ \\ \\ \\ \\ \\

ℎ ℎ ℎ𝑇 𝑡 𝑡+1 … ℎ𝑡−1

𝑥 𝑥 𝑥𝑇 𝑡 𝑡+1

Time

127

128

# Long Short-Term Memory (LSTM)

𝑦ො𝑡

Forget gate 𝑦ො 𝑦ො𝑡 𝑦ො 𝑡−1 𝑡+1

\\ 𝑐𝑡 𝑐𝑡−1 Long-term Output gate

Input gate 𝑐 𝑐 𝑐 𝑡−1 𝑡 𝑡+1 tanh \\

memory state \\

Short-term 𝑓𝑡 𝑔 𝑖 𝑜

𝑡 𝑡 𝑡 ℎ𝑡−1 ℎ𝑡 ℎ

𝑡+1 tanh sigmoid sigmoid

memory state sigmoid

FC FC FC FC

𝑥𝑡−1 𝑥𝑡 𝑥

𝑡+1 ℎ

ℎ 𝑡 𝑡−1

𝑥𝑡  𝑔 = tanh(ℎ 𝑊 + 𝑥 𝑈 + 𝑏 )

𝑡 𝑡−1 𝑡  Regression

𝑓 𝑓 𝑓  Forget gate: 𝑓 = 𝜎 𝑥 𝑈 + ℎ 𝑊 + 𝑏

𝑡 𝑡 𝑡−1 𝑦ො𝑡 = 𝒉𝒕𝑽 + 𝒄

𝑖 𝑖 𝑖  Input gate: 𝑖 = 𝜎(𝑥 𝑈 + ℎ 𝑊 + 𝑏)

𝑡 𝑡 𝑡−1  Classification

 LSTM long-term state: 𝑐 = 𝑓 ⊙ 𝑐 + 𝑔 ⊙ 𝑖

𝑡 𝑡 𝑡−1 𝑡 𝑡 𝑦ො𝑡 = softmax(𝒉𝒕𝑽 + 𝒄)

𝑜 𝑜 𝑜  Output gate: 𝑜 = 𝜎 𝑥 𝑈 + ℎ 𝑊 + 𝑏

𝑡 𝑡 𝑡−1  LSTM short-term state: ℎ = 𝑜 ⊙ tanh(𝑐)

𝑡 𝑡 𝑡  LSTM output: 𝒚ෝ = 𝒉 𝑽 + 𝒄 or 𝒚ෝ = softmax(𝒉 𝑽 + 𝒄)

𝒕 𝒕 𝒕 𝒕

# Long Short-Term Memory (LSTM)

Summary

 LSTM belongs to a class of gated RNN models  LSTM introduces self-loops to create paths where the

gradient can flow for long durations 𝑦𝑡  Improve over basic RNN cell

Forget gate o Can capture long-term dependency

𝑐 \\ 𝑡−1 𝑐𝑡 o Faster and more robust to train, often with quicker convergence

Output gate Input gate

tanh \\ \\  LSTM cells manage two state vectors, and for

performance reasons they are kept separate by 𝑓 𝑔 𝑖 𝑜

𝑡 𝑡 𝑡 𝑡 sigmoid tanh sigmoid sigmoid default

FC FC FC FC ▪ ℎ𝑡 as the short-term state ▪ 𝑐𝑡 as the long-term state ℎ ℎ𝑡

𝑡−1  Gates can remove or add information to the cell state:

forget, input, output

𝑥𝑡

129

130

# Gated Recurrent Unit

[Cho et. al, 2014]

### ❑ Proposed by Cho et al. 2014

❑ Was introduced for Encoder-Decoder network

### ❑ Can be viewed as a simplified version of the LSTM:

❑ Both long-term 𝑐𝑡 and short-term ℎ𝑡 are merged to a single state ℎ𝑡 ❑ Single update gate controller 𝑧𝑡 is used to control both forget and input gates

❑ = 1: open input gate, close forget gate ❑ = 0: close input gate, open forget gate ❑ i.e., whenever a memory must be stored, the location to be stored will be erased first! ❑ Output gate is removed, but an additional reset gate 𝑟𝑡 controls how much previous state should be carried

forward

Cho et al., Learning Phrase Representation using RNN Encoder-Decoder for Statistical Machine Translation, Proceedings of EMNLP, 2014.

# Gated Recurrent Unit

[Cho et. al, 2014]

 Update gate 𝑧 decides how much the unit

𝑡 updates its state:

𝑧 𝑧 𝑦ො 𝑧𝑡 = 𝜎 𝑥𝑡𝑈 + ℎ𝑡−1𝑊 𝑡  Reset gate controls which parts of the state get

used to compute the next target state ℎ \\ ℎ

𝑡−1 𝑡 𝑟 𝑟 𝑟𝑡 = 𝜎 𝑥𝑡𝑈 + ℎ𝑡−1𝑊 1-

\\ 𝑔𝑡 Res et gate  The memory state ℎ is a linear interpolation

𝑡 tanh

\\ between ℎ and 𝑔 𝑧 Update gate FC

𝑡−1 𝑡 𝑟𝑡 𝑡

sigmoid sigmoid ℎ𝑡 = 1 − 𝑧𝑡 ⊙ ℎ𝑡−1 + 𝑧𝑡 ⊙ 𝑔𝑡 FC FC

where the candidate 𝑔𝑡 is pre-computed

𝑔 𝑔 𝑔𝑡 = tanh 𝑥𝑡𝑈 + 𝑟𝑡 ⊙ ℎ𝑡−1 𝑊 𝑥

𝑡

 When 𝑧 and 𝑟 are close to 1, GRU will be

𝑡 𝑡 reduced to Basic RNN

131

132

# Sentiment analysis

###  Movie review dataset

≥ 0.5(pos) and < 0.5(neg)

1. I like that movie (pos:1). Output layer (sigmoid)
2. This is a bad movie to watch (neg:0) RNN RNN RNN RNN RNN
3. I love the movie (pos: 1) cell cell cell cell cell
4. I do not recommend you to watch this movie (neg:0)
5. This movie is fantastic (pos:1) This movie is fantastic \<pad>

How to feed symbolic words to an RNN?

133

# Sentiment analysis

≥ 0.5(pos) and < 0.5(neg)  Movie review dataset

Output layer (sigmoid)

1. I like this movie (pos:1). embedding size =6
2. This is a bad movie to watch (neg:0) RNN RNN RNN RNN RNN cell cell cell cell cell
3. I love this movie (pos: 1)
4. I do not recommend you to watch this movie (neg:0) This movie is fantastic \<pad>
5. This movie is fantastic (pos:1)

 Build up vocabulary

1. Like (index: 1) Embedding matrix (𝑬 [𝟖×𝟔])
2. Love (index: 2) 𝑬𝟏 1 2 1.5 -1.2 1.3 1
3. Bad (index: 3) 𝑬𝟐 -1 1.3 -2.5 -1.2 1.6 -1
4. Fantastic (index: 4) 𝑬𝟑 1 3.3 -3.5 -1.0 2.6 1
5. Not (index: 5)
6. Recommend (index: 6) 𝑬 -1 1.3 -2.5 -1.2 1.8 -1 𝟒 𝑬𝟓 -1.2 2.3 -2.5 -1.2 -1.6 1

❑ Not in vocabulary (out of vocabulary bucket: 2)

𝑬𝟔 -1.7 -1.3 -2.5 -1.2 3.6 1.2

1. I, movie, to, pad (index: 7) 𝑬𝟕 -4.2 2.3 -3.5 4.3 1.8 -2
2. This, is, watch (index: 8) 𝑬𝟖 -1.7 -1.3 -4.5 -2.2 -3.6 1

134

# Sentiment analysis

 Movie review dataset ≥ 0.5(pos) and < 0.5(neg)

1. I like this movie (pos:1). Output layer (sigmoid) embedding size =6
2. This is a bad movie to watch (neg:0) 𝑊 𝑊 𝑊 𝑊 ℎ ℎ ℎ ℎ ℎ5
3. I love this movie (pos: 1) 1 2 3 4
4. I do not recommend you to watch this movie (neg:0) ×𝐸 ×𝐸 ×𝐸 ×𝐸 ×𝐸
5. This movie is fantastic (pos:1) 𝟖 𝟕 𝟖 𝟒 𝟕 This movie is fantastic \<pad> [0,0,0,0,0,0,0,1] [0,0,0,0,0,0,1,0] [0,0,0,0,0,0,0,1] [0,0,0,1,0,0,0,0] [0,0,0,0,0,0,1,0]

 Build up vocabulary

1. Like (index: 1) Embedding matrix (𝑬 [𝟖×𝟔])
2. Love (index: 2) 𝑬𝟏 1 2 1.5 -1.2 1.3 1
3. Bad (index: 3) 𝑬𝟐 -1 1.3 -2.5 -1.2 1.6 -1
4. Fantastic (index: 4)
5. Not (index: 5) 𝑬𝟑 1 3.3 -3.5 -1.0 2.6 1
6. Recommend (index: 6) 𝑬𝟒 -1 1.3 -2.5 -1.2 1.8 -1 𝑬𝟓 -1.2 2.3 -2.5 -1.2 -1.6 1

❑ Not in vocabulary (out of vocabulary bucket: 2)

𝑬𝟔 -1.7 -1.3 -2.5 -1.2 3.6 1.2

1. I, movie, to, pad (index: 7) 𝑬𝟕 -4.2 2.3 -3.5 4.3 1.8 -2
2. This, is, watch (index: 8) 𝑬𝟖 -1.7 -1.3 -4.5 -2.2 -3.6 1

# Sentiment analysis

≥ 0.5(pos) and < 0.5(neg)

###  The word/item embedding

Output layer (sigmoid) 1×6 o 𝑒1 = 18𝐸 = 𝐸8 ∈ ℝ

𝑊 𝑊 𝑊 𝑊 ▪ 1×8 × 8×6 = 1×6 ℎ ℎ ℎ ℎ ℎ5

1 2 3 4 ▪ 1𝑖 is one-hot vector in which the 𝑖 − 𝑡ℎ element is 1

𝑈 𝑈 𝑈 and the rest are 0.

1×6 Embedding layer 𝑒 𝑒 𝑒 𝑒 𝑒

1 2 3 4 5 o 𝑒2 = 17𝐸 = 𝐸7 ∈ ℝ

embedding size =6 1×6 o 𝑒 = 1 𝐸 = 𝐸 ∈ ℝ ×𝐸 ×𝐸 ×𝐸 ×𝐸 ×𝐸

3 8 8

1×6 𝟖 𝟕 𝟖 𝟒 𝟕 o 𝑒4 = 14𝐸 = 𝐸4 ∈ ℝ

This movie is fantastic \<pad> 1×6 o 𝑒 = 1 𝐸 = 𝐸 ∈ ℝ

5 7 7 [0,0,0,0,0,0,0,1] [0,0,0,0,0,0,1,0] [0,0,0,0,0,0,0,1] [0,0,0,1,0,0,0,0] [0,0,0,0,0,0,1,0]

Embedding matrix (𝑬 [𝟖×𝟔])

###  The sequence embedding

𝑬𝟏 1 2 1.5 -1.2 1.3 1 𝑒1

𝑬𝟐 -1 1.3 -2.5 -1.2 1.6 -1 𝑒2 𝑒 5×6 𝑠𝑒𝑞_𝑙𝑒𝑛 ×𝑒𝑚𝑏𝑒𝑑_𝑠𝑖𝑧𝑒 o 𝑒 = 3 ∈ ℝ = ℝ 𝑬𝟑 1 3.3 -3.5 -1.0 2.6 1

𝑒4

𝑬𝟒 -1 1.3 -2.5 -1.2 1.8 -1 𝑒5

𝑬𝟓 -1.2 2.3 -2.5 -1.2 -1.6 1 o Embedding lookup operation

𝑬𝟔 -1.7 -1.3 -2.5 -1.2 3.6 1.2 ▪ Pick rows with indices 8, 7, 8, 4, 7

𝑬𝟕 -4.2 2.3 -3.5 4.3 1.8 -2 𝑬𝟖 -1.7 -1.3 -4.5 -2.2 -3.6 1

135

≥ 0.5(pos) and < 0.5(neg)

# Sentiment analysis Output layer (sigmoid)

𝑊 𝑊 𝑊 𝑊 ℎ ℎ ℎ ℎ ℎ5

###  The word/item embedding 1 2 3 4

1×6 o 𝑒1 = 18𝐸 ∈ ℝ

1×6 o 𝑒 = 1 𝐸 ∈ ℝ Embedding layer

2 7 𝑒 𝑒 𝑒 𝑒 𝑒

1 2 3 4 5 1×6 embedding size =6 o 𝑒3 = 18𝐸 ∈ ℝ

1×6 ×𝐸 ×𝐸 ×𝐸 ×𝐸 ×𝐸 o 𝑒4 = 14𝐸 ∈ ℝ

𝟖 𝟕 𝟖 𝟒 𝟕 1×6 o 𝑒 = 1 𝐸 ∈ ℝ

5 7 This movie is fantastic \<pad>

[0,0,0,0,0,0,0,1] [0,0,0,0,0,0,1,0] [0,0,0,0,0,0,0,1] [0,0,0,1,0,0,0,0] [0,0,0,0,0,0,1,0]

###  The sequence embedding

𝑒1

≥ 0.5(pos) and < 0.5(neg) 𝑒2

5×6 𝑠𝑒𝑞_𝑙𝑒𝑛 ×𝑒𝑚𝑏𝑒𝑑_𝑠𝑖𝑧𝑒 Output layer (sigmoid) 𝑒3 o 𝑒 = ∈ ℝ = ℝ

𝑒4

𝑊 𝑊 𝑊 𝑊 ℎ ℎ ℎ ℎ ℎ5 𝑒 1 2 3 4 5

###  The sequence batch embedding

Embedding layer 𝑒 𝑒 𝑒 𝑒 𝑒 o The embedding for one sequence: 1 2 3 4 5

embedding size =6 𝑠𝑒𝑞_𝑙𝑒𝑛 ×𝑒𝑚𝑏𝑒𝑑_𝑠𝑖𝑧𝑒 ℝ.

batch_size=2 ×𝐸 ×𝐸 ×𝐸 ×𝐸 ×𝐸 o The embedding for entire batch with

8 7 8 4 7 Input: 𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒 sequences: 3 2 3 5 4

𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒×𝑠𝑒𝑞_𝑙𝑒𝑛 ×𝑒𝑚𝑏𝑒𝑑_𝑠𝑖𝑧𝑒 ℝ

[𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒, 𝑠𝑒𝑞_𝑙𝑒𝑛]

136

# Shape Transformation in RNNs

batch_size

batch_size

batch_size

ℎ ℎ ℎ ℎ ℎ5 1 2 3 4

seq_len … batch_size

𝑊 𝑊 𝑊 ℎ ℎ ℎ ℎ ℎ5 [batch_size, seq_len, hidden_size 1] 1 2 3 4

𝑈 𝑈 𝑈 𝑈 𝑈

… seq_len batch_size

Embedding Layer (E: [vocab_size, embed_size])

batch_size [batch_size, seq_len, embed_size] 8 7 8 4 7 Input:

3 2 3 5 4

[𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒, 𝑠𝑒𝑞_𝑙𝑒𝑛]

137

138

# Sentiment analysis

(batch_size, seq_len)

(batch_size, seq_len, embed_size)

(batch_size, seq_len, 128) (batch_size, 128)

(batch_size, 1)

## Word2Vec

# Word embedding (Word2Vec)

Wikipedia text corpus

We desire…

𝑑 vector space ℝ

King further

Queen

Football

Australia

Canberra

Paris

Canberra : Australia = Paris : ??? 𝑎𝑟𝑔𝑚𝑖𝑛𝑣 𝑣𝐶𝑎𝑛𝑏𝑒𝑟𝑟𝑎 − 𝑣𝐴𝑢𝑠𝑡𝑟𝑎𝑙𝑖𝑎 + 𝑣𝑃𝑎𝑟𝑖𝑠 − 𝑣 = 𝑣𝐹𝑟𝑎𝑛𝑐𝑒

 We have: Many texts in Wikipedia  We want: Learn vector representations for words that preserve semantic and linguistic relationship carried in the text corpus  We need: Devise pretext task to cast the learning word representation to supervised learning.

140

141

# Word2Vec: Pretext task

Pretext task

###  What is the pretext task of Word2Vec?

The quick brown fox jumps over the lazy dogs. Original sentence

target word context word context word The quick brown fox jumps over the lazy dogs. Skip-gram

𝒑𝒓𝒆𝒅𝒊𝒄𝒕 ❑ Target word context words

The quick brown fox jumps over the lazy dogs.

Continuous Bag of Words (CBOW)

𝒑𝒓𝒆𝒅𝒊𝒄𝒕 ❑ Context words target word The quick brown fox jumps over the lazy dogs.

The quick brown fox jumps over the lazy dogs.

The quick brown fox jumps over the lazy dogs.

142

# Skip-gram

Pretext task

###  What is the pretext task of Skip-gram?

Skip-gram The quick brown fox jumps over the lazy dogs.

𝒑𝒓𝒆𝒅𝒊𝒄𝒕 ❑ Target word context words target word context word context word

(brown, the), (brown, quick), (brown, for), The quick brown fox jumps over the lazy dogs.

(brown, jumps)

(fox, quick), (fox, brown), (fox, jumps), (fox, The quick brown fox jumps over the lazy dogs.

overs)

(jumps, brown), (jumps, fox), (jumps, over), The quick brown fox jumps over the lazy dogs.

(jumps, the)

(over, fox), (over, jumps), (over, the), (over, The quick brown fox jumps over the lazy dogs.

lazy)

(the, jumps), (the, over), (the, lazy), (the, The quick brown fox jumps over the lazy dogs.

dogs)

143

# Skip-gram

Modelling

brow  Current window n

o The quick brown fox jumps  𝑃 𝑡ℎ𝑒, 𝑞𝑢𝑖𝑐𝑘, 𝑓𝑜𝑥, 𝑗𝑢𝑚𝑝𝑠 𝑏𝑟𝑜𝑤𝑛 = 𝑃 𝑡ℎ𝑒 𝑏𝑟𝑜𝑤𝑛 ×𝑃 𝑞𝑢𝑖𝑐𝑘 𝑏𝑟𝑜𝑤𝑛 ×𝑃 𝑓𝑜𝑥 𝑏𝑟𝑜𝑤𝑛 ×𝑃(𝑗𝑢𝑚𝑝𝑠 ∣

𝑏𝑟𝑜𝑤𝑛) the quick fox jumps  log 𝑃 𝑡ℎ𝑒, 𝑞𝑢𝑖𝑐𝑘, 𝑓𝑜𝑥 𝑗𝑢𝑚𝑝𝑠 𝑏𝑟𝑜𝑤𝑛 = log 𝑃 𝑡ℎ𝑒 𝑏𝑟𝑜𝑤𝑛 + ⋯ +

log 𝑃(𝑗𝑢𝑚𝑝𝑠 ∣ 𝑏𝑟𝑜𝑤𝑛)  (brown, the), (brown, quick), (brown, for), (brown, jumps). Let consider

𝒉 𝑡𝑤 = 𝑏𝑟𝑜𝑤𝑛 and 𝑐𝑤 = 𝑡ℎ𝑒.  Two matrices

𝑵×𝒅 o 𝑼 ∈ ℝ (𝑁 is vocabulary size, 𝑑 is embedding size) One-hot

𝒅×𝑵 vector o 𝑽 ∈ ℝ 𝒑

𝑁×𝑑 𝑑×𝑁 𝟏 𝑈 ∈ ℝ 𝑉 ∈ ℝ 𝒕  Assume that indices of 𝑡𝑤 and 𝑐𝑤 are 1 ≤ 𝑡, 𝑐 ≤ 𝑁 respectively. The

forward propagation is as follows:

𝑟 1×𝑑 1×𝑁 1×𝑁 o ℎ = 1𝑡𝑈 = 𝑈 ∈ ℝ, 𝑜 = ℎ𝑉 ∈ ℝ, 𝑝 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑜) ∈ ℝ

𝑡 o 𝑃 𝑐𝑤 = 𝑡ℎ𝑒 𝑡𝑤 = 𝑏𝑟𝑜𝑤𝑛 = 𝑝𝑐

𝑑 𝑟 𝑐 𝑁 𝑟 𝑐 o log 𝑃 𝑐𝑤 = 𝑡ℎ𝑒 𝑡𝑤 = 𝑏𝑟𝑜𝑤𝑛 = log 𝑝 = 𝑈 𝑉 − log(σ exp(𝑈 𝑉))

𝑐 𝑡 𝑐 𝑘=1 𝑡 𝑘

𝑁 𝑁  Train the model by maximizing log likelihood.

# Toy example

##  Corpus: the quick brown fox jumps over the lazy dog

### o Tokens: {'brown’: 1, 'lazy’: 2, 'over’: 3, 'fox’: 4, 'dog’: 5, 'quick’: 6, 'the’: 7, 'jumps’: 8}

### o Number of tokens N = 8 quick

### o Context (window) size C = 3 [0 0 0 0 0 1 0 0]

one-hot encoding

### o Size of embedded vectors d = 3

### o U&V: collections of input & output vectors

144

145

# Skip-gram, forward propagation

1xN Nxd 1xd dxN 1xN 1xN

𝑇.7.2.5 1 quick 𝒉 = 𝟏 𝑼 𝒐 = 𝒉𝑽 𝑝(𝑐𝑤|𝑡𝑤) 0

𝟔.7.4.0 0... 𝑇 𝑇 brown 8 7 3 0.45.09 0 [0 0 0 0 0 1 0 0] 0

... 0.53.10 1 6 3 0

0. 0\. 8. 3. 0. 9. 7. 7. 0.89.14

.5.8.7 ℎ = 0.7 0.7,0.6

0.48.09 0

3. 5\. 3. 3. 7. 9. 4. 2.

... 0.67 softmax.11 CE loss 0 7 7 6

4. 3\. 2. 1. 3. 1. 0. 3..21 1.32

.8.1.3 0.77.12 𝑤

𝑡+1.9.6.5

0.81.13

𝑇 0 0 𝑤𝑡 = 16 U V

0 0 the 0 0 1 0

𝑤𝑡−1 center

the quick brown context {'brown’: 1, 'lazy’: 2, 'over’: 3, 'fox’: 4, 'dog’: 5, 'quick’: 6, 'the’: 7, 'jumps’: 8}

146

# Skip-gram

Drawback

###  Two matrices

𝑵×𝒅 o 𝑼 ∈ ℝ (𝑁 is vocabulary size, 𝑑 is embedding size)

𝒅×𝑵 o 𝑽 ∈ ℝ

###  Assume that indices of 𝑡𝑤 and 𝑐𝑤 are 1 ≤ 𝑡, 𝑐 ≤ 𝑁

𝒉

### respectively. The forward propagation is as follows:

𝑟 1×𝑑 1×𝑁 o ℎ = 1𝑡𝑈 = 𝑈 ∈ ℝ, 𝑜 = ℎ𝑉 ∈ ℝ, 𝑝 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑜) ∈

𝑡 One-hot 1×𝑁 ℝ vector

𝒑 𝑁×𝑑 𝑑×𝑁 𝟏 𝑈 ∈ ℝ 𝑉 ∈ ℝ o 𝑃 𝑐𝑤 = 𝑡ℎ𝑒 𝑡𝑤 = 𝑏𝑟𝑜𝑤𝑛 = 𝑝 𝒕

𝑐

𝑟 𝑐 o log 𝑃 𝑐𝑤 = 𝑡ℎ𝑒 𝑡𝑤 = 𝑏𝑟𝑜𝑤𝑛 = log 𝑝𝑐 = 𝑈𝑡 𝑉𝑐 −

𝑁 𝑟 𝑐 log(σ exp(𝑈 𝑉))

𝑘=1 𝑡 𝑘

###  Some drawbacks 𝑑

o 𝑝 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑜) is computationally expensive

𝑁 𝑁 1×𝑁 o 𝑝 ∈ ℝ is a distribution over the vocabulary with size 𝑵 (usually very

big) → the values of 𝑝𝑖 are very tiny → hard to train ▪ Hierarchical SoftMax ▪ Negative sampling (more popular and efficient)

147

# Skip-gram

Negative sampling

###  Transform 𝑵-class prediction to binary prediction with

### negative examples

###  Consider a positive (true) pair (tw=brown, cw = the) 𝒉

o [(brown, the),1] o Sample randomly some (two) words

One-hot ▪ [(brown, 𝑛𝑔 = hello), 0] and [(brown, 𝑛𝑔 = awsome), 0] vector

1 2 𝒑

𝑑×𝑁 𝑁×𝑑 𝟏 𝑈 ∈ ℝ 𝑉 ∈ ℝ ▪ Let denote the indices of 𝑛𝑔 and 𝑛𝑔 by 1 ≤ 𝑛, 𝑛 ≤ 𝑁. 𝒕

1 2 1 2

###  The forward propagation

𝑟 1×𝑑 1×𝑁 1×𝑁 o ℎ = 1𝑡 𝑈 = 𝑈 ∈ ℝ, 𝑜 = ℎ𝑉 ∈ ℝ, 𝑝 = 𝑠𝑖𝑔𝑚𝑜𝑖𝑑(𝑜) ∈ ℝ

𝑡 o 𝑃 𝑦 = 1 tw=brown, cw =𝑡ℎ𝑒 = 𝑝 𝑑

𝑐 o 𝑃 𝑦 = 1 tw=brown, 𝑛𝑔1=ℎ𝑒𝑙𝑙𝑜 = 𝑝𝑛

1 𝑁

𝑁 o 𝑃 𝑦 = 1 tw=brown, 𝑛𝑔2=𝑎𝑤𝑠𝑜𝑚𝑒 = 𝑝𝑛

2

###  Optimization problem

o 𝐦𝐚𝐱 log 𝑝𝑐 − 𝛼 log 𝑝𝑛 − 𝛼 log 𝑝𝑛 where 𝛼>0 is a trade-off parameter.

1 2

148

# Continuous Bag of Words (CBOW)

Pretext task

###  What is the pretext task of CBOW?

Continuous Bag of Words (CBOW) The quick brown fox jumps over the lazy dogs.

𝒑𝒓𝒆𝒅𝒊𝒄𝒕 ❑ Context words target word target word context word context word

(the | quick | for | jumps, brown) The quick brown fox jumps over the lazy dogs.

(quick | brown | jumps | over, fox) The quick brown fox jumps over the lazy dogs.

(brown | fox | over | the, jumps) The quick brown fox jumps over the lazy dogs.

(for | jumps | the | lazy, over) The quick brown fox jumps over the lazy dogs.

(jumps | over | lazy| dog, the) The quick brown fox jumps over the lazy dogs.

149

# Continuous Bag of Words (CBOW)

Modelling

 Current window

fox jumps the quick o The quick brown fox jumps  Need to formulate: 𝑃 𝑏𝑟𝑜𝑤𝑛 𝑡ℎ𝑒, 𝑞𝑢𝑖𝑐𝑘, 𝑓𝑜𝑥, 𝑗𝑢𝑚𝑝𝑠  (the | quick | fox | jumps, brown). Let consider 𝑡𝑤 = 𝑏𝑟𝑜𝑤𝑛 and 𝑐𝑤 =

1 brow 𝑡ℎ𝑒, 𝑐𝑤 = 𝑞𝑢𝑖𝑐𝑘, 𝑐𝑤 = 𝑓𝑜𝑟, 𝑐𝑤 = 𝑗𝑢𝑚𝑝𝑠. n

2 3 4  Two matrices

𝑵×𝒅 o 𝑼 ∈ ℝ (𝑁 is vocabulary size, 𝑑 is embedding size)

𝒅×𝑵 𝒉 o 𝑽 ∈ ℝ

Vector  Assume that indices of 𝑡𝑤 is 1 ≤ 𝑡 ≤ 𝑁 and 𝑐𝑤 are 1 ≤ 𝑐 ≤ 𝑁

1:4 1:4 respectively. The forward propagation is as follows: 𝟏

(𝟏𝒄

𝟏 1𝑐 +⋯+1𝑐 1 𝟒 1 4 𝑟 𝑟 𝑟 1×𝑑 1×𝑁 𝒑 o ℎ = 𝑈 = 𝑈𝑐 + ⋯ + 𝑈𝑐 = 𝑈 ∈ ℝ, 𝑜 = ℎ𝑉 ∈ ℝ, 𝑝 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑜) ∈ 𝑁×𝑑 𝑑×𝑁

4 4 1 4 + 𝟏 𝑈 ∈ ℝ 𝑉 ∈ ℝ

𝒄𝟐 1×𝑁 ℝ

- 𝟏𝒄 𝟑

o 𝑃 𝑏𝑟𝑜𝑤𝑛 𝑡ℎ𝑒, 𝑞𝑢𝑖𝑐𝑘, 𝑓𝑜𝑥 𝑗𝑢𝑚𝑝𝑠 = 𝑝𝑡

+𝟏𝒄)

𝟒 𝑟 𝑐 o log 𝑃 𝑏𝑟𝑜𝑤𝑛 𝑡ℎ𝑒, 𝑞𝑢𝑖𝑐𝑘, 𝑓𝑜𝑥 𝑗𝑢𝑚𝑝𝑠 = log 𝑝𝑡 = 𝑈 𝑉 −

𝑡 𝑁 𝑟 𝑐 log(σ exp(𝑈 𝑉))

𝑘=1 𝑘

𝑑  Train the model by maximizing log likelihood.

𝑁 𝑁

150

# CBOW: forward propagation

brown

𝑇 1 0 0 Nxd 1xd dxN 1xN 1xN 1xN 0

.7.2.5 0   11 + 17 𝑜 = ℎ𝑉 𝑝(𝑜|𝑖)

... ℎ = 𝑈 quick 0 7 4 0 2

𝑇 𝑇 𝑇 0 0.55.11 0

.8.7.3 0

.3.0.1.3.1.2.3.4 0.44.10 0.1.6.3 𝑤𝑡+1 0.93.16 0

.2.4.9.7.3.3.5.3...        0.40.09 0 5 8 7 ℎ = 0.75 0.15 0.4 softmax

.7.7.9.0.3.8.0.0 0.46.10 0.7.7.6 the   0.94.16 1

𝑇... 0.60.11 0

8 1 3 0

0.97.17 0 0.9.6.5 0

CE loss 0 0 U V 𝑤

𝑡 0 1 0 𝑤𝑡−1

context center

## Advanced Sequential Models

# Encoder-Decoder Models

Context or encode 𝒄

Target structured data Source structured data

𝒄 = 𝒉𝒍𝒂𝒔𝒕

𝒄

Machine translation Image captioning

Many more applications ❖ C++ programs to Java programs, texts to images, question answering (questions to answers), and etc.

152

153

# Introduction

### Problem statement of seq2seq

o Source sequence: 𝑥 = 𝑥1, 𝑥2, … , 𝑥𝑇 where 𝑥𝑖𝜖𝑉𝑥

𝑥 o Target sequence: y = 𝑦1, 𝑦2, … , 𝑦𝑇 where 𝑦𝑖𝜖𝑉𝑦

𝑦 o Training set:

𝑁 (1) (1) (2) (2) (𝑁) (𝑁) (𝑖) (𝑖) ▪ 𝔇 = 𝑥, 𝑦, 𝑥, 𝑦, … , 𝑥, 𝑦 = 𝑥, 𝑦

𝑖=1 (𝑖) ▪ 𝑆𝑥: set of possible source sequences ( 𝑥 ⊆ 𝑆𝑥)

(𝑖) ▪ 𝑆𝑦: set of possible target sequences ( 𝑦 ⊆ 𝑆𝑦) o Task

▪ Learn a function 𝑓: 𝑆𝑥 → 𝑆𝑦

### Applications

o Machine Translation o Image Captioning

154

# Machine translation

Paper: paper link

### Translate a sentence in a language to another

### language

o Input: He loved to eat. (English) o Output: Il aimait manger. (French)

𝐼𝑙 𝑎𝑖𝑚𝑎𝑖𝑡 𝑚𝑎𝑛𝑔𝑒𝑟 < 𝑬𝑶𝑺 >......... ... vocabulary size+1

ℎ1 ℎ2 ℎ3 𝑐 = ℎ4 𝑞 𝑞 𝑞 𝑞

0 1 2 3

𝐻𝑒 𝑙𝑜𝑣𝑒𝑑 𝑡𝑜 𝑒𝑎𝑡 \<BOS> 𝐼𝑙 𝑎𝑖𝑚𝑎𝑖𝑡 𝑚𝑎𝑛𝑔𝑒𝑟

155

# Encoder-decoder model for seq2seq

Fixed context vector

### Encoder

o Produces the context vector 𝒄 = 𝒉𝑻 of the input sequence

𝒙 o Context vector c summarizes input sequence [𝐱𝟏, … , 𝐱𝐓 ].

𝐱

### Decoder

o Decodes the encode c to the output sequence

### Special symbols

o \<EOS> signifies the end of a sequence o \<BOS> signifies the beginning of a sequence

ℎ ℎ ℎ ℎ ℎ 𝑐 = ℎ𝑇 1 2 𝑖−1 𝑖 𝑇𝑥−1 𝑥

𝑥 𝑥 𝑥 𝑥 𝑥 𝑥𝑇 1 2 𝑖−1 𝑖 𝑇𝑥−1 𝑥 \<EOS> 𝑦𝑗+1 𝑦𝑗 𝑦3 𝑦 𝑦

2 1

𝑞𝑇 𝑞 𝑞 𝑞 𝑞 𝑞

𝑦 𝑗 𝑗−1 2 1 0

𝑦 𝑦 𝑦 𝑦 𝑦 \<BOS> 𝑇𝑦 𝑗 𝑗−1 2 1

156

# Training of seq2seq

𝑦1 𝑦2 𝑦 \<EOS>

𝑗

ℎ 𝒄 = 𝒉𝑻 𝒒 𝑞 𝑞 𝑞 1 𝒙 𝟎 1 𝑗−1 𝑇

𝑦

𝑥 𝑥𝑇 \<BOS> 𝑦1 𝑦 1 𝑥 𝑗−1 𝑦

𝑇𝑦 ❑ We need to maximize the log-likelihood:

max J θ = ෍ log 𝑃 𝒚|𝒙, 𝜃 θ

𝒙,𝒚 𝜖𝔇 where 𝜃 = [𝜃𝑒, 𝜃𝑑] and 𝜃𝑒, 𝜃𝑑 are encoding and decoding parameters respectively. ❑ Product rule:

𝑃 𝒚|𝒙, 𝜃 = 𝑃 𝒚𝟏:𝑻 𝒙𝟏:𝑻, 𝜃 = 𝑃 𝒚𝟏:𝑻 𝒄, 𝜃

𝒚 𝒙 𝒚

𝑇𝑦 = 𝑃 𝒚𝟏 𝒄, 𝜃 𝑃 𝒚𝟐 𝒚𝟏, 𝒄, 𝜃 … 𝑃 𝒚𝒋 𝒚𝟏:𝒋−𝟏, 𝒄, 𝜃 … 𝑃 𝒚𝑻 𝒚𝟏:𝑻 −𝟏, 𝒄, 𝜃 = ෑ 𝑃 𝒚𝒋 𝒚𝟏:𝒋−𝟏, 𝒄, 𝜃

𝒚 𝒚

𝑗=1 𝑇𝑦 𝑇𝑦 log 𝑃 𝒚|𝒙, 𝜃 = log 𝑃 𝒚|𝒄, 𝜃 = ෍ log 𝑃 𝒚𝒋|𝒚𝟏:𝒋−𝟏, 𝒄, 𝜃 = ෍ log 𝑃 𝒚𝒋|𝒒𝒋−𝟏, 𝒄, 𝜃

𝑗=1 𝑗=1 o We can compute 𝑃 𝒚𝒋 𝒒𝒋−𝟏, 𝒄 = 𝑔 𝒚𝒋, 𝒒𝒋−𝟏, 𝒄 where 𝑔 is a nonlinear, potentially multi-layered NN that

outputs the probability of 𝑦𝑗. Pay attention on how 𝑐 is used in every step during decoding

157

# Training of seq2seq

### ❑ We need to maximize the log-

### likelihood:

(1) [Sutskever et al. 2014] max J θ = ෍ log 𝑃 𝒚|𝒙, 𝜃 θ

𝒙,𝒚 𝜖𝔇 where 𝜃 = [𝜃𝑒, 𝜃𝑑 ] and 𝜃𝑒, 𝜃𝑑 are encoding and decoding parameters, respectively.

### ❑ Product rule:

𝑇𝑦 log 𝑃 𝒚|𝒙, 𝜃 = log 𝑃 𝒚|𝒄, 𝜃 = σ log 𝑃 𝒚 |𝒚, 𝒄, 𝜃 =

𝑗=1 𝒋 𝟏:𝒋−𝟏 𝑇𝑦 σ log 𝑃 𝒚 |𝒒, 𝒄, 𝜃 𝑗=1 𝒋 𝒋−𝟏 ❖ 𝒒𝒋−𝟏 contains the information of 𝒄. ❖ (1): 𝑃 𝒚𝒋|𝒒𝒋−𝟏, 𝒄, 𝜃 = 𝑃 𝒚𝒋|𝒒𝒋−𝟏, 𝜃

𝑞𝑗−1 = 𝑓(𝑞𝑗−2, 𝑦𝑗−1) ❖ (2): 𝑃 𝒚𝒋|𝒒𝒋−𝟏, 𝒄, 𝜃 = 𝑃 𝒚𝒋|𝒒𝒋−𝟏, 𝜃

(2) 𝑞𝑗−1 = 𝑓(𝑞𝑗−2, 𝑐𝑜𝑛𝑐𝑎𝑡(𝑐, 𝑦𝑗−1))

Big drawback: the context vector 𝒄 is fixed across timesteps. ❖ 𝑓 is a memory cell (e.g., LSTM or GRU)

158

# Inference

ℎ 𝒄 = 𝒉𝑻 1 𝒙

𝑥 𝑥𝑇 1 𝑥

We know 𝑃(𝑦1 = ∘∣ 𝑐) We know 𝑃(𝑦 = ∘∣ 𝑐)

2 𝑦 =? 𝑦1 𝑦 𝑦 1 𝑦 𝑦 =? 2 𝑗 \<EOS>

1 2

𝒄 = 𝒉 𝒒 𝒄 = 𝒉 𝒒 𝑞 𝒄 = 𝒉𝑻 𝒒 𝑞 𝑞 𝑞

𝑻𝒙 𝟎 𝑻 𝟎 1 𝒙 𝟎 1 𝑗−1 𝑇

𝒙 𝑦

\<BOS> \<BOS> 𝑦1 \<BOS> 𝑦 𝑦 𝑦

1 𝑗−1 𝑇𝑦

### Given a trained model and input sequence 𝒙

o We need to infer the corresponding output sequence 𝒚

### Two common strategies

o Greedy Decoding and Beam Search Decoding

159

# Inference

Greedy decoding

### Greedy Decoding

o Given x, find word 𝒚𝟏 with highest probability o Given 𝒚𝟏 and x, find word 𝒚𝟐 with highest probability o … o Stop when seeing \<EOS>

𝑃(𝑦 = ∘∣ 𝑐) 𝑃(𝑦 = ∘∣ 𝑐) 𝑃(𝑦𝑇 +1 = ∘∣ 𝑐)

1 2 𝑦

𝑦1 𝑦2 \<EOS>

ℎ 𝑐 = ℎ𝑇 𝑞 𝑞 𝑞 1 𝑥 0 1 𝑇𝑦

𝑥 𝑥𝑇 \<BOS> 𝑦1 𝑦 1 𝑥 𝑇

𝑦

160

# Inference

Beam search

### ❑ Beam Search Decoding with beam width 𝒌

o Given x, find k candidates for 𝒚 with highest probability

𝟏 o Given x, for each candidate 𝒚𝟏, find k candidates for word 𝒚𝟐 with highest probability o Pick top-k sequences 𝑦1𝑦2 with highest joint probability o For each 𝑦 𝑦, find k candidates for word 𝒚 with highest probability

1 2 𝟑 o Pick top-k sequences 𝑦1𝑦2𝑦3 with highest joint probability o ….. o Stop when see \<EOS> on each beam o Finally, pick 1 sequence with highest probability from top-k sequences

𝑦1 𝑦2 \<EOS>

ℎ ℎ𝑇 𝑞 𝑞 𝑞 1 𝑥 0 1 𝑇𝑦

𝑥 𝑥𝑇 \<BOS> 𝑦1 𝑦 1 𝑥 𝑇

𝑦

161

# Inference

Beam search

, (𝑝11) Joint Probability Hi 𝑝 𝑝 …

How (𝑝) 1 11

12 (𝑝1) 𝑝 𝑝 …

1 12 What (𝑝) 𝑝1𝑝13…

13

𝑝2𝑝11… sure (𝑝21) … Input Yes

certainly (𝑝22) 𝑥 (𝑝2)

Beam width = 3 I(𝑝23) - We always

choose three I (𝑝31)

sentences with Please see (𝑝) highest joint

32 (𝑝3) probabilities

come (𝑝33)

# Drawback of fixed context

Fixed context vector 𝒄 is easily

Paper: paper link overwhelmed by long inputs or long outputs.

Fixed context vector 𝒄 At a specific timestep 𝑗, some words or items in the input sequence might possibly contribute more to the generation of next item or word in the output sequence. o I want to see you every day → Je veux te voir

chaque jour o I want to see you every day → Je veux te ?

(voir) …. How to timely adapt the context vector 𝑐𝑗? o 𝑐𝑗 = 𝛼(ℎ1, . . , ℎ𝑇, 𝑞𝑗−1)

𝑥 o Computed using attention mechanism

Timely varied context vectors 𝒄𝒋

162

# Attention mechanism

Bahdanau, Cho, Bengio, Neural Machine Translation by Luong, Pham, Manning, Effective Approach Attention-based Neural Jointly Learning to Align and Translate, ICLR 2015 Machine Translation, EMNLP, 2015

### Attention mechanism allows the decoding network to refer to the input.

o Global attention

▪ Use all input hidden states of the encoder when deriving the context 𝑐𝑡. o Local attention

▪ Use a selective window of input hidden states of the encoder when deriving the context 𝑐𝑡.

163

164

# Global attention

### ❑ Main idea

o Consider all input hidden states of the encoder when deriving the

3 context: 𝑐 = σ 𝑎 (𝑠) ℎ

𝑡 𝑠=1 𝑡 𝑠

keys

### Alignment weights:

exp 𝑠𝑐𝑜𝑟𝑒 𝑞𝑡, ℎ𝑠 𝑎𝑡 𝑠 = align 𝑞𝑡, ℎ𝑠 =

σ exp 𝑠𝑐𝑜𝑟𝑒 𝑞, ℎ 𝑠′ 𝑡 𝑠′ where 𝑞𝑡 is current target state, ℎ𝑠 is each source states.

query

### Alignment score function:

T 𝑞𝑡 ℎ𝑠 𝑑𝑜𝑡 𝑝𝑟𝑜𝑑𝑢𝑐𝑡 T 𝑠𝑐𝑜𝑟𝑒 𝑞𝑡, ℎ𝑠 = ൦ 𝑞 𝑊 ℎ 𝑔𝑒𝑛𝑒𝑟𝑎𝑙 𝑚𝑒𝑡𝑟𝑖𝑐

𝑡 𝑎 𝑠 T 𝑐𝑜𝑛𝑐𝑎𝑡 𝑣𝑎 𝑡𝑎𝑛ℎ 𝑊𝑎 𝑞𝑡; ℎ𝑠

165

# Global attention

𝑦ො𝑡

### Context vector:

3 𝑐𝑡 = ෍ 𝑎𝑡(𝑠) ℎ𝑠

𝑠=1

### Attentional hidden state:

𝑞෤𝑡 = 𝑡𝑎𝑛ℎ 𝑊𝑐 𝑐𝑡; 𝑞𝑡

### Predictive distribution:

𝑝 𝑦𝑡|𝑦\<𝑡, 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 𝑊𝑠 𝑞෤𝑡

166

# Global attention

Example

### 1\. Convert into alignment weights.

𝑒𝑥𝑝 𝑠𝑐𝑜𝑟𝑒 𝑞𝑡, ℎ𝑠 𝑎𝑡 𝑠 =

σ 𝑒𝑥𝑝 𝑠𝑐𝑜𝑟𝑒 𝑞, ℎ 𝑞 𝑠′ 𝑡 𝑠′ 𝑡

### 2\. Build context vector: weighted average

𝑐𝑡 = ෍ 𝑎𝑡(𝑠) ℎ𝑠

𝑠

### 3\. Compute the next hidden state.

𝑞෤𝑡 = 𝑡𝑎𝑛ℎ 𝑊𝑐 𝑐𝑡; 𝑞𝑡

### 4\. Predictive distribution:

𝑝 𝑦𝑡|𝑦\<𝑡, 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 𝑊𝑠 𝑞෤𝑡

𝑞𝑡

167

# Global attention

Drawback

### ❑ Drawback: employ all items on the source

### side for deriving each target item.

## o expensive computation

## o impractical to translate longer

## sequences

## Local attention

168

# Local attention

### Main idea

### o Selectively focuses on a small window of

### context and is differentiable.

### 𝑐𝑡 is then derived as a weighted average over

### the set of source hidden states within the

### window 𝒑𝒕 − 𝑫; 𝒑𝒕 + 𝑫 , 𝐷 is empirically

### selected.

169

# Local attention

Main idea o Selectively focuses on a small window of context and is differentiable For current target word, predict aligned position 𝑝𝑡

𝑇 o 𝑝𝑡 = 𝑆 ∙ sigmoid 𝑣𝑝 tanh 𝑊𝑝𝑞𝑡

where 𝑊𝑝 and 𝑣𝑝 are the model parameters, 𝑆 is the source sentence length, and hence 0 ≤ 𝑝𝑡 ≤ 𝑆 Alignment weights:

2 𝑠−𝑝𝑡

𝐷 o 𝑎 𝑠 = align 𝑞, ℎ exp − with 𝜎 = Τ

𝑡 𝑡 𝑠 2 2

2𝜎

exp 𝑠𝑐𝑜𝑟𝑒 𝑞𝑡,ℎ𝑠 o align 𝑞𝑡, ℎ𝑠 = (temporary alignment

σ exp 𝑠𝑐𝑜𝑟𝑒 𝑞,ℎ 𝑠′ 𝑡 𝑠′ weights)

### Context is now computed as before but within a

### windows of size D, centered at 𝑝𝑡

## Transformers

## A step back: Recurrent Networks (RNNs)

### The cat jumps the wall Encoder

hstart h0 h1 h2 h3 h4 Context vector 𝒄

### \<BOS> Il gatto salta il muro

### Decoder

h5

h6 h7 h8 h9 h10

### Il gatto salta il muro \<EOS>

171

# Problems

### 1\. We forget tokens too far in the past in the context vector c

### 2\. We need to wait the previous token to compute the next hidden-state

### The cat jumps the wall Encoder

hstart h0 h1 h2 h3 h4 Context vector 𝒄

### \<BOS> Il gatto salta il muro

### Decoder

h5

h6 h7 h8 h9 h10

### Il gatto salta il muro \<EOS>

172

# Solving problem 1

"We forget tokens too far in the past" Solution: Add an attention mechanism

### The cat jumps the wall Encoder

hstart h0 h1 h2 h3 h4 Context vector 𝒄

𝛼0 𝛼 𝛼 𝛼

1 2 3

𝛼4 Attention

- \+ + + = Context 𝑐𝑡

### \<BOS> Il gatto salta il muro

### Decoder

h5

h6 h7 h8 h9 h10

### Il gatto salta il muro \<EOS>

173

## Solving problem 2

"We need to wait the previous token to compute the next hidden-state"

### The cat jumps the wall

### Solution

### Throw away recurrent

### connections

hstart h0 h1 h2 h3 h4

2017 paper Attention

"Attention Is All You Need"

- \+ + + = Context 𝑐𝑡

### \<BOS> Il gatto salta il muro

h5

h6 h7 h8 h9 h10

### Il gatto salta il muro \<EOS>

174

# Transformer Motivations

### Problem using RNNs for Seq2Seq models

o Slow due to sequential nature o Poor long-range dependency modelling

### Transformer

o Enable parallel computation, hence exploit the power of GPU. How? o Remove the dependency between words o But position/location matters → resolve by positional encoding o Extremely good at capturing long-range dependency

175

# Transformer – General Architecture

[Source: main paper]

Decoder Encoder

[Source: https://jalammar.github.io/illustrated-transformer/]

### Some building blocks

o Positional encoding Model architecture

o Self-attention and Multi-Head Self-attention Attention o Masked Multi-Head Attention o Residual add and layer norm

176

# Overview

[batch_size,seq_len, d_model]

# Architecture of

Add & Normalize(Layer-Norm +Skip Connection)

# Transformer

Point-wise Feed-forward NN

# Encoder

[batch_size,seq_len, d_model]

Encoder output

Add & Normalize(Layer-Norm +Skip Connection)

× n Encoders

[batch_size,seq_len, d_model] seq_len

batch_size Self-Attention (SA)/ Multi-head SA [batch_size,seq_len, d_model]

[batch_size,seq_len, d_model]

seq_len

batch_size

Embedding Layer (E: [vocab_size, embed_size]) I love deep learning

8 7 9 4 Input: Hello the world \<pad> 3 2 5 0

[𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒, 𝑠𝑒𝑞_𝑙𝑒𝑛]

177

[Source: https://jalammar.github.io/illustrated-transformer/]

# Transformer – Self Attention

𝑧1 = 0.88×𝑣1 + 0.12×𝑣2 Self attention among a sequence of items/tokens o Keys and queries for computing self attention weights 𝑸 𝑲 𝑽 𝑄 𝐾 𝑉 Multiplying 𝒙𝟏, 𝒙𝟐 with 𝑾, 𝑾, 𝑾 to gain o 𝑊, 𝑊, 𝑊 are learnable matrices.

queries, keys, and values.

178

# Transformer

Self-Attention

For each token 𝑥𝑖, calculate its query, key, and value

𝑄 𝐾 𝑉 𝑞𝑖 = 𝑥𝑖𝑊, 𝑘𝑖 = 𝑥𝑖𝑊, 𝑣𝑖 = 𝑥𝑖𝑊

𝑥1 Matrix/stacked version (𝑋 = …, 𝐿 = 𝑠𝑒𝑞_𝑙𝑒𝑛)

𝑥𝐿 𝑄 𝐾 𝑉 𝑄 = 𝑋𝑊, 𝐾 = 𝑋𝑊, 𝑉 = 𝑋𝑊 Calculate the attention probabilities between query and key o Scaled dot-product attention

𝑇 𝑄𝐾 𝐴 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥

𝑑𝑘 Take a weighted sum of values

𝑍 = 𝐴𝑉

179

[batch_size,seq_len, d_model] 𝑍

# Self-Attention 𝑂

Output: 𝑍 = 𝑍𝑊

𝑂 𝑊 𝑍 = 𝐴𝑉 [batch_size,seq_len, 𝑑𝑉]

seq_len 𝑍

𝐴 [batch_size,seq_len, seq_len] Attention probs: 𝐴 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 𝐵, 𝑑𝑖𝑚 = 2

𝑇 [batch_size,seq_len, seq_len] 𝑄𝐾 Attention scores: 𝐵 = 𝐵

𝑑 seq_len 1 1 1 𝑇 1 1 𝐾 𝐵 = 𝑄 𝑖, : 𝐾 𝑗, : → 𝑠𝑖𝑚(𝑥, 𝑥)

𝑖𝑗 𝑖 𝑗

𝑄 𝐾 𝑉 𝑄 = 𝑋𝑊 [𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒, 𝑠𝑒𝑞_𝑙𝑒𝑛, 𝑑𝑄 = 𝑑𝐾] 𝐾 = 𝑋𝑊 [𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒, 𝑠𝑒𝑞_𝑙𝑒𝑛, 𝑑 = 𝑑 ] 𝑉 = 𝑋𝑊 [𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒, 𝑠𝑒𝑞_𝑙𝑒𝑛, 𝑑 ]

𝑄 𝐾 𝑉

seq_len seq_len seq_len

𝑑𝑉 𝑑𝑄 = 𝑑𝐾 𝑑 = 𝑑

𝐾 𝑄

𝑄 𝑉 𝑊 𝐾 𝑊

𝑊

𝑋

𝑑_𝑚𝑜𝑑𝑒𝑙 𝑑_𝑚𝑜𝑑𝑒𝑙 𝑑_𝑚𝑜𝑑𝑒𝑙 2

1 𝑋 𝑋

[batch_size,seq_len, d_model]

seq_len

batch_size

Embedding Layer (E: [vocab_size, embed_size]) I love deep learning

8 7 9 4 Input: Hello the world \<pad> 3 2 5 0

[𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒, 𝑠𝑒𝑞_𝑙𝑒𝑛]

180

# Transformer

Multi-Head Self-Attention

### Main idea:

o Perform self-attention multiple times in parallel and combine results

### Multiple attention heads through multiple Q, K, V matrices

### Each attention head performs attention independently

o Allow attending to parts of the sequence differently

181

# Transformer – Self Attention/Multi-Head Attention

[Source: https://jalammar.github.io/illustrated-transformer/]

182

# Transformer – Self Attention/Multi-Head Attention

Matrix calculation for One-Head Matrix calculation for Multi-Head

[Source: https://jalammar.github.io/illustrated-transformer/]

183

# Transformer – Residual Connection/Layer Norm

One 𝑿 One 𝒁 Batch 𝑿 Batch 𝒁 seq_len

Multi- embed_size Head Multi-

attention Head

attention [𝒃𝒂𝒕𝒄𝒉_𝒔𝒊𝒛𝒆, 𝒔𝒆𝒒_𝒍𝒆𝒏, 𝒆𝒎𝒃𝒆𝒅_𝒔𝒊𝒛𝒆]

𝑿 𝒁 𝒀

[𝒃𝒂𝒕𝒄𝒉_𝒔𝒊𝒛𝒆, 𝒔𝒆𝒒_𝒍𝒆𝒏, 𝒆𝒎𝒃𝒆𝒅_𝒔𝒊𝒛𝒆]

𝑌𝑚 = 𝑚𝑒𝑎𝑛 𝑌, 𝑎𝑥𝑖𝑠 = 𝑒𝑚𝑏𝑒𝑑_𝑠𝑖𝑧𝑒, 𝑘𝑒𝑒𝑝_𝑑𝑖𝑚𝑠 = 𝑇𝑟𝑢𝑒 𝑌𝜎 = 𝑠𝑡𝑑 𝑌, 𝑎𝑥𝑖𝑠 = 𝑒𝑚𝑏𝑒𝑑_𝑠𝑖𝑧𝑒, 𝑘𝑒𝑒𝑝_𝑑𝑖𝑚𝑠 = 𝑇𝑟𝑢𝑒

𝑌 − 𝑌𝑚 𝐿𝑎𝑦𝑒𝑟𝑁𝑜𝑟𝑚 𝑌 = 𝛾 + 𝛽

𝑌𝜎

Layer normalization (LN) is the same as batch normalization except that LN normalizes across the feature dimension. o Batch normalization is usually empirically less effective than layer normalization in natural language processing tasks, whose inputs are often variable-

length sequences.

184

# Transformer- Pointwise

# Feed-forward NN

𝟎 𝑾

## LayerNorm( )

Pointwise Feed-forward NN 𝑍ҧ = 𝑅𝑒𝐿𝑈 𝑍𝑊 + 𝑏 𝑊 + 𝑏

1 1 2 2

𝑍ҧ 𝑍ҧ

## FFN FFN FFN … FFN LayerNorm()

𝑧 𝑧2 𝑧3 𝑧 1 𝑇

Multi-head Self-attention

Next Encoder 𝑥 𝑥 𝑥3 … 𝑥 1 2 𝑇 185

# Transformer – Positional Encoding

Use to capture position of a word in a sentence o Transformer has no recurrence and no convolution. o For the model to make use of the order of the sequence, some

information about the relative or absolute position of the tokens in the sequence needs to be injected

2𝑖/𝑒𝑚𝑏𝑒𝑑_𝑠𝑖𝑧𝑒 𝑃𝐸(𝑝𝑜𝑠,2𝑖) = sin(𝑝𝑜𝑠/1000)

2𝑖/𝑒𝑚𝑏𝑒𝑑_𝑠𝑖𝑧𝑒 𝑃𝐸(𝑝𝑜𝑠,2𝑖+1) = cos(𝑝𝑜𝑠/1000)

𝑃𝐸(1,1) 𝑃𝐸 𝑃𝐸(1,3) 𝑃𝐸 𝑃𝐸 𝑃𝐸 𝑃𝐸 𝑃𝐸 𝑃𝐸 𝑃𝐸 𝑃𝐸 𝑃𝐸

(1,2) (2,1) (2,2) (2,3) (3,1) (3,2) (3,3) (4,1) (4,2) (4,3)

embed_size=3 𝐸 𝐸 𝐸 𝐸 𝐸 𝐸 𝐸 𝐸 𝐸 𝐸81 𝐸82 𝐸83

31 32 33 21 22 23 51 52 53 pos=1 pos=2 pos=3 pos=4

I am a student index=3 index=2 index=5 index=8

𝐸3 𝐸 𝐸 𝐸 No information of word order

2 5 8 I am a student

All words or word embeddings are inputted simultaneously to the Transformer

[Source: Dive into DL] 186

# Transformer's Attention Mechanism

Queries Il FFN Target tokens “from Target

gatto FFN the point of view” of Sequence Dot

FFN product the source salta

sequence …

∙ ∙ ∙

Norm &Softmax Keys & Values

∙ The FFN Source cat FFN Sequence

jumps FFN the FFN …

187

## Transformer's Attention Mechanism

From a different perspective

Query Il “gatto” token, built by Target

gatto “N5”

aggregating value vectors Sequence

salta in the source dictionary

Soft-matching …

Lookup Table Key Value The “A4” Source cat Weighted

“N9” Sequence average

jumps “O7” the “A4” wall

“N2”

188

## Vision Transformers (ViT)

# Transformers in Computer Vision

Can we use the self-attention mechanism in images?

# Transformers in Computer Vision

###  The transformer works with a set of tokens

###  What are tokens/visual words in images?

224px

4

### 2 224

### 224

### 224px pixels calculations

### Impossible!

# Transformers in Computer Vision

### • Tokens as the features from an object detector

ROI Pooling

ROI Pooling Tokens!

ROI Pooling

# Vision Transformers (ViTs)

“An image is worth 16x16 words”

14 segments 16px 224px

16px

Image to Flatten Patches + Patches Linear Projection

Tokens! 224px

14 segments

“An image is worth 16x16 words” | Dosovitskiy et al., 2020

Transformer Encoder

# Vision Transformers (ViTs)

MLP CLASS

Head

\*

class token Transformer Encoder

\*

1 2 3 4 5 6 7 8 9 1 2 3

Linear Projection of Flattened Patches 4 5 6

7 8 9

# Vision Transformers (ViTs)

# What happens during training?

Learned Knowledge

Not Improving

Anymore Still Improving

Convolutional Neural

Vision Transformer Network

# Why are they different?

Locality Need very large Sensitive dataset for training

Convolutional

Neural Network Translation

Learns Invariant

inductive biases

Vision Transformers

Lack of global Able to find understanding long-term dependencies

# A different point of view

ViTs are both local and global!

Heads focus on farther patches

0.0 1

0.7

The ViT learns only global information

with low amount of data

# A different point of view

ViTs are both local and global!

Higher layers heads still focus on farther

patches

0.0 1

0.7

Lower layers heads focus on both farther

and closer patches

0.4

0.0 0.3

6 0.01 0.6

The ViT learns also local information

with more data

## Model Fine-Tuning

# ViT: Model Fine-Tuning

MLP Pretrained ViT/16 models trained on CLASS

Head ImageNet 21K and JFT-300M datasets

\*

class token Transformer Encoder

\*

1 2 3 4 5 6 7 8 9 1 2 3

Linear Projection of Flattened Patches 4 5 6

7 8 9

# ViT: Model Fine-Tuning

New dataset

CLASS New MLP Head

Fine-tuning

New class token Transformer Encoder

1 2 3 4 5 6 7 8 9 1 2 3

Linear Projection of Flattened Patches 4 5 6

7 8 9

## ViT: Model Fine-Tuning with Additional Components

New dataset

Principle: insert to a pre-trained ViT some additional components that favour the computations of ViT and fine-tune them.

- …

LoRA

Adapter

LoRA class token

LoRA

learnable … \*

prompts 1 2 3 1 2 3 4 5 6 7 8 9

4 5 6 Linear Projection of Flattened Patches

7 8 9

## ViT: Model Fine-Tuning with Prompts

New dataset

- …

Fine-tuning

class token

learnable … \*

prompts 1 2 3 1 2 3 4 5 6 7 8 9

4 5 6 Linear Projection of Flattened Patches

Fine-tuning 7 8 9

## ViT: Model Fine-Tuning with Adapters

New dataset Fine-tuning

\* Fine-tuning

𝒇 𝑿 = 𝒈 𝑿 + 𝝈 𝑿𝑾𝒅𝒐𝒘𝒏 𝑾𝒖𝒑

ℎ 𝑋 = 𝜎 𝑋𝑊𝑑𝑜𝑤𝑛 𝑊𝑢𝑝 𝑔 𝑋 = 𝜎 𝑋𝑊1 𝑊2

𝑊 𝑊𝑢𝑝 class token 2 Fixed

Learnable Point-wise 𝑊 Adapter 1 FFN 𝑊𝑑𝑜𝑤𝑛

𝑿

\* 1 2 3 1 2 3 4 5 6 7 8 9

4 5 6 Linear Projection of Flattened Patches

7 8 9

## ViT: Model Fine-Tuning with Additional LoRA

Fine-tuning New dataset

\* Fine-tuning

𝑟 ≪ 𝑚, 𝑛 [𝑚, 𝑛] [𝑚, 𝑟] [𝑟, 𝑛] 𝑊𝑄 + 𝑩𝑸×𝑨𝑸

𝑊𝐾 + 𝑩𝑲×𝑨𝒌 class token

𝑊𝑉 + 𝑩𝑽×𝑨𝑽

\* 1 2 3 1 2 3 4 5 6 7 8 9

4 5 6 Linear Projection of Flattened Patches

7 8 9

## Generative Adversarial Networs

209

# General formulation for a Deep Generative

# Models (DGM)

𝐷 = 𝒙1, 𝒙2, … , 𝒙𝑁

### Given a training set 𝐷 =

### 𝒙1, 𝒙2, … , 𝒙𝑁 where each 𝒙𝑖~𝑝𝑑 𝒙 .

o 𝑝𝑑 𝒙 exists but unknown.

### Learn a generator 𝑮 mapping from

### the ‘noise’/latent space 𝒵 to data

### 𝐺(𝒛) Deep NNs

### space

o 𝒛~𝑝(𝒛) → 𝒙෥ = 𝐺(𝒛)~𝑝 𝒙

𝑑

### o 𝒙෥ = 𝐺(𝒛) looks ‘similar’ to those in 𝐷.

### Use 𝑮 to generate novel and new 𝒵

𝒛~𝑝(𝒛)

### data samples.

210

# Current DGM methods

Image credit: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

# Generative Adversarial Networks

NeurIPS 2014

Yann LeCun

212

# Generative Adversarial Networks

𝑓𝑎𝑘𝑒 𝑟𝑒𝑎𝑙

### A game of two players [Goodfellows et al., 2014]

o Generator 𝑮: generate fake samples indistinguishable from real

### 𝐷

samples o Discriminator 𝑫: discriminate real and fake samples

𝒙𝑓𝑎𝑘𝑒 𝒙

𝑟𝑒𝑎𝑙

### 𝐺

### Min-max problem

min max 𝐽 𝐺, 𝐷 = 𝔼𝒙 ~ 𝑝 𝒙 log 𝐷 𝒙 + 𝔼𝒛 ~ 𝑝 𝒛 log(1 − 𝐷(𝐺 𝒛 )

𝑑 𝒛 ~ 𝑝 𝑑𝑎𝑡𝑎 𝐺 D 𝒛

### Issue & challenge

o Mode collapse, min-max problem

213

# GAN: analogy to minimax game

Termination at Nash equilibrium point

Task: distinguish between real and counterfeit money No player can improve any more

Goal: try to sabotage the counterfeit maker

Discriminator

Task: produce counterfeit money

Goal: try to fool/confuse the police officer

Final output Generator Target/data distribution from GANs

### Adversarial or Cooperative

214

# Generative Adversarial Networks (GAN)

How to formulate this?

Introduce a noise variable 𝒛 ~ 𝑝𝒛 o This is the prior distribution; think of it as the

resource used to make money. A generator 𝐺 that maps 𝒛 to 𝒙෥ in data space:

𝒙෥ = 𝐺 𝒛 o Parameterize 𝐺 by a deep neural network with

parameter 𝜽𝑔.

A discriminator 𝐷 𝒙 maps 𝒙 to a probability in [0, 1] representing the probability that 𝒙 comes from the real data rather than 𝐺. o Parameterize 𝐷(𝒙) by another deep neural

network 𝜽𝑑 . o 𝐷(𝑥) close to 1 (real) o 𝐷(𝑥) close to 0 (fake) min max 𝐽 𝐺, 𝐷 = 𝔼𝒙 ~ 𝑝 𝒙 log 𝐷 𝒙 + 𝔼𝒛 ~ 𝑝 𝒛 log(1 − 𝐷(𝐺 𝒛 )

𝑑 𝐺 D

(Figure credit: from Goodfellow, NIPS workshop’16)

215

# Training GAN

- Gradient ascent to train discriminator 𝐷𝜃: 𝑑 max 𝔼 log 𝐷𝜃 𝒙 + 𝔼 log(1 − 𝐷𝜃 (𝐺 𝒛 ) 𝑑 𝑑 𝜃𝑑 𝒙~𝑝𝑑𝑎𝑡𝑎(𝒙) 𝒛~𝑝𝒛

(𝑖) (𝑖) Ascent update 𝐷: 𝒛 ~𝑝𝑧, 𝒙 ~𝑝𝑑𝑎𝑡𝑎 Training 𝐷 is to maximize the

1 𝑀

𝑖 𝑖 𝛻 σ log 𝐷 𝒙 + log 1 − 𝐷 𝐺 𝒛 𝜽d 𝑖=1 probability of detecting correct labels M

### min max 𝐽 𝐺, 𝐷 = 𝔼 log 𝐷 𝒙 + 𝔼 log(1 − 𝐷(𝐺 𝒛 )

𝐺 D 𝒙~𝑝𝑑𝑎𝑡𝑎(𝒙) 𝒛~𝑝𝒛

216

# Training GAN

### min max 𝐽 𝐺, 𝐷 = 𝔼 log 𝐷 𝒙 + 𝔼 log(1 − 𝐷(𝐺 𝒛 )

𝐺 D 𝒙~𝑝𝑑𝑎𝑡𝑎(𝒙) 𝒛~𝑝𝒛

Training 𝐺 is to minimize or fool the • Gradient descent to train the generator 𝐺𝜃

𝑔 discriminator 𝐷, i.e., minimize (1 − min 𝔼 log(1 − 𝐷𝜃 (𝐺 𝒛 )

𝑑 𝜃𝑔 𝒛~𝑝𝒛 𝐷 𝐺 𝒛 ) to 0

(𝑖) Descent update 𝐺: 𝒛 ~𝑝𝑧

𝑀 1

𝑖 − 𝛻𝜽 ෍ log 1 − 𝐷 𝐺 𝒛

g M

𝑖=1

217

# Training GAN

(𝑖) (𝑖) Ascent update 𝐷: 𝒛 ~𝑝𝑧, 𝒙 ~𝑝𝑑𝑎𝑡𝑎 Training 𝐷 is to maximize the

1 𝑀

𝑖 𝑖 𝛻 σ log 𝐷 𝒙 + log 1 − 𝐷 𝐺 𝒛 𝜽d 𝑖=1 probability of detecting correct labels M

### min max 𝐽 𝐺, 𝐷 = 𝔼 log 𝐷 𝒙 + 𝔼 log(1 − 𝐷(𝐺 𝒛 )

𝐺 D 𝒙~𝑝𝑑𝑎𝑡𝑎(𝒙) 𝒛~𝑝𝒛

(𝑖) Descent update 𝐺: 𝒛 ~𝑝𝑧 Training 𝐺 is to minimize or fool the 𝑀

1

𝑖 discriminator 𝐷, i.e., minimize 1 − − 𝛻𝜽 ෍ log 1 − 𝐷 𝐺 𝒛

g M

𝑖=1 𝐷 𝐺 𝒛 to 0

218

# Basic theory of GAN

𝑝 (𝒙) y=1 𝑑 𝑝 (𝒙)

𝑑

Gap between two distributions

𝒙𝑓𝑎𝑘𝑒 = 𝐺𝜃 𝒛 ~𝑝𝑔 𝒙

y=-1 𝑝𝑔 𝒙

❖ D(x) = P(y=1|x): probability x is true data. ❖ 1-D(x) = P(y=-1|x): probability x is fake data. 𝐺𝜃(. )

The log likelihood function need to be maximized

𝐽 𝐷, 𝐺 = 𝔼𝒙~𝑃 log 𝐷(𝒙) + 𝔼𝒙~𝑃 log 1 − 𝐷(𝒙)

𝑑 𝑔 = 𝔼𝒙~𝑃 log 𝐷(𝒙) + 𝔼𝒛~𝑃 log 1 − 𝐷(𝐺(𝒛)) 𝑑 𝑑 𝑧 𝒛~𝑝 𝒛 = 𝑁 0, 𝐼 𝑜𝑟 𝑈𝑛𝑖( 0,1 )

Source of randomness

We want to train NN-based function 𝐺𝜃(. ) such that 𝑝𝑔 𝒙 = 𝑝𝑑 𝒙 , but we do not have 𝑝𝑑 𝒙 at hand

### o Introduce a strong discriminator 𝐷 𝒙 to implicitly justify how far 𝑝𝑑 𝒙 and 𝑝𝑔 𝒙 are.

219

# Basic theory of GAN

y=1

𝑝𝑑(𝒙)

∗ ∗ ∗ 𝐽 𝐷, 𝐺 = 𝔼𝒙~𝑃 log 𝐷 (𝒙) + 𝔼𝑥~𝑃 log 1 − 𝐷 (𝒙) is large

𝑑 𝑔

y=-1 𝑝𝑔 𝒙

y=1

∗ ∗ ∗ 𝐽 𝐷, 𝐺 = 𝔼𝒙~𝑃 log 𝐷 (𝒙) + 𝔼𝑥~𝑃 log 1 − 𝐷 (𝒙) decreases

𝑑 𝑔 𝑝𝑑(𝒙)

y=-1

𝑝𝑔 𝒙

y=1

𝑝 (𝒙) ∗ ∗ ∗ 𝑑 𝐽 𝐷, 𝐺 = 𝔼𝒙~𝑃 log 𝐷 (𝒙) + 𝔼𝒙~𝑃 log 1 − 𝐷 (𝒙) is smallest

𝑑 𝑔

y=-1 𝑝𝑔 𝒙

It comes with the minimax problem

𝒎𝒊𝒏𝑮𝒎𝒂𝒙𝑫𝑱 𝑫, 𝑮

220

# Optimal discriminator solution

When training is perfect, discriminator D Recall the GAN objective:

can NO longer tell the fake (money) from the real (money), i.e., 0.5 probability 𝐽 𝐺, 𝐷 = 𝔼 log 𝐷 𝒙 + 𝔼 log(1 − 𝐷(𝐺 𝒛 )

𝑓𝑎𝑘𝑒 𝑟𝑒𝑎𝑙 𝒙~𝑝𝑑𝑎𝑡𝑎(𝒙) 𝒛~𝑝𝒛 Let 𝑝𝑔 be the distribution over 𝐺(𝑧) when 𝑧~𝑝𝑧 and let 𝑝𝑑 denotes 𝑝𝑑𝑎𝑡𝑎 :

### 𝐷

𝐽 𝐺, 𝐷 = 𝔼 log 𝐷 𝒙 + 𝔼 log(1 − 𝐷(𝑥)

𝒙~𝑝𝑑 𝒙~𝒑𝒈 = න𝑝𝑑 𝑥 log 𝐷 𝑥 𝑑𝑥 + න𝑝𝑔 𝑥 log 1 − 𝐷 𝑥 𝑑𝑥 𝒙

𝑓𝑎𝑘𝑒 𝒙𝑟𝑒𝑎𝑙 𝑥 𝑥

### We need to maximize D, let’s do this pointwise

### 𝐺

### at each position of 𝑥:

max{ 𝑝 𝑥 log 𝐷 𝑥 + 𝑝 𝑥 log 1 − 𝐷 𝑥 } 𝒛 ~ 𝑝 𝑑𝑎𝑡𝑎

𝑑 𝑔 𝒛 𝐷 𝑝 𝒙 𝑝𝑔 𝒙 𝑑 o 𝛻𝐷 = − = 0

𝐷 1−𝐷

𝑝𝑑(𝒙) ∗ o Optimal discriminator: 𝐷 (𝒙) =

𝑝𝑑 𝒙 +𝑝𝑔(𝒙) o When G generates perfect samples, 𝑝𝑔 = 𝑝𝑑 at which 𝐷 𝑥 =

0.5 ∀𝑥

221

# Optimal generator solution

When training is perfect, discriminator D can NO longer tell the fake (money) from ∗ ∗ Fix 𝐷 in: 𝒎𝒊𝒏𝑮𝒎𝒂𝒙𝑫𝑱 𝑫, 𝑮 = 𝒎𝒊𝒏𝑮𝑱(𝑫, 𝑮) the real (money), i.e., 0.5 probability Recall Jensen-Shannon divergence:

1 1 1 𝑓𝑎𝑘𝑒 𝑟𝑒𝑎𝑙 𝐷𝐽𝑆 𝑑, 𝑔 = 𝐾𝐿 𝑑 𝑚 + 𝐾𝐿 𝑔 𝑚 , 𝑚 = (𝑑 + 𝑔)

2 2 2

### 𝐷

∗ ∗ ∗ 𝐽(𝐷, 𝐺)=𝔼𝒙~𝑝 log 𝐷 (𝒙) + 𝔼𝒙~𝑝 log 1 − 𝐷 (𝒙)

𝑑 𝑔

𝑝𝑑(𝒙) 𝑝𝑑(𝒙) = 𝔼 log + 𝔼 log 1 − 𝒙𝑓𝑎𝑘𝑒 𝒙

𝒙~𝑝 𝒙~𝑝 𝑟𝑒𝑎𝑙

𝑑 𝑝 𝒙 +𝑝 (𝒙) 𝑔 𝑝 𝒙 +𝑝 (𝒙)

### 𝑑 𝑔 𝑑 𝑔 𝑝 ∗ 𝑝

𝑔 = 𝑑 𝑝 (𝒙) 𝑝𝑔(𝒙) 𝑑

### = 𝔼𝒙~𝑝 log + 𝔼𝒙~𝑝 log 𝐺

𝑑 𝑝 𝒙 +𝑝 (𝒙) 𝑔 𝑝 𝒙 +𝑝 (𝒙)

𝑑 𝑔 𝑑 𝑔 = 𝐷 𝑝𝑑 𝒙 ||𝑝𝑔(𝒙) − 2 log 2

𝐽𝑆

𝒛 ~ 𝑝 𝑑𝑎𝑡𝑎

𝒛

### Therefore,

o 𝑚𝑖𝑛 𝐷 𝑝 𝒙 ||𝑝 (𝒙) → 𝑝 ∗ = 𝑝

𝐺 𝐽𝑆 𝑑 𝑔 𝑔 𝑑

Optimal G

222

# The Nash equilibrium point

## The solution of the minimax problem 𝒎𝒊𝒏𝑮𝒎𝒂𝒙𝑫𝑱 𝑫, 𝑮

∗ ∗ o Nash equilibrium point 𝐷, 𝐺 which satisfies

▪ 𝑝 ∗ = 𝑝

𝑔 𝑑

𝑝𝑑(𝒙) ∗ ▪ 𝐷 (𝒙) = = 0.5, 𝑓𝑜𝑟 𝑎𝑙𝑙 𝑥

𝑝𝑑 𝒙 +𝑝 ∗(𝒙)

𝑔

𝑝 𝑝 ∗ = 𝑝 𝑑 𝑔 𝑑

∗ 𝐷 𝐷 𝑥

𝑝𝑔

= 0.5, ∀𝑥

Our own simulation [Goodfellow et al, NIPS 14]

223

# GAN: example results

nearest neighbour MNIST

samples from training CIFAR-10

(from Goodfellow, et al, NIPS14’s paper)

224

# GAN: example results

### For a complex dataset such as ImageNet

o GANs can generate sharp images, but look unrealistic o Hard to know what objects they are

### Conditional GANs with labels work much better in practice!

225

# Issues with GAN

### Two serious technical problems with GAN

### \#1: mode collapsing problem

o Impeach the ability to generate diverse, realistic data/images

### Generated (fake) examples 𝐺(𝒛) where 𝒛~𝑝(𝒛)

o Can only cover a few modes in the data distribution

𝑝𝑑(𝒙) o Miss many other modes in the data distribution.

226

(Image credit: deeplearning.ai)

227

# Issues with GAN

### Two serious technical problems with GAN

### \#2: Convergence is hard due to minimax formulation

o Training GAN is very challenging!

Generator loss Discriminator loss

##  Mini-max problem

- min max 𝐽 𝐺, 𝐷 = 𝔼 log 𝐷 𝒙 + 𝔼 log(1 − 𝐷(𝐺 𝒛 ) 𝐺 D 𝒙~𝑝𝑑(𝒙) 𝒛~𝑝𝒛

##  No unique loss function

228

(image credit: deeplearning.ai)

229

# How to evaluate GAN: Inception score

Except for trivial cases, we don’t know the true data distribution 𝑝𝑑 so how can we say how good the generated distribution 𝑝𝑔 is? How to know that a generative model can generate diverge examples that cover many modes in data? How to compare two generative models trained on the same training set?

y=1 y=2 y=1 y=2

y=3 y=4 y=3 y=4

𝐺2(𝒛) 𝐺1(𝒛)

𝒵 𝒵 𝒛~𝑝(𝒛)

𝒛~𝑝(𝒛)

Two generators 𝑮𝟏 and 𝑮𝟐, which one is better?

230

# How to evaluate GAN: Inception score

Except for trivial cases, we don’t know the true data distribution 𝑝𝑑𝑎𝑡𝑎 so how can we say how good the generated distribution 𝑝𝑔 is? How to know that a generative model can generate diverge examples that cover many modes in data? How to compare two generative models trained on the same training set?

y=1 y=2 y=1 y=2

y=3 y=4 y=3 y=4

𝐺2(𝒛) 𝐺1(𝒛)

𝒵 𝒵 𝒛~𝑝(𝒛)

𝒛~𝑝(𝒛)

𝑮𝟐 is better and generate more diverge examples, while 𝑮𝟏 misses the data mode 𝑦 = 4.

231

# How to evaluate GAN: Inception score

### Assume that training examples have labels

o y=1,2,3, and 4.

### We train a good classifier 𝐶 on the training set with labels

4 o 𝐶 𝒙 = 𝑝 𝑦 𝒙 = 𝑝 𝑦 = 𝑘 𝒙 is the prediction probabilities where

𝑘=1 𝑝 𝑦 = 𝑘 𝒙 is the probability to classify 𝑥 to the class 𝑘. y=1 y=2

### Given 𝒙෥ = 𝐺 𝒛, 𝑖 = 1, … , 𝑇, we wish

𝑖 𝒊 o 𝐶 𝒙෥ = 𝑝(𝑦 ∣ 𝒙෥ = 𝐺 𝒛) is close to an one-hot vector

𝑖 𝑖 𝒊

y=3 y=4 ▪ 𝐶 𝒙෥ is confident and has a small entropy.

𝑖

𝐺(𝒛) 1 𝑇 1 𝑇 o σ 𝐶 𝒙෥ = σ 𝑝(𝑦 ∣ 𝒙෥ = 𝐺 𝒛) is close to the uniform distribution

𝑖=1 𝑖 𝑖=1 𝑖 𝒊 𝑇 𝑇 [0.25, 0.25,0.25,0.25] (largest entropy).

𝒵

𝒛~𝑝(𝒛)

## Inception score

1 𝑇 1 𝑇 o σ 𝐾𝐿 𝐶 𝒙෥, σ 𝐶 𝒙෥ ≈ 𝔼 [𝐾𝐿 𝑝 𝑦 𝒙෥ = 𝐺 𝒛 , 𝑝(𝑦) ]

𝑖=1 𝑖 𝑗=1 𝑗 𝑧 𝑇 𝑇 o Higher is better → more diversity of generated images

Salimans et al, Improved Techniques for Training GANs, 2016

232

# GAN: Summary

### Adversarial training: a game of two players [Goodfellows et al.,

### 2014]

o Generator 𝐺: generate fake samples indistinguishable from real samples o Discriminator 𝐷: discriminate real and fake samples

### Min-max problem

min max 𝐽 𝐺, 𝐷 = 𝔼𝒙 ~ 𝑝 𝒙 log 𝐷 𝒙 + 𝔼𝒛 ~ 𝑝 𝒛 log(1 − 𝐷(𝐺 𝒛 )

𝑑 𝐺 D

### Issues and challenges

o Mode collapse, min-max problem

Yann Lecun

# Thanks for your attention!

### Question time