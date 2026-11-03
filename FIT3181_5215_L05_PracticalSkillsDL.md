COPYRIGHT WARNING: Copyright in these original lectures is owned by Monash University. You may transcribe, take notes, download or stream lectures for the purpose of your research and study only. If used for any other purpose, (excluding exceptions in the Copyright Act 1969 (Cth)) the University may take legal action for infringement of copyright.

Do not share, redistribute, or upload the lecture to a third party without a written permission!

### FIT3181/5215 Deep Learning

Week 05: Practical skills in deep learning

CE/Lecturer: Dr Trung Le (Clayton)

Lecturer: Dr Binh Nguyen (Clayton) Lecturer: Dr Ting Fung Fung (Malaysia) Lecturer: Dr Arghya Pal (Malaysia) Lecturer: Dr Sailaja Rajanala (Malaysia)

Email: trunglm@monash.edu

Department of Data Science and AI Faculty of Information Technology, Monash University, Australia

2

## Outline

### Setting of a machine learning problem

o General loss versus empirical loss

### Gradient vanishing/exploding and network initialization.

### Overfitting and underfitting

### Recipe for overfitting

o Use regularization term, dropout, batch norm, data augmentation, transfer learning o Label smoothing, data mix-up, cut-mix

###  Further reading recommendation

o [Deep Learning, Sections 4.1-4.3, 8.1 -8.5, 11.3, 11.4]. o [Dive into Deep Learning, Chapters 5 and 11].

## Machine learning setting

Data-label generative process Dataset collecting process 𝑥′, 𝑦′ ~𝑝 𝑥, 𝑦 = 𝑝 𝑥 𝑝 (𝑦 ∣ 𝑥) 𝐷 = { 𝑥1, 𝑦1 , … , (𝑥𝑁, 𝑦𝑁)}

𝑑𝑎𝑡𝑎 𝑑𝑎𝑡𝑎 𝑑𝑎𝑡𝑎

𝑁 𝑦′ = 𝟏 𝒐𝒓 𝒄𝒂𝒕

𝑁→∞ 1 𝑦′~𝑝 (𝑦 ∣ 𝑥′) 𝑥𝑖, 𝑦𝑖 ~𝑝𝑑𝑎𝑡𝑎(𝑥, 𝑦) 𝑝෤𝑑𝑎𝑡𝑎 𝑥, 𝑦 = ෍ 𝛿(𝑥) 𝑥, 𝑦

𝑑𝑎𝑡𝑎 𝑖,𝑦

N 𝑖

𝑖=1 Cat Dog Lion Panda

𝑦 ~𝑝 (𝑦 ∣ 𝑥) 𝑦𝑁~𝑝𝑑𝑎𝑡𝑎(𝑦 ∣ 𝑥𝑁) (1) (2) (3) (4) 1 𝑑𝑎𝑡𝑎 1 0.7 0.2 0.05 0.05 𝑦2~𝑝𝑑𝑎𝑡𝑎(𝑦 ∣ 𝑥2) 𝑝𝑑𝑎𝑡𝑎(𝑦 ∣ 𝑥′) Ground-truth labelling mechanism

𝒙 𝒙𝑵 𝒙𝟏 𝟐 𝒙′

Data 𝒙1, … , 𝑥𝑁~𝑝𝑑𝑎𝑡𝑎(𝑥) Data 𝒙′~𝑝𝑑𝑎𝑡𝑎(𝑥)

space 𝓧 Data distribution space 𝓧

Data distribution

3

Empirical data/label distribution 𝒑෥

𝒅𝒂𝒕𝒂

## Machine learning setting

(𝒙, 𝒚) (𝒙𝟏, 𝒚𝟏) (𝒙𝟐, 𝒚𝟐) … (𝒙𝑵, 𝒚𝑵) 𝑝 1/N 1/N … 1/N

Data-label generative process Dataset collecting process 𝑥′, 𝑦′ ~𝑝𝑑𝑎𝑡𝑎 𝑥, 𝑦 = 𝑝𝑑𝑎𝑡𝑎 𝑥 𝑝𝑑𝑎𝑡𝑎 (𝑦 ∣ 𝑥)

𝐷 = { 𝑥1, 𝑦1 , … , (𝑥𝑁, 𝑦𝑁)} 𝑦′ = 𝟏 𝒐𝒓 𝒄𝒂𝒕 𝑁

𝑁→∞ 1 𝑦′~𝑝𝑑𝑎𝑡𝑎(𝑦 ∣ 𝑥′) 𝑥, 𝑦 ~𝑝 (𝑥, 𝑦) 𝑝෤ 𝑥, 𝑦 = ෍ 𝛿 𝑥, 𝑦

𝑖 𝑖 𝑑𝑎𝑡𝑎 𝑑𝑎𝑡𝑎 (𝑥𝑖,𝑦)

N 𝑖

𝑖=1 Cat Dog Lion Panda

𝑦 ~𝑝 (𝑦 ∣ 𝑥) 𝑦𝑁~𝑝𝑑𝑎𝑡𝑎(𝑦 ∣ 𝑥𝑁) (1) (2) (3) (4) 1 𝑑𝑎𝑡𝑎 1 0.7 0.2 0.05 0.05 𝑦2~𝑝𝑑𝑎𝑡𝑎(𝑦 ∣ 𝑥2) 𝑝𝑑𝑎𝑡𝑎(𝑦 ∣ 𝑥′) Ground-truth labelling mechanism

𝒙 𝒙𝑵 𝒙𝟏 𝟐 𝒙′

Data 𝒙~𝑝𝑑𝑎𝑡𝑎(𝑥) Data 𝒙′~𝑝𝑑𝑎𝑡𝑎(𝑥)

space 𝓧 Data distribution space 𝓧

Data distribution

4

## Machine learning setting

Empirical loss Generalization (general) loss

(loss on a collected training set D) (loss on data/label distribution) 𝑁

𝑵 → ∞ 1

ℒ𝑒𝑚𝑝 𝜃 = 𝔼 𝑙 𝑓 𝑥; 𝜃 , 𝑦 = ෍ 𝑙 𝑓 𝑥𝑖; 𝜃 , 𝑦𝑖 ℒ𝑔𝑒𝑛 𝜃 = 𝔼 𝑙 𝑓 𝑥; 𝜃 , 𝑦

𝑝෤𝑑𝑎𝑡𝑎 𝑁 𝑝𝑑𝑎𝑡𝑎 Law of large numbers

𝑖=1 ∗ ∗ Reality: 𝜃 = argmin ℒ𝑒𝑚𝑝 𝜃 Ideal: 𝜃 = argmin ℒ𝑔𝑒𝑛 𝜃

𝜃 𝜃

𝐷 = { 𝑥1, 𝑦1 , … , (𝑥𝑁, 𝑦𝑁)}

𝑁 𝑁→∞ 1 𝑥′, 𝑦′ ~𝑝𝑑𝑎𝑡𝑎 𝑥, 𝑦 = 𝑝𝑑𝑎𝑡𝑎 𝑥 𝑝𝑑𝑎𝑡𝑎(𝑦 ∣ 𝑥)

𝑥𝑖, 𝑦𝑖 ~𝑝𝑑𝑎𝑡𝑎(𝑥, 𝑦) 𝑝෤𝑑𝑎𝑡𝑎 𝑥, 𝑦 = ෍ 𝛿(𝑥) 𝑥, 𝑦

𝑖,𝑦 N 𝑖 𝑦′~𝑝𝑑𝑎𝑡𝑎(𝑦 ∣ 𝑥′) 𝑖=1

𝑦1~𝑝𝑑𝑎𝑡𝑎(𝑦 ∣ 𝑥1)

𝑦𝑁~𝑝𝑑𝑎𝑡𝑎(𝑦 ∣ 𝑥𝑁) 𝑝𝑑𝑎𝑡𝑎(𝑦 ∣ 𝑥′)

𝑦2~𝑝𝑑𝑎𝑡𝑎(𝑦 ∣ 𝑥2) Labelling mechanism

𝒙 𝒙𝑵 𝒙𝟏 𝟐 𝒙′

Data 𝒙~𝑝𝑑𝑎𝑡𝑎(𝑥) Data 𝒙′~𝑝𝑑𝑎𝑡𝑎(𝑥)

space 𝓧 Data distribution space 𝓧

Data distribution

5

6

## How Learning Differs from Pure Optimization?

###  Some important notations:

o 𝑝𝑑𝑎𝑡𝑎(𝑥, 𝑦): existed, but unknown, distribution of data and label

1 𝑁 o 𝑝෤ 𝑥, 𝑦 = σ 𝛿 (𝑥, 𝑦): empirical data/label distribution - this is what we

𝑑𝑎𝑡𝑎 𝑖=1 (𝑥𝑖,𝑦)

N 𝑖

𝑵 iid observed from training data 𝑫 = (𝒙𝒊, 𝒚𝒊) where 𝒙𝒊, 𝒚𝒊 ∼ 𝑝𝑑𝑎𝑡𝑎(𝑥, 𝑦).

𝒊=𝟏 o Per-sample loss: 𝑙(𝑓 𝑥; 𝜃 , 𝑦)

###  Empirical loss minimisation (pure

### optimisation):

𝑁 1 ℒ𝑒𝑚𝑝 𝜃 = 𝔼 𝑙 𝑓 𝑥; 𝜃 , 𝑦 = ෍ 𝑙 𝑓 𝑥𝑖; 𝜃 , 𝑦𝑖

### 𝑝෤ 𝑁 Empirical loss (maths)

𝑑𝑎𝑡𝑎

𝑖=1

### vs.

###  But what ML wants is true generalisation

### Generalisation loss (ML)

### loss:

ℒ𝑔𝑒𝑛 𝜃 = 𝔼 𝑙 𝑓 𝑥; 𝜃 , 𝑦 How to achieve this when we only

𝑝𝑑𝑎𝑡𝑎

have access to empirical data?

7

## Optimization Problem in ML and DL

###  Most of optimization problems in machine learning (deep learning) has the

### following form:

𝑵 Guiding principles: 1 1. Occam Razer: prefer the min𝜃 𝐽 𝜃 = 𝛺 𝜃 + ෍ 𝑙 𝑦, 𝑓 𝑥; 𝜃

𝑖 𝑖 simplest model that can do 𝑁

𝒊=𝟏 well.

Regularization term Empirical loss

2 2 𝐤 𝐤 - Work well on training set

- Ω θ = 𝜆 σ σ 𝑊 = 𝜆 σ 𝑾 𝑘 𝑖,𝑗 𝑖,𝑗 𝑘 𝐹
- Encourage simple models
- Avoid overfitting

How to efficiently solve this optimization problem?

6 N is the training size and might be very big (e.g., 𝑁 ≈ 10)

Works well for Deep Learning (non-convex)

Works well for convex problems, but not DL First-order iterative methods Second-order iterative methods

gradient descent, steepest descent Newton and quasi Newton methods Use the gradient (first derivative) g = 𝛻𝛉𝐽(𝛉) to Use the Hessian matrix (second derivative) H =

2 update parameters: 𝛻 𝐽(𝛉) to update parameters

𝛉 𝑤 = 𝑤 − learning rate × gradient

8

## Optimization Problem in ML and DL

###  Most of optimization problems in machine learning (deep learning) has the

### following form:

𝑵 1 min𝜃 𝐽 𝜃 = 𝛺 𝜃 + ෍ 𝑙 𝑦, 𝑓 𝑥; 𝜃

𝑖 𝑖 𝑁

𝒊=𝟏

What loss function should one use?

### Some typical loss functions used in the classification and

### regression problems

- 0-1 loss, Hinge loss, Logistic loss (binary classification)
- L1 loss, L2 loss, ε −insensitive loss (regression)
- Popular surrogate loss: cross-entropy loss (multi-class classification, deep learning)

9

## How Learning Differs from Pure Optimization?

###  Achieving generalisation capacity is the holy-grail of machine learning.

o Empirical risk is prone to over-fitting o Some times, it is not really feasible if the loss function does not have useful derivatives (e.g., 0-1 loss),

hence we usually resort to surrogate loss function, e.g., cross-entropy, Hinge loss, etc.

###  In most cases, DL algorithm doesn’t halt at local minimum

o It halts when certain convergence criterion is met (e.g., based on early stopping when overfitting start

to occur, reach certain budgets, number of epochs, etc). o For training DL models, it might stop when the loss function still has large derivates, which is different

from a pure optimisation setting.

Reality Ideality

Train loss Train loss

Grad Norm Grad Norm

Epochs Epochs

### Challenges in deep learning optimization: gradient

### vanishing, gradient exploding

12

## Gradient vanishing

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

13

## Gradient vanishing

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

14

## Gradient exploding

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

= × × ⋯ × Tip: 𝜕ℎ 𝜕ℎ 𝜕ℎ 𝜕ℎ

0 𝑇 𝑇−1 0

- Let simplify 𝑊 as a scalar in the real-valued set 𝑚 𝑊
- 𝑊 → 0 if 𝑊 < 1. 𝑊 𝑚
- 𝑊 → ∞ if 𝑊 > 1. Multiplication of many matrices 𝑊

15

## Gradient Clipping

###  One way to lessen the exploding

### gradients problem is to simply clip

### the gradients during backpropagation

### so that they never exceed some

### threshold

o Either clip by values (direction might change) o or clip by norms (keep direction but rescale the

magnitude)

###  Widely applied to Recurrent Neural

### Networks

###  Widely used for NLP tasks, not so

### much used for CNNs.

16

## Observing Gradient Vanishing and Exploding

Log the histogram of gradients to TensorBoard Using TensorBoard

### Weight initialization

18

## He and Xavier weight initialization

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

19

## He and Xavier weight initialization

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

20

## He and Xavier weight initialization

2  Xavier initialization

𝑤𝑋𝑎~𝑁 0,

𝑛𝑖𝑛 + 𝑛𝑜𝑢𝑡 o Ensure the variance of the outputs of each layer equal to the variance of its inputs o Also need the gradients to have equal variance before and after flowing through a

𝑛𝑜𝑢𝑡 = 2 layer in the reverse direction

𝑛𝑖𝑛 = 3 o Good for sigmoid and tanh activation functions o Not good for ReLU

1 if sigmoid 2 𝑤𝐻𝑒~𝑁 0, 𝛼 × 𝛼 = ቐ4 if tanh  He initialization 𝑛𝑖𝑛 + 𝑛𝑜𝑢𝑡

2 if ReLU o A variant of Xavier initialization where 𝛼 = 1 o Works better for ReLU.

Paper link: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf

22

## Where do we go from here?

###  Challenges in deep learning optimisation and how to address them:

o Local minima, saddle points and complex loss surfaces o Gradient vanishing and exploding o What we don’t cover in this unit (see DL section 8.2):

▪ Ill-conditioning problem ▪ Long-term dependencies ▪ Poor correspondence between local and global structures ▪ Theoretical limits of optimisation (but they usually have little use in practice of deep learning)

###  Initialization Strategies

###  Regularization in deep learning

o Parameter norm penalty: l1, l2 regularization o Early stopping o Dropout o Batch normalization

###  Choice of optimizers:

o Basic algorithms: SGD, Momentum, Nesterov Momentum o Algorithms with adaptive learning rate: AdaGrad, RMSProp, Adam

### Overfitting and Regularization in Deep Learning

24

## Deep Learning Pipeline

We want to train our DL model on a training set such that the trained model can predict well unseen data in a separate testing set.

Original dataset

Validation set

Testing set Training set (10%)

(10%) (80%)

DL model 𝜂 = 0.1 DL model 𝜂 = 0.01 DL model 𝜂 = 0.001

Model parameters: weight matrices, biases, filters which will be learnt Hyper-parameters to consider: learning rate, #layers, #neurons which need to be tuned

25

## Deep Learning Pipeline

We want to train our DL model on a training set such that the trained model can predict well unseen data in a separate testing set.

(1)

A) Extremely high B) Too high C) Too small D) Just right (2) (3) (4)

Hyper-parameters to consider: learning rate, #layers, #neurons

26

## Deep Learning Pipeline

We want to train our DL model on a training set such that the trained model can predict well unseen data in a separate testing set.

Hyper-parameters to consider: learning rate, #layers, #neurons

27

## Regularization and Overfitting

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

28

## Overfitting & Underfitting

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

## Overfitting in Deep NNs

Overfitting

Overfit: tendency of the network to “memorize” all training samples, leading to poor generalization

Testing data

Overfitting with classification [Source: https://www.ibm.com/cloud/learn/overfitting]

 Overfitting occurs when your network models the training data too well and fails to generalize

to your test (validation) data. o Performance measured by errors on “unseen” data. o Minimize error alone on training data is not enough o Causes: too many layers, too many hidden nodes, and overtrained.

29

We can also experiment with model capacity itself in parallel.

[DL, ch11]

30

31

## Recipe for Overfitting

### ❑ Early stopping

❑ Stopping the training on time before it becomes overtrained and overly complex.

### ❑ Train with more data

❑ Expanding the training set to include more data

### ❑ Data augmentation

❑ Creating many variations of clean data to make the model to generalize better to unseen

examples.

### ❑ Regularization

❑ Overfitting occurs when a model is too complex comparing with data. ❑ Using regularization terms (L1, L2), batch norm, dropout, data mix-up, label smoothing, VAT

(virtual adversarial training).

### ❑ Ensemble methods (not cover in this lecture)

❑ Ensemble learning methods are made up of a set of classifiers and their predictions are

aggregated to identify the most popular result. ❑ The most well-known ensemble methods are bagging and boosting.

32

## Early stopping

Early stopping point

Underfitting Overfitting

Generalization gap

### At first, the train and valid losses gradually drop, but the model is not good enough to

### characterise data

o Underfitting is happening

### At a certain point, the train loss still decreases, while the valid loss starts increasing

o Overfitting starts happening and we need to do early stopping at this point

## Add Regularization

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

33

34

## Dropout

Reduce Overfitting

Without dropout With dropout

(Source: chatbotslife) (Source: Analytics Vidhya)

 This is a cheap technique to reduce model capacity

o Reduce overfitting  In each iteration, at each layer, randomly choose some neurons and drop all connections from

these neurons o dropout_rate= 1 – keep_prob

## Dropout

Reduce Overfitting

###  Computationally efficient

###  Can be considered a bagged ensemble of an

𝑁

### exponential number (2) of neural networks.

𝑙+1

###  Training 𝒉 × × ×

### 𝒓 ~ Bernoulli 𝜇

### ෩𝑙 𝑙

𝑙

### 𝒉 = 𝒉 ⨀ 𝒓 𝒉 × × × ×

⊤

### 𝑙+1 𝑙 ෩𝑙 𝑙

### 𝒉 = 𝜎 𝑊 𝒉 + 𝒃

𝑙−1 𝒉

###  Testing

o No dropout (dropout_rate =0).

35

36

## Internal covariate shift problem

Input batch Representation batch

𝟏 𝟏 𝟏 𝟏 𝟏 𝟏 𝟏 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟐 𝟐 𝟐 𝟐 𝟐 𝟐 𝟐 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝟑 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑 𝟒 𝟒 𝟒 𝟒 𝟒 𝟒 𝟒 𝒙 𝒙 𝒙 𝒙 𝐳 𝐳 𝐳 𝟏 𝟐 𝟑 𝟒 𝟏 𝟐 𝟑

𝑊 𝑊

2 1 , 𝑊, 𝑏 𝑊 𝑏

2 1 3 4

, , 𝑏 𝑏 3 4

Feature extractor Classifier

Paper link: http://proceedings.mlr.press/v37/ioffe15.pdf

37

## Internal covariate shift problem

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

38

## Batch Normalization

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

39

## Batch Normalization

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

## Batch Normalization 𝑘 𝑘 𝑘

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

Standardize column-wise 40

41

## Batch Normalization

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

## Data Augmentation

### Reality

o You have a tiny dataset of 10K images, and you need

to train a good deep net.

What are the criteria of a qualified training set?

Quality? Quantity?

- More diverge data?
- Collect more and more data?

- More qualified data?

Prof Dinh Phung Create many variants

42

43

## Data Augmentation

### Use simple transformations to augment

### data examples. Models will be challenged with

diverge data examples which might be encountered in the testing set o Rotation, Width Shifting, Height Shifting, Brightness, Shear

Intensity, Zoom, Channel Shift, Horizontal Flip, Vertical Flip

### This will reduce overfitting, making this a

### regularization technique. The trick is to

### generate realistic training instances

[Source: Stanford CS 230 Deep Learning]

## Data Augmentation in PyTorch

Without augmentation

With augmentation

44

### Some additional regularization techniques

## Label smoothing

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

𝛼 o 𝑙 𝑦, 𝑦ො = 𝐶𝐸(𝟏෩, 𝒑(𝒙)) with 𝟏෩ = 1 − 𝛼 × 𝟏 + × 𝟏 where 𝟏 is a vector of all 1 and 0 < 𝛼 < 1.

𝑦 𝑦 𝑦

𝑀

47

## Data mix-up

Paper link: https://openreview.net/pdf?id=r1Ddp1-Rb

[Source: https://medium.com/] (𝑥1, 𝟏𝑦)

1

𝜆~Beta(𝛼, 𝛼) Blended image 𝑥෤ = 𝜆 × 𝑥1 + 1 − 𝜆 × 𝑥2 Blended label 𝑦෤ = 𝜆 × 𝟏 + 1 − 𝜆 × 𝟏 (𝑥2, 𝟏𝑦)

𝑦1 𝑦2 2 𝐦𝐢𝐧 𝑪𝑬(𝒚෥, 𝒑(𝒙෥))

### ❑ for (𝑥1, 𝑦1), (𝑥2, 𝑦2) in zip(batch1, batch 2)

1. 𝜆~Beta 𝛼, 𝛼
2. 𝑥෤ = 𝜆 × 𝑥1 + 1 − 𝜆 × 𝑥2
3. 𝑦෤ = 𝜆 × 𝟏𝑦 + 1 − 𝜆 × 𝟏𝑦 1 2
4. Update optimizer to minimize 𝑪𝑬(𝒚෥, 𝒑(𝒙෥))

48

## Cut-mix

[Source: https://encord.com/blog/data-augmentation-guide/]

[Source: https://arxiv.org/pdf/1905.04899 ] 𝑴 𝐶×𝑊×𝐻 1 1 1 1 1 1 𝐶×𝑊×𝐻 𝑥, 𝟏, 𝑥 ∈ ℝ 𝐵 = [𝑟, 𝑟, 𝑟, 𝑟 ] 𝑥, 𝟏, 𝑥 ∈ ℝ 2 𝑦2 2 𝑥෤ 1 𝑥 𝑦 𝑤 ℎ 1 𝑦1 1 1

0 0 0 1 1 𝜆~Beta(𝛼, 𝛼) 0 0 0

𝐻×𝑊 1 1 𝑎𝑟𝑒𝑎(𝐵) 𝑊𝐻(1 − 𝜆) 𝑥෤ = 𝑀 ⊙ 𝑥1 + 1 − 𝑀 ⊙ 𝑥2, 𝑀 ∈ 0,1 = = 1 − 𝜆

1 0 0 0 1

𝑎𝑟𝑒𝑎(𝑖𝑚𝑎𝑔𝑒) 𝑊𝐻 𝑦෤ = 𝜆 × 𝟏𝑦 + 1 − 𝜆 × 𝟏𝑦

1 2 1 1

1 1 1 1 1 1 𝐦𝐢𝐧 𝑪𝑬(𝒚෥, 𝒑(𝒙෥)) ❑ for (𝑥1, 𝑦1), (𝑥2, 𝑦2) in zip(batch1, batch 2)

1. 𝜆~Beta 𝛼, 𝛼 ⊙ 𝑀 =
2. Sample a bounding box 𝐵 = [𝑟𝑥, 𝑟𝑦, 𝑟𝑤, 𝑟ℎ]

1. 𝑟𝑥 ∼ 𝑈𝑛𝑖[0, 𝑊], 𝑟𝑤 ∼ 𝑊 1 − 𝜆
2. 𝑟𝑦 ∼ 𝑈𝑛𝑖[0, 𝐻], 𝑟ℎ ∼ 𝐻 1 − 𝜆 𝑾×𝑯
3. 𝑴 ∈ 𝟎, 𝟏 by filling 1 outside B and 0 otherwise
4. 𝑥෤ = 𝑀 ⊙ 𝑥1 + 1 − 𝑀 ⊙ 𝑥2 ⊙ (1 − 𝑀) =
5. 𝑦෤ = 𝜆 × 𝟏𝑦 + 1 − 𝜆 × 𝟏𝑦 1 2
6. Update optimizer to minimize 𝑪𝑬(𝒚෥, 𝒑(𝒙෥)) 49

### Transfer learning and fine-tuning

52

## Transfer Learning

Motivation

Large-scale dataset (ImageNet) Your small-scale dataset

- over 1.2 million images and 1,000 (Flower-17)

possible object categories -17 category dataset with 80

images per class

Transfer learning

- Fine tune

Powerful pretrained model

- VGG16, VGG19 Not enough data to train a good model.
- Inception V3 HOW?
- Xception
- ResNet

Fits the target dataset

Low-level filers Mid-level filers High-level filers

53

## Transfer Learning

How to Do That?

###  Remove FC layers from the pretrained model

###  Replace them with a brand-new FC head.

o These new FC layers can then be fine-tuned to the specific dataset o The old FC layers are no longer used

## Transfer Learning

How to Do That?

###  Freeze all CONV layers in the network

o Only allow the gradient to backpropagate through the FC layers o Doing this allows our network to warm up

Freeze these layers

###  Training data is forward propagated through

### the network

o However, the backpropagation is stopped after the FC layers o Allows these layers to start to learn patterns from Fine-tune these layers the highly discriminative CONV layers

54

55

## Transfer Learning

How to Do That?

### ❑ After the FC layers have had a chance to

### warm up, we may choose to unfreeze all

### layers in the network

❑ Allow each of them to be fine-tuned. ❑ Continue training the entire network, but with a very small learning rate ❑ We do not want to deviate our CONV filters dramatically. Training is then allowed to continue until sufficient accuracy is obtained.

56

## Model zoo supported by PyTorch

## Transfer learning with PyTorch

Load pretrained VGG19

Will be replaced by a fresh new linear layer

57

58

## Transfer learning with PyTorch

Warm-up the fresh new linear layer

Fine-tune the entire model

59

## Summary

###  Setting of a machine learning problem

o General loss versus empirical loss

###  Gradient vanishing/exploding and network initialization.

###  Overfitting and underfitting

###  Recipe for overfitting

o Use regularization term o Dropout, batch norm o Data augmentation o Transfer learning o Label smoothing, data mix-up

## Thanks for your attention!

Question time