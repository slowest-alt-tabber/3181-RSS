COPYRIGHT WARNING: Copyright in these original lectures is owned by Monash University. You may transcribe, take notes, download or stream lectures for the purpose of your research and study only. If used for any other purpose, (excluding exceptions in the Copyright Act 1969 (Cth)) the University may take legal action for infringement of copyright.

Do not share, redistribute, or upload the lecture to a third party without a written permission!

### FIT3181/5215 Deep Learning

Week 01: Machine Learning and Mathematical

Background Revisit

CE/Lecturer: Dr Trung Le

Lecturer: Dr Binh Nguyen

Email: trunglm@monash.edu

Department of Data Science and AI Faculty of Information Technology, Monash University, Australia

## Outline

### Mathematical Background Revisit

### Machine Learning Revisit

### Logistic Regression

### Summary

### Mathematical Background Revisited

### Linear algebra revision

## Why do we need vectors?

Hand-crafted feature

###  Traditional approach (hand-crafted feature learning)

Input Feature extraction (SIFT, SURF, HOG) Learning algorithms (SVM, XGBoost)

Hand-crafted features

[Source: Internet] 0.1 0.5 -1 2.1 1.3 SIFT, SURF, HOG

0.6 0.2 1.3 2.5 3.3

Vision dataset

0.7 0.5 -2 2.2 2.3 Prediction

0.8 0.4 0.9 2.4 1.2

0.6 0.9 1.5 2.0 1.6 ML model

(SVM, k-NN, decision tree, XG Boost) Feature vectors

5

## Vector

n-dimensional vector

dimension

𝑥1 𝐱 = 𝑥2

𝑥3

x x

𝑥𝑖 : 𝑖-th element

## Vector

Operations on vector

transpose: column vector to row vector

x

x

x y addition

inner product x y

## Vector

p-norm

1 𝑝 𝑝 𝑝 𝑝

### 𝐱 𝑝 = 𝑥1 + 𝑥2 + ⋯ + 𝑥𝑛 with 𝑝 > 0

### p = 0 : how many elements in x are non-zeros (sparsity)

### p = 1 : sum of absolute values of elements

### p = 2 : length of the vector

### ❑ 𝐱 𝑝 ≥ 0 and 𝐱 𝑝 = 0 if only if 𝐱 = 𝟎.

### ❑ k𝐱 𝑝 = |𝑘| 𝐱 𝑝 for any scalar 𝑘 ∈ ℝ.

### ❑ 𝐱 + 𝒚 𝑝 ≤ 𝐱 𝑝 + 𝒚 𝑝 (triangle inequality).

## Vector

Length of a vector

𝑥1 𝐱 = 𝑥2 2 2 2 length 𝐱 = 𝑥 + 𝑥 + ⋯ + 𝑥𝑛

1 2 𝑥

3

2 2 2 𝐱 2 = 𝑥 + 𝑥 + ⋯ + 𝑥𝑛

1 2

## Vector

Distance between two vectors

𝐱

Euclidean distance

2 2 |𝐱 − 𝒚| = 𝑥1 − 𝑦1 + ⋯ + 𝑥𝑛 − 𝑦𝑛

𝐲

Cosine similarity Cosine distance is defined as

𝑇 𝑇 𝐱 𝐲 𝐱 𝐲 cos 𝜃 = cos(𝐱, 𝐲) = = 𝑑 𝐱, 𝐲 = 1 − cos 𝐱, 𝐲

𝑐𝑜𝑠𝑖𝑛𝑒 𝐱 | 𝐲 | 𝑇 𝑇

𝐱 𝐱 𝐲 𝐲 −1 ≤ cos 𝐱, 𝐲 ≤ 1

Cosine distance can be computed via Euclidean distance if vectors are made unit vectors! (why?)

## Example of cosine similarity

Word2Vec

### Assume that a Word2Vec model

𝑇 𝑣𝐾 = 1, −2,1

### transforms three words King, King

𝑇

### Queen, and Football to three 𝑣𝑄 = 0.9, −2.2,1.2

Queen

### vectors 𝑣, 𝑣, and 𝑣 as shown in 𝑇

𝐾 𝑄 𝐹 Football 𝑣 = 1,2.1, −0.9

𝐹

### the figure. Between two words King

Australia

### and Football, which word is more

Canberra

### similar to word Queen?

Paris

𝑇 𝑣 𝑣𝑄 1×0.9+ −2 × −2.2 +1×1.2 𝐾 𝑐𝑜𝑠_𝑠𝑖𝑚 𝐾𝑖𝑛𝑔, 𝑄𝑢𝑒𝑒𝑛 = = = 0.996

𝑣 𝑣 2 2 2 2 2 2 𝐾 𝑄 1 + −2 +1 0.9 + −2.2 +1.2

𝑇 𝑣 𝑣𝑄 1×0.9+2.1× −2.2 +(−0.9)×1.2 𝐹 𝑐𝑜𝑠_𝑠𝑖𝑚 𝐹𝑜𝑜𝑡𝑏𝑎𝑙𝑙, 𝑄𝑢𝑒𝑒𝑛 = = = −0.72

𝑣 𝑣 2 2 2 2 2 2 𝐹 𝑄 1 +2.1 +(−0.9) 0.9 + −2.2 +1.2 King is more similar to Queen than Football.

11

## Matrix (2D tensor)

number of columns

number of columns

### 𝑎11 ⋯ 𝑎1𝑛

### ⋮ ⋱ ⋮

### 𝐴 = 𝑎𝑖1 … 𝑎𝑖𝑛

### ⋮ ⋱ ⋮

### 𝑎𝑚1 ⋯ 𝑎𝑚𝑛

number of rows

number of rows

column vector

### 𝑎11 ⋯ 𝑎1𝑛

### ⋮ ⋱ ⋮ 𝑎 … 𝑎1𝑗 … 𝑎

11 1𝑛

### 𝐴 = 𝑎 … 𝑎 row vector 𝐴 = … … … … …

𝑖1 𝑖𝑛

### ⋮ ⋱ ⋮ 𝑎 … 𝑎𝑚𝑗 … 𝑎

𝑚1 𝑚𝑛

### 𝑎𝑚1 ⋯ 𝑎𝑚𝑛

## Matrix operation

𝒂 𝒂 𝒃 ⋯ 𝒃 𝟏𝟏 ⋯ 𝟏𝒏 𝟏𝟏 𝟏𝒏 ⋮ ⋱ ⋮ ⋮ ⋱ ⋮

𝒎×𝒏 𝒂 𝒂 𝒎×𝒏 𝑩 = 𝒃 … 𝒃 ∈ ℝ 𝑨 = 𝒊𝟏 … 𝒊𝒏 ∈ ℝ 𝒊𝟏 𝒊𝒏

⋮ ⋱ ⋮ ⋮ ⋱ ⋮ 𝒂 ⋯ 𝒂 𝒃𝒎𝟏 ⋯ 𝒃𝒎𝒏 𝒎𝟏 𝒎𝒏

𝒂𝟏𝟏 ⋯ 𝒂𝒎𝟏 𝜶𝒂 𝟏𝟏 ⋯ 𝜶𝒂𝟏𝒏 ⋮ ⋱ ⋮ ⋮ ⋱ ⋮ 𝑻 𝒏×𝒎 𝒎×𝒏 𝑪 = 𝑨 = 𝒂𝟏𝒋 … 𝒂𝒎𝒋 ∈ ℝ 𝜶𝑨 = 𝜶𝒂𝒊𝟏 … 𝜶𝒂𝒊𝒏 ∈ ℝ

⋮ ⋱ ⋮ ⋮ ⋱ ⋮ 𝒂 ⋯ 𝒂 𝜶𝒂𝒎𝟏 ⋯ 𝜶𝒂𝒎𝒏 𝟏𝒏 𝒎𝒏 𝑻 𝑪 = 𝑨 if only if 𝑪𝒊𝒋 = 𝑨𝒋𝒊 for any 1 ≤ 𝑖 ≤ 𝑛, 1 ≤ 𝑗 ≤ 𝑚

𝒂 + 𝒃 ⋯ 𝒂 + 𝒃 𝟏𝟏 𝟏𝟏 𝟏𝒏 𝟏𝒏

⋮ ⋱ ⋮

𝒎×𝒏 𝑪 = 𝑨 + 𝑩 = 𝒂 + 𝒃 … 𝒂 + 𝒃 ∈ ℝ

𝒊𝟏 𝒊𝟏 𝒊𝒏 𝒊𝒏

⋮ ⋱ ⋮ 𝒂𝒎𝟏 + 𝒃𝒎𝟏 ⋯ 𝒂𝒎𝒏 + 𝒃𝒎𝒏

13

## Matrix multiplication

𝑎11 ⋯ 𝑎1𝑛

𝑏 … 𝑏1𝑗 … 𝑏1𝑝 ⋮ ⋱ ⋮ 11 𝑎 𝑎 𝐵 = … … … … … 𝐴 = 𝑖1 … 𝑖𝑛

𝑏 … 𝑏𝑛𝑗 … 𝑏𝑛𝑝 ⋮ ⋱ ⋮ 𝑛1 𝑎𝑚1 ⋯ 𝑎𝑚𝑛

𝑛 𝑐𝑖𝑗 = 𝐴 𝑖, : 𝐵 : , 𝑗 = ෍ 𝑎𝑖𝑘𝑏𝑘𝑗

𝑘=1 𝑚×𝑛 𝑛×𝑝

###  Given two matrices 𝐴 = 𝑎 ∈ ℝ and 𝐵 = 𝑏 ∈ ℝ, we

𝑖𝑗 𝑖𝑗

### can compute the multiplication 𝐶 = 𝐴𝐵 of 𝐴 and 𝐵 as

𝑚×𝑝 o 𝐶 = 𝑐𝑖𝑗 ∈ ℝ

𝑛 o 𝑐 = σ 𝑎 𝑏 for 1 ≤ 𝑖 ≤ 𝑚 and 1 ≤ 𝑗 ≤ 𝑝

𝑖𝑗 𝑘=1 𝑖𝑘 𝑘𝑗 o 𝑐𝑖𝑗 = 𝐴 𝑖, : 𝐵[: , 𝑗] which is the dot product of the row 𝒊 in A and the column 𝒋 in B.

## Matrix multiplication with numpy

Declare matrix 𝑨 with the size 𝟐 × 𝟑 Declare matrix 𝑫 with the size 𝟑 × 𝟒

𝑬 = 𝑨 × 𝑫 has the size 𝟐 × 𝟒

## Tensor

###  A tensor is simply a

### multidimensional array

o 0D tensor: a scalar o 1D tensor: a vector o 2D tensor: a matrix o 3D tensor: a 3D hyperrectangle of numerals

Examples of tensors. [Source: Internet]

### Information theory revision

## Why need to work with discrete distributions?

Ground-truth one-hot label 0 1 0 0

lion car chair airplane

How diverge Image classification task and distant?

Prediction probability 0.6 0.2 0.1 0.1

𝒇(𝒙)

Neural Net or machine learning model 𝒇

Input x

## Revision of basic informatic quantities

𝑑 𝑑

###  Given two discrete distributions 𝑝 = 𝑝 (𝑝 ≥ 0 and σ 𝑝 = 1) and 𝑞 =

𝑖 𝑖=1 𝑖 𝑖=1 𝑖 𝑑 𝑑

### 𝑞 (𝑞 ≥ 0 and σ 𝑞 = 1).

𝑖 𝑖=1 𝑖 𝑖=1 𝑖

###  Kullback-Leibler (KL) divergence between 𝒑, 𝒒

𝑑 𝑝𝑖 ❑ 𝐾𝐿 𝑝, 𝑞 = σ 𝑝 log

𝑖=1 𝑖

𝑞𝑖 ❑ 𝐾𝐿 𝑝, 𝑞 ≥ 0 and 𝐾𝐿 𝑝, 𝑞 = 0 if only if 𝑝 = 𝑞

### ❑ Entropy of the distribution 𝒑

𝑑 1 𝑑 ❑ 𝐻 𝑝 = σ 𝑝 log = − σ 𝑝 log 𝑝 where 0 × log 0 = 0

𝑖=1 𝑖 𝑖=1 𝑖 𝑖

𝑝𝑖

###  Cross-entropy (CE) divergence between 𝒑, 𝒒

𝑑 ❑ 𝐶𝐸 𝑝, 𝑞 = − σ 𝑝 log 𝑞.

𝑖=1 𝑖 𝑖 ❑ 𝐶𝐸(𝑝, 𝑞) = 𝐾𝐿 𝑝, 𝑞 + 𝐻(𝑝).

19

## Example

###  Consider 𝑝 = [0.1, 0.4, 0.5] and q = [0.2, 0.5, 0.3]. Compute 𝐾𝐿(𝑝, 𝑞),

### 𝐶𝐸(𝑝, 𝑞), and 𝐻(𝑝).

0.1 0.4 0.5 o 𝐾𝐿 𝑝, 𝑞 = 0.1 × log + 0.4 × log + 0.5 × log ≈ 0.042

0.2 0.5 0.3 o 𝐶𝐸 𝑝, 𝑞 = −0.1 × log 0.2 − 0.4 × log 0.5 − 0.5 × log 0.3 ≈ 0.452

1 1 1 o 𝐻 𝑝 = 0.1 × log + 0.4 × log + 0.5 × log ≈ 0.41

0.1 0.4 0.5

###  Consider 𝑝 = [0,1,0] and q = [0.2,0.3,0.5]. Compute 𝐶𝐸(𝑝, 𝑞).

o 𝐶𝐸 𝑝, 𝑞 = −0 × log 0.2 − 1 × log 0.3 − 0 × log 0.5 ≈ −log 0.3=0.523

Question: when the entropy is maximized or minimized?

20

### Machine Learning Revisited

## Machine learning

 “Field of study that gives computers the ability to learn without being explicitly programmed” (Arthur Samuel, 1959).  “Machine learning sits at the crossroads of computer science, statistics and a variety of other disciplines concerned with automatic improvement over time, and inference and decision-making under uncertainty.” (Jordan & Michell, 2015).  “A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at

(from Eric Xing lecture notes) tasks in T, as measured by P, improves with experience E”, (Tom Mitchell, 1997) M.I. Jordan and T. Mitchell, “Machine Learning: Trends, perspectives, and prospects”, Science, 17 July 2015.

## A brief history of ML

Successful applications Symbolic concept induction

IR & ranking Data mining Multi strategy learning MIML

Active & online learning Minsky criticism Transfer learning

NN, GA, EBL, CBL

Kernel methods

Sparse learning Abduction, Pattern Recognition emerged Analogy

Bayesian methods Revival of non-symbolic learning

PAC learning ILP Semi-supervised Deep learning

learning

Dimensionality reduction Experimental comparisons Math discovery AM

Probabilistic graphical models Supervised learning

Statistical learning Neural modeling

Nonparametric Bayesian Unsupervised learning

Ensemble methods Rote learning Reinforcement learning Structured prediction

19501941 19491960  1956 19701958  19681980  1970 19721990  1982 19862000  1990 1997

2010 ICML (1982) ECML (1989) KDD (1995) PAKDD (1997) ACML (2009)

enthusiasm

dark age renaissance explosion fast development

[credit: Prof Bao Ho, JAIST]

## Machine learning

Experience E Task T Performance P dataset of thousands of given a new photo, how accurate the known faces recognise the name of recognition is

the face Measure: Accuracy Label set: names of all people.

Face recognition system

Experience E Task T Performance P Examples of spam To assign a label how accurate spam emails and not-spam “spam” or “not- email can be emails spam” to an email detected

Label set: spam and Measure: accuracy Detecting Spam Emails not spam

Experience E Task T Performance P Data collected for To assign a label ‘fraud’ how accurate a credit- credit-card transactions or “not fraud” to a card fraud transaction deemed as fraud and given credit-card can be detected. not-fraud transaction Measure: accuracy Detect credit card fraud Label set: fraud and not

fraud

## Elements of machine learning

# 1 Data 2 Model 3 Assessment

𝑑 ▪ 𝑓: 𝑋 → 𝑌 ▪ How good is the model doing Vector 𝒙 ∈ ℝ

▪ 𝑋 is data space, 𝑌 is label space its tasks? Image 𝒙

▪ Classification: 𝑌 = {1,2, … , 𝑀} ▪ Performance metrics Sentence 𝒙

▪ Regression: 𝑌 = ℝ - Accuracy, Recall, Precision, F-score Speech 𝒙

\#𝐶𝑜𝑟𝑟𝑒𝑐𝑡 𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛𝑠 ▪ Prediction: 𝑓 𝑥 = 𝑦ො ∈ 𝑌 Ground-truth label 𝑦 ∈ 𝑌 - 𝐴𝑐𝑐𝑢𝑟𝑎𝑐𝑦 =

\#𝐴𝑙𝑙 𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛𝑠: % of correctly classified examples Labelled Dataset

Training set Test set

Learning Learned

Model Model performance Process

Machine learning pipeline.

26

## Classification and Regression

 Two main tasks in machine learning

1. Classification and regression
2. Other tasks in ML include density estimation, clustering, anomaly detection, recommendation, forecasting, and ranking airplane?(0.4) ML model 𝑓 automobile(0.0)?

 Classification

bird?(0.0)

1. The label set 𝑌 = {1,2, … , 𝑀} and the data space 𝑋 (vectors, cat?(0.05) images). deer?(0.2) 𝑑
2. Training set 𝑫 = { 𝒙𝟏, 𝑦1 , … , 𝒙𝑵, 𝑦𝑁 } where 𝒙𝒊 ∈ ℝ and 𝑥 dog?(0.05) 𝑦𝑖 ∈ 𝑌 = {1,2, … , 𝑀} True label 𝑦 = ℎ𝑜𝑟𝑠𝑒 frog?(0.0)
3. Learn a model 𝑓: 𝑋 → 𝑌 horse?(0.3)
4. Given data example 𝒙, render the prediction 𝑦ො = 𝑓 𝒙 ∈ ship?(0.0) {1,2, … , 𝑀} truck?(0.0)

ML model 𝑓  Regression

1. The label set 𝑌 = ℝ and the data space 𝑋 (vectors, images). Human info 𝑑
2. Training set 𝑫 = { 𝒙, 𝑦, … , 𝒙, 𝑦 } where 𝒙 ∈ ℝ and - Age, position, job, Income? 𝟏 1 𝑵 𝑁 𝒊 qualification, and etc. 𝑦𝑖 ∈ 𝑌 = ℝ 𝑥
3. Learn a model 𝑓: 𝑋 → 𝑌 True label 𝑦 = 90𝐾 𝐴𝑈𝐷𝑠
4. Given data example 𝒙, render the prediction 𝑦ො = 𝑓 𝒙 ∈ ℝ

## Supervised learning

Problem: Learn a function from data to relate

the inputs with outputs. Training data include output information (labels).

Data: 𝐷 = { 𝒙𝟏, 𝑦1 , … , (𝒙𝑵, 𝑦𝑁)}

Function: 𝑓: 𝑋 → 𝑌 e.g., linear functions

𝑑 𝑥 ∈ ℝ = feature 𝑦 ∈ {1,2, … , 𝑀} = a discrete label (classification), 𝑦 ∈ ℝ = a continuous value (regression)

e.g., nonlinear functions

28

## Supervised learning

Labelled vision dataset Convolution Neural Network Vision Transformers

Recurrent neural nets Transformers Labelled text corpus

Multilayered feedforward nets Labelled tabular dataset

### Discriminative Machine Learning and Logistic Regression

## Discriminative machine learning

Classification

Discriminative ML model 𝒉

ℎ1 𝒙 ℎ2 𝒙 feed predict Data point 𝒙 ℎ(𝒙) …… 𝑓 𝒙 = 𝑦ො ≔ argmax ℎ (𝒙)

1≤𝑚≤𝑀 𝑚 Ground-truth label ……

the loss suffers if we predict 𝑥 with 𝑦 ∈ {1, . . , 𝑀} ℎ𝑀 𝒙

label 𝑦ො while its true label is 𝑦.

Suffer loss 𝑙(𝑦, 𝑦ො)

Training set 𝟏 𝑵

Suffer loss σ 𝒍(𝒚, 𝒚ෝ)

𝒊=𝟏 𝒊 𝒊 𝐷 = { 𝒙, 𝑦, … , (𝒙, 𝑦)} 𝑵

𝟏 1 𝑵 𝑁

 A data point 𝑥 receives discriminative values/logits ℎ 𝒙 , … , ℎ 𝒙 ∈ ℝ from the model

1 𝑀  ℎ (𝒙) represents the possibility to classify 𝑥 to class 𝒎 for 1 ≤ 𝑚 ≤ 𝑀

𝑚  We use these discriminative values/logits to predict the label 𝑦ො as

 𝑦ො = argmax ℎ (𝒙), meaning the class with highest discriminative value/logit

1≤𝑚≤𝑀 𝑚  The prediction 𝑥 with the predicted label 𝑦ො suffers a loss

 𝑙(𝑦 ො, 𝑦) where 𝑙 is a loss function (if 𝑦ො = 𝑦 then 𝑙 𝑦ො, 𝑦 = 0).  Given a training set 𝐷 = { 𝒙, 𝑦, … , (𝒙, 𝑦)}, the loss incurred is

𝟏 1 𝑵 𝑁 𝟏 𝑵  σ 𝒍(𝒚, 𝒚ෝ)

𝒊=𝟏 𝒊 𝒊 31 𝑵

## Softmax activation function

 Transform real-valued discriminative scores to discrete probabilities

𝑀 𝑀 exp{ℎ𝑚} o ℎ = ℎ𝑚 → 𝑝 = softmax ℎ ≔

𝑚=1 𝑀

σ exp{ℎ } 𝑖=1 𝑖

𝑚=1 𝑀 o 𝑝 = 𝑝𝑚 becomes a distribution over classes {1, … , 𝑀}

𝑚=1

𝑀 ▪ 𝑝 ≥ 0 (1 ≤ 𝑚 ≤ 𝑀) and σ 𝑝 = 1 .

𝑚 𝑚=1 𝑚

 Assume that we do sentiment classification (analysis)

o Data point 𝑥 is a sentence (e.g., 𝑥 = "𝑖𝑡 𝑖𝑠 𝑎 𝑏𝑒𝑎𝑢𝑡𝑖𝑓𝑢𝑙 𝑑𝑎𝑦 𝑡𝑜𝑑𝑎𝑦“), we need to predict sentiment of this sentence o Three sentiment labels: positive (happy, class 1), negative (sad, class 2), neutral (none happy and sad, class 3) o The model gives three discriminative values for 𝑥

▪ ℎ1 = 2, ℎ2 = −1, ℎ3 = 1 means that highest possibility to classify 𝑥 to the class 𝟏 with the label positive. o We apply softmax function on the discriminative scores ℎ

▪ 𝑝 = softmax(ℎ)

exp{2} exp{−1} exp{1} o 𝑝1 = ≈ 0.705, p2 = ≈ 0.035, p3 = ≈ 0.259

exp{2}+exp −1 +exp{1} exp{2}+exp −1 +exp{1} exp{2}+exp −1 +exp{1} ▪ 𝑝 = [0.70538451,0.03511903,0.25949646] are the probabilities to classify 𝑥 to the classes 𝟏, 𝟐, 𝟑 respectively.

h (logits) p (pred prob)

Softmax transforms real-valued discriminative softmax

scores (ranged in (−∞, +∞)) to probability values (ranged in [0, 1]) but preserving the order.

## Cross-entropy loss

###  Given 𝑀 labels 1,2, … , 𝑀, a label 𝑦 ∈ {1, … , 𝑀} has two equivalent forms:

o Numeric form: 𝑦 ∈ {1, … , 𝑀} is an index in {1, … , 𝑀}. o One-hot vector form: 1𝑦 = [0, . . , 0,1,0, … , 0]- the vector of zeros except a unique 𝟏 in the 𝑦-th position.

𝑦-th position

###  Label set {ℎ𝑎𝑝𝑝𝑦, 𝑠𝑎𝑑, 𝑛𝑒𝑢𝑟𝑎𝑙} with 1: ℎ𝑎𝑝𝑝𝑦, 2: 𝑠𝑎𝑑, and 3: 𝑛𝑒𝑢𝑡𝑟𝑎𝑙

o [1,0,0] → happy (numeric label 1), [0,1,0]→sad (numeric label 2), [0,0,1]→neural (numeric label 3)

𝑀 𝑀

###  Given two discrete distributions over classes: 𝑝 = 𝑝 and q = 𝑞,

𝑚 𝑚=1 𝑚 𝑚=1

### the cross-entropy divergence between 𝑝 and 𝑞

𝑀 o CE 𝑝, 𝑞 ≔ − σ 𝑝 log 𝑞 which measures how far (divergent) it is between 𝑝 and 𝑞.

𝑚=1 𝑚 𝑚

𝑀 o CE 𝑝, 𝑞 ≥ H 𝑝 ≔ − σ 𝑝 log 𝑝 (H is the Shannon entropy).

𝑚=1 𝑚 𝑚

## Cross-entropy loss

###  Cross-entropy loss

o 𝑙 𝑦, 𝑦ො = CE 1𝑦, 𝑝 = − log 𝑝𝑦 o 𝑦 ∈ {1, … , 𝑀} is the ground-truth label of a given 𝑥 and 𝑝 is the prediction probabilities of 𝑥 to classes

###  Given a sentence 𝑥 with the label positive/happy (the class 1), assume that our

### model predicts it with prediction probabilities 𝑝 = [0.3,0.4,0.3], the cross-

### entropy loss for this prediction

o 𝑙 𝑦, 𝑦ො = 𝐶𝐸 1,0,0 , 0.3,0.4,0.3 = −1. log 0.3 − 0. log 0.4 − 0. log 0.3 = − log 0.3 ≈ 1.204 o Correct or Incorrect prediction?

Question: why is the cross-entropy loss always non-negative and when it is 0?

## Discriminative machine learning

Classification with the cross-entropy loss

Discriminative Prediction probabilities ML model 𝒉 𝑝 𝒙 = softmax(ℎ(𝒙))

ℎ 𝒙 𝑝1 𝒙 1 feed ℎ 𝒙 softmax 𝑝2 𝒙 predict

2 p(𝑥) Data point 𝒙 ℎ(𝑥) 𝑓 𝑥 = 𝑦ො ≔ argmax1≤𝑚≤𝑀𝑝𝑚 (𝒙)

………… ………… Ground-truth label

ℎ 𝒙 𝑝𝑀 𝒙 𝑀 𝑦 ∈ {1, . . , 𝑀}

Suffer loss 𝑙 𝑦, 𝑦ො = 𝐶𝐸(1𝑦, 𝑝(𝒙))

 We apply the softmax function: 𝐩 𝒙 = 𝐬𝐨𝐟𝐭𝐦𝐚𝐱(𝐡(𝐱)) to make 𝑝(𝑥) prediction

probabilities o 𝑝𝑚 𝑥 = 𝑝(𝑦 = 𝑚|𝑥) is the probability to classify 𝑥 to the class 𝑚 for 1 ≤ 𝑚 ≤ 𝑀  The loss incurred for this prediction

o 𝑙 𝑦, 𝑦ො = 𝐶𝐸 1𝑦, 𝑝 𝒙 = − log 𝑝𝑦 (𝒙) (negative log likelihood)  The loss for the training set 𝐷 = { 𝒙, 𝑦, … , (𝒙, 𝑦)}

𝟏 1 𝑵 𝑁 1 𝑁 1 𝑁 o 𝐿 𝐷, ℎ = σ 𝐶𝐸(1, 𝑝(𝒙)) = − σ log 𝑝 (𝒙)

𝑖=1 𝑦𝑖 𝒊 𝑖=1 𝑦𝑖 𝒊 𝑁 𝑁

## Discriminative machine learning

Regression

Discriminative ML model 𝒉

feed predict Data point 𝒙 ℎ(𝑥) 𝑓 𝒙 = 𝑦ො ≔ ℎ(𝒙) Ground-truth label

𝑦 ∈ ℝ

Suffer loss 𝑙(𝑦, 𝑦ො)

❑ Only one discriminative value ℎ 𝒙 ∈ ℝ, we use this to predict 𝑦ො = 𝑓(𝒙) ≔ ℎ(𝒙). ❑ The potential losses could be

1

2 ❑ L2 loss: 𝑙 𝑦, 𝑦ො = 𝑦 − 𝑦ො

2 ❑ L1 loss: 𝑙 𝑦, 𝑦ො = 𝑦 − 𝑦ො ❑ 𝝐-insensitive loss: 𝑙 𝑦, 𝑦ො = max{0, 𝑦 − 𝑦ො − 𝜖} where 𝜖 ≥ 0 .

## Logistic regression

A simple feed-forward neural network

ℎ 𝑥 𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ(𝑥))

1 2 𝑑 𝑑 𝑥 = 𝑥 𝑥 … 𝑥 ∈ ℝ

𝑦 ∈ {1; 2}

𝑑×2 𝑊 ∈ ℝ

1×2 𝑏 ∈ ℝ  Computational process

o ℎ 𝑥 = [ℎ1 𝑥 ℎ2(𝑥)] = 𝑥𝑊 + 𝑏

1×2 ▪ ℎ 𝑥 ∈ ℝ contains discriminative scores with respect to classes 1 and 2. o 𝑝 𝑥 = [𝑝1 𝑥 𝑝2(𝑥)] = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ 𝑥 )

▪ 𝑝(𝑥) is the prediction probabilities with respect to classes 1 and 2.

𝟏, 𝒑𝟏 𝒙 ≥ 𝒑𝟐 𝒙 o 𝒚ෝ = ቊ

𝟐, 𝒑𝟏 𝒙 < 𝒑𝟐 𝒙

o 𝑙 𝑦, 𝑦ො = 𝐶𝐸 1𝑦, 𝑝 𝑥 = − log 𝑝𝑦 (𝑥) → the cross-entropy loss.  Training process

o Training set 𝐷 = { 𝒙𝟏, 𝑦1 , … , (𝒙𝑵, 𝑦𝑁)}

1 𝑁 1 𝑁 o 𝐿 𝐷; 𝑊, 𝑏 = σ 𝐶𝐸(1, 𝑝(𝒙)) = − σ log 𝑝 (𝒙)

𝑖=1 𝑦𝑖 𝒊 𝑖=1 𝑦𝑖 𝒊 𝑁 𝑁

∗ ∗ o min 𝐿(𝐷; 𝑊, 𝑏) → (𝑊, 𝑏) = argmin𝑊,𝑏𝐿(𝐷; 𝑊, 𝑏)

𝑊,𝑏

## Logistic Regression Pipeline

Training set 1 epoch

𝒚 = 𝒄𝒂𝒕 𝒚𝟐 = 𝒅𝒐𝒈 𝒚𝟔𝟒 = 𝒍𝒊𝒐𝒏 𝒚 = 𝒔𝒑𝒊𝒅𝒆𝒓 𝒚 = 𝒄𝒐𝒘 𝒚 = 𝒄𝒂𝒕 𝟏 ∗ ∗+𝟔𝟑 𝟔𝟎𝟎𝟎𝟎

𝒙 𝒙𝟐 … 𝒙 𝒙 𝟏 … 𝒙 … … ∗ … 𝒙 𝟔𝟎𝟎𝟎𝟎

… 𝟔𝟒 ∗+𝟔𝟑 … …

… … …

A mini-batch Another mini-batch

One vector: [1, 16] → Our Logistic

Batch: [64, 16] Regression 64

1 Model parameters 𝜃 = [𝑊, 𝑏] 𝑏𝑎𝑡𝑐ℎ_𝑙𝑜𝑠𝑠(𝜃) = ෍ 𝐶𝐸(1𝑦, 𝑝 𝑥𝑖, 𝜃 )

𝑖 … 64 𝑖=1

ℎ 𝑥 𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ(𝑥)) … 64 Optimizer

………

- Adam,

… - RMSProp, 16 𝑑×2 - SGD

𝑊 ∈ ℝ

1×2 𝑏 ∈ ℝ

min 𝑏𝑎𝑡𝑐ℎ_𝑙𝑜𝑠𝑠(𝜃) 𝜃 Testing set

… … … ……. … … …

Test accuracy 𝒙′ 𝒙′ 𝒙′𝟏𝟎𝟎𝟎

𝟏 𝟐

38

### Appendix

# Other ML Books

## Machine learning: five tribes

The Symbolist School

Tribes Origins Master Algorithm

Symbolists Logic, Inverse deduction

philosophy

Tom Mitchell Steve Muggleton Ross Quinlan

Decision trees, C4.5, first-order propositional logic, rule-based

reasoning systems, etc.

## Machine learning: five tribes

The Evolutionary School

Tribes Origins Master Algorithm

Symbolists Logic, Inverse deduction

philosophy

John Koza John Holland Hod Lipson Evolutionary Evolutionary Genetic

biology programming

Genetic programming, genetic

algorithms, fuzzy systems

## Machine learning: five tribes

The Connectionist

Tribes Origins Master Algorithm

Symbolists Logic, Inverse deduction

philosophy

Evolutionary Evolutionary Genetic

Geoff Hinton Yann LeCun Yoshua Bengio biology programming

Connectionists Neuroscience Backpropagation

Deep Neural Networks, Convolutional Neural

Networks, Deep learning, etc.

## Machine learning: five tribes

The Bayesians

Tribes Origins Master Algorithm

Symbolists Logic, Inverse deduction

philosophy

Evolutionary Evolutionary Genetic David Heckerman Judea Pearl Michael Jordan

biology programming

Connectionists Neuroscience Backpropagation

Bayesians Statistics Bayes’ theorem

Bayesian networks classifiers, graphical models, Hidden Markov models, Conditional Random Fields, most of statistical machine

learning approaches …

## Machine learning: five tribes

The Analogizers

Tribes Origins Master Algorithm

Symbolists Logic, Inverse deduction

philosophy

Evolutionary Evolutionary Genetic

Vladimir Vapnik Peter Hart Douglas Hofstadter biology programming

Connectionists Neuroscience Backpropagation

Bayesian Statistics Bayes’ theorem

Analogizers Psychology Support Vector

Machines

K-NN classifiers, Support vector machines, kernel machines, max-margin machines,

instance-based classification, etc.

## Machine learning: five tribes

Approaches Problem Solved Solution Origin Solution Symbolists Knowledge acquisition Logic, philosophy Inverse deduction

Evolutionary Structure discovery Evolutionary Genetic programming

(nature) biology Connectionists Credit assignment Neuroscience Backpropagation and

(nurture) deep learning Bayesian Uncertainty Statistics Probabilistic Inference

Analogizers Similarity Psychology Kernel machines

## Machine learning: five tribes

https://www.youtube.com/watch?v=E8rOVwKQ5-8