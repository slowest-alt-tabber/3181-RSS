COPYRIGHT WARNING: Copyright in these original lectures is owned by Monash University. You may transcribe, take notes, download or stream lectures for the purpose of your research and study only. If used for any other purpose, (excluding exceptions in the Copyright Act 1969 (Cth)) the University may take legal action for infringement of copyright.

Do not share, redistribute, or upload the lecture to a third party without a written permission!

### FIT3181/5215 Deep Learning

Week 07: DL for time-series and temporal data-

RNNs and LSTMs

CE/Lecturer: Dr Trung Le (Clayton)

Lecturer: Dr Binh Nguyen (Clayton) Lecturer: Dr Ting Fung Fung (Malaysia) Lecturer: Dr Arghya Pal (Malaysia) Lecturer: Dr Sailaja Rajanala (Malaysia)

Email: trunglm@monash.edu

Department of Data Science and AI Faculty of Information Technology, Monash University, Australia

2

# Outline

##  Introduction of sequential

## data

##  Recurrent neural networks

## (RNNs)

##  Memory cells

### o Long-short term memory (LSTM) cell

### o Gated recurrent unit (GRU) cell

##  Applications of RNNs

### o Sentiment analysis

### o Text generation

## Sequence modelling

## Time series and Speech

## sequential data

Video surveillance

Stock market prices

 We live in a time-space universe  All data collected has a timestamp  Time-series/sequential data

= collection of sequential data points indexed by time order!

EEG brain signal Electrocardiography signals = electrical activity of the heart over

time

Texts, messages from media

4

5

## Sequential data examples

### • Data can also be viewed from different, more subtle angles

### I love deep learning

Word level o I, love, deep, learning Character level o I, l, o, v, e, d, e, e, p, l, e, a, r, n, i, n, g

I love deep learning How words/items 𝑥1, 𝑥2, 𝑥3, 𝑥4 cooccur in a meaningful text? Items 𝑥 𝑥 𝑥 𝑥4

1 2 3 𝑝 𝑥, 𝑥, 𝑥, 𝑥 = 𝑝 𝑥 𝑝 𝑥 𝑥 𝑝 𝑥 𝑥, 𝑥 𝑝 𝑥 𝑥, 𝑥, 𝑥

1 2 3 4 1 2 1 3 1 2 4 1 2 3

= 𝑝 𝑥1 𝑝 𝑥2 𝑥1 𝑝 𝑥3 𝑥1:2 𝑝 𝑥4 𝑥1:3

I l o v … … n g Items 𝑥 𝑥 𝑥 𝑥19 𝑥

1 2 3 𝑥 20

4

6

## Sequential data examples

### • Data can also be viewed from different, more subtle angles

### I love deep learning

Word level o I, love, deep, learning Character level o I, l, o, v, e, d, e, e, p, l, e, a, r, n, i, n, g

I love deep learning How words/items 𝑥1, 𝑥2, 𝑥3, 𝑥4 cooccur in a meaningful text? Items 𝑥 𝑥 𝑥 𝑥4

1 2 3 𝑝 𝑥, 𝑥, 𝑥, 𝑥 = 𝑝 𝑥 𝑝 𝑥 𝑥 𝑝 𝑥 𝑥, 𝑥 𝑝 𝑥 𝑥, 𝑥, 𝑥

1 2 3 4 1 2 1 3 1 2 4 1 2 3

= 𝑝 𝑥1 𝑝 𝑥2 𝑥1 𝑝 𝑥3 𝑥1:2 𝑝 𝑥4 𝑥1:3

I l o v … … n g Items 𝑥 𝑥 𝑥 𝑥19 𝑥

1 2 3 𝑥 20

4 𝑝 𝑥1, 𝑥2, … , 𝑥20 = 𝑝 𝑥1 𝑝 𝑥2 𝑥1 𝑝 𝑥3 𝑥1:2 𝑝 𝑥4 𝑥1:3 … 𝑝 𝑥19 𝑥1:18 𝑝(𝑥20 ∣ 𝑥1:19)

## Sequential data examples

𝑥 𝑥 𝑥 𝑥4 … … 1 2 3 𝑥 Row 1 𝑥1 10 … …

Row 2 𝑥2 𝑥19 … …

… 𝑥 … … 28

… 𝑥37… … 𝑥 … … … 𝑥53 𝑥 46 54

Row 6 𝑥6  Sequence of pixels or  Sequence of pixels  Sequence of rows

rows of pixels 𝑝 𝑥 = 𝑝 𝑥 𝑝 𝑥 𝑥 … 𝑝(𝑥 ∣ 𝑥) 𝑝 𝑥1:6 = 𝑝 𝑥1 𝑝 𝑥2 𝑥1 … 𝑝(𝑥6 ∣ 𝑥1:5)

1:54 1 2 1 54 1:53

𝑥1 𝑥2 𝑥3 𝑥4 𝑥5

 Video as a sequence of frames/images (Source: medium.com) 𝑝 𝑥1, … , 𝑥5 = 𝑝 𝑥1 𝑝 𝑥2 𝑥1 𝑝 𝑥3 𝑥1:2 𝑝 𝑥4 𝑥1:3 𝑝(𝑥5 ∣ 𝑥1:4)

7

## Time series data

𝑝 𝑥1:10 = 𝑝 𝑥1 𝑝 𝑥2 𝑥1 … 𝑝(𝑥10 ∣ 𝑥1:9) 𝑥 Too simple and naïve modelling (i.e., 10

underfitting modelling) Historical data are not sufficient Many external factors 𝑥9 o Human behaviour, the success of Tesla, climate

change, weather, and so on 𝑥8 𝑥5 𝑥 𝑥 𝑥 6 7 𝑝 𝑥 ∣ 𝑥, … , 𝑥 =? ? ? 𝑥 𝑥 𝑥3 4 𝑡 𝑡−1 𝑡−𝑘 1 2

If we consider the stock prices of the last 𝒌 years, what the stock price of the current year?

𝑝 𝑥1995:2020 = 𝑝 𝑥1995 𝑝 𝑥1996 𝑥1995 … 𝑝(𝑥2020 ∣ 𝑥1995:2019) 𝑥𝑖 is sea level rise in year 𝑖 𝑝 𝑥𝑡 ∣ 𝑥𝑡−1, … , 𝑥𝑡−𝑘 =? ? ?

8

9

# How to model sequential data?

###  Markov models

o Hidden Markov Models (HMM), Dynamic Bayesian Networks (DBN), etc.

first-order Markov

assumption … … 𝑥 𝑥 𝑥𝑇−1 𝑥 𝑥 𝑥 𝑥 𝑥 1 2 𝑇 1 2 𝑇−1 𝑇 𝑝 𝑥1, . . , 𝑥𝑇 𝑝 𝑥, . . , 𝑥

1 𝑇 = 𝑝 𝑥1 𝑝 𝑥2 𝑥1 𝑝 𝑥3 𝑥1:2 … 𝑝(𝑥𝑇 ∣ 𝑥1:𝑇−1) = 𝑝 𝑥 𝑝 𝑥 𝑥 𝑝 𝑥 𝑥 … 𝑝(𝑥 ∣ 𝑥)

1 2 1 3 2 𝑇 𝑇−1

###  Autoregressive model  The hidden state ℎ stores information of past

𝑡 observations 𝑥0:𝑡 = lossy historical summary 𝑦ො 𝑦ො 𝑦ො 𝑦ො𝑡 𝑦ො 0 1 𝑡−1 𝑡+1 𝑦ො𝑇

 ℎ is used to predict 𝑦ෝ.

𝑡 𝑡  ℎ = 𝑔(ℎ, 𝑥) depends on 𝒉 and 𝒙.

𝑡 𝑡−1 𝑡 𝐭−𝟏 𝒕 ℎ0 ℎ ℎ ℎ ℎ ℎ

1 𝑡−1 𝑡 𝑡+1 𝑇

 𝑝 𝑥, . . , 𝑥 = 𝑝 𝑥 𝑝 𝑥 𝑥 𝑝 𝑥 𝑥 … 𝑝 𝑥 𝑥

0 𝑇 0 1 0 2 0:1 𝑇 0 ℎ𝑇−1) 𝑥 𝑥1 𝑥 𝑥 𝑥 0 𝑡−1 𝑡 𝑡+1 𝑥

𝑇

 RNNs belong to the family of autoregressive models

10

# What are time-series and sequential data?

###  Time-series/sequential data = collection of sequential data

### points indexed by time order

o Traditionally, it is often the collection of measurements recorded repeatedly over time

for the same object of interest ▪ e.g., stock market prices, position of the missile, location of the car o However, modern machine learning problems often deal with timestamped data

collected from heterogeneous sources: ▪ e.g., analysing stock market prices together with real-world events, fusion of missile location

bearings + weather information for object tracking ▪ DL revolution has enabled much more sophisticated/complex problems

###  What to model?

o When historical observations influence on the future

…

### 𝑥 𝑥 𝑥𝑇−1 𝑥

1 2 𝑇 The central question is to model the joint distribution 𝑝(𝑥1, 𝑥2, … , 𝑥𝑇)

11

# Sequence Modelling: Summary

###  Pre-Deep Learning Models:

o Typically exploit or make assumption on the model structures to facilitate inference

and model training. o Hidden Markov Models (discrete state)

▪ Factorial Hidden Markov Models, Coupled HMM, Hierarchical HMMs o State-space models (Kalman filters, continuous states):

▪ Hidden state is a continuous random variable, usually with some Gaussian assumptions o Dynamic Bayesian Networks o Conditional Random Fields (CRFs), etc

###  Deep learning:

o Leverage on the power of DNNs to capture time-varying dependency o Autoregressive models, Deep Recurrent Neural Networks o Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) o Seq2Seq models, memory networks, transformer, etc.

## Recurrent NNs (RNNs)

14

# DNN and CNNs

𝑦

Output layer

𝐿 ℎ (𝑥) 𝐿 𝐿 𝑊, 𝑏

Hidden layer

𝐿−1 ℎ (𝑥) 𝐿−1 𝐿−1 𝑊, 𝑏

2 2 𝑊, 𝑏

Hidden layer

1 ℎ (𝑥)

1 1 𝑊, 𝑏

Input layer 0 ℎ (𝑥) = 𝒙  DNNs: build networks for deep static structures  CNNs: mainly builds network to exploit spatial patterns  How to build network for sequences?

15

# Recurrent Neural Networks

[Rumelhart, et al., 1986]

Simplest RNN with two time-slices (no output)

ℎ0 ℎ1

𝑥 𝑥1 0

values known hidden nodes, values to be from data inferred from our models

16

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

## Recurrent Neural Networks

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

17

18

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

𝑡 ℎ𝑡 = 𝑓 ℎ𝑡−1 , 𝑥𝑡 = 𝑓 𝑓 ℎ𝑡−2, 𝑥𝑡−1 , 𝑥𝑡 = ⋯ = summary 𝑥1:𝑡, ℎ0  ℎ can be considered as a kind of lossy summary of the history 𝑥

𝑡 1:𝑡

19

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

20

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

21

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

22

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

## Training RNNs with Back Propagation

## Through Time (BPTT)

25

## Forward propagation through the time

1×𝑖𝑛𝑝𝑢𝑡_𝑠𝑖𝑧𝑒  Input: 𝑥, 𝑥, . . , 𝑥, … ∈ ℝ, 𝑦, 𝑦, … , 𝑦, … ∈ 𝑌

0 1 𝑡 0 1 𝑡 ത ത  ℎ = 𝑥 𝑈 + 𝑏, ℎ = tanh(ℎ)

0 0 0 0

ℎ0𝑉 + 𝑐 (regression)  𝒚ෝ = ቊ

0

softmax (ℎ0𝑉 + 𝑐) (classification) 𝑦0 𝑦 𝑦𝑡−1 𝑦 𝑦

1 𝑡 𝑡+1  Suffer loss: loss = 𝑙( 𝑦ො, 𝑦)

0 0 0 𝑙(𝑦 ො, 𝑦) 𝑙(𝑦 ො, 𝑦) … 𝑙(𝑦 ො, 𝑦) …  for t=1,2,...

0 0 1 1 𝑡 𝑡

Pre-activation: ℎത = ℎ 𝑊 + 𝑥 𝑈 + 𝑏 𝑦ො 𝑦ො 𝑦ො 𝑦ො 𝑦ො 𝑡 𝑡−1 𝑡 0 1 𝑡−1 𝑡 𝑡+1

After-activation: ℎ = tanh(ℎത) 𝑉 𝑉 𝑉 𝑡 𝑡 𝑉 𝑉

𝑊 𝑊 𝑊 ℎ 𝑉 + 𝑐 (regression)

𝑡 ℎ0 ℎ ℎ ℎ ℎ Prediction: 𝒚ෝ = ቊ

1 𝑡−1 𝑡 𝑡+1 𝑡

softmax (ℎ𝑡 𝑉 + 𝑐) (classification) 𝑈 𝑈 𝑈 𝑈 𝑈

Suffer loss: lo𝑠𝑠𝑡 = 𝑙(𝑦 ො𝑡, 𝑦𝑡) 𝑥 𝑥1 𝑥 𝑥 𝑥 0 𝑡−1 𝑡 𝑡+1

❑ Total loss

❑ 𝑇𝑜𝑡𝑎𝑙 − 𝑙𝑜𝑠𝑠 = σ 𝑙𝑜𝑠𝑠

𝑡 𝑡

26

## Back propagation through the time

###  Total loss

o 𝐿 = σ 𝑙𝑜𝑠𝑠

𝑡 𝑡

𝜕𝐿

###  Let us compute ?

𝑙𝑜𝑠𝑠0 𝑙𝑜𝑠𝑠1 𝑙𝑜𝑠𝑠𝑡−1 𝑙𝑜𝑠𝑠 𝑙𝑜𝑠𝑠 𝜕ℎ

𝑡 𝑡+1 0

𝜕𝐿 𝜕𝑙𝑜𝑠𝑠𝑡 𝑦ො 𝑦ො 𝑦ො 𝑦ො𝑡 𝑦ො 0 1 𝑡−1 𝑡+1

###  = σ

𝑡 𝜕ℎ0 𝜕ℎ0 𝑉 𝑉 𝑉 𝑉 𝑉

𝜕𝑙𝑜𝑠𝑠 𝜕 𝑦ො 𝜕ℎ 𝜕ℎഥ 𝜕ℎ 𝜕ℎഥ

𝑡 𝑡 𝑡 𝑡 1 1

### ℎ ℎ ℎ ℎ = σ …

0 1 𝑡−1 𝑡 ℎ𝑡+1 𝑡

𝑊 𝑊 𝜕 𝑦ො 𝜕ℎ 𝜕ℎഥ 𝜕ℎ 𝜕ℎഥ 𝜕ℎ

𝑊 𝑡 𝑡 𝑡 𝑡−1 1 0 𝑈 𝑈 𝑈 𝑈 𝑈

𝑥 𝑥1 𝑥 𝑥 𝑥 0 𝑡−1 𝑡 𝑡+1 𝜕𝐿 𝜕𝑙𝑜𝑠𝑠 𝜕 𝑦ො

𝑡 𝑡 2 2 = ෍ 𝑑𝑖𝑎𝑔 1 − ℎത 𝑊 … 𝑑𝑖𝑎𝑔 1 − ℎത 𝑊

𝑡 1 𝜕ℎ0 𝜕 𝑦ො𝑡 𝜕ℎ𝑡

𝑡

Multiplication of many 𝑾 Pre-activation: ℎത = ℎ 𝑊 + 𝑥 𝑈 + 𝑏 ❖ Gradient vanishing/exploding

𝑡 𝑡−1 𝑡

After-activation: ℎ = tanh ℎത

𝑡 𝑡

ℎ𝑡𝑉 + 𝑐 (regression) Prediction: 𝒚ෝ𝑡 = ቊ

softmax (ℎ𝑡𝑉 + 𝑐) (classification) Suffer loss lo𝑠𝑠𝑡 = ℓ( 𝑦ො𝑡, 𝑦𝑡)

## Training RNN

Summary

###  Use BackProp Through Time (BPTT)

Total loss o Create unrolled network

o Forward pass, store values and calculate loss at each time slide

𝒍𝒐𝒔𝒔 𝒍𝒐𝒔𝒔 l𝒐𝒔𝒔𝒕 𝒍𝒐𝒔𝒔

𝟎 𝟏 𝑳 o Backward propagation through time, starting from the last step, to

𝑦 𝑦 𝑦𝑡 𝑦 0 1 𝐿 calculate the local gradients at each time step.

𝑉

𝑊 ℎ ℎ ℎ ℎ o Sum up local gradients to obtain the end gradients for each 0 1 𝑡 𝐿

𝑈 parameters

𝑥0 𝑥1 𝑥𝑡 𝑥

𝐿 o Update the parameters via GD/SGD

27

## RNNs applications

30

# RNNs Architecture Zoo

###  RNN’s architecture is very flexible for many real-world tasks

𝒚

𝒉 𝒉 𝒉

Sentiment analysis [https://www.twilio.com]

𝒙 𝒙 𝒙 many to one (sentiment analysis, image classification)

Image captioning 𝒚 𝒚 𝒚 [https://www.pcworld.com]

𝒉 𝒉 𝒉

𝒙 one to many (image captioning)

31

# RNNs Architecture Zoo

###  RNNs architecture is very flexible for many real-world tasks

𝒚 𝒚 𝒚 𝒚 𝒚 𝒚

𝒉 𝒉 𝒉 𝒉 𝒉 𝒉 𝒉 𝒉

𝒙 𝒙 𝒙 𝒙 𝒙 𝒙

many to many (1) many to many (2)

(video classification) (machine translation)

… more to come in our next lectures

### Problems of simple RNN and

### rethinking the memory cell

## Problems of RNN

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

34

## Memory cells

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

35

### Long-Short Term Memory Cells (LSTM)

[Hochreiter and Schimdhuber ‘97]

37

## Long Short-Term Memory (LSTM)

[Hochreiter and Schimdhuber ‘97]

𝑦ො 𝑦ො𝑡 𝑦ො 𝑡−1 𝑡+1 RNN

ℎ𝑡−1 ℎ𝑡 ℎ

𝑡+1

𝑥𝑡−1 𝑥𝑡 𝑥

𝑡+1

38

## Long Short-Term Memory (LSTM)

[Hochreiter and Schimdhuber ‘97]

𝑦ො 𝑦ො𝑡 𝑦ො 𝑡−1 𝑡+1 RNN

ℎ𝑡 = 𝑓 ℎ𝑡−1 , 𝑥𝑡 = 𝑓 𝑓 ℎ𝑡−2, 𝑥𝑡−1 , 𝑥𝑡 = ⋯ = summary 𝑥1:𝑡, ℎ0 Short-term ℎ𝑡−1 ℎ𝑡 ℎ

𝑡+1

memory state

𝑥𝑡−1 𝑥𝑡 𝑥

𝑡+1

 ℎ = tanh(ℎ 𝑊 + 𝑥 𝑈 + 𝑏 )

𝑡 𝑡−1 𝑡  𝑦ො = softmax(ℎ 𝑉 + 𝑐)

𝑡 𝑡

39

## Long Short-Term Memory (LSTM)

[Hochreiter and Schmiddhuber ‘97]

𝑦ො𝑡

𝑦ො 𝑦ො𝑡 𝑦ො 𝑡−1 𝑡+1

𝑐 𝑐 RNN 𝑡−1 𝑡

LSTM Short-term ℎ𝑡−1 ℎ𝑡 ℎ

𝑡+1

memory state

ℎ𝑡−1 ℎ𝑡

𝑥𝑡−1 𝑥𝑡 𝑥

𝑡+1

𝑥𝑡

LSTM  Introduced in 1997 by Hohreiter and

Schmidhuber and improved over the years: Sak et. al 2014, Zaremba, 2015, etc.  Address the long-term dependency problem

by introducing a long-term state memory 𝑐𝑡  Help the gradient flow significantly over a

long duration, hence capture long-term dependency

40

## Long Short-Term Memory (LSTM)

[Hochreiter and Schmiddhuber ‘97]

𝑦ො 𝑦ො𝑡 𝑦ො 𝑦ො 𝑦ො 𝑦ො 𝑡−1 𝑡+1 𝑡−1 𝑡 𝑡+1

LSTM RNN

Long-term 𝑐 𝑐 𝑐 Short-term 𝑡−1 𝑡 𝑡+1

memory state ℎ𝑡−1 ℎ𝑡 ℎ

𝑡+1

memory state

Short-term ℎ𝑡−1 ℎ𝑡 ℎ

𝑡+1

memory state 𝑥𝑡−1 𝑥𝑡 𝑥

𝑡+1

𝑥𝑡−1 𝑥𝑡 𝑥

𝑡+1

LSTM  Introduced in 1997 by Hohreiter and

Schmidhuber and improved over the years: Sak et. al 2014, Zaremba, 2015, etc.  Address the long-term dependency problem

by introducing a long-term state memory 𝑐𝑡  Help the gradient flow significantly over a

long duration, hence capture long-term dependency

41

## Long Short-Term Memory (LSTM)

Output Gate

𝑦ො𝑡

𝑦ො𝑡

Forget gate 𝑐 \\ 𝑐𝑡 𝑐 𝑐 𝑡−1 𝑡−1 𝑡

Output gate Input gate

tanh \\ \\ ℎ𝑡−1 ℎ𝑡

𝑓𝑡 𝑔 𝑖 𝑜

𝑡 𝑡 𝑡 sigmoid tanh sigmoid sigmoid 𝑥 FC FC FC FC 𝑡

ℎ ℎ𝑡 𝑡−1

𝑥𝑡  Introduce three gate controllers:

 𝑔 = tanh(ℎ 𝑊 + 𝑥 𝑈 + 𝑏 )

𝑡 𝑡−1 𝑡 o Use sigmoid to ensure outputs’ range between 0 and 1

𝑓 𝑓 𝑓  Forget gate: 𝑓 = 𝜎 𝑥 𝑈 + ℎ 𝑊 + 𝑏

𝑡 𝑡 𝑡−1 o Forget gate 𝑓𝑡 controls how much 𝑐𝑡−1 be ‘forgotten’

𝑖 𝑖 𝑖  Input gate: 𝑖 = 𝜎(𝑥 𝑈 + ℎ 𝑊 + 𝑏)

𝑡 𝑡 𝑡−1 o Input gate 𝑖𝑡 controls which how much 𝑔𝑡 be

 LSTM long-term state: 𝑐 = 𝑓 ⊙ 𝑐 + 𝑔 ⊙ 𝑖 remembered 𝑡 𝑡 𝑡−1 𝑡 𝑡

𝑜 𝑜 𝑜  Output gate: 𝑜 = 𝜎 𝑥 𝑈 + ℎ 𝑊 + 𝑏 o Output gate 𝑜 controls how much long-term 𝑐 should 𝑡 𝑡 𝑡−1

𝑡 𝑡 be carried on to the next time slice:

 LSTM short-term state: ℎ = 𝑜 ⊙ tanh(𝑐)

𝑡 𝑡 𝑡 ▪ to contribute to short-term state: ℎ𝑡

42

## Long Short-Term Memory (LSTM)

Output Gate

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

## Long Short-Term Memory (LSTM)

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

43

## Long Short-Term Memory (LSTM)

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

44

### Gated Recurrent Unit (GRU)

46

## Gated Recurrent Unit

[Cho et. al, 2014]

### ❑ Proposed by Cho et al. 2014

❑ Was introduced for Encoder-Decoder network

### ❑ Can be viewed as a simplified version of the LSTM:

❑ Both long-term 𝑐𝑡 and short-term ℎ𝑡 are merged to a single state ℎ𝑡 ❑ Single update gate controller 𝑧𝑡 is used to control both forget and input gates

❑ = 1: open input gate, close forget gate ❑ = 0: close input gate, open forget gate ❑ i.e., whenever a memory must be stored, the location to be stored will be erased first! ❑ Output gate is removed, but an additional reset gate 𝑟𝑡 controls how much previous state should be carried

forward

Cho et al., Learning Phrase Representation using RNN Encoder-Decoder for Statistical Machine Translation, Proceedings of EMNLP, 2014.

## Gated Recurrent Unit

[Cho et. al, 2014]

 Update gate 𝑧 decides how much the unit

𝑡

ℎ𝑡−1 updates its state:

𝑧 𝑧 𝑧𝑡 = 𝜎 𝑥𝑡𝑈 + ℎ𝑡−1𝑊

𝑧 Update gate 𝑡 sigmoid FC

𝑥𝑡

47

## Gated Recurrent Unit

[Cho et. al, 2014]

 Update gate 𝑧 decides how much the unit

𝑡 updates its state: ℎ𝑡−1

𝑧 𝑧 𝑧𝑡 = 𝜎 𝑥𝑡𝑈 + ℎ𝑡−1𝑊

Res et gate Update gate 𝑟 𝑧𝑡 𝑡  Reset gate controls which parts of the state get

sigmoid sigmoid used to compute the next target state FC FC

𝑟 𝑟 𝑟𝑡 = 𝜎 𝑥𝑡𝑈 + ℎ𝑡−1𝑊

𝑥𝑡

48

## Gated Recurrent Unit

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

49

### Applications of Recurrent Neural Networks

- Sentiment Analysis
- Text generation

51

## Sentiment analysis

###  Movie review dataset

≥ 0.5(pos) and < 0.5(neg)

1. I like that movie (pos:1). Output layer (sigmoid)
2. This is a bad movie to watch (neg:0) RNN RNN RNN RNN RNN
3. I love the movie (pos: 1) cell cell cell cell cell
4. I do not recommend you to watch this movie (neg:0)
5. This movie is fantastic (pos:1) This movie is fantastic \<pad>

How to feed symbolic words to an RNN?

52

## Sentiment analysis

≥ 0.5(pos) and < 0.5(neg)  Movie review dataset

Output layer (sigmoid)

1. I like this movie (pos:1). embedding size =6
2. This is a bad movie to watch (neg:0) RNN RNN RNN RNN RNN cell cell cell cell cell
3. I love this movie (pos: 1)
4. I do not recommend you to watch this movie (neg:0) This movie is fantastic \<pad>
5. This movie is fantastic (pos:1)

 Build up vocabulary

1. Like (index: 1) Embedding matrix (𝑬 [𝟖 × 𝟔])
2. Love (index: 2) 𝑬𝟏 1 2 1.5 -1.2 1.3 1
3. Bad (index: 3) 𝑬𝟐 -1 1.3 -2.5 -1.2 1.6 -1
4. Fantastic (index: 4) 𝑬𝟑 1 3.3 -3.5 -1.0 2.6 1
5. Not (index: 5)
6. Recommend (index: 6) 𝑬 -1 1.3 -2.5 -1.2 1.8 -1 𝟒 𝑬𝟓 -1.2 2.3 -2.5 -1.2 -1.6 1

❑ Not in vocabulary (out of vocabulary bucket: 2)

𝑬𝟔 -1.7 -1.3 -2.5 -1.2 3.6 1.2

1. I, movie, to, pad (index: 7) 𝑬𝟕 -4.2 2.3 -3.5 4.3 1.8 -2
2. This, is, watch (index: 8) 𝑬𝟖 -1.7 -1.3 -4.5 -2.2 -3.6 1

53

## Sentiment analysis

 Movie review dataset ≥ 0.5(pos) and < 0.5(neg)

1. I like this movie (pos:1). Output layer (sigmoid) embedding size =6
2. This is a bad movie to watch (neg:0) 𝑊 𝑊 𝑊 𝑊 ℎ ℎ ℎ ℎ ℎ5
3. I love this movie (pos: 1) 1 2 3 4
4. I do not recommend you to watch this movie (neg:0) × 𝐸 × 𝐸 × 𝐸 × 𝐸 × 𝐸
5. This movie is fantastic (pos:1) 𝟖 𝟕 𝟖 𝟒 𝟕 This movie is fantastic \<pad> [0,0,0,0,0,0,0,1] [0,0,0,0,0,0,1,0] [0,0,0,0,0,0,0,1] [0,0,0,1,0,0,0,0] [0,0,0,0,0,0,1,0]

 Build up vocabulary

1. Like (index: 1) Embedding matrix (𝑬 [𝟖 × 𝟔])
2. Love (index: 2) 𝑬𝟏 1 2 1.5 -1.2 1.3 1
3. Bad (index: 3) 𝑬𝟐 -1 1.3 -2.5 -1.2 1.6 -1
4. Fantastic (index: 4)
5. Not (index: 5) 𝑬𝟑 1 3.3 -3.5 -1.0 2.6 1
6. Recommend (index: 6) 𝑬𝟒 -1 1.3 -2.5 -1.2 1.8 -1 𝑬𝟓 -1.2 2.3 -2.5 -1.2 -1.6 1

❑ Not in vocabulary (out of vocabulary bucket: 2)

𝑬𝟔 -1.7 -1.3 -2.5 -1.2 3.6 1.2

1. I, movie, to, pad (index: 7) 𝑬𝟕 -4.2 2.3 -3.5 4.3 1.8 -2
2. This, is, watch (index: 8) 𝑬𝟖 -1.7 -1.3 -4.5 -2.2 -3.6 1

## Sentiment analysis

≥ 0.5(pos) and < 0.5(neg)

###  The word/item embedding

Output layer (sigmoid) 1×6 o 𝑒1 = 18𝐸 = 𝐸8 ∈ ℝ

𝑊 𝑊 𝑊 𝑊 ▪ 1 × 8 × 8 × 6 = 1 × 6 ℎ ℎ ℎ ℎ ℎ5

1 2 3 4 ▪ 1𝑖 is one-hot vector in which the 𝑖 − 𝑡ℎ element is 1

𝑈 𝑈 𝑈 and the rest are 0.

1×6 Embedding layer 𝑒 𝑒 𝑒 𝑒 𝑒

1 2 3 4 5 o 𝑒2 = 17𝐸 = 𝐸7 ∈ ℝ

embedding size =6 1×6 o 𝑒 = 1 𝐸 = 𝐸 ∈ ℝ × 𝐸 × 𝐸 × 𝐸 × 𝐸 × 𝐸

3 8 8

1×6 𝟖 𝟕 𝟖 𝟒 𝟕 o 𝑒4 = 14𝐸 = 𝐸4 ∈ ℝ

This movie is fantastic \<pad> 1×6 o 𝑒 = 1 𝐸 = 𝐸 ∈ ℝ

5 7 7 [0,0,0,0,0,0,0,1] [0,0,0,0,0,0,1,0] [0,0,0,0,0,0,0,1] [0,0,0,1,0,0,0,0] [0,0,0,0,0,0,1,0]

Embedding matrix (𝑬 [𝟖 × 𝟔])

###  The sequence embedding

𝑬𝟏 1 2 1.5 -1.2 1.3 1 𝑒1

𝑬𝟐 -1 1.3 -2.5 -1.2 1.6 -1 𝑒2 𝑒 5×6 𝑠𝑒𝑞_𝑙𝑒𝑛 ×𝑒𝑚𝑏𝑒𝑑_𝑠𝑖𝑧𝑒 o 𝑒 = 3 ∈ ℝ = ℝ 𝑬𝟑 1 3.3 -3.5 -1.0 2.6 1

𝑒4

𝑬𝟒 -1 1.3 -2.5 -1.2 1.8 -1 𝑒5

𝑬𝟓 -1.2 2.3 -2.5 -1.2 -1.6 1 o Embedding lookup operation

𝑬𝟔 -1.7 -1.3 -2.5 -1.2 3.6 1.2 ▪ Pick rows with indices 8, 7, 8, 4, 7

𝑬𝟕 -4.2 2.3 -3.5 4.3 1.8 -2 𝑬𝟖 -1.7 -1.3 -4.5 -2.2 -3.6 1

54

≥ 0.5(pos) and < 0.5(neg)

## Sentiment analysis Output layer (sigmoid)

𝑊 𝑊 𝑊 𝑊 ℎ ℎ ℎ ℎ ℎ5

###  The word/item embedding 1 2 3 4

1×6 o 𝑒1 = 18𝐸 ∈ ℝ

1×6 o 𝑒 = 1 𝐸 ∈ ℝ Embedding layer

2 7 𝑒 𝑒 𝑒 𝑒 𝑒

1 2 3 4 5 1×6 embedding size =6 o 𝑒3 = 18𝐸 ∈ ℝ

1×6 × 𝐸 × 𝐸 × 𝐸 × 𝐸 × 𝐸 o 𝑒4 = 14𝐸 ∈ ℝ

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

batch_size=2 × 𝐸 × 𝐸 × 𝐸 × 𝐸 × 𝐸 o The embedding for entire batch with

8 7 8 4 7 Input: 𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒 sequences: 3 2 3 5 4

𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒×𝑠𝑒𝑞_𝑙𝑒𝑛 ×𝑒𝑚𝑏𝑒𝑑_𝑠𝑖𝑧𝑒 ℝ

[𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒, 𝑠𝑒𝑞_𝑙𝑒𝑛]

55

## Shape Transformation in RNNs

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

56

57

## Sentiment analysis

(batch_size, seq_len)

(batch_size, seq_len, embed_size)

(batch_size, seq_len, 128) (batch_size, 128)

(batch_size, 1)

58

## Text generation at the character level

𝒙 = 𝑰 𝒙 = 𝒍 𝒙 = 𝒐 𝒙𝟒 = 𝒗 𝒙𝟓 = 𝒆 𝒙 = 𝑫 𝒙𝟕 = 𝑳 𝒙𝟖 = \$ 𝟏 𝟐 𝟑 𝟔

27 neurons

ℎ ℎ ℎ ℎ ℎ ℎ ℎ6 ℎ7 Hidden: 0 1 2 3 4 5

𝒙 =∗ 𝒙𝟏 = 𝑰 𝒙 = 𝒍 𝒙 = 𝒐 𝒙 = 𝒗 𝒙 = 𝒆 𝒙 = 𝑫 𝒙𝟕 = 𝑳 Input: 𝟎 𝟐 𝟑 𝟒 𝟓 𝟔

 Consider a sentence: x=“I love DL” in our dataset

o 𝑥0 =∗, 𝑥1 = 𝐼, 𝑥2 = 𝑙, 𝑥3 = 𝑜, 𝑥4 = 𝑣, 𝑥5 = 𝑒, 𝑥6 = 𝐷, 𝑥7 = 𝐿, 𝑥8 = \$  The joint distribution

o 𝑝 𝑥0, 𝑥1, 𝑥2, … , 𝑥7, 𝑥8 = 𝑝(𝑥0)𝑝 𝑥1 ∣ 𝑥0 𝑝 𝑥2 𝑥0:1 𝑝 𝑥3 𝑥0:2 … 𝑝 𝑥7 𝑥0:6 𝑝(𝑥8 ∣ 𝑥0:7) o 𝑝 𝑥0, 𝑥1, 𝑥2, … , 𝑥7, 𝑥8 = 𝑝 𝑥1 ℎ0 𝑝 𝑥2 ℎ1 𝑝 𝑥3 ℎ2 … 𝑝 𝑥7 ℎ6 𝑝(𝑥8 ∣ ℎ7) o 𝐦𝐚𝐱 log 𝑝 𝑥0, 𝑥1, 𝑥2, … , 𝑥7, 𝑥8 is equivalent to 𝐦𝐢𝐧 −[log 𝑝 𝑥1 ℎ0 + log 𝑝 𝑥2 ℎ1 + log 𝑝 𝑥3 ℎ2 + ⋯ + log 𝑝 𝑥8 ℎ7 ] o Given ℎ1, we need to maximize the probability to predict 𝑥2 and so on. o Cast to the problem of prediction to train our RNN

▪ The loss is sum of loss at each time step

59

## Summary

## How to define and model sequential data

## Recurrent neural networks (RNNs)

## Memory cells

### o Long-short term memory (LSTM) cell

### o Gated recurrent unit (GRU) cell

## Applications of RNNs

### o Sentiment analysis

### o Text generation

## Thanks for your attention!