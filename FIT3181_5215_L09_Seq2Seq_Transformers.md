COPYRIGHT WARNING: Copyright in these original lectures is owned by Monash University. You may transcribe, take notes, download or stream lectures for the purpose of your research and study only. If used for any other purpose, (excluding exceptions in the Copyright Act 1969 (Cth)) the University may take legal action for infringement of copyright.

Do not share, redistribute, or upload the lecture to a third party without a written permission!

## FIT3181/5215 Deep Learning

Week 09: Deep Learning for Sequential Data (II):

Seq2Seq and Transformers

CE/Lecturer: Dr Trung Le (Clayton)

Lecturer: Dr Binh Nguyen (Clayton) Lecturer: Dr Ting Fung Fung (Malaysia) Lecturer: Dr Arghya Pal (Malaysia) Lecturer: Dr Sailaja Rajanala (Malaysia)

Email: trunglm@monash.edu

Department of Data Science and AI Faculty of Information Technology, Monash University, Australia

2

# Outline

##  Encoder-Decoder models

##  Sequence to sequence models

o Machine Translation and Image Captioning

##  Attention mechanism

o Global attention and Local attention

##  Transformer and BERT

o Self-Attention Mechanism & Multi-Head Self-Attention Mechanism o Pre-trained Language Models: BERT

# Encoder-Decoder Models

- Image Captioning
- Machine Translation

# Encoder-Decoder Models: Application Motivations

Context or encode 𝒄

Target structured data Source structured data

𝒄 = 𝒉𝒍𝒂𝒔𝒕

𝒄

Machine translation Image captioning

Many more applications ❖ C++ programs to Java programs, texts to images, question answering (questions to answers), and etc.

4

# Sequence to Sequence Models

7

# Introduction

##  Problem statement of seq2seq

o Source sequence: 𝑥 = 𝑥1, 𝑥2, … , 𝑥𝑇 where 𝑥𝑖𝜖𝑉𝑥

𝑥 o Target sequence: y = 𝑦1, 𝑦2, … , 𝑦𝑇 where 𝑦𝑖𝜖𝑉𝑦

𝑦 o Training set:

𝑁 (1) (1) (2) (2) (𝑁) (𝑁) (𝑖) (𝑖) ▪ 𝔇 = 𝑥, 𝑦, 𝑥, 𝑦, … , 𝑥, 𝑦 = 𝑥, 𝑦

𝑖=1 (𝑖) ▪ 𝑆𝑥: set of possible source sequences ( 𝑥 ⊆ 𝑆𝑥)

(𝑖) ▪ 𝑆𝑦: set of possible target sequences ( 𝑦 ⊆ 𝑆𝑦) o Task

▪ Learn a function 𝑓: 𝑆𝑥 → 𝑆𝑦

##  Applications

o Machine Translation o Image Captioning

8

# Machine translation

Paper: paper link

###  Translate a sentence in a language to another

### language

o Input: He loved to eat. (English) o Output: Il aimait manger. (French)

𝐼𝑙 𝑎𝑖𝑚𝑎𝑖𝑡 𝑚𝑎𝑛𝑔𝑒𝑟 < 𝑬𝑶𝑺 >......... ... vocabulary size+1

ℎ1 ℎ2 ℎ3 𝑐 = ℎ4 𝑞 𝑞 𝑞 𝑞

0 1 2 3

𝐻𝑒 𝑙𝑜𝑣𝑒𝑑 𝑡𝑜 𝑒𝑎𝑡 \<BOS> 𝐼𝑙 𝑎𝑖𝑚𝑎𝑖𝑡 𝑚𝑎𝑛𝑔𝑒𝑟

9

# Encoder-decoder model for seq2seq

### Fixed context vector

###  Encoder

o Produces the context vector 𝒄 = 𝒉𝑻 of the input sequence

𝒙 o Context vector c summarizes input sequence [𝐱𝟏, … , 𝐱𝐓 ].

𝐱

###  Decoder

o Decodes the encode c to the output sequence

###  Special symbols

o \<EOS> signifies the end of a sequence o \<BOS> signifies the beginning of a sequence

ℎ ℎ ℎ ℎ ℎ 𝑐 = ℎ𝑇 1 2 𝑖−1 𝑖 𝑇𝑥−1 𝑥

𝑥 𝑥 𝑥 𝑥 𝑥 𝑥𝑇 1 2 𝑖−1 𝑖 𝑇𝑥−1 𝑥

𝑦𝑗+1 𝑦𝑗 𝑦 𝑦 𝑦 \<EOS> 3 2 1

𝑞𝑇 𝑞 𝑞 𝑞 𝑞 𝑞

𝑦 𝑗 𝑗−1 2 1 0

𝑦𝑇 𝑦 𝑦𝑗−1 𝑦 𝑦 \<BOS>

𝑦 𝑗 2 1

10

# Training of seq2seq

𝑦1 𝑦2 𝑦 \<EOS>

𝑗

ℎ 𝒄 = 𝒉𝑻 𝒒 𝑞 𝑞 𝑞 1 𝒙 𝟎 1 𝑗−1 𝑇

𝑦

𝑥 𝑥𝑇 \<BOS> 𝑦 𝑦 1 𝑥 1 𝑗−1 𝑦

𝑇𝑦 ❑ We need to maximize the log-likelihood:

max J θ = ෍ log 𝑃 𝒚|𝒙, 𝜃 θ

𝒙,𝒚 𝜖𝔇 where 𝜃 = [𝜃𝑒, 𝜃𝑑] and 𝜃𝑒, 𝜃𝑑 are encoding and decoding parameters respectively. ❑ Product rule:

𝑃 𝒚|𝒙, 𝜃 = 𝑃 𝒚𝟏:𝑻 𝒙𝟏:𝑻, 𝜃 = 𝑃 𝒚𝟏:𝑻 𝒄, 𝜃

𝒚 𝒙 𝒚

𝑇𝑦 = 𝑃 𝒚𝟏 𝒄, 𝜃 𝑃 𝒚𝟐 𝒚𝟏, 𝒄, 𝜃 … 𝑃 𝒚𝒋 𝒚𝟏:𝒋−𝟏, 𝒄, 𝜃 … 𝑃 𝒚𝑻 𝒚𝟏:𝑻 −𝟏, 𝒄, 𝜃 = ෑ 𝑃 𝒚𝒋 𝒚𝟏:𝒋−𝟏, 𝒄, 𝜃

𝒚 𝒚

𝑗=1 𝑇𝑦 𝑇𝑦 log 𝑃 𝒚|𝒙, 𝜃 = log 𝑃 𝒚|𝒄, 𝜃 = ෍ log 𝑃 𝒚𝒋|𝒚𝟏:𝒋−𝟏, 𝒄, 𝜃 = ෍ log 𝑃 𝒚𝒋|𝒒𝒋−𝟏, 𝒄, 𝜃

𝑗=1 𝑗=1 o We can compute 𝑃 𝒚𝒋 𝒒𝒋−𝟏, 𝒄 = 𝑔 𝒚𝒋, 𝒒𝒋−𝟏, 𝒄 where 𝑔 is a nonlinear, potentially multi-layered NN that

outputs the probability of 𝑦𝑗.  Pay attention on how 𝑐 is used in every step during decoding

11

# Training of seq2seq

### ❑ We need to maximize the log-

### likelihood:

(1) [Sutskever et al. 2014] max J θ = ෍ log 𝑃 𝒚|𝒙, 𝜃 θ

𝒙,𝒚 𝜖𝔇 where 𝜃 = [𝜃𝑒, 𝜃𝑑] and 𝜃𝑒, 𝜃𝑑 are encoding and decoding parameters, respectively.

### ❑ Product rule:

log 𝑃 𝒚|𝒙, 𝜃 = log 𝑃 𝒚|𝒄, 𝜃 = 𝑇𝑦 σ log 𝑃 𝒚 |𝒚, 𝒄, 𝜃 = 𝑗=1 𝒋 𝟏:𝒋−𝟏 𝑇𝑦 σ log 𝑃 𝒚 |𝒒, 𝒄, 𝜃 𝑗=1 𝒋 𝒋−𝟏 ❖ 𝒒𝒋−𝟏 contains the information of 𝒄. ❖ (1): 𝑃 𝒚𝒋|𝒒𝒋−𝟏, 𝒄, 𝜃 = 𝑃 𝒚𝒋|𝒒𝒋−𝟏, 𝜃

(2) 𝑞𝑗−1 = 𝑓(𝑞𝑗−2, 𝑦𝑗−1) ❖ (2): 𝑃 𝒚 |𝒒, 𝒄, 𝜃 = 𝑃 𝒚 |𝒒, 𝜃 Big drawback: the context vector 𝒄 is fixed across timesteps.

𝒋 𝒋−𝟏 𝒋 𝒋−𝟏 𝑞𝑗−1 = 𝑓(𝑞𝑗−2, 𝑐𝑜𝑛𝑐𝑎𝑡(𝑐, 𝑦𝑗−1)) ❖ 𝑓 is a memory cell (e.g., LSTM or GRU)

12

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

##  Given a trained model and input sequence 𝒙

o We need to infer the corresponding output sequence 𝒚

##  Two common strategies

o Greedy Decoding and Beam Search Decoding

13

# Inference

### Greedy decoding

###  Greedy Decoding

o Given x, find word 𝒚𝟏 with highest probability o Given 𝒚𝟏 and x, find word 𝒚𝟐 with highest probability o … o Stop when seeing \<EOS>

𝑃(𝑦 = ∘∣ 𝑐) 𝑃(𝑦 = ∘∣ 𝑐) 𝑃(𝑦𝑇 +1 = ∘∣ 𝑐)

1 2 𝑦

𝑦1 𝑦2 \<EOS>

ℎ 𝑐 = ℎ𝑇 𝑞 𝑞 𝑞 1 𝑥 0 1 𝑇𝑦

𝑥 𝑥𝑇 \<BOS> 𝑦 𝑦 1 𝑥 1 𝑇

𝑦

14

# Inference

### Beam search

## ❑ Beam Search Decoding with beam width 𝒌

o Given x, find k candidates for 𝒚𝟏 with highest probability o Given x, for each candidate 𝒚𝟏, find k candidates for word 𝒚𝟐 with highest probability o Pick top-k sequences 𝑦1𝑦2 with highest joint probability o For each 𝑦1𝑦2, find k candidates for word 𝒚𝟑 with highest probability o Pick top-k sequences 𝑦1𝑦2𝑦3 with highest joint probability o ….. o Stop when see \<EOS> on each beam o Finally, pick 1 sequence with highest probability from top-k sequences

𝑦1 𝑦2 \<EOS>

ℎ 𝑐 = ℎ𝑇 𝑞 𝑞 𝑞 1 𝑥 0 1 𝑇𝑦

𝑥 𝑥𝑇 \<BOS> 𝑦 𝑦 1 𝑥 1 𝑇

𝑦

15

# Inference

### Beam search

, (𝑝11) Hi

How (𝑝12) (𝑝1)

What (𝑝13)

sure (𝑝21) Input Yes

certainly (𝑝22) 𝑥 (𝑝2)

Beam width = 3 I (𝑝23) - We always

choose three I (𝑝31)

sentences with Please see (𝑝) highest joint

32 (𝑝3) probabilities

come (𝑝33)

16

# Inference

### Beam search

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

17

# Attention mechanism

20

# Attention mechanism

 So far, the input sequence is summarised by a

single context c !

 Fixed-length context could be problematic as

it is easily overwhelmed by long inputs or long outputs o The fixed-length context c might not be

powerful enough to capture long input sequences

 Some specific items in an input sequence

might be more relevant and contributing in Input Output

Sequence Encoder c Decoder Sequence generating a given item in output sequence.

“bottleneck”, context c

- Gratefully acknowledge the excellent visualizations used from Jay Alammar blog at:

- https://jalammar.github.io/
- https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/

# Attention mechanism

Bahdanau, Cho, Bengio, Neural Machine Translation by Luong, Pham, Manning, Effective Approach Attention-based Neural Jointly Learning to Align and Translate, ICLR 2015 Machine Translation, EMNLP, 2015

### Attention mechanism allows the decoding network to refer to the input.

o Global attention

▪ Use all input hidden states of the encoder when deriving the context 𝑐𝑡. o Local attention

▪ Use a selective window of input hidden states of the encoder when deriving the context 𝑐𝑡.

21

22

# Global attention

## ❑ Main idea

o Consider all input hidden states of the encoder when

3 deriving the context: 𝑐 = σ 𝑎 (𝑠) ℎ keys

𝑡 𝑠=1 𝑡 𝑠

##  Alignment weights:

exp 𝑠𝑐𝑜𝑟𝑒 𝑞𝑡, ℎ𝑠 𝑎𝑡 𝑠 = align 𝑞𝑡, ℎ𝑠 =

σ exp 𝑠𝑐𝑜𝑟𝑒 𝑞, ℎ 𝑠′ 𝑡 𝑠′

query where 𝑞𝑡 is current target state, ℎ𝑠 is each source states.

##  Alignment score function:

T 𝑞𝑡 ℎ𝑠 𝑑𝑜𝑡 𝑝𝑟𝑜𝑑𝑢𝑐𝑡 T 𝑠𝑐𝑜𝑟𝑒 𝑞𝑡, ℎ𝑠 = ൦ 𝑞 𝑊 ℎ 𝑔𝑒𝑛𝑒𝑟𝑎𝑙 𝑚𝑒𝑡𝑟𝑖𝑐

𝑡 𝑎 𝑠 T 𝑐𝑜𝑛𝑐𝑎𝑡 𝑣𝑎 𝑡𝑎𝑛ℎ 𝑊𝑎 𝑞𝑡; ℎ𝑠

23

# Global attention

𝑦ො𝑡

##  Context vector:

3 𝑐𝑡 = ෍ 𝑎𝑡(𝑠) ℎ𝑠

𝑠=1

##  Attentional hidden state:

𝑞෤𝑡 = 𝑡𝑎𝑛ℎ 𝑊𝑐 𝑐𝑡; 𝑞𝑡

##  Predictive distribution:

𝑝 𝑦𝑡|𝑦\<𝑡, 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 𝑊𝑠 𝑞෤𝑡

24

# Global attention

### Example

## 1\. Convert into alignment weights.

𝑒𝑥𝑝 𝑠𝑐𝑜𝑟𝑒 𝑞𝑡, ℎ𝑠 𝑎𝑡 𝑠 =

σ 𝑒𝑥𝑝 𝑠𝑐𝑜𝑟𝑒 𝑞, ℎ 𝑞 𝑠′ 𝑡 𝑠′ 𝑡

## 2\. Build context vector: weighted

## average

𝑐𝑡 = ෍ 𝑎𝑡(𝑠) ℎ𝑠

𝑠

## 3\. Compute the next hidden state.

𝑞෤𝑡 = 𝑡𝑎𝑛ℎ 𝑊𝑐 𝑐𝑡; 𝑞𝑡

𝑞𝑡

## 4\. Predictive distribution:

𝑝 𝑦𝑡|𝑦\<𝑡, 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 𝑊𝑠 𝑞෤𝑡

25

# Global attention

### Drawback

## ❑ Drawback: employ all items on

## the source side for deriving each

## target item.

## o expensive computation

## o impractical to translate longer

## sequences

## Local attention

26

# Local attention

##  Main idea

### o Selectively focuses on a small window of

### context and is differentiable.

##  𝑐 is then derived as a weighted

𝑡

## average over the set of source hidden

## states within the window [𝒑 −

𝒕

## 𝑫; 𝒑 + 𝑫 ], 𝐷 is empirically selected.

𝒕

27

# Local attention

 Main idea

o Selectively focuses on a small window of context and is

differentiable  For current target word, predict aligned position 𝑝

𝑡

𝑇 o 𝑝𝑡 = 𝑆 ∙ sigmoid 𝑣𝑝 tanh 𝑊𝑝𝑞𝑡

where 𝑊𝑝 and 𝑣𝑝 are the model parameters, 𝑆 is the source sentence length, and hence 0 ≤ 𝑝𝑡 ≤ 𝑆  Alignment weights:

2 𝑠−𝑝𝑡

𝐷 o 𝑎 𝑠 = align 𝑞, ℎ exp − with 𝜎 = Τ

𝑡 𝑡 𝑠 2 2

2𝜎

exp 𝑠𝑐𝑜𝑟𝑒 𝑞𝑡,ℎ𝑠 o align 𝑞𝑡, ℎ𝑠 = (temporary alignment

σ exp 𝑠𝑐𝑜𝑟𝑒 𝑞𝑡,ℎ𝑠′ 𝑠′ weights)

###  Context is now computed as before but

### within a windows of size D, centered at 𝑝𝑡

28

# Other types of attention mechanism

###  Self-attention/Multiple Heads/Transformer Networks

o "Attention is all you need. Vaswani, Ashish, et al. ." Advances in neural information

processing systems, 2017.

###  Pyramidal encoders

o “Listen Attend and Spell”. William Chan, Navdeep Jaitly, Quoc Le, Oriol Vinyals,

ICASSP 2015.

###  Hierarchical Attention

o “Learning efficient algorithms with hierarchical attentive memory”. Andrychowicz,

Marcin, and Karol Kurach, arXiv:1602.03218, 2016.

###  Soft/Hard Attention

o “Show, attend and tell: Neural image caption generation with visual attention”. Xu,

Kelvin, et al, ICML 2015.

# Transformer and BERT

# Question

# What does GPT stands for in ChatGPT?

# Transformer and BERT

Paper: paper link Paper: paper link

### Transformer proposed self-attention mechanism

### BERT was developed based on Transformer

o BERT currently achieves SOTA performances in most tasks of NLP.

### Driving force of generative AI revolution

32

# Transformer Motivations

### Problem using RNNs for Seq2Seq models

o Slow due to sequential nature o Poor long-range dependency modelling

### Transformer

o Enable parallel computation, hence exploit the power of GPU. How? o Remove the dependency between words o But position/location matters → resolve by positional encoding o Extremely good at capturing long-range dependency

33

# Transformer – General Architecture

[Source: main paper]

Decoder Encoder

[Source: https://jalammar.github.io/illustrated-transformer/]

### Some building blocks

o Positional encoding Model architecture

o Self-attention and Multi-Head Self-attention Attention o Masked Multi-Head Attention o Residual add and layer norm

34

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

35

[Source: https://jalammar.github.io/illustrated-transformer/]

# Transformer – Self Attention

𝑧1 = 0.88 × 𝑣1 + 0.12 × 𝑣2 Self attention among a sequence of items/tokens o Keys and queries for computing self attention weights 𝑸 𝑲 𝑽 𝑄 𝐾 𝑉 Multiplying 𝒙, 𝒙 with 𝑾, 𝑾, 𝑾 to gain o 𝑊, 𝑊, 𝑊 are learnable matrices.

𝟏 𝟐 queries, keys, and values.

36

# Transformer

Self-Attention

For each token 𝑥𝑖, calculate its query, key, and value

𝑄 𝐾 𝑉 𝑞𝑖 = 𝑥𝑖𝑊, 𝑘𝑖 = 𝑥𝑖𝑊, 𝑣𝑖 = 𝑥𝑖𝑊

𝑥1 Matrix/stacked version (𝑋 = …, 𝐿 = 𝑠𝑒𝑞_𝑙𝑒𝑛)

𝑥𝐿 𝑄 𝐾 𝑉 𝑄 = 𝑋𝑊, 𝐾 = 𝑋𝑊, 𝑉 = 𝑋𝑊 Calculate the attention probabilities between query and key o Scaled dot-product attention

𝑇 𝑄𝐾 𝐴 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥

𝑑𝑘 Take a weighted sum of values

𝑍 = 𝐴𝑉

37

[batch_size,seq_len, d_model] 𝑍

# Self-Attention 𝑂

Output: 𝑍 = 𝑍𝑊

𝑂 𝑊 𝑍 = 𝐴𝑉 [batch_size,seq_len, 𝑑𝑉]

seq_len 𝑍

𝐴 [batch_size,seq_len, seq_len] Attention probs: 𝐴 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥 𝐵, 𝑑𝑖𝑚 = 2

𝑇 [batch_size,seq_len, seq_len] 𝑄𝐾 Attention scores: 𝐵 = 𝐵

𝑑 seq_len 1 1 1 1 1 𝐾 𝐵 = 𝑄 𝑖, : 𝐾: , 𝑗 → 𝑠𝑖𝑚(𝑥, 𝑥)

𝑖𝑗 𝑖 𝑗

𝑄 𝐾 𝑉 𝑄 = 𝑋𝑊 [𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒, 𝑠𝑒𝑞_𝑙𝑒𝑛, 𝑑𝑄 = 𝑑𝐾] 𝐾 = 𝑋𝑊 [𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒, 𝑠𝑒𝑞_𝑙𝑒𝑛, 𝑑 = 𝑑 ] 𝑉 = 𝑋𝑊 [𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒, 𝑠𝑒𝑞_𝑙𝑒𝑛, 𝑑𝑉]

𝑄 𝐾

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

38

# Transformer

Multi-Head Self-Attention

### Main idea:

o Perform self-attention multiple times in parallel and combine results

### Multiple attention heads through multiple Q, K, V matrices

### Each attention head performs attention independently

o Allow attending to parts of the sequence differently

39

# Transformer – Self Attention/Multi-Head Attention

[Source: https://jalammar.github.io/illustrated-transformer/]

40

# Transformer – Self Attention/Multi-Head Attention

Matrix calculation for One-Head Matrix calculation for Multi-Head

[Source: https://jalammar.github.io/illustrated-transformer/]

41

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

42

# Transformer- Pointwise

# Feed-forward NN

𝟎 𝑾

## LayerNorm( )

Pointwise Feed-forward NN 𝑍ҧ = 𝑅𝑒𝐿𝑈 𝑊 𝑍 + 𝑏 𝑊 + 𝑏

1 1 2 2

𝑍ҧ 𝑍ҧ

## FFN FFN FFN … FFN LayerNorm()

𝑧 𝑧2 𝑧3 𝑧𝑇 1

Multi-head Self-attention

Next Encoder 𝑥 𝑥 𝑥3 … 𝑥 1 2 𝑇 43

# Transformer – Positional Encoding

Use to capture position of a word in a sentence o Transformer has no recurrence and no convolution. o For the model to make use of the order of the sequence, some

information about the relative or absolute position of the tokens in the sequence needs to be injected

2𝑖/𝑒𝑚𝑏𝑒𝑑_𝑠𝑖𝑧𝑒 𝑃𝐸(𝑝𝑜𝑠,2𝑖) = sin(𝑝𝑜𝑠/1000)

2𝑖/𝑒𝑚𝑏𝑒𝑑_𝑠𝑖𝑧𝑒 𝑃𝐸(𝑝𝑜𝑠,2𝑖+1) = cos(𝑝𝑜𝑠/1000)

𝑃𝐸(1,1) 𝑃𝐸 𝑃𝐸(1,3) 𝑃𝐸 𝑃𝐸 𝑃𝐸 𝑃𝐸 𝑃𝐸 𝑃𝐸 𝑃𝐸 𝑃𝐸

(1,2) (2,1) (2,2) (2,3) (3,1) 𝑃𝐸(3,2) (3,3) (4,1) (4,2) (4,3)

embed_size=3 𝐸 𝐸 𝐸 𝐸 𝐸 𝐸 𝐸 𝐸 𝐸 𝐸81 𝐸82 𝐸83

31 32 33 21 22 23 51 52 53 pos=1 pos=2 pos=3 pos=4

I am a student index=3 index=2 index=5 index=8

𝐸3 𝐸 𝐸 𝐸 No information of word order

2 5 8 I am a student

All words or word embeddings are inputted simultaneously to the Transformer

[Source: Dive into DL] 44

45

# Transformer Decoder

Putting them together

Cross-Attention

Decoder

Encoder

Decoder Output target vocab size + 1

# Transformer Decoder predict

𝑙𝑜𝑔𝑖𝑡1 …

J'adore Putting them together 𝑙𝑜𝑔𝑖𝑡 … predict

2 les J'adore les mathématiques \<EOS>

𝑙𝑜𝑔𝑖𝑡3 …

predict mathématiques 𝑙𝑜𝑔𝑖𝑡4 …

predict \<EOS>

𝑥1(𝑁) 𝑥2(𝑁)

Encoder Output

𝑦1(𝑁) Decoder needs to be a 𝑥3(𝑁)

Cross-Attention casual model

𝑦2(𝑁)

- 𝑦3(𝑁) only depends

𝑥4(𝑁)

𝑦 (𝑁) on 𝑦1 0 , 𝑦2 0 , 𝑦3(0) 3

- Masking self-attention

𝑦4(𝑁) Encoder

Decoder

𝑥1(0) 𝑦 (0)

1 𝑥2(0) 𝑦 (0)

2 𝑥3(0) 𝑦 (0)

3 𝑥4(0) 𝑦 (0)

4 Encoder Input Decoder Input

I love mathematics \<pad> \<BOS> J'adore les mathématiques

46

47

# Transformer Decoder

Putting them together

Cross-Attention

Decoder

Encoder

[batch_size,seq_len, 𝑑𝑉]

𝑂

# Masked Self-Attention 𝑊 𝑍

𝑂 𝑍 Output: 𝑌 = 𝑍𝑊

seq_len Casual model [batch_size,seq_len, d_model]

𝑍 = 𝐴𝑉

0 0 𝐴

[batch_size,seq_len, seq_len] Attention probs: 𝐴 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝐵, 𝑑𝑖𝑚 = 2)

-inf -inf [batch_size,seq_len, seq_len] mask 𝐵

Filled attention scores: 𝐵 = 𝐵. 𝑚𝑎𝑠𝑘_𝑓𝑖𝑙𝑙(𝑚𝑎𝑠𝑘 == 0, −𝑖𝑛𝑓)

[batch_size,seq_len, seq_len]

1 1 1 1 1 𝑇 𝐵 = 𝑄 𝑖, : 𝐾: , 𝑗 → 𝑠𝑖𝑚(𝑥, 𝑥) 𝑄𝐾 𝑖𝑗 𝑖 𝑗 seq_len Attention scores: 𝐵 =

seq_len 𝑑𝐾

𝑄 𝐾 𝑉 𝑄 = 𝑌𝑊 [𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒, 𝑠𝑒𝑞_𝑙𝑒𝑛, 𝑑𝑄 = 𝑑𝐾] 𝐾 = 𝑌𝑊 [𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒, 𝑠𝑒𝑞_𝑙𝑒𝑛, 𝑑 = 𝑑 ] 𝑉 = 𝑌𝑊 [𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒, 𝑠𝑒𝑞_𝑙𝑒𝑛, 𝑑 ]

𝑄 𝐾 𝑉

seq_len seq_len seq_len

𝑑𝑉 𝑑 = 𝑑 𝑑𝐾 = 𝑑𝑄 𝑄 𝐾

𝑉 𝐾 𝑊 𝑄 𝑊 𝑌 𝑊

𝑑_𝑚𝑜𝑑𝑒𝑙 𝑑_𝑚𝑜𝑑𝑒𝑙 𝑑_𝑚𝑜𝑑𝑒𝑙 2

1 𝑌 𝑌

[batch_size,seq_len, d_model]

seq_len

[batch_size,seq_len, size, d_model]

batch_size

Embedding Layer (E: [vocab_size, embed_size]) BOS Elle aime livres

8 7 9 4 Input: BOS Je suis 8 2 5 0

ici

[𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒, 𝑠𝑒𝑞_𝑙𝑒𝑛]

48

49

# Transformer Decoder

Putting them together

Cross-Attention

Decoder

Encoder

[batch_size,tgt_seq_len,d_model] [batch_size,tgt_seq_len, 𝑑𝑉]

# Cross-Attention 𝑍

𝑂 𝑍 Output: 𝑌 = 𝑍𝑊

𝑂 𝑊 tgt_seq_len Casual model 𝑍 = 𝐴𝑉

0 0 𝐴

[batch_size,tgt_seq_len, src_seq_len] Attention probs: 𝐴 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝐵, 𝑑𝑖𝑚 = 2) mask -inf -inf [batch_size,tgt_seq_len, src_seq_len]

𝐵 Filled attention scores: 𝐵 = 𝐵. 𝑚𝑎𝑠𝑘_𝑓𝑖𝑙𝑙(𝑚𝑎𝑠𝑘 == 0, −𝑖𝑛𝑓)

[batch_size,tgt_seq_len, src_seq_len]

1 1 1 1 1 𝑇 𝐵 = 𝑄 𝑖, : 𝐾: , 𝑗 → 𝑠𝑖𝑚(𝑥, 𝑥) 𝑄𝐾 𝑖𝑗 𝑖 𝑗 tgt_seq_len Attention scores: 𝐵 =

tgt_seq_len 𝑑𝐾

𝑄 𝐾 𝑉 𝑄 = 𝑌𝑊 [𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒, 𝑡𝑔𝑡_𝑠𝑒𝑞_𝑙𝑒𝑛, 𝑑𝑄 = 𝑑𝐾] 𝐾 = 𝑋𝑊 [𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒, 𝑠𝑟𝑐_𝑠𝑒𝑞_𝑙𝑒𝑛, 𝑑 = 𝑑 ] 𝑉 = 𝑋𝑊 [𝑏𝑎𝑡𝑐ℎ_𝑠𝑖𝑧𝑒, 𝑠𝑟𝑐_𝑠𝑒𝑞_𝑙𝑒𝑛, 𝑑 ]

𝑄 𝐾 𝑉

seq_len seq_len seq_len

𝑑𝑉 𝑑 = 𝑑 𝑑𝐾 = 𝑑𝑄 𝑄 𝐾

𝑉 𝐾 𝑊 𝑄 𝑊 𝑊

𝑑_𝑚𝑜𝑑𝑒𝑙 𝑑_𝑚𝑜𝑑𝑒𝑙 𝑑_𝑚𝑜𝑑𝑒𝑙

𝑋 𝑌 𝑋

[batch_size,tgt_seq_len,d_model] [batch_size,src_seq_len,d_model]

Decoder current tokens Encoder output

50

## Pretrained Language Models

# Bidirectional Encoder Representations from

# Transformers (BERT)

Developed based on the architecture of Transformer Two steps in the framework o Pretrain and fine tuning Pretrain o The model is trained on unlabeled data over different pre-training tasks o Masked word prediction and next sentence prediction Fine tuning o The model for a downstream task is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labeled data

from the downstream tasks. o Some downstream tasks, for example, sequence-level classification, predicting the answer text span in the passage, and sentence-pair

completion

52

# BERT - Pretrain Predict Predict

Predict next sentence masked word masked word

Task 1: Masked words prediction o 15% of the words are masked at random o Not all tokens were masked in the same way. Given a

masked word, it happens

Sequence Token ▪ With an 80% of chance, this word is replaced by the \<MASK> representation representations

\<MASK> \<MASK> token ▪ With a 10% of chance, this word is replaced by a random word ▪ With a 10% of change, this word is left intact o Predict the indices of masked words on top of

representations of those words.

Task 2: Next sentence prediction

Start symbol Separation symbol o When choosing the sentences A and B for each pretraining

example, 50% of the time B is the actual next sentence following A (labeled as IsNext), and 50% of the time it is a random sentence from the corpus (labeled as NotNext). o The final hidden vector of the special [CLS] token as 𝐶 ∈

𝐻 ℝ is used to predict two labels: IsNext and NotNext.

53

# BERT – Fine Tuning

Sequence level classification

### Input a single sentence to pretrain BERT.

ℎ

### Observe the representations 𝐶, 𝑇1, … , 𝑇𝑁 ∈ ℝ

ℎ o 𝐶 ∈ ℝ aggregates the information of tokens/words 1,2, … , 𝑁,

hence can be viewed as a sentence representation.

### On top of the sentence representation 𝐶, we

### predict sentence label

𝑀×ℎ o 𝑊 ∈ ℝ is an additional weight matrix where 𝑀 is the

number of classes. o Prediction probabilities 𝑃 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑊𝐶)

### Fine tuning 𝑾 and pretrain BERT parameters.

54

# BERT – Fine Tuning

### Predicting the answer text span

Task description o Input Question:

Where do water droplets collide with ice crystals to form precipitation? o Input Paragraph:

.... Precipitation forms as smaller droplets coalesce via collision with other raindrops or ice crystals within a cloud. ... o Output Answer:

within a cloud

❖ Represent the input question and input paragraph as a single packed sequence ❖ The question uses the A embedding and the paragraph uses the B embedding

ℎ ℎ ❖ New parameters to be learned in fine-tuning are start vector 𝑆 ∈ ℝ and end vector 𝐸 ∈ ℝ ❖ The probability of word i being the start of the answer span

𝑇 exp{𝑆 𝑇𝑖} ❖ 𝑃𝑖 =

𝑇 σ exp{𝑆 𝑇 } 𝑗∈𝑖𝑛𝑝𝑢𝑡 𝑝𝑎𝑟𝑎𝑔𝑟𝑎𝑝ℎ 𝑗 ❖ The probability word i being the start and word j being the end of the answer span

𝑇 𝑇 exp{𝑆 𝑇𝑖+𝐸 𝑇𝑗} ❖ 𝑃𝑖,𝑗 =

𝑇 𝑇 σ exp{𝑆 𝑇 ′+𝐸 𝑇 }

𝑖′,𝑗′ :𝑗′>𝑖′ 𝑖 𝑗′ ❖ Train 𝑺, 𝑬 and pretrain BERT parameters by maximizing log-likelihood.

55

56

# Why does pretraining-then-finetuning work?

### 1\. Transformer architecture

### 2\. Large scale training data

### 3\. Large number of parameters

# Further Reading Recommendation

Seq2Seq o https://jalammar.github.io/ o https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/ o https://trungtran.io/2019/02/27/neural-machine-translation-with-tensorflow-model-creation/ Transformer o https://ai.plainenglish.io/transformers-visual-guide-153af370693f o https://jalammar.github.io/illustrated-transformer/ BERT o https://jalammar.github.io/illustrated-bert/ o https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270

### ViT

o https://arxiv.org/pdf/2010.11929.pdf

### Swin

o https://arxiv.org/pdf/2103.14030.pdf

### Dive into Deep Learning: Chapter 10

57

58

# Summary

### Encoder-Decoder models

### Sequence to sequence models

### Attention mechanism

o Motivation o Global attention o Local attention

### Transformer and BERT

# Thanks for your attention!

### Question time