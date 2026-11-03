COPYRIGHT WARNING: Copyright in these original lectures is owned by Monash University. You may transcribe, take notes, download or stream lectures for the purpose of your research and study only. If used for any other purpose, (excluding exceptions in the Copyright Act 1969 (Cth)) the University may take legal action for infringement of copyright.

Do not share, redistribute, or upload the lecture to a third party without a written permission!

### FIT3181/5215 Deep Learning

Week 10: Vision Transformer and Model Fine-Tuning Techniques

CE/Lecturer: Dr Trung Le (Clayton)

Lecturer: Dr Binh Nguyen (Clayton) Lecturer: Dr Ting Fung Fung (Malaysia) Lecturer: Dr Arghya Pal (Malaysia) Lecturer: Dr Sailaja Rajanala (Malaysia)

Email: trunglm@monash.edu

Department of Data Science and AI Faculty of Information Technology, Monash University, Australia

2

# Outline

##  Revision of Transformers

##  Vision Transformer

##  Swin Transformer

##  Model Fine-Tuning for Transformer

### o Prompt-Tuning

### o LoRA

### o Adapter

###  Acknowledgment:

o https://web.eecs.umich.edu/~justincj/slides/eecs498/WI2022/598_WI2022_lecture18.pdf o https://www.slideshare.net/slideshow/transformers-in-vision-from-zero-to-hero-

dlipptx/253343336

# What do you think when you hear the

# word «Transformer»?

## The Transformer «today»

Salta (90%) | Odia (9%) | Perchè (1%)

Linear + Softmax

Decoder

Memory Add & Norm

Feed Forward Encoder

Add & Norm Add & Norm

Multi-Head Feed Forward

Attention V K Q

Mx Nx

Add & Norm Add & Norm

Multi-Head Multi-Head Attention Attention V K Q V K Q

Positional Positional

- \+

Encoding Encoding

Input Output

The cat jumps the wall Il gatto

# Text History

2017 Introduced transformers in NLP

2020 Vision Transformers Transformers

### Images Transformers for video

### Videos 2021

understanding

Now Computer Vision Revolution!

## A step back: Recurrent Networks (RNNs)

### The cat jumps the wall Encoder

hstart h0 h1 h2 h3 h4 Context vector 𝒄

### \<BOS> Il gatto salta il muro

### Decoder

h5

h6 h7 h8 h9 h10

### Il gatto salta il muro \<EOS>

6

## Problems

### 1\. We forget tokens too far in the past in the context vector c

### 2\. We need to wait the previous token to compute the next hidden-state

### The cat jumps the wall Encoder

hstart h0 h1 h2 h3 h4 Context vector 𝒄

### \<BOS> Il gatto salta il muro

### Decoder

h5

h6 h7 h8 h9 h10

### Il gatto salta il muro \<EOS>

7

## Solving problem 1

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

8

## Solving problem 2

"We need to wait the previous token to compute the next hidden-state"

### The cat jumps the wall

Solution Throw away recurrent

connections

hstart h0 h1 h2 h3 h4

2017 paper Attention

"Attention Is All You Need"

- \+ + + = Context 𝑐𝑡

### \<BOS> Il gatto salta il muro

h5

h6 h7 h8 h9 h10

### Il gatto salta il muro \<EOS>

9

## Transformer's Attention Mechanism

Queries Il FFN Target tokens “from Target

gatto FFN the point of view” of Sequence Dot

FFN product the source salta

sequence …

∙ ∙ ∙

Norm &Softmax Keys & Values

∙ The FFN Source cat FFN Sequence

jumps FFN the FFN …

10

## Transformer's Attention Mechanism

From a different perspective

Query Il “gatto” token, built by Target

gatto “N5”

aggregating value vectors Sequence

salta in the source dictionary

Soft-matching …

Lookup Table Key Value The “A4” Source cat “N9” Weighted Sequence average

jumps “O7” the “A4” wall

“N2”

11

## Cross-Attention and Self-Attention

### Self-Attention

### Cross-Attention

I gave my dog Charlie some food I gave my dog Charlie some food

Who? To whom? What? Ho dato da mangiare al mio cane Charlie

Source = Target Source ≠ Target Multi-Head Multi-Head

Attention Attention

o Key, Queries, Values o Queries from Target

V K Q V K Q

obtained from the o Key, Values from Source

same sentence Source Target

Captures inter-sequence dependencies

Captures intra-sequence dependencies

12

# Self-Attention

2

### Attention calculation is O(n)

I gave my dog Charlie some food I

gave

my

dog

Charlie

some

food

13

[batch_size,seq_len, 𝑑𝑉]

𝑂

# Masked Self-Attention 𝑊 𝑌

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

14

[batch_size,tgt_seq_len,d_model] [batch_size,tgt_seq_len, 𝑑𝑉]

# Cross-Attention

𝑌 𝑂 𝑍 Output: 𝑌 = 𝑍𝑊

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

15

## Full Transformer Architecture

“Salta” (90%) | “Odia” (9%) | “Perchè” (1%) Lookup Table (source sequence)

Key Value

Linear + Softmax “A4”

“N9”

Decoder “O7” “A4” Memory Add & Norm “N2”

Feed Forward Encoder

Add & Norm Cross-attention Add & Norm

The elements of the memory Multi-Head constitute the dictionary used to Feed Forward

Attention contextualize every token in the

deconding stage V K Q

Mx Nx

Self-attention Add & Norm Add & Norm

Input tokens are contextualized Multi-Head Multi-Head with tokens from the same Attention Attention sentence. This is for understanding

the meaning of the sentences V K Q V K Q

Positional Positional

- \+

Encoding Encoding

Input Output

“The cat jumps the wall” “Il gatto”

16

## Full Transformer Architecture

“Salta” (90%) | “Odia” (9%) | “Perchè” (1%)

Linear + Softmax

Decoder

Memory Add & Norm

Feed Forward Encoder

Add & Norm Add & Norm

Multi-Head Feed Forward

Attention V K Q

Mx Nx

Add & Norm Add & Norm

Multi-Head Multi-Head Attention Attention V K Q V K Q

Positional Positional

- \+

Encoding Encoding

Input Output

“The cat jumps the wall” “Il gatto”

17

### Vision Transformer

## Transformers in Computer Vision

Can we use the self-attention mechanism in images?

## Transformers in Computer Vision

###  The transformer works with a set of tokens

###  What are tokens/visual words in images?

224px

4 2 224 224 224px pixels calculations

Impossible!

## Transformers in Computer Vision

### • Tokens as the features from an object detector

ROI Pooling

ROI Pooling Tokens!

ROI Pooling

## Vision Transformers (ViTs)

“An image is worth 16x16 words”

14 segments 16px 224px

16px

Image to Flatten Patches + Patches Linear Projection

Tokens! 224px

14 segments

“An image is worth 16x16 words” | Dosovitskiy et al., 2020

Transformer Encoder

## Vision Transformers (ViTs)

MLP CLASS

Head

\*

class token Transformer Encoder

\*

1 2 3 4 5 6 7 8 9 1 2 3

Linear Projection of Flattened Patches 4 5 6

7 8 9

## Vision Transformers (ViTs)

## What happens during training?

Learned Knowledge

Not Improving

Anymore Still Improving

Convolutional Neural

Vision Transformer Network

## Why are they different?

Locality Need very large Sensitive dataset for training

Convolutional

Neural Network Translation

Learns Invariant

inductive biases

Vision Transformers

Lack of global Able to find understanding long-term dependencies

## A different point of view

ViTs are both local and global!

Heads focus on farther patches

0.0 1

0.7

The ViT learns only global information

with low amount of data

## A different point of view

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

## A different point of view

Vision Transformers are very robust!

Occlusion Distribution Shift Adversarial Perturbation Permutation

Intriguing Properties of Vision Transformers | Muzammal Naseer et al.

### Swin Transformer

## Swin Transformers

Shifted Window based Self-Attention

Vision Transformer Swin Transformer

High-level

16x 16x

8x 16x

4x 16x

(4x4 patches)

Low-level (pixels)

## Swin Transformers

𝑯 𝑾 𝑯 𝑾 𝑯 𝑾 𝑯 𝑾 𝑯 𝑾 𝑯 𝑾 𝑯 𝑾

𝑯 𝑾 𝑯 𝑾 𝟒𝟖 × × 𝑪 × × 𝟐𝑪 × × 𝟐𝑪 × × 𝟒𝑪 × × 𝟒𝑪 × × 𝟖𝑪 × ×

𝑪 × × 𝟖𝑪 × × 𝟒 𝟒 𝟒 𝟒 𝟖 𝟖 𝟖 𝟖 𝟏𝟔 𝟏𝟔 𝟏𝟔 𝟏𝟔 𝟑𝟐 𝟑𝟐

𝟒 𝟒 𝟑𝟐 𝟑𝟐 4

4

Stage 1 Stage 2 Stage 3 Stage 4

𝟑 × 𝑯 × 𝑾

Swin Swin Swin Swin Transformer Transformer Transformer Transformer

Block Block Block Block

Patch Merging Patch Merging Patch Merging Patch Partition Linear Embedding

x2 x2 x6 x2

4 4

Swin Transformer: Hierarchical Vision Transformer using Shifted Windows | Ze Liu et al., 2021

## Swin Transformers

34

## Swin Transformers

35

## Swin Transformers

36

## Swin Transformers

37

## Swin Transformers

38

## Swin Transformer Block

### ✓ Two cascading Transformer Encoder Blocks:

Local window

### ✓ Window Self Attention

### ✓ Shifted Window Self Attention

4x4 patch

Window-MSA ShiftedW-MHA Add & Norm Add & Norm

Feed Forward Feed Forward

Add & Norm Add & Norm

W-MSA SW-MSA

Problem: tokens only interact with other tokens within the same window; no communication across windows Solution: Alternate between normal windows and shifted windows in successive Transformer blocks

## Swin Transformer: Speed and Accuracy

40

### Model Fine-Tuning

## ViT: Model Fine-Tuning

MLP Pretrained ViT/16 models trained on CLASS

Head ImageNet 21K and JFT-300M datasets

\*

class token Transformer Encoder

\*

1 2 3 4 5 6 7 8 9 1 2 3

Linear Projection of Flattened Patches 4 5 6

7 8 9

## ViT: Model Fine-Tuning

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

𝑟 ≪ 𝑚, 𝑛 [𝑚, 𝑛] [𝑚, 𝑟] [𝑟, 𝑛] 𝑊𝑄 + 𝑩𝑸 × 𝑨𝑸

𝑊𝐾 + 𝑩𝑲 × 𝑨𝒌 class token

𝑊𝑉 + 𝑩𝑽 × 𝑨𝑽

\* 1 2 3 1 2 3 4 5 6 7 8 9

4 5 6 Linear Projection of Flattened Patches

7 8 9

48

## Summary

### Revision of Transformers

### Vision Transformer

### Swin Vision Transformer

### Model Fine-Tuning for Transformer

o Prompt-Tuning o LoRA o Adapter

## Thanks for your attention!

Question time