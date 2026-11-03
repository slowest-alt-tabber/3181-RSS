COPYRIGHT WARNING: Copyright in these original lectures is owned by Monash University. You may transcribe, take notes, download or stream lectures for the purpose of your research and study only. If used for any other purpose, (excluding exceptions in the Copyright Act 1969 (Cth)) the University may take legal action for infringement of copyright.

Do not share, redistribute, or upload the lecture to a third party without a written permission!

### FIT3181/5215 Deep Learning

Week 11: Deep Generative Models

CE/Lecturer: Dr Trung Le (Clayton)

Lecturer: Dr Binh Nguyen (Clayton) Lecturer: Dr Ting Fung Fung (Malaysia) Lecturer: Dr Arghya Pal (Malaysia) Lecturer: Dr Sailaja Rajanala (Malaysia)

Email: trunglm@monash.edu

Department of Data Science and AI Faculty of Information Technology, Monash University, Australia

## Generative AI teaser: pre-2022

High Low

Realistic Image Generation Super Resolution Sketch, sematic image Realistic image

Text to Image Semantic-Image-to-Photo Translation

Image Painting Image Editing

## Generative AI teaser: pre-2022, GAN-based

Style Transfer

Photo Blending Face Aging Neural Photo Editor

## Generative AI teaser: pre-2022

DeepDream – I created these images around 2016-2017 using two images

## 2022 – 2023: years of Generative AI

Two minutes Papers channel

# …

## DeepDream in 2023

## Beyond image generation …

Image credit: https://www.midjourney.com/showcase

## Beyond image generation …

1. 3D generation from a single image: https://3d-diffusion.github.io/
2. Video generation: https://video-diffusion.github.io/
3. Text to 3D: https://dreamfusion3d.github.io/
4. Subject-driven Image generation: https://dreambooth.github.io/
5. Image generation with visual prompt: https://github.com/lllyasviel/ControlNet

Text to 3D Visual prompt with ControlNet

## and of course, ChatGPT and LLMs

https://businessday.ng/technology/article/chatgpt-is-fastest-app-to-hit-100m-users-in-history/

## The current rise of diffusion methods …

Image credit: https://cvpr2023-tutorial-diffusion-models.github.io/

## Outline

Introduction to generative models o Explicit versus implicit density estimation o What is a generative model? o Why do we need to be generative? Generative Adversarial Networks o Adversarial training and minimax two-player game o Formulation and optimal solution o Issues and challenges with GAN Diffusion Models

## Explicit vs Implicit Density Estimation

## Explicit Density Estimation

𝒅 𝒙𝒊~𝒑𝒅(𝒙) ℝ

𝒅 Dataset 𝐷 = {𝒙𝟏, 𝒙𝟐, … , 𝒙𝑵} where 𝒙𝒊 ∈ ℝ and 𝒙𝒊~𝒑𝒅(𝒙) (data distribution existed but unknown)

###  Given training set 𝐷 = {𝒙, 𝒙, … , 𝒙 } where 𝒙 ~𝒑 (𝒙)

𝟏 𝟐 𝑵 𝒊 𝒅

- 𝒑𝒅(𝒙) is existed but unknown data distribution.
- Learn 𝒑𝒎(𝒙) from 𝐷 such that 𝑝𝑚(𝑥) ≈ 𝑝𝑑(𝑥).

###  Gaussian Mixture Model

𝐾

- 𝑝 𝒙 = σ 𝜋 𝑁(𝒙 ∣ 𝝁, 𝚺) 𝑚 𝑖=1 𝑘 𝒌 𝒌
- We need to estimate 𝜋𝑘, 𝝁𝒌, 𝚺𝒌 for 𝑘 = 1, … , 𝐾 ▪ Using Expectation-Maximization (EM) approach

## Explicit Density Estimation

Structural data High-dimensional data

Data manifold

[Source: https://lvdmaaten.github.io/tsne/]

###  When does explicit density estimation fail?

- High-dimensional data (curse of dimensionality)
- Structural data ▪ Images, texts, speeches, videos, biological data, and etc.

## Implicit Density Estimation

𝐷 = 𝒙1, 𝒙2, … , 𝒙𝑁

Resolution: 3x32x32= 3072

Deep learning 𝐺(𝒛)

Deep learning model 𝐺(𝒛)

model

𝒵 𝒛~𝑝(𝒛) 𝒵 𝒛~𝑝(𝒛)

 Given a training set 𝐷 = 𝒙, 𝒙, … , 𝒙 where each 𝒙 ~𝑝 𝒙 .

1 2 𝑁 𝑖 𝑑

- 𝑝𝑑 𝒙 exists but unknown but we do not learn 𝑝𝑚(𝑥) ≈ 𝑝𝑑(𝑥)

 Learn a generator 𝑮 mapping from the noise space 𝒵 to data space

- 𝒛~𝑝(𝒛) → 𝒙෥ = 𝐺(𝒛)~𝑝 𝒙 𝑑
- 𝒙෥ = 𝐺(𝒛) mimics those in 𝐷.

 This is the setting of generative model

- Generative Adversarial Network (GAN), Variational Auto-Encoder (VAE)

## Generative models

 Generative models learn how data are

generated to re-generate new data.

- Synthesize poem with Markov process

 Texts, speech, videos, interactions

 Simulations

- Robot learning without damage

- Learning tasks with few or expensive labels

 Create new art, music, or fashion

 Synthetic biology

 Many more applications …

## Generative models

 Explain cause for actions beyond associating inputs to outputs.

- Explainable AI, patients need to know why they will be treated this way or that way!

 Understand and predict the world - how it evolves!

- Self-driving needs the ability to predict what happens next on the street
- Robots need the ability to foresee the environments
- Fintech AI needs to ‘simulate’ the future economy, world events, etc.
- Biologists wish to understand the evolution from data

 Imagine and generate rich plans for the future.  Recognize objects in the world and their factors of variation.  Establish concepts useful for reasoning and decision making.  Detect novelty, surprising events in the world.  Test our ability to manipulate high-dimensional distributions

## Generative Models

Yann LeCun

## Generative Adversarial Networks

Yann LeCun

A B

Which face is NOT generated by AI?

A B

Which face is NOT generated by AI?

A B

Which face is NOT generated by AI?

A B

Which face is NOT generated by AI?

http://www.whichfaceisreal.com

## Deep Generative Models (DGM)

 Given a training set 𝐷 =

𝒙1, 𝒙2, … , 𝒙𝑁 where each 𝒙𝑖~𝑝𝑑 𝒙 . 𝐷 = 𝒙1, 𝒙2, … , 𝒙𝑁

- 𝑝𝑑 𝒙 exists but unknown.

 Learn a generator 𝑮 mapping from the

noise space 𝒵 to data space

- 𝒛~𝑝(𝒛) → 𝒙෥ = 𝐺(𝒛)~𝑝 𝒙 𝑑
- 𝒙෥ = 𝐺(𝒛) ‘mimics’ those in 𝐷. 𝐺(𝒛) Deep NNs

 Two most important and popular

families of DGM:

- Variational Auto Encoder (Kingma et al., 2013)
- Generative Adversarial Networks (Goodfellow et 𝒵 𝒛~𝑝(𝒛) al., 2014)

## Our Task in Generative Models

Source of data (Images, waves, texts) Train

Trained model (Generator)

Deep learning Generate 𝐺(𝒛)

model

𝒵 𝒛~𝑝(𝒛)

## Our Task in Training Generative Models

𝐷 = 𝒙𝟏, 𝒙𝟐, … , 𝒙𝑵

𝑑 𝑚×𝑛 ❑ 𝑥𝑖 ∈ ℝ, 𝑥𝑖 ∈ ℝ, 𝑥𝑖 = 𝐺𝑖 = 𝑉𝑖, 𝐸𝑖 ❑ Assume that 𝑥𝑖(s) are drawn from an unknown distribution with p.d.f. 𝑝𝑑(𝑥)

min𝜃 𝐷𝐽𝑆 𝑝𝑑 𝑥 ||𝑝𝑔 𝑥 → GAN  How to use 𝑝 (𝒙) since it is unknown?

𝑑

𝒙𝒓𝒆𝒂𝒍~𝑝𝑑(𝒙)  How to be aware of how far 𝑝 (𝒙) and

𝑑

Gap between two distributions min𝜃 𝐷𝑓 𝑝𝑑 𝑥 ||𝑝𝑔 𝑥 → fGAN 𝑝𝑔(𝒙) ? o Use a discriminator (classifier) to classify the 𝒙 = 𝐺 𝒛 ~𝑝 𝒙

𝑓𝑎𝑘𝑒 𝜃 𝑔

min𝜃 𝐷𝑊𝑆 𝑝𝑑 𝑥 ||𝑝𝑔 𝑥 → WGAN fake data and true data. o If the accuracy is high and the loss is small,

𝑝𝑔 𝒙 they are far away 𝐺 (. ) Data space

𝜃 o If the accuracy is low (50%) and the loss is

high, they are very close and even

𝑑 overlapping 𝒛~𝑝 𝒛 = 𝑁 0, 𝐼 𝑜𝑟 𝑈𝑛𝑖( 0,1 ) 𝑝 𝒛

Noise space Source of randomness

## Generative Adversarial Networks

𝑘 𝑟  A game of two players [Goodfellows et al., 2014]

o Generator 𝑮: generate fake samples indistinguishable from real samples

𝐷 o Discriminator 𝑫: discriminate real and fake samples

𝒙𝑓𝑎𝑘𝑒 𝒙

𝑒𝑎  Min-max problem

𝐺 min max 𝐽 𝐺, 𝐷 = 𝔼𝒙 ~ 𝑝 𝒙 log 𝐷 𝒙 + 𝔼𝒛 ~ 𝑝 𝒛 log(1 − 𝐷(𝐺 𝒛 )

𝑑 𝐺 D

𝒛 ~ 𝑝𝒛

 Issue & challenge

o Mode collapse, min-max problem

## GAN: analogy to minimax game

Termination at Nash equilibrium point

Task: distinguish between real and counterfeit money No player can improve any more

Goal: try to sabotage the counterfeit maker

Discriminator

Task: produce counterfeit money

Goal: try to fool/confuse the police officer

Final output Generator Target/data distribution from GANs

Adversarial or Cooperative

## Training GAN

- Gradient ascent to train discriminator 𝐷𝜃: 𝑑 max 𝔼 log 𝐷𝜃 𝒙 + 𝔼 log(1 − 𝐷𝜃 (𝐺 𝒛 ) 𝑑 𝑑 𝜃𝑑 𝒙~𝑝𝑑𝑎𝑡𝑎(𝒙) 𝒛~𝑝𝒛

(𝑖) (𝑖) Ascent update 𝐷: 𝒛 ~𝑝𝑧, 𝒙 ~𝑝𝑑𝑎𝑡𝑎 Training 𝐷 is to maximize the

1 𝑀

𝑖 𝑖 𝛻 σ log 𝐷 𝒙 + log 1 − 𝐷 𝐺 𝒛 𝜽d 𝑖=1 probability of detecting correct labels M

min max 𝐽 𝐺, 𝐷 = 𝔼 log 𝐷 𝒙 + 𝔼 log(1 − 𝐷(𝐺 𝒛 ) 𝐺 D 𝒙~𝑝𝑑𝑎𝑡𝑎(𝒙) 𝒛~𝑝𝒛

(𝑖) Descent update 𝐺: 𝒛 ~𝑝𝑧 Training 𝐺 is to minimize or fool the 𝑀

1

𝑖 discriminator 𝐷, i.e., minimize 1 − − 𝛻𝜽 ෍ log 1 − 𝐷 𝐺 𝒛

g M

𝑖=1 𝐷 𝐺 𝒛 to 0

- Gradient descent to train the generator 𝐺𝜃 𝑔 min 𝔼 log(1 − 𝐷𝜃 (𝐺 𝒛 ) 𝑑 𝜃𝑔 𝒛~𝑝𝒛

## GAN algorithm

𝑘 = 1 is stable

In practice, this objective doesn’t work well since gradient signal is dominated by region where the sample looks real, and relatively flat in samples look more fake!

Instead, we can use: max 𝔼 log(𝐷𝜃 (𝐺 𝒛 )

𝑑 𝜃𝑔 𝒛~𝑝𝒛

In general, training GAN using minimax strategy is challenging!

(Lecture 13, Stanford, CS 231n)

## Theory of GAN

y=1 𝑝𝑑(𝒙) 𝑝𝑑(𝒙)

Gap between two distributions

y=-1 𝒙𝒇𝒂𝒌𝒆 = 𝐺𝜃 𝒛 ~𝑝𝑔 𝒙

𝑝𝑔 𝒙

❖ D(x) = P(y=1|x): probability x is true data. 𝐺𝜃 (. ) ❖ 1-D(x) = P(y=-1|x): probability x is fake data.

The log likelihood function need to be maximized

𝐽 𝐷, 𝐺 = 𝔼𝒙~𝑃 log 𝐷(𝒙) + 𝔼𝑥 ෤~𝑃 log 1 − 𝐷( 𝑥෤) 𝒛~𝑝(𝒛) 𝑑 𝑔

= 𝔼𝒙~𝑃 log 𝐷(𝒙) + 𝔼𝒛~𝑃 log 1 − 𝐷(𝐺(𝒛))

𝑑 𝑧 Source of randomness

 We want to train NN-based function 𝐺 (. ) such that 𝑝 𝒙 = 𝑝 𝒙 , but we do not have

𝜃 𝑔 𝑑 𝑝𝑑 𝒙 in hand o Invoke a strong discriminator 𝐷 𝒙 to implicitly justify how far 𝑝𝑑 𝒙 and 𝑝𝑔 𝒙 are.

## Theory of GAN

y=1

𝑝𝑑(𝒙)

∗ ∗ ∗ 𝐽 𝐷, 𝐺 = 𝔼 log 𝐷 (𝒙) + 𝔼෥ log 1 − 𝐷 (෥𝒙) is large

𝒙~𝑃𝑑 𝒙~𝑃𝑔

y=-1 𝑝𝑔 𝒙

y=1

∗ ∗ ∗ 𝐽 𝐷, 𝐺 = 𝔼 log 𝐷 (𝒙) + 𝔼෥ log 1 − 𝐷 (෥𝒙) decreases

𝒙~𝑃𝑑 𝒙~𝑃𝑔 𝑝𝑑(𝒙)

y=-1

𝑝𝑔 𝒙

y=1

𝑝 (𝒙) ∗ ∗ ∗ 𝑑 𝐽 𝐷, 𝐺 = 𝔼 log 𝐷 (𝒙) + 𝔼 log 1 − 𝐷 (෥𝒙) is smallest

𝒙~𝑃𝑑 ෥𝒙~𝑃𝑔

y=-1 𝑝𝑔 𝒙

It comes with the minimax problem

𝒎𝒊𝒏𝑮𝒎𝒂𝒙𝑫𝑱 𝑫, 𝑮

## Optimal discriminator

𝐽 𝐷, 𝐺 y=1 𝑝𝑑(𝒙) = 𝔼 log 𝐷(𝒙) + 𝔼 log 1 − 𝐷(𝒙)

𝒙~𝑝𝑑 𝒙~𝑝𝑔 = 𝔼𝒙~𝑝 log 𝐷(𝒙) + 𝔼𝑧~𝑝 log 1 − 𝐷(𝐺(𝒛))

𝑑 𝑧  𝒎𝒊𝒏 𝒎𝒂𝒙 𝑱 𝑮, 𝑫

𝑮 𝑫 y=-1 𝑝 𝒙 o Given G, we need to solve: 𝑔

∗ ∗ 𝑫 = 𝑫 (𝑮) = 𝒂𝒓𝒈𝒎𝒂𝒙𝑫𝑱(𝑫, 𝑮)

❖ The optimal discriminator has the following form:

𝑝𝑑(𝒙) ∗ ❖ 𝐷 𝒙 = , 𝑜𝑟 𝒙

𝑝𝑑 𝒙 +𝑝𝑔(𝒙)

## Optimal discriminator

𝐽 𝐷, 𝐺 = 𝔼 log 𝐷(𝒙) + 𝔼 log 1 − 𝐷(𝒙) 𝑘 𝑟

𝒙~𝑃𝑑 𝒙~𝑃𝑔

= න[𝑝𝑑(𝒙) log 𝐷(𝒙) + 𝑝𝑔 𝒙 𝑜𝑔(1 − 𝐷 𝒙 )] 𝒙

𝐷  We need to solve:

𝒙𝑓𝑎𝑘𝑒 𝒙

𝑒𝑎 o 𝑚 𝑥𝐷 ׬[𝑝𝑑(𝒙) log 𝐷(𝒙) + 𝑝𝑔 𝒙 𝑜𝑔(1 − 𝐷 𝒙 )] 𝒙  We need to maximize D, let’s do this pointwise at each

𝐺 position of 𝑥: o 𝑚 𝑥𝐷 ℎ 𝐷 : = 𝑝𝑑 𝒙 log 𝐷 + 𝑝𝑔 𝒙 log 1 − 𝐷

𝒛 ~ 𝑝𝒛 𝑝 𝒙 𝑝𝑔 𝒙 𝑑 o 𝛻ℎ 𝐷 = − = 0

𝐷 1−𝐷

𝑝𝑑(𝒙) ∗ o It means that 𝐷 (𝒙) =

𝑝𝑑 𝒙 +𝑝𝑔(𝒙)

## Optimal generator

When training is perfect, discriminator D can ∗ NO longer tell the fake (money) from the real  𝒎𝒊𝒏 𝒎𝒂𝒙 𝑱 𝑫, 𝑮 = 𝒎𝒊𝒏 𝑱(𝑫 𝑮 , 𝑮)

𝑮 𝑫 𝑮

(money), i.e., 0.5 probability ∗ ∗  𝐽(𝐷 𝐺 , 𝐺)=𝔼 log 𝐷 (𝒙) +

𝒙~𝑝𝑑

∗ 𝑘 𝑟 𝔼𝒙~𝑝 log 1 − 𝐷 (𝒙) =

𝑔

𝑝𝑑(𝒙) 𝔼 log + 𝔼 ൤log ൤1 − 𝒙~𝑝𝑑 𝒙~𝑝𝑔

𝑝𝑑 𝒙 +𝑝𝑔(𝒙)

𝐷 𝑝𝑑(𝒙) 𝑝𝑑(𝒙)

൨൨ = 𝔼 log +

𝒙~𝑝𝑑 𝑝𝑑 𝒙 +𝑝𝑔(𝒙) 𝑝𝑑 𝒙 +𝑝𝑔(𝒙)

𝒙𝑓𝑎𝑘𝑒 𝒙 𝑝 (𝒙) 𝑒𝑎 𝑔 ∗ 𝔼 log = 𝑝𝑔 = 𝑝 𝒙~𝑝 𝑑

𝑔 𝑝 𝒙 +𝑝 (𝒙)

𝑑 𝑔 0.5,0.5 𝐺 𝐷 𝑝𝑑 𝒙 ||𝑝𝑔(𝒙) − 2 log 2 𝐽𝑆  We come with 𝒛 ~ 𝑝

𝒛 0.5,0.5 o 𝑚𝑖𝑛 𝐷 𝑝 𝒙 ||𝑝 (𝒙) → 𝑝 ∗ = 𝑝

𝐺 𝐽𝑆 𝑑 𝑔 𝑔 𝑑

Optimal G

## The Nash equilibrium point

##  The solution of the minimax problem 𝒎𝒊𝒏 𝒎𝒂𝒙 𝑱 𝑫, 𝑮

𝑮 𝑫 ∗ ∗

- Nash equilibrium point 𝐷, 𝐺 which satisfies ▪ 𝑝 ∗ = 𝑝 𝑔 𝑑 ∗ 𝑝𝑑(𝒙) ▪ 𝐷 (𝒙) = = 0.5, 𝑜𝑟 𝑥 𝑝𝑑 𝒙 +𝑝 ∗(𝒙) 𝑔

𝑝 𝑝 ∗ = 𝑝 𝑑 𝑔 𝑑

∗ 𝐷 𝐷 𝑥

𝑝𝑔

= 0.5, ∀𝑥

Our own simulation [Goodfellow et al, NIPS 14]

## The Nash equilibrium point

##  The solution of the minimax problem 𝒎𝒊𝒏 𝒎𝒂𝒙 𝑱 𝑫, 𝑮

𝑮 𝑫 ∗ ∗

- Nash equilibrium point 𝐷, 𝐺 which satisfies ▪ 𝑝 ∗ = 𝑝 𝑔 𝑑 ∗ 𝑝𝑑(𝒙) ▪ 𝐷 (𝒙) = = 0.5, 𝑜𝑟 𝑥 𝑝𝑑 𝒙 +𝑝 ∗(𝒙) 𝑔

𝑝 𝑝 ∗ = 𝑝 𝑑 𝑔 𝑑

∗ 𝐷 𝐷 𝑥

𝑝𝑔

= 0.5, ∀𝑥

Our own simulation [Goodfellow et al, NIPS 14]

## GAN: example results

 For a complex dataset such as ImageNet

- GANs can generate sharp images, but look unrealistic
- Hard to know what objects they are

 Conditional GANs with labels work much better in practice!

## Issues with GAN

Two serious technical problems with GAN  #1: mode collapsing problem

- Impeach the ability to generate diverse, realistic data/images

 Generated (fake) examples 𝐺(𝒛) where 𝒛~𝑝(𝒛)

- Can only cover a few modes in the data distribution 𝑝𝑑(𝒙)
- Miss many other modes in the data distribution.

## Issues with GAN

Two serious technical problems with GAN

 #2: Convergence is hard due to minimax formulation

- Training GAN is very challenging!

Generator loss Discriminator loss

###  Mini-max problem

- min max 𝐽 𝐺, 𝐷 = 𝔼 log 𝐷 𝒙 + 𝔼 log(1 − 𝐷(𝐺 𝒛 ) 𝐺 D 𝒙~𝑝𝑑(𝒙) 𝒛~𝑝𝒛

###  No unique loss function

## How to evaluate GAN: Inception score

 Except for trivial cases, we don’t know the true data distribution 𝑝 so how can we say how

𝑑 good the generated distribution 𝑝𝑔 is?  How to know that a generative model can generate diverge examples that cover many modes in

data?  How to compare two generative models trained on the same training set?

y=1 y=2 y=1 y=2

y=3 y=4 y=3 y=4

𝐺2(𝒛) 𝐺1(𝒛)

𝒵 𝒵 𝒛~𝑝(𝒛)

𝒛~𝑝(𝒛)

Two generators 𝑮𝟏 and 𝑮𝟐, which one is better?

## How to evaluate GAN: Inception score

 Except for trivial cases, we don’t know the true data distribution 𝑝 so how can we say how

𝑑𝑎𝑡𝑎 good the generated distribution 𝑝𝑔 is?  How to know that a generative model can generate diverge examples that cover many modes in

data?  How to compare two generative models trained on the same training set?

y=1 y=2 y=1 y=2

y=3 y=4 y=3 y=4

𝐺2(𝒛) 𝐺1(𝒛)

𝒵 𝒵 𝒛~𝑝(𝒛)

𝒛~𝑝(𝒛)

𝑮𝟐 is better and generate more diverge examples, while 𝑮𝟏 misses the data mode y=4.

## How to evaluate GAN: Inception score

 Assume that training examples have labels

- y=1,2,3, and 4.

 We train a good classifier 𝐶 on the training set with labels

4

- 𝐶 𝒙 = 𝑝 𝑦 𝒙 = 𝑝 𝑦 = 𝑘 𝒙 is the prediction 𝑘=1 probabilities where 𝑝 𝑦 = 𝑘 𝒙 is the probability to classify 𝑥 to y=1 y=2 the class 𝑘.

 Given 𝒙෥𝑖 = 𝐺 𝒛𝒊 , 𝑖 = 1, … , 𝑇, we wish

- 𝐶 𝒙෥ = 𝑝(𝑦 ∣ 𝒙෥ = 𝐺 𝒛) is close to an one-hot vector 𝑖 𝑖 𝒊 y=4 y=3 ▪ 𝐶 𝒙෥𝑖 is confident and has a small entropy. 𝐺(𝒛) 1 𝑇 1 𝑇
- σ 𝐶 𝒙෤ = σ 𝑝(𝑦 ∣ 𝒙෤ = 𝐺 𝒛 is close to the uniform 𝑖=1 𝑖 𝑖=1 𝑖 𝒊 𝑇 𝑇 distribution [0.25, 0.25,0.25,0.25] (big entropy). 𝒵 𝒛~𝑝(𝒛)

###  Inception score

1 𝑇 1 𝑇

- σ 𝐾𝐿 𝐶 𝒙෤, σ 𝐶 𝒙෤ ≈ 𝔼 [𝐾𝐿 𝑝 𝑦 𝒙෥ = 𝐺 𝒛 , 𝑝(𝑦) ] 𝑖=1 𝑖 𝑗=1 𝑗 𝑧 𝑇 𝑇
- Higher is better → more diversity of generated images

Salimans et al, Improved Techniques for Training GANs, 2016

## GAN: Summary

###  Adversarial training: a game of two players [Goodfellows et al.,

### 2014]

- Generator 𝐺: generate fake samples indistinguishable from real samples
- Discriminator 𝐷: discriminate real and fake samples

###  Min-max problem

min max 𝐽 𝐺, 𝐷 = 𝔼𝒙 ~ 𝑝 𝒙 log 𝐷 𝒙 + 𝔼𝒛 ~ 𝑝 𝒛 log(1 − 𝐷(𝐺 𝒛 )

𝑑 𝐺 D

###  Issues and challenges

- Mode collapse, min-max problem

Yann Lecun

## Deep Convolutional GAN (DCGAN)

Radford et al, “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks”, 2015

## Introducing Conv2DTranspose

###  Why do we need the Conv2DTranspose layer?

DCGAN architecture

## Introducing Conv2DTranspose

 Standard Conv2D

- Conv2D: input = [3,3], kernel = [2,2], stride = [1,1] → output = [2,2]

0 0 1

1 4 0 1 0 1 2 0 0

1 1 4 4 1 2 0 0 0

output filter input  Conv2Dtranpose: it is opposite of conv2D

- Input =[2,2], kernel = [2,2], stride = [1,1] → output = [3,3]

0 1 0 1

x 2 3 2 3

input filter

## Introducing Conv2DTranspose

 Standard Conv2D

- Conv2D: input = [3,3], kernel = [2,2], stride = [1,1] → output = [2,2]

0 0 1

1 4 0 1 0 1 2 0 0

1 1 4 4 1 2 0 0 0

output filter input  Conv2Dtranpose: it is opposite of conv2D

0 1

- Input =[2,2], kernel = [2,2], stride = [1,1] → output = [3,3] 2 3

0 1 0 1

x 2 3 2 3

input filter

## Introducing Conv2DTranspose

 Standard Conv2D

- Conv2D: input = [3,3], kernel = [2,2], stride = [1,1] → output = [2,2]

0 0 1

1 4 0 1 0 1 2 0 0

1 1 4 4 1 2 0 0 0

output filter input  Conv2Dtranpose: it is opposite of conv2D

0 1

- Input =[2,2], kernel = [2,2], stride = [1,1] → output = [3,3] 2 3

0 1 0 1

x 2 3 2 3 0 2

input filter 4 6

## Introducing Conv2DTranspose

 Standard Conv2D

- Conv2D: input = [3,3], kernel = [2,2], stride = [1,1] → output = [2,2]

0 0 1

1 4 0 1 0 1 2 0 0

1 1 4 4 1 2 0 0 0

filter output input  Conv2Dtranpose: it is opposite of conv2D

0 1

- Input =[2,2], kernel = [2,2], stride = [1,1] → output = [3,3] 2 3

0 1 0 1

x 2 3 2 3 0 2

input filter 4 6

0 3

6 9

## Introducing Conv2DTranspose

 Standard Conv2D

- Conv2D: input = [3,3], kernel = [2,2], stride = [1,1] → output = [2,2]

0 0 1

1 4 0 1 0 1 2 0 0

1 1 4 4 1 2 0 0 0

filter output input  Conv2Dtranpose: it is opposite of conv2D

0 1

- Input =[2,2], kernel = [2,2], stride = [1,1] → output = [3,3] 2 3

0 0 1 0 1 0 1

x 0 4 6 2 3 2 3 0 2

4 12 9 input filter 4 6

output

0 3

6 9

## DCGAN architecture

Radford et al, 2015

 Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions

(generator).  Use batch normalization in both the generator and the discriminator.  Remove fully connected hidden layers for deeper architectures.  Use ReLU activation in generator for all layers except for the output which uses tanh.  Use LeakyReLU activation in the discriminator for all layers.

## DCGAN’s generated images

Generated bedroom after 5 epochs

Radford et al, “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks”, 2015

## DCGAN: arithmetic on latent space

## DCGAN: arithmetic on latent space

## Progressive growing of GAN

(Kerras et. al NVIDIA, 2017)

## Further Reading Recommendation

❖ Hands-On: chapter 17 ❖ Deep learning: chapter 20 ❖ DC GAN: https://arxiv.org/pdf/1511.06434.pdf ❖ Wasserstein GAN: https://arxiv.org/pdf/1701.07875.pdf ❖ Wasserstein GAN with Gradient Penalty:

https://arxiv.org/pdf/1704.00028.pdf ❖ Cycle GAN: https://arxiv.org/pdf/1703.10593.pdf ❖ Progressive GAN: https://arxiv.org/pdf/1710.10196.pdf

## Diffusion Models

61

## Motivation

q(𝑥)

###  Non-equilibrium thermodynamics

forward

backward

backward - forward

𝒵

z ~ 𝑝(𝒛)

Image credit: Sohl-Dickstein et al., Deep Unsupervised Learning using Nonequilibrium Thermodynamics, ICML 2015

### General Diffusion Framework

###  Forward diffusion process:

- Adding noise to ‘destruct’ the image

###  Backward (reverse) diffusion process

- Denoising to reconstruct the image
- Then, used as generator

Key papers: Sohl-Dickstein et al., Deep Unsupervised Learning using Nonequilibrium Thermodynamics, ICML 2015 Ho et al., Denoising Diffusion Probabilistic Models, NeurIPS 2020 Song et al., Generating Modeling by Estimating Gradients of the Data Distribution, NeurIPS 2020 Image credit: https://cvpr2023-tutorial-diffusion-models.github.io/

## Forward Diffusion Process

Formulated as Markov chain with 𝑇 steps (usually large)  At time 0:

- Sample data point 𝑥0 from real data distribution 𝑞(𝑥) (i.e., from training data)

 At step :

- 𝑥𝑡 = 𝑥𝑡−1 + Gaussian noise

- 𝑥𝑡 is now distributed according to 𝑞 𝑥𝑡 𝑥𝑡−1 = 𝒩 ⋅∣ 𝜇𝑡, Σ𝑡 with 𝜇𝑡 = 1 − 𝛽𝑡𝑥𝑡−1, Σ𝑡 = 𝛽𝑡 𝑰 and 0 < 𝛽𝑡 < 1

𝑞(𝑥𝑡 ∣ 𝑥𝑡−1)

𝑥 … 𝑥 𝑥 … 𝑥 0 𝑡−1 𝑡 𝑇

Forward Diffusion Process

Image adapted from Ho et al. “Denoising Diffusion Probabilistic Models”, NeurIPS, 2020.

### Forward Diffusion Process

Formulated as Markov chain with 𝑇 steps (usually large)

At time 0: o Sample data point 𝑥0 from real data distribution 𝑞(𝑥) (i.e., from training data) At step : o 𝑥𝑡 = 𝑥𝑡−1 + Gaussian noise

o 𝑥𝑡 is now distributed according to 𝑞 𝑥𝑡 𝑥𝑡−1 = 𝒩 ⋅∣ 𝜇𝑡, Σ𝑡 with 𝜇𝑡 = 1 − 𝛽𝑡𝑥𝑡−1, Σ𝑡 = 𝛽𝑡 𝑰 and 0 < 𝛽𝑡 < 1

 Question 1: How to sample 𝑥 from 𝒩 𝑥; 𝜇 = 1 − 𝛽 𝑥, Σ = 𝛽 𝑰 ?

𝑡 𝑡 𝑡 𝑡 𝑡−1 𝑡 𝑡  Answer: use reparameterization trick

- Sample randomness comes from the prior distribution 𝑧 = 𝜇 + 𝜎 ∙ 𝜖 where ϵ~𝒩 0, 𝐼 is equivalent with 𝑧~𝒩 𝜇, 𝜎 𝑥𝑡 = 1 − 𝛽𝑡𝑥𝑡−1 + 𝛽𝑡𝜖𝑡−1

Image credit: Ho et al. “Denoising Diffusion Probabilistic Models”, NeurIPS, 2020.

### Forward Diffusion Process

Formulated as Markov chain with 𝑇 steps (usually large)

At time 0: o Sample data point 𝑥0 from real data distribution 𝑞(𝑥) (i.e., from training data) At step : o 𝑥𝑡 = 𝑥𝑡−1 + Gaussian noise

o 𝑥𝑡 is now distributed according to 𝑞 𝑥𝑡 𝑥𝑡−1 = 𝒩 ⋅∣ 𝜇𝑡, Σ𝑡 with 𝜇𝑡 = 1 − 𝛽𝑡𝑥𝑡−1, Σ𝑡 = 𝛽𝑡 𝑰 and 0 < 𝛽𝑡 < 1

 Question 2: Is there an analytical form for 𝑥 directly from 𝑥?

𝑡 0  Answer: yes, 𝑥 ~𝑞 𝑥 𝑥 = 𝒩 𝑥; 𝛼ത 𝑥, 1 − 𝛼ത 𝑰

𝑡 𝑡 0 𝑡 𝑡 0 𝑡

𝑡

- define 𝛼 = 1 − 𝛽, 𝛼ത = ς 𝛼 where 𝜖, … , 𝜖 ~𝒩 0, 𝑰 one can prove that: 𝑡 𝑡 𝑡 𝑠=0 𝑠 0 𝑡−1 𝑥𝑡 = 1 − 𝛽𝑡𝑥𝑡−1 + 𝛽𝑡𝜖𝑡−1 𝑥𝑡 = 𝛼𝑡𝛼𝑡−1𝑥𝑡−2 + 1 − 𝛼𝑡𝛼𝑡−1  𝜖𝑡ҧ −2 𝑥𝑡 = 𝛼ത𝑡𝑥0 + 1 − 𝛼ത𝑡  𝜖0ҧ where  𝜖𝑡ҧ −2 is sampled a sum of two Gaussians, which is another Gaussian.

Image credit: Ho et al. “Denoising Diffusion Probabilistic Models”, NeurIPS, 2020.

## Forward Diffusion Process

Importantly

𝑥𝑡~𝑞 𝑥𝑡 𝑥0 = 𝒩 𝑥𝑡; 𝛼ത𝑡𝑥0, 1 − 𝛼ത𝑡 𝑰 𝑥𝑡 = 𝛼ത𝑡𝑥0 + 1 − 𝛼ത𝑡𝜖 where 𝜖 ∼ 𝑁(𝟎, 𝑰)

𝛼ത𝑡 → 0, as → ∞, therefore 𝑥𝑡~𝒩 𝟎, 𝑰 , as → ∞

Exercise: prove this!

### Backward Diffusion Process

Iteratively denoise from time step T to 0

At step 𝑇: o Sample a random vector 𝑥𝑇 from 𝒩 0, 𝑰 At step : o 𝑥𝑡−1 = 𝑥𝑡 − amount of noise o 𝑥𝑡−1 is now distributed according to 𝑞 𝑥𝑡−1 𝑥𝑡

𝑞(𝑥𝑡 ∣ 𝑥𝑡−1)

𝑥 … 𝑥 𝑥 … 𝑥 0 𝑡−1 𝑡 𝑇

… …

𝑞(𝑥𝑡−1 ∣ 𝑥𝑡)

Backward Diffusion Process Image credit: Ho et al. “Denoising Diffusion Probabilistic Models”, NeurIPS, 2020.

## Backward Diffusion Process

Iteratively denoise from time step T to 0

At step 𝑇: o Sample a random vector 𝑥𝑇 from 𝒩 0, 𝑰 At step : o 𝑥𝑡−1 = 𝑥𝑡 − amount of noise o 𝑥𝑡−1 is now distributed according to 𝑞 𝑥𝑡−1 𝑥𝑡

Issue: 𝑞 𝑥𝑡−1 𝑥𝑡 is intractable Solution (Ho’ et al.): conditioning on 𝑥0 so that 𝑞 𝑥𝑡−1 𝑥𝑡, 𝑥0 is tractable Question: How to known 𝑞 𝑥𝑡−1 𝑥𝑡 (or 𝑞 𝑥𝑡−1 𝑥𝑡, 𝑥0 ) so that we can sample 𝑥𝑡−1? Answer: Approximate by 𝑝𝜃 𝑥𝑡−1 𝑥𝑡 which is a neural network parameterized by 𝜽. Question: Can we get the 𝑝𝜃(𝑥0)?

𝑇 Answer: 𝑝 𝑥 = 𝑝(𝑥) ς 𝑝 (𝑥 ∣ 𝑥) where 𝑝 𝑥 𝑥 = 𝒩 𝑥; 𝜇 (𝑥, ), Σ (𝑥, )

𝜃 0:𝑇 𝑇 𝑡=1 𝜃 𝑡−1 𝑡 𝜃 𝑡−1 𝑡 𝑡−1 𝜃 𝑡 𝜃 𝑡

## How to train diffusion models?

- Fix a forward process 𝑡 ▪ Fix 𝛽, recall 𝛼 = 1 − 𝛽, 𝛼ത = ς 𝛼, and 𝑥 = 𝛼ത 𝑥 + 1 − 𝛼ത 𝜖~𝑞 𝑥 𝑥 = 𝑡 𝑡 𝑡 𝑡 𝑠=0 𝑠 𝑡 𝑡 0 𝑡 𝑡 0 𝒩 𝑥𝑡; 𝛼ത𝑡𝑥0, 1 − 𝛼ത𝑡 𝑰 ▪ Ideally, start from 𝑥𝑇~𝑁(0, 𝐼), we want 𝑞(𝑥𝑡−1 ∣ 𝑥𝑡) to generate new sample, but it is intractable! ▪ Solution (Ho’ et al.): it turns out that conditioning on 𝑥0 then: 2
- 𝑞(𝑥𝑡−1|𝑥𝑡, 𝑥0) is tractable, indeed is a Gaussian with mean 𝜇෤𝑡 𝑥𝑡, 𝑥0 and covariance 𝜎𝑡 𝑰
- But 𝑥0 is problematic for generation as we don’t have it during generation process (why?)
- Good news: maximise its expected negative log-likelihood, 𝐸[−ln(𝑥0)] yields a tractable variational distribution 𝑝𝜃(𝑥𝑡−1 ∣ 𝑥𝑡) to approximate 𝑞 𝑥𝑡−1 𝑥𝑡, 𝑥0 in the upper bound (similar to VAE) by minimising: 𝐿𝑡−1 = 𝐷𝐾𝐿 𝑞 𝑥𝑡−1 𝑥𝑡, 𝑥0 ∥ 𝑝𝜃 𝑥𝑡−1 𝑥𝑡
- How to parameterise and learn 𝑝𝜃 𝑥𝑡−1 𝑥𝑡 ? 2 ▪ Assume it is a Gaussian with 𝜇𝜃(𝑥𝑡, ) and fixed covariance Σ 𝑥𝑡, = 𝜎𝑡 𝑰 1−𝛼ഥ𝑡 2 2
- Empirically, both 𝜎𝑡 = 𝛽𝑡 or 𝜎𝑡 = 𝛽𝑡 works well 1−𝛼ഥ𝑡 2

▪ Since both distributions in the argument are Gaussians with same covariance 𝜎𝑡 𝑰

1

2 𝐿𝑡−1 = 𝐷𝐾𝐿 𝑞 𝑥𝑡−1 𝑥𝑡, 𝑥0 ∥ 𝑝𝜃 𝑥𝑡−1 𝑥𝑡 = 𝔼𝑞 𝜇෤𝑡 𝑥𝑡, 𝑥0 − 𝜇𝜃 𝑥𝑡,

2 2𝜎

𝑡 1 𝛽𝑡 ▪ Since 𝜇෤𝑡 𝑥𝑡, 𝑥0 = 𝑥𝑡 − 𝜖 explicitly, introduce a ‘denoise’ network 𝜖𝜃(𝑥𝑡, ) so

1−𝛽 1−𝛼ഥ𝑡

𝑡 1 𝛽𝑡 2 that 𝜇𝜃 𝑥𝑡, = 𝑥𝑡 − 𝜖𝜃 𝑥𝑡, , hence now: 𝐿𝑡−1 = 𝔼𝑥 ~𝑞 𝑥,𝜖~𝒩(0,𝐼) 𝜖 − 𝜖𝜃 𝑥𝑡,

1−𝛼ഥ 0 0 1−𝛽𝑡 𝑡

## Training and Sampling Algorithms

Sample a starting noise 𝑥𝑇 from the At each training iteration:

Normal distribution  Sample 𝑥 from the training set

0

Iteratively remove noise:  Sample time step and noise 𝜖

o Predict amount of noise (mean) to be removed

𝜖𝜃 𝑥𝑡,  Train the U-Net with the denoising

o Remove noise from 𝑥𝑡 by subtracting the predicted matching loss at time step

noise (subtracting the mean) o Add small amount of noise to simulate sampling

𝛽𝑡(1−𝛼ന𝑡) operation. 𝜎𝑡 = or 𝜎𝑡 = 𝛽𝑡

1−𝛼𝑡 Return the last step 𝑥0

### Forward Diffusion Process – Noise Scheduler

How to add noise in each step

Scheduling noise 𝛽𝑡 is an art! Cannot add too much noise / too little noise at each step

𝑞 𝑥𝑡 𝑥𝑡−1 = 𝒩 𝑥𝑡; 1 − 𝛽𝑡𝑥𝑡−1, 𝛽𝑡𝑰 Two common approaches

𝑡 Linear schedule: 𝛼ത = ς (1 − 𝛽) where 𝛽 is a constant

𝑡 𝑠=0 𝑠 𝑠

𝑡 2 𝑓(𝑡) +𝑠 𝜋

𝑇 Cosine schedule: 𝛼ത𝑡 = where = cos ∙

𝑓(0) 1+𝑠 2

𝛼ത𝑡 throughout process

Latent samples from linear (top) and cosine (bottom) schedules. Samples in the second half of the linear schedule are almost noise, whereas those of cosine schedule are noising slowly. Image source: Nichol, Alexander Quinn, and Prafulla Dhariwal. "Improved denoising diffusion probabilistic models.", ICML, 2021

### Architecture and Time Embedding

The denoising network 𝜖𝜃(𝑥𝑡, ):  Uses U-Net architectures with residual blocks and self-attention layers  Input: previous sample 𝑥 and time embedding

𝑡  Output: amount of noise to be removed 𝜖 (𝑥, )

𝜃 𝑡 Time Embedding: Represent time step 𝒕 by a vector via FC layers and embed to U-Net (concatenate to each intermediate representation of U-Net 𝜖𝜃 with input 𝑥𝑡)

Image credit: https://cvpr2022-tutorial-diffusion-models.github.io/

## Latent Diffusion Model

The architecture empowers Stable Diffusion, ImageGen, Midjourney Three important contributions:  Perceptual image compression via Encoder 𝜀 and

Decoder 𝒟  Latent diffusion process with denoising U-Net 𝜖

𝜃  Conditioning mechanism so that we can generate images

by a prompt

Objective function: Attention mechanism at layer 𝒊:

𝑇 2 𝑄𝐾 DM: 𝔼𝑥 ~𝑞 𝑥,𝜖~𝒩(0,𝐼) 𝜖 − 𝜖𝜃 𝑥𝑡, where 𝑥𝑡 = 𝛼ത𝑡𝑥0 + 1 − 𝛼ത𝑡𝜖

0 0

𝐴 𝑛 𝑖𝑜𝑛 𝑄, 𝐾, 𝑉 = 𝑠𝑜 𝑚 𝑥 𝑉 2 LDM: 𝔼𝑧 ~𝜀 𝑥,𝜖~𝒩(0,𝐼) 𝜖 − 𝜖𝜃 𝑧𝑡, where 𝑧𝑡 = 𝛼ത𝑡𝑧0 + 1 − 𝛼ത𝑡𝜖

0 0

(𝑖) (𝑖) 𝑖 and 𝑧0~𝜀 𝑥0 is sampled from Encoder 𝜀 𝑄 = 𝑊 𝜑 (𝑧), 𝐾 = 𝑊 𝜏 (𝑦), V = 𝑊 𝜏 𝑦

𝑄 𝑖 𝑡 𝐾 𝜃 𝑉 𝜃 2 Conditional LDM: 𝔼𝑧 ~𝜀 𝑥,𝜖~𝒩(0,𝐼) 𝜖 − 𝜖𝜃 𝑧𝑡, , 𝜏𝜃(𝑦) where 𝑦 is

0 0 Here 𝜑𝑖(𝑧𝑡) denotes a flattened intermediate condition (i.e., a caption/sentence) and 𝜏𝜃 is a pre-trained domain representation at layer 𝒊 of the U-Net with input encoder 𝑧.

𝑡

Image source: Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models.”, CVPR, 2022

## Diffusion Models

###  A trendy and effective generative AI method

###  Consists of two processes

- Forward diffusion process: Iteratively adding noise to an input image, having closed- form and can be pre-computed
- Backward diffusion process: Iteratively removing noise to reconstruct the input image, can be approximated by a U-Net model

###  Learning is formulated as a probabilistic denoising process

- Objective function is a denoising matching term
- Need noise scheduler, time embedding, U-Net with self-attention to make it work

###  Latent diffusion models works on the latent space

- Diffusion is performed on the latent space hence deal with lower-dimensional space
- Faster, and gives better overall results
- Further enabling conditional generation – which is hugely applicable in real-world applications.

## Further Recommended Readings

- Ho et al., Denoising Diffusion Probabilistic Models, NeurIPS 2020
- Song et al., Generating Modeling by Estimating Gradients of the Data Distribution, NeurIPS 2020
- Variational Autoencoders and Diffusion Models, by Dr. Ruiqi Gao from Google DeepMind
- What are Diffusion Models? By Dr. Lilian Weng

## Summary

###  Introduction to generative models

- Explicit versus implicit density estimation
- What is a generative model?
- Why do we need to be generative?

###  Generative Adversarial Networks

- Adversarial training and minimax two-player game
- Formulation and optimal solution
- Issues and challenges with GAN

###  Diffusion Models

## Thanks for your attention!

Question time