COPYRIGHT WARNING: Copyright in these original lectures is owned by Monash University. You may transcribe, take notes, download or stream lectures for the purpose of your research and study only. If used for any other purpose, (excluding exceptions in the Copyright Act 1969 (Cth)) the University may take legal action for infringement of copyright.

Do not share, redistribute, or upload the lecture to a third party without a written permission!

### FIT3181/5215 Deep Learning

Week 06:Advanced Convolutional Neural Networks

CE/Lecturer: Dr Trung Le (Clayton)

Lecturer: Dr Binh Nguyen (Clayton) Lecturer: Dr Ting Fung Fung (Malaysia) Lecturer: Dr Arghya Pal (Malaysia) Lecturer: Dr Sailaja Rajanala (Malaysia)

Email: trunglm@monash.edu

Department of Data Science and AI Faculty of Information Technology, Monash University, Australia

2

# Outline

###  Deep learning before and after 2012

###  Revision of building-blocks, rationales and drawbacks of CNNs

###  SOTA Convolutional Neural Networks

###  Visualization in CNNs

###  Adversarial Machine Learning

###  Further Reading Recommendation

o [FDL, ch05], [HandsOn-2nd, ch14], [Dive Into Deep Learning, ch07] o “CS231n: Convolutional Neural Networks for Visual Recognition”, Stanford University, 2016

## Deep Learning before and after 2012

4

# Deep learning renaissance

### Before 2012 After 2012

 Some milestones before 2012  AlexNet won ImageNet competition

o Backprop was proposed in 1986

 Missing Ingredient: Data o CNN was proposed in 1995

o In 2009, the ImageNet dataset was released,

challenging researchers to learn models from 1  People did not believe that NNs work

million examples, 1000 each from 1000 distinct o Could not train NNs for real-world datasets

categories of objects o Easy to get overfitting

 Missing Ingredient: Hardware  In machine learning (computer vision),

o Graphical processing units (GPUs) proved to be a hand-crafted feature approach dominated game changer in making deep learning feasible. o SIFT (scale-invariant feature transform), SURF o These chips had long been developed to

(speeded up robust features) optimize for high throughput 4×4 matrix-vector

products, which are needed for many computer graphics tasks and calculate convolutional layers  The renaissance of Deep Learning

## Hand-crafted feature

Visual data

Learning algorithm and feature extractor are disconnected !

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

## Automatic feature extractor

Visual data

Classification and feature extractor are

connected !

###  Deep learning approach (automatic feature learning)

Input Feature extraction Classification

Automatic features

Prediction Vision dataset

6

## Hand-crafted and automatic feature extractor

Learning algorithm Visual data

and feature extractor are disconnected !

###  Traditional approach (hand-crafted feature learning)

Input Feature extraction (SIFT, SURF, HOG) Learning algorithms (SVM, XGBoost)

Hand-crafted features

Classification and feature extractor are

connected !

###  Deep learning approach (automatic feature learning)

Input Feature extraction Classification

Automatic features

7

## Achievements of CNNs

SOTA in object detection and semantic segmentation Win human in object recognition on ImageNet dataset

### Some fiction and imaginary tasks

©Dinh Phung, 2017 Generative AI Tesla self-driving cars Image captioning

DALL.E/Mid Journey

8

## Revision of CNN building-blocks

## and How CNN works?

## Convolution layer with multiple filters and feature maps

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

11

## Convolution layer with multiple filters and feature maps

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

out_channels, in_channels, fil_height, fil_width, 12

Filters [4,3,4,4]

## CNN in

Filter 1

## Operation

𝑜𝑢𝑡_𝑠𝑖𝑧𝑒 𝑜𝑢𝑡_𝑠𝑖𝑧𝑒

15 + 2 × 1 − 2 32 + 0 − 4 softmax [3,4,4] =

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

13

Filters [4,3,4,4]

4x8x8

## CNN in

Filter 1 2

10

## Operation

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

14

## Automatic feature extractor

Deep learning for visual data

receptive receptive

field field neuron

Neurons on higher layers have larger receptive fields (patches) on input images High-level feature/representation

- objects

Mid-level feature/representation

- circles, triangles, boxes

Low-level feature/representation

- edges, pixels, corners

15

16 Feature map 1

# How CNNs work?

Human face recognition ☺

Filter 1 (depth=1)

Filter 1 (depth=2) Filter 2 (depth=1)

Feature map

Feature map 2

 Higher level filters combine lower level filters to represent more abstract objects

o The context is locally expanded

17 Feature map 1

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

## Pooling layer

###  Max/average-pooling is locally invariant

### and applied independently to each

### feature map

###  Reduce the size of feature map to be

### more manageable

o Smaller size for flattening

###  Reduce the difference of output tensors

18

down-sampling and sharpening

## Why Do We Need Pooling Layer?

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

19

## Some comments from Hinton

Paper link: https://arxiv.org/pdf/1710.09829.pdf

(Source: medium.com)

 CNNs cannot capture the spatial

relationships among objects, hence it wrongly predicts every face with two eyes, one nose, and one mouth as a human face.

(Source: original paper of Capsule Net)

20

## Global pooling layer

###  Like pooling layer, but the kernel

### size is set to input size

o Each feature map becomes a single neuron o Global average pooling (GAP) o Global max pooling (GMP) o 4D input tensor [batch_size, in_channel, in_height, in_width]→ 2D output tensor [batch_size, in_channel]

21

Filters [4,3,4,4]

## Global AdaptiveAvgPool2d((1, 1))

Filter 1

Flatten(1)

## Pooling Layer

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

22

## SOTA CNNs

25

# GoogleNet

### Szegedy et al. ‘14

###  Won ImageNet competition in 2014.

###  Address the question of which sized

### convolution kernels are best.

o Previous popular networks employed choices as small

as 1×1 and as large as 7×7. o Sometimes it can be advantageous to employ a combination

of variously-sized kernels.

###  The building-block is Inception block

Inception block (Source:Dive into DL)

26

# GoogleNet

### Szegedy et al. ‘14

### 22 layers

(Source: Hands On, Ch. 15)

## ResNet

###  Won ImageNet competition in 2015

###  Many ResNet blocks

###  Each ResNet block contains many

### residual blocks

o Number of residual blocks o Number of channels for residual blocks

###  Each ResNet block

o From the second block, the first residual block

▪ Use 1x1 Conv skip connection o Otherwise

▪ Use standard skip connection

ResNet architecture

27

28

# ResNet

### Residual Block

(Source:Dive into DL)

Without skip connection With skip connection

###  Learn 𝑓 𝑥 = 𝑔 𝑥 + 𝑥 where 𝑔 𝑥 = 𝑓 𝑥 − 𝑥

o The model expressiveness is the same as previous o The gradient ∇𝑓 𝑥 = ∇𝑔 𝑥 + 𝟏 looks better where 𝟏 is the vector of all 1 with the

same shape as 𝑥

29

# ResNet

### Residual Block

Not use 1x1 Conv Use 1x1 Conv

 ResNet follows VGG’s full 3×3 convolutional layer design.

o The residual block has two 3×3 convolutional layers with the same number of output channels. o Each convolutional layer is followed by a batch normalization layer and a ReLU activation function. o Skip these two convolution operations and add the input directly before the final ReLU activation

function. o Requires that the output of the two convolutional layers must be of the same shape as the input, so

that they can be added together. ▪ If we want to change the number of channels, we need to introduce an additional 1×1 convolutional layer to

transform the input into the desired shape for the addition operation.

30

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

31

# ResNet

(16,32,32)

### Residual Block

padding=1 (16,32,32)

strides= (1,1) (16,32,32)

\#filters=16

padding=1 strides= (2,2) \#filters=16 padding=1

(16,32,32) strides= (2,2)

\#filters=16 (8,64,64) Not use 1x1 Conv Use 1x1 Conv

 ResNet follows VGG’s full 3×3 convolutional layer design.

o The residual block has two 3×3 convolutional layers with the same number of output channels. o Each convolutional layer is followed by a batch normalization layer and a ReLU activation function. o Skip these two convolution operations and add the input directly before the final ReLU activation

function. o Requires that the output of the two convolutional layers must be of the same shape as the input, so

that they can be added together. ▪ If we want to change the number of channels, we need to introduce an additional 1×1 convolutional layer to

transform the input into the desired shape for the addition operation.

32

# Implementation of Residual Block

(Source:Dive into DL)

(Source:Dive into DL)

## Implementation of ResNet

Residual block

ResNet block

Residual block

down sample Input size by 2

Residual block

ResNet block

Residual block

33

(Source:Dive into DL)

## Implementation of ResNet

Residual block

ResNet block

Residual block

down sample Input size by 2 (

1, 64,112,112) Residual block (1, 64,112,112)

(1, 64,112,112) (1, 64,56,56) ResNet block (1, 64,56,56)

(1, 128,28,28) (1, 256,14,14) (1, 256,1,1) Residual block (1,256)

(1,10)

34

## Visualizing CNNs: Activation Map

## Visualization and GradCam

## First layer visualization

First layer visualization

Visualization of filters at higher layers is not that interesting

Tiny 3x3 filters

(CS231n: Convolutional Neural Networks for Visual Recognition”, Stanford University, 2016) 37

Feature volume Filters [4,3,4,4]

## Feature maps Visualize a Filter

Filter 1

### Activation Map Visualization

Feature How to visualize map 1 (𝒇)

𝟏 filter 1?

softmax [3,4,4]

Filter 2

Feature map 2 (𝒇𝟐)

Some layers Some layers

[3,4,4] Filter 3

Feature map 3 (𝒇𝟑) Input Layer

[3,4,4]

Output Filter 4

layer Feature map 4 (𝒇) 10 neurons

𝟒

for 10 classes FC layer [3,4,4]

8x8x4 neurons Conv2D 1 padding= 0 strides = (2,2)

38

Feature volume Filters [4,3,4,4]

## Feature maps Visualize a Filter

Filter 1

### Activation Map Visualization

Feature ❖ How to visualize filter 1? map 1 (𝒇𝟏)

∗ ❖ Find input 𝑰 that maximally

softmax fires filter 1. [3,4,4] ∗ ❖ 𝑰 = 𝒂𝒓𝒈𝒎𝒂𝒙𝑰 𝒎𝒆𝒂𝒏(𝒇𝟏)

Filter 2

Feature map 2 (𝒇𝟐)

Some layers Some layers

[3,4,4] Filter 3

Feature map 3 (𝒇𝟑) Input Layer

[3,4,4]

Output Filter 4

layer Feature map 4 (𝒇𝟒) 10 neurons

for 10 classes FC layer [3,4,4]

8x8x4 neurons Conv2D 1 padding= 0 strides = (2,2)

39

Feature volume Filters [4,3,4,4]

## ❖ How to visualize filter 1? Feature maps Visualize a Filter

∗ ❖ Find input 𝑰 that maximally Filter 1

### Activation Map Visualization

fires filter 1. Feature ∗ map 1 (𝒇) ❖ 𝑰 = 𝒂𝒓𝒈𝒎𝒂𝒙 𝒎𝒆𝒂𝒏(𝒇) 𝟏

𝑰 𝟏

softmax [3,4,4] 𝐼0 initialized randomly

Filter 2

Feature map 2 (𝒇𝟐)

Some layers Some layers 𝐼1 [3,4,4]

Filter 3

Feature map 3 (𝒇𝟑) …

𝐼𝑡 [3,4,4]

Output Filter 4

layer Feature map 4 (𝒇𝟒) 10 neurons

for 10 classes 𝐼𝑡+1 = 𝐼𝑡 + 𝜂∇𝐼𝒜(𝐼𝑡)

FC layer 𝒜 𝐼 = 𝑚𝑒𝑎𝑛(𝑓) [3,4,4]

1

8x8x4 neurons Conv2D 1 …

padding= 0 strides = (2,2)

40

## Activation Map Visualization

Filter 1 Feature map 1 (𝒇𝟏) 1 1 1 -1 23 21 1 1 1 1 26 17 1 1 1

-1 8 13 Feed to our CNN

Filter 2 Feature map 2 (𝒇𝟐) -1 -1 -1 1 -23 -21 -1 -1 -1 -1 -26 -17 Input image I -1 -1 -1 1 -8 -13

∗

###  Find the input image 𝑰 that maximally activate a feature map corresponding to a filter

∗ o 𝑰 shows the patterns in input images that can fire (maximally activate) a certain filter

∗

###  To visualize filter 1, we find the input 𝑰 that maximizes average of feature map 𝑓

1 ∗ o 𝐼 = 𝑎𝑟𝑔𝑚𝑎𝑥 𝒜(𝐼) where 𝓐 𝑰 : = 𝒎𝒆𝒂𝒏 𝒇 − 𝜶𝜴(𝑰)

𝐼 𝟏 o Ω(𝐼) is a norm over 𝐼 to encourage the sparsity

###  Using gradient ascent for updating

o 𝐼𝑡+1 = 𝐼𝑡 + 𝜂∇𝐼𝒜(𝐼𝑡) o 𝜂 > 0 is the learning rate

41

## Activation map visualization of VGG19

42

Feature volume Filters [4,3,4,4]

## Feature maps Visualize a Class

Filter 1

Feature ❖ How to visualize class 1? map 1 (𝒇𝟏)

∗ ❖ Find input 𝑰 that maximizes

softmax prediction probability 𝒑𝟏. [3,4,4] ∗ ❖ 𝑰 = 𝒂𝒓𝒈𝒎𝒂𝒙𝑰 𝒑𝟏

Filter 2 𝑝

1 Feature

𝑝2 map 2 (𝒇𝟐)

. Some layers Some layers

. [3,4,4]

. Filter 3

Feature. map 3 (𝒇𝟑) Input Layer

𝑝10 [3,4,4]

Output Filter 4

layer Feature map 4 (𝒇𝟒) 10 neurons

for 10 classes FC layer [3,4,4]

8x8x4 neurons Conv2D 1 padding= 0 strides = (2,2)

43

## GradCam Model Explanability

### ❑ Which part of image that a CNN (e.g., ResNet 50) highly bases on to make

### prediction?

Original image Gradcam Guided Gradcam

44

## GradCam Model

𝒚ෝ: predicted label

𝒑𝒚ෝ

## Explainability

𝜕𝑝𝑦ො 𝜕𝑥𝑖𝑗

𝑝 = [𝑝1, … , 𝑝𝑦ො, … , 𝑝1000] 𝑦ො is a predicted label

A layer with 3D tensor feature maps [100,7,7]

𝐴 = 𝐴1, 𝐴2, … , 𝐴100 with shape [100,7,7] 𝐴𝑖 with shape [7,7] is a feature map. 7 𝜕𝑝

100 𝑦ෝ 7 ℎ𝑒𝑎𝑡𝑚𝑎𝑝 = σ 𝑠𝑢𝑚 × 𝐴 has shape [7,7]

𝑖=1 𝑖

𝜕𝐴𝑖

Normalize heatmap and resize to [224, 224]

Salience/Heat map

Salience/Heat map

𝒙𝒊𝒋

224

224 (𝒙, 𝑦) Which part of images mostly contribute Link: https://arxiv.org/pdf/1610.02391 Link: https://arxiv.org/pdf/1312.6034

to the model prediction? 45

## Adversarial Machine Learning

48

# Some notions

𝑓 𝑥; 𝜃 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ(𝑥; 𝜃)) (prediction probabilities) softmax

ℎ(𝑥; 𝜃) (logits)

𝑀 𝑓 𝑥; 𝜃 = 𝑓𝑚 𝑥; 𝜃

𝑚=1 𝒉 𝑝 𝑦 = 𝑚 𝑥 = 𝑓𝑚(𝑥; 𝜃) for 1 ≤ 𝑚 ≤ 𝑀 𝜽

𝑙 𝑓 𝑥; 𝜃 , 𝑦 = 𝐶𝐸 𝟏𝑦, 𝑓 𝑥; 𝜃 = − log 𝑓𝑦 (𝑥; 𝜃)

𝑥, 𝑦 ∈ {1,2, … , 𝑀}

49

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

50

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

51

# Adversarial examples

(Source: OpenAI) (Source: Internet) 𝒙 𝒙𝒂𝒅𝒗 = 𝒙 + 𝒏𝒐𝒊𝒔𝒆

Small perturbation

Misleading autonomous car driving system

not difference difference

###  Adversarial example 𝒙 of a clean data example 𝒙 w.r.t a model

𝒂𝒅𝒗

### 𝑓(. ; 𝜃)

o 𝑑(𝑥𝑎𝑑𝑣, 𝑥) = ‖𝑥𝑎𝑑𝑣 − 𝑥‖ ≤ 𝜖 where . is a norm (e.g., . 1, . 2, . ∞) o 𝑓 predicts 𝑥 and 𝑥𝑎𝑑𝑣 with different labels, i.e., 𝑎𝑟𝑔𝑚𝑎𝑥𝑖𝑓𝑖 𝑥; 𝜃 ≠ 𝑎𝑟𝑔𝑚𝑎𝑥𝑖𝑓𝑖 𝑥𝑎𝑑𝑣; 𝜃

52

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

53

# Decision regions of deep learning model

𝒙(0,0) 𝒙(0,0) 𝒙(0,0) 𝒙(0,0) 𝒙(0,0) ⊥ 𝑔

Each colour represents a class

𝑇 ⊥ 𝑔 𝑔 = 0

𝒙(0,0) 𝑔 = ∇ 𝑙(𝑓(𝑥; 𝜃), 𝑦)

𝑥

###  Deep learning models are fragile and easy to be attacked

⊥ o Start from 𝑥 = 𝑥 + 0𝑔 + 0𝑔 (coordinate (𝟎, 𝟎)), if we go along with the gradient

⊥ direction 𝑔 = ∇𝑥𝑙(𝑓(𝑥; 𝜃), 𝑦), we can reach 𝑥𝑎𝑑𝑣 = 𝑥 + 𝛼𝑔 + 0𝑔 such that ▪ 𝑥𝑎𝑑𝑣 is very close to 𝑥 → human cannot tell how they are different→ the same class from

human perceptron ▪ 𝑥𝑎𝑑𝑣 is predicted to different class from 𝑥 by the classifier ▪ 𝑥𝑎𝑑𝑣 is called adversarial example of 𝑥

Delving into Transferable Adversarial Examples and Black-box Attacks Yanpei L., et al., ICLR 2016

54

# Loss surface of deep learning model

𝒈 𝒈 ⊥ ⊥ 𝒈 𝒈

Zoom out Zoom in

⊥ The loss surface of DL model around 𝒙. This is plotted on 𝒙′ = 𝒙 + 𝝐𝟏𝒈 + 𝝐𝟐𝒈.

###  At the data point 𝒙, if we move along with the gradient direction

### 𝑔, the loss function is maximally locally increased.

###  Taylor expansion around 𝑥

𝑇 o 𝑙 𝑓(𝑥 + ℎ; 𝜃), 𝑦 ≈ 𝑙 𝑓(𝑥; 𝜃), 𝑦 + 𝑔 ℎ where 𝑔 = ∇𝑥𝑙(𝑓(𝑥; 𝜃), 𝑦). o The gradient direction is the steepest direction to increase the loss function locally.

Ensemble Adversarial Training Attacks and Defences Tramer et al., ICLR 2017

55

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

56

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

## Implementation of Attacks

FGSM attack PGD attack 57

## Adversarial Examples and Perturbations

58

59

# Adversarial training

Loss surface 𝒍(𝒇 𝒙′; 𝜽 , 𝒚)

We wish a smooth (or flat) loss surface around 𝒙 𝒙

###  The ideal optimization problem for adversarial training

′ o min 𝔼 𝑥,𝑦 ~𝑝(𝑥,𝑦) max 𝑙(𝑓 𝑥; 𝜃 , 𝑦)

𝜃 ′

𝑥 ∈𝐵𝜖(𝑥) o 𝑝(𝑥, 𝑦) is the data distribution that generates the pair of data point 𝑥 and label 𝑦 o Minimize the loss of over most violated adversarial examples

###  PGD adversarial training

o for epoch in n_epochs

▪ for iter in range(n_iter_per_epoch)

o Sample mini-batch 𝑥1, 𝑦1 , … , (𝑥𝑏, 𝑦𝑏) from the training set

𝑎𝑑𝑣 𝑎𝑑𝑣 o Find PGD untargeted adversarial examples 𝑥, … , 𝑥 for 𝑥1, … , 𝑥𝑏 w.r.t labels 𝑦1, … , 𝑦𝑏

1 𝑏 1 𝑏 1 𝑏 𝑎𝑑𝑣 o 𝑏𝑎𝑡𝑐ℎ_𝑙𝑜𝑠𝑠 = σ 𝑙(𝑓 𝑥; 𝜃 , 𝑦) + σ 𝑙(𝑓 𝑥; 𝜃 , 𝑦)

𝑖=1 𝑖 𝑖 𝑖=1 𝑖 𝑖 𝑏 𝑏 𝜕𝑏𝑎𝑡𝑐ℎ_𝑙𝑜𝑠𝑠 o 𝜃 = 𝜃 − 𝜂

𝜕𝜃

60

# Adversarial training in practice

Principle to develop machine learning models ❖ Training condition must match

testing condition

###  PGD adversarial training in practice

o for epoch in n_epochs

▪ for iter in range(n_iter_per_epoch)

o Sample mini-batch 𝑥1, 𝑦1 , … , (𝑥𝑏, 𝑦𝑏) from the training set

𝑝𝑟𝑒𝑑 𝑝𝑟𝑒𝑑 𝑝𝑟𝑒𝑑 o Evaluate the predicted labels 𝑦, … , 𝑦 by the current model, i.e., 𝑦 = 𝑎𝑟𝑔𝑚𝑎𝑥𝑚𝑓𝑚(𝑥1; 𝜃) , and etc.

1 𝑏 1

𝑎𝑑𝑣 𝑎𝑑𝑣 𝑝𝑟𝑒𝑑 𝑝𝑟𝑒𝑑 o Find PGD untargeted adversarial examples 𝑥, … , 𝑥 for 𝑥1, … , 𝑥𝑏 w.r.t labels 𝑦, … , 𝑦

1 𝑏 1 𝑏 1 𝑏 1 𝑏 𝑎𝑑𝑣 o 𝑏𝑎𝑡𝑐ℎ_𝑙𝑜𝑠𝑠 = σ 𝑙(𝑓 𝑥; 𝜃 , 𝑦) + σ 𝑙(𝑓 𝑥; 𝜃 , 𝑦)

𝑖=1 𝑖 𝑖 𝑖=1 𝑖 𝑖 𝑏 𝑏 𝜕𝑏𝑎𝑡𝑐ℎ_𝑙𝑜𝑠𝑠 o 𝜃 = 𝜃 − 𝜂

𝜕𝜃

61

# Attack vs Defense: Cat-and-mouse game

attack

defense

evolves evolves

(Source: Internet)

###  Attack is stronger than defense

###  No defense method can really defend on ImageNet dataset.

62

# Summary

###  Deep learning before and after 2012

###  Revision of building-blocks of CNN

o Conv layer, pooling layer, global pooling layer o Drawback of CNNs

###  SOTA Convolutional Neural Networks

o GoogLeNet and ResNet

###  Visualization in CNNs

o Activation map, visualizing filters, last layer nearest neighbour

###  Adversarial Machine Learning

o Adversarial examples, attack and defense

## Thanks for your attention!