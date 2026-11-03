COPYRIGHT WARNING: Copyright in these original lectures is owned by Monash University. You may transcribe, take notes, download or stream lectures for the purpose of your research and study only. If used for any other purpose, (excluding exceptions in the Copyright Act 1969 (Cth)) the University may take legal action for infringement of copyright.

Do not share, redistribute, or upload the lecture to a third party without a written permission!

## FIT3181/5215 Deep Learning

Week 03: Convolutional Neural Networks

CE/Lecturer: Dr Trung Le (Clayton)

Lecturer: Dr Binh Nguyen (Clayton) Lecturer: Dr Ting Fung Fung (Malaysia) Lecturer: Dr Arghya Pal (Malaysia) Lecturer: Dr Sailaja Rajanala (Malaysia)

Email: trunglm@monash.edu

Department of Data Science and AI Faculty of Information Technology, Monash University, Australia

2

# Outline

## ◻ Origin of human vision

## ◻ Fundamental of computer vision

## ◻ Convolutional Neural Networks

o Convolutional operators in 1D, 2D, 3D o Standard CNN architecture:

▪ Convolutional layer, Pooling layer, Fully-connected layer o CNN-based image recognition system

▪ Batch norm and dropout layers ▪ LeNet, AlexNet, and VGG

## ◻ Further reading recommendation

o [HandsOn, chapter 14], [FDL, chapter 5], [Dive into DL, chapter 6]. o “CS231n: Convolutional Neural Networks for Visual Recognition”, Stanford University, 2016.

## Origin of Human Vision

# Human vision

### ◻ The vision begins with eyes, but truly take places in

### the brain.

### ◻ The collaboration between the eyes and the brain,

### called the primary visual pathway, is the reason

### we can make sense of the world around us.

4

6

# Recognition is easy and fundamental

(Source: https://www.youtube.com/watch?v=40riCqvRoMs)

A boy A boy is playing A boy is playing in the park

Biological neurons in the visual cortex respond to specific patterns in small regions of the visual field called receptive fields; as the visual signal makes its way through consecutive brain modules, neurons respond to more complex patterns in (Source: Hands On, Ch. 14) larger receptive fields.

(Source: www.pngwing.com)

7

# Recognition is much harder for machines

Human vision

8

# Human vision vs computer vision

pixel

Human vision Computer vision

o Computer ‘see’ in a different way than

we do. They are pixels! o Every image can be represented as a 2-

dimensional arrays of numbers, each represents a pixel.

9

# Pixels: the building block of images

1,000 pixels wide

### ◻ Pixels are the raw building blocks of an

### image. Every image consists of a set of 750

pixels tall

### pixels.

### ◻ Most pixels are represented in two ways:

o Grayscale/single channel

\# pixels= 750 x1,000= 750,000 ▪ A scalar between 0 (black) and 255 (white) o Color

▪ Color pixels are represented in RGB system

o Red, green, and blue channels

10 255

# RGB colour space

◻ 0

\+ = + \=

blue green red

HSV and YCbCr are other ways to represent colors.

11

# Image coordinate system

An image of letter I. Each pixel has a coordinate.

width width

height

height

A colour image can be viewed as a 3D A grayscale image can viewed as a 2D tensor tensor

12

# Images as Tensor

2 images

height

width

RGB channels

## Convolutional Neural Networks

# How to simulate human’s vision?

visual receptive field

high level

mid level low level

High-level feature David H. Hubel and Torsten Wiesel performed a series of - Object (Source: Hands On, Ch. 14)

experiments on cats and monkeys in 1958 and 1959 (the Nobel Prize in Physiology or Medicine in 1981). ❖ Low-level neurons can capture small visual receptive fields

with low-level patterns. ❖ Higher-level neurons are based on the outputs of neighboring Mid-level feature

- Shape

lower-level neurons that can detect all sorts of complex patterns in any area of the visual field.

Low-level feature ◻ Building-blocks of Convolutional Neural Networks

- Edge

o Convolutional layers o Pooling layers

▪ Max pooling or average pooling o Fully-connected layers o Batch normalization layers

15

16

# General architecture of CNNs

Histogram of oriented gradients (HoG)

### 1\. Convolutional layer

### 2\. Pooling layer

### 3\. Fully connected layer

# Why do we need Convolution layers?

softmax

What layers should we place in the middle? Can we still use the linear layers?

[3,32,32]

Output layer 10 neurons for 10 classes

17

# Why do we need Convolution layers? ReLU (Rectified Linear Unit) is an activation

function commonly used in neural networks. ReLU ReLU

softmax

[3,32,32]

Flatten to [1, 3x32x32]

Output layer Cannot take advantage of spatial

10 neurons Information carried in pixels.

for 10 classes Input layer 3x32x32 FC layer FC layer neurons 128 neurons 64 neurons

18

## Convolutional Layer

# Convolution operation

◻

20

21

# Convolution operation – 1D filter

size = 1x14 0 1 2 1 1 2 0 1 0 1 2 0 0 1

filter = 1x2

0.5 0.5

22

# Convolution operation – 1D filter

size = 1x14 0 1 2 1 1 2 0 1 0 1 2 0 0 1

filter = 1x2

0.5 0.5

0.5

23

# Convolution operation – 1D filter

size = 1x14 0 1 2 1 1 2 0 1 0 1 2 0 0 1

filter = 1x2

0.5 0.5 stride = 2

0.5 1.5

24

# Convolution operation – 1D filter

size = 1x14 0 1 2 1 1 2 0 1 0 1 2 0 0 1

filter = 1x2

0.5 0.5 stride = 2

0.5 1.5 1.5

25

# Convolution operation – 1D filter

size = 14 0 1 2 1 1 2 0 1 0 1 2 0 0 1

filter = 2x1

0.5 0.5 stride = 2

0.5 1.5 1.5 0.5 0.5 1.0 0.5

26

# Convolution operation – 1D filter

Zero-padding What to do when it is not perfect fit? size = 1x14 0 1 2 1 1 2 0 1 0 1 2 0 0 1 0 filter = 1x2

0.5 0.5 0.5 0.5 stride = 1

0.5 0.5

0.5 0.5 …

Exercise 1: what is the output size in this case? Exercise 2: if stride = 3, do we need zero-padding, and if yes, what would be the output size?

# Convolution layer with zero padding

strides = (2,2) 0 0 0 0 0 0 0 0 0 feature map 0 1 -2 -1 5 3 2 1 0 1 1 1 3 8 19 7 0 1 3 -1 4 3 3 1 0 1 1 1

3 16 30 10 0 1 -2 1 6 3 3 2 0 1 1 1

6 11 22 5 0 2 -2 2 5 2 1 0 0

2 -2 8 3 0 0 3 -2 5 4 1 2 0 0 1 2 -3 1 1 2 -1 0

◻◻  TheThe  sliding windowsliding window  moves frommoves from  leftleft  🡪to  rightright, , toptop  🡪 tobottom bottom with strides. with strides. 0 1 -2 -1 1 2 1 1 0

◻◻  WeWe  convolveconvolve  thethe  filterfilter  and theand the  sliding windowssliding windows  to work out theto work out the  neuronsneurons 0 0 0 0 0 0 0 0 0 on theon the  feature mapfeature map..

27

# Convolution layer with zero padding

strides = (2,2) 0 0 0 0 0 0 0 0 0 feature map 0 1 -2 -1 5 3 2 1 0 1 1 1 3 8 19 7 0 1 3 -1 4 3 3 1 0 1 1 1

3 15 30 10 0 1 -2 1 6 3 3 2 0 1 1 1

6 11 22 5 0 2 -2 2 5 2 1 0 0

2 -2 9 3 0 0 3 -2 5 4 1 2 0 0 1 2 -3 1 1 2 -1 0

◻ The sliding window moves from left to right, top to bottom with strides. 0 1 -2 -1 1 2 1 1 0

◻ We convolve the filter and the sliding windows to work out the neurons 0 0 0 0 0 0 0 0 0 on the feature map.

(Source: https://towardsdatascience.com/)

28

# Convolution layer

strides = (2,2) 0 0 0 0 0 0 0 0 0 feature map 0 1 -2 -1 5 3 2 1 0 1 1 1 3 8 19 7 0 1 3 -1 4 3 3 1 0 1 1 1

3 15 30 10 0 1 -2 1 6 3 3 2 0 1 1 1

6 11 22 5 0 2 -2 2 5 2 1 0 0

2 -2 9 3 0 0 3 -2 5 4 1 2 0 0 1 2 -3 1 1 2 -1 0

◻◻  TheThe  sliding windowsliding window  moves frommoves from  leftleft  🡪to  rightright, , toptop  🡪 tobottom bottom with strides. with strides. 0 1 -2 -1 1 2 1 1 0

◻◻  WeWe  convolveconvolve  thethe  filterfilter  and theand the  sliding windowssliding windows  to work out theto work out the  neuronsneurons 0 0 0 0 0 0 0 0 0 on theon the  feature mapfeature map..

29

# Convolution layer

strides = (2,2) 1 -2 -1 5 3 2 1 1 1 3 -1 4 3 3 1 -1 feature map 1 -2 1 6 3 3 2 2 1 1 1 1 23 21 2 -2 2 5 2 1 0 -1 1 1 1 3 26 18 0 3 -2 5 4 1 2 1 1 1 1

-1 8 13 1 2 -3 1 1 2 -1 2 1 -2 -1 1 2 1 1 3

p=0 zero padding

30

# Convolution layer

strides = (2,2) 1 -2 -1 5 3 2 1 1 1 3 -1 4 3 3 1 -1 feature map 1 -2 1 6 3 3 2 2 1 1 1 1 23 21 2 -2 2 5 2 1 0 -1 1 1 1 3 26 18 0 3 -2 5 4 1 2 1 1 1 1

-1 8 13 1 2 -3 1 1 2 -1 2 1 -2 -1 1 2 1 1 3

p=0 zero padding

(Source: Internet)

31

32

# Convolution 2D without zero padding

1 -2 -1 5 3 2 1 1

1 3 -1 4 3 3 1 -1

1 -2 1 6 3 3 2 2

2 -2 2 5 2 1 0 -1 1 1 1

0 3 -2 5 4 1 2 1 1 1 1

1 2 -3 1 1 2 -1 2 1 1 1

1 -2 -1 1 2 1 1 3

33

# Convolution 2D without zero padding

1 -2 -1 5 3 2 1 1

1 3 -1 4 3 3 1 -1

feature 1 -2 1 6 3 3 2 2

map 1x1 1x-2 1x-1

1 2 -2 2 5 2 1 0 -1 1 1 1

1x1 1x3 1x-1 0 3 -2 5 4 1 2 1 1 1 1 =

1x1 1x-2 1x1 1 2 -3 1 1 2 -1 2 1 1 1

1-2-1+1+3-1+1-2+1 = 1 1 -2 -1 1 2 1 1 3

34

# Convolution 2D without zero padding

move left 2 move down

2 1 -2 -1 5 3 2 1 1

1 3 -1 4 3 3 1 -1

feature 1 -2 1 6 3 3 2 2

map 1x1 1x6 1x3

1 2 -2 2 5 2 1 0 -1 1 1 1

1x2 1x5 1x2 0 3 -2 5 4 1 2 1 1 1 1 =

1x-2 1x5 1x4 1 2 -3 1 1 2 -1 2 1 1 1

1+6+3+2+5+2-2+5+4 = 26 1 -2 -1 1 2 1 1 3

35

# Convolution 2D without zero padding

move left 2 move down

2 1 -2 -1 5 3 2 1 1

1 3 -1 4 3 3 1 -1

feature 1 -2 1 6 3 3 2 2

map 1x1 1x6 1x3

1 2 -2 2 5 2 1 0 -1 1 1 1

1x2 1x5 1x2 0 3 -2 5 4 1 2 1 1 1 1 = 26

1x-2 1x5 1x4 1 2 -3 1 1 2 -1 2 1 1 1

1+6+3+2+5+2-2+5+4 = 26 1 -2 -1 1 2 1 1 3

36

# Convolution 2D without zero padding

1 -2 -1 5 3 2 1 1

1 3 -1 4 3 3 1 -1

feature 1 -2 1 6 3 3 2 2

map 1 2 -2 2 5 2 1 0 -1 1 1 1 0 3 -2 5 4 1 2 1 1 1 1 3 26 1 2 -3 1 1 2 -1 2 1 1 1

1 -2 -1 1 2 1 1 3

37

# Convolution 2D without zero padding

1 -2 -1 5 3 2 1 1

1 3 -1 4 3 3 1 -1

feature map 1 -2 1 6 3 3 2 2

1 2 -2 2 5 2 1 0 -1 1 1 1 0 3 -2 5 4 1 2 1 1 1 1 3 26 1 2 -3 1 1 2 -1 2 1 1 1 -1 1 -2 -1 1 2 1 1 3

38

# Convolution 2D without zero padding

1 -2 -1 5 3 2 1 1

1 3 -1 4 3 3 1 -1

feature map 1 -2 1 6 3 3 2 2 2 -2 2 5 2 1 0 -1 1 1 1 1 23 21 0 3 -2 5 4 1 2 1 1 1 1 3 26 17 1 2 -3 1 1 2 -1 2 1 1 1

-1 8 13 1 -2 -1 1 2 1 1 3

39

# Convolution 2D with zero padding

0 0 0 0 0 0 0 0 0 0 1 -2 -1 5 3 2 1 0 feature

map 0 1 3 -1 4 3 3 1 0 3 8 20 7

1 1 1 0 1 -2 1 6 3 3 2 0

3 16 30 10 1 1 1 0 2 -2 2 5 2 1 0 0

6 13 22 7 1 1 1 0 0 3 -2 5 4 1 2 0

2 -2 7 3 0 1 2 -3 1 1 2 -1 0 0 1 -2 -1 1 2 1 1 0 0 0 0 0 0 0 0 0 0

(Source: https://towardsdatascience.com/)

40

# Convolution layer

Example

output

input

s=1, p=0, f=3 s=2, p=1, f=3

### s = stride, p = padding, f = filter size

### (square) (Source: An intuitive guide to Convolutional Neural Networks)

# Convolution layer with multiple filters and feature maps

strides = (2,2)

feature map1 1 -2 -1 5 3 2 1 1

1 1 1 1 23 21 1 3 -1 4 3 3 1 -1

1 1 1 3 26 18 feature volume (2,3,3) 1 -2 1 6 3 3 2 2 feature maps (2,3,3)

1 1 1

-1 8 13 2 -2 2 5 2 1 0 -1

feature map2 0 3 -2 5 4 1 2 1

-1 -1 -1 -1 -23 -21 1 2 -3 1 1 2 -1 2

-1 -1 -1 -3 -26 -18 1 -2 -1 1 2 1 1 3

-1 -1 -1

1 -8 -13

41

# Convolution layer with multiple filters and feature maps

strides = (2,2)

feature map1 1 -2 -1 5 3 2 1 1

1 1 1 1 23 21 1 3 -1 4 3 3 1 -1

1 1 1 3 26 18 feature volume (2,3,3) 1 -2 1 6 3 3 2 2 feature maps (2,3,3)

1 1 1

-1 8 13 2 -2 2 5 2 1 0 -1

feature map2 0 3 -2 5 4 1 2 1

-1 -1 -1 -1 -23 -21 1 2 -3 1 1 2 -1 2

-1 -1 -1 -3 -26 -18 1 -2 -1 1 2 1 1 3

-1 -1 -1

1 -8 -13

Input images (batch_size,3,32,32)

Inputs (batch_size,4,7,8) Feature volume (batch_size,2,3,3)

CONV with filters (2,4,3,3) Feed through some layers

batch_size, in_channels , height, width,

out_channels, in_channels, fil_height, fil_width, 42

43

# Example of convolutional operation with PyTorch

44

# Effects of filters/kernels to images

CNNs learn meaningful filters automatically from data!

45

# Summary: full convolutional layer

•

filters f

p

stride s

s = stride, p = padding, f = filter size (square)

46

# General architecture of CNNs

### 1\. Convolutional layer

### 2\. Pooling layer

### 3\. Fully connected layer

## Pooling layer

48

# Pooling operation

## ◻ Makes the representations smaller

## and more manageable

## ◻ Subsample the image

## ◻ Operates over each feature map

## independently

(Source: FDL Ch. 5)

49

# Max/average pooling

•

(Source: medium.com)

50

# Pooling operation

### ◻ Makes the representations smaller and

### more manageable

### ◻ Operates over each channel

### independently

51

# Example of pooling with PyTorch

Max pooling

Average pooling

## Fully connected layer

53

# Fully connected layer

Feature extractor Classifier

Fully connected layers [3,32,32]

flatten [5,5,10]

[1, 5x5x10]

◻

## Put them all together

55

# General architecture of CNNs

### • Convolution stage

### • Pooling stage

### • Fully connected stage

“CS231n: Convolutional Neural Networks for Visual Recognition”, Stanford, 2016

Filters [4,3,4,4]

# CNN in

Filter 1

# Operation

softmax [3,4,4]

Filter 2

Feature Feature

volume volume

Feature Feature

Pooling maps maps [3,4,4] layer

Filter 3

pool-size=(2,2) strides= (2,2) Input Layer padding= 1 [3,32,32]

[4,8,8] [4,15,15] [3,4,4] Latent Latent

representation representation Output Filter 4

layer 10 neurons for 10 classes FC layer [3,4,4]

4x8x8 neurons Conv2D 1 padding= 0 strides = (2,2)

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

## Batch Normalization Layer

60

# Covariate shift vs internal covariate shift

Covariate shift problem Internal covariate shift

(Source: DL course Andrew Ng)

Covariate shift problem

- The distribution (nature) of training data is

different from that of testing data.

# Batch Normalization

### 1\. Cope with internal covariate shift

### 2\. Reduce gradient

### vanishing/exploding

### 3\. Reduce overfitting

### 4\. Make training more stable

### 5\. Converge faster

1. Allow us to train with bigger learning rate

0.1 0.2 … …

Mini-batch size 0.5 0.4

(Source: medium.com) Standardize column-wise

61

62

# Batch Normalization

Testing Phase

•

# Implementation of BatchNorm2D in PyTorch

Self-study

BatchNorm2D

Normalize

…. ….

63

## Dropout Layer

65

# Dropout

Reduce Overfitting

Without dropout

With dropout

(Source: chatbotslife) (Source: Analytics Vidhya)

◻ This is a cheap technique to reduce model capacity

o Reduce overfitting ◻ In each iteration, at each layer, randomly choose some neurons and drop all connections from

these neurons o dropout_rate= 1 – keep_prob

# Dropout

Reduce Overfitting

◻

66

## Some popular CNN architectures

## Network Architectures

AlexNet

VGG16

ResNet

70

# LeNet and AlexNet

(Source:Dive into DL)

### • The design philosophies of AlexNet and LeNet

### are very similar, but there are also significant

### differences.

o AlexNet is much deeper than the comparatively small LeNet5. o AlexNet consists of eight layers

▪ Five convolutional layers, two fully-connected hidden layers, and

one fully-connected output layer. o AlexNet used the ReLU instead of the sigmoid o In AlexNet’s first layer, the convolution filter shape is 11×11

because objects in ImageNet data tend to occupy more pixels.

LeNet AlexNet

71

# VGG Network

VGG block nd

### • Won 2 in ImageNet competition in 2014

### • Moving from thinking in terms of individual neurons

### 🡪 whole layers 🡪 blocks (repeating patterns of

### layers)

### • VGG block

o One VGG block consists of a sequence of convolutional layers, VGG network followed by a max pooling layer for spatial downsampling.

### • VGG network

o Consists of many VGG blocks, followed by fully-connected layers o The original VGG network had 5 convolutional blocks

▪ The first two have one convolutional layer each and the latter three contain two convolutional layers each. ▪ The first block has 64 output channels and each subsequent block doubles the number of output channels, until that number reaches 512\.

(Source: Dive into DL)

72

# MiniVGG for Cifar10

Our Tutorial

60,000 images 10 classes

## Thanks for your attention!