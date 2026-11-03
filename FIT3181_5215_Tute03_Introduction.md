### FIT3181/5215 Deep Learning

# Convolutional Neural Networks

### Trung Le and Teaching team

Department of Data Science and AI Faculty of Information Technology, Monash University

Email: trunglm@monash.edu

2

# Outline

## ❑ Fundamentals of convolutional neural networks (Tute 3a)

## ❑ Implementing CNNs in PyTorch (Tute 3b)

### o MiniVGG network

## ❑ Implementing CNNs with Data Augmentation (Tute 3c)

## ❑ Visualize feature maps and filters (Tute 3c)

# Fundamentals of convolutional neural networks

# Conv2D in PyTorch

Examples of tensors. [Source: datacamp]

𝑖𝑛_ℎ𝑒𝑖𝑔ℎ𝑡 + 2𝑝 − 𝑘𝑒𝑟𝑛𝑒𝑙_𝑠𝑖𝑧𝑒 𝑖𝑛_𝑤𝑖𝑑𝑡ℎ + 2𝑝 − 𝑘𝑒𝑟𝑛𝑒𝑙_𝑠𝑖𝑧𝑒 𝑜𝑢𝑡_ℎ𝑒𝑖𝑔ℎ𝑡 = + 1 𝑜𝑢𝑡_𝑤𝑖𝑑𝑡ℎ = + 1

𝑠𝑡𝑟𝑖𝑑𝑒 𝑠𝑡𝑟𝑖𝑑𝑒

4

5

# Example of pooling with PyTorch

Max pooling

Average pooling

# Additional reading

## ❑ The effects of kernels to the input image

### ❑ Small blur, large blur filters

### ❑ Sharpening filter

### ❑ Laplacian filter

### ❑ SobelX, sobelY filters

### ❑ Emboss filter

6

# Implementing CNNs in PyTorch

8

# MiniVGG for CIFAR-10

## Our Tutorial

50,000 images 10 classes

# MiniVGG for CIFAR-10 dataset

Download CIFAR-10 Visualize dataset Process dataset

9

Build up MiniVGG Declare loss and optimizer

# MiniVGG for

# CIFAR-10

# Train the model dataset

10

# Data augmentation

## ❑ Apply the simple transformations over training images to augment data. Model will be

## challenged with diverge data examples which might be encountered in the testing set

❑ Rotation, Width Shifting, Height Shifting, Brightness, Shear Intensity, Zoom, Channel Shift, Horizontal Flip, Vertical Flip ❑ https://medium.com/mlait/image-data-augmentation-image-processing-in-tensorflow-part-2-b77237256df0

Original image Flip image Grey-scale image Change image brightness

Rotate image

Center crop image

11

# Data augmentation for training our model

Without augmentation

With augmentation

12

# Visualize feature maps and filters

Visualize filters Visualize feature maps

13

# Thanks for your attention!

### Any question?