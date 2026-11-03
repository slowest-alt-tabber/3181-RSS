### FIT3181/5215 Deep Learning

# Advanced Convolutional Neural

# Networks

### Trung Le and Tutor team

Department of Data Science and AI Faculty of Information Technology, Monash University

Email: trunglm@monash.edu

# Summary

## ❑ Tutorial 6a: Implementation of ResNet (\*\*\*\*\*)

## ❑ Tutorial 6b: Adversarial Machine Learning (\*\*\*\*\*)

### ❑ Questions 3.6 and 3.7 in Assignment 1.

## ❑ Tutorial 6c:

### ❑ Visualization of Filters and Feature Maps of CNNs (\*\*\*)

### ❑ Gradcam and GuidedGradcam

# ResNet

Residual block

Residual block

ResNet block

Residual block

down sample Input size by 2

Residual block

ResNet block

ResNet block

Residual block

Our ResNet

# Attacks

### FGSM, PGD, MIM, TRADES

𝒙 = 𝒂𝒓𝒈𝒎𝒂𝒙 ′ 𝒍(𝒇 𝒙′; 𝜽 , 𝒚) 𝒂𝒅𝒗 𝒙 ∈𝑩 𝒙

𝝐

 Projected Gradient Descent (Ascent)

(PGD) o 𝑥0 = 𝑥 + 𝑈𝑛𝑖𝑓𝑜𝑟𝑚([−𝜖, 𝜖]) o 𝒙෥𝑡+1 = 𝑥𝑡 + 𝜂𝑠𝑖𝑔𝑛(∇𝑥𝑙 𝑓 𝑥𝑡; 𝜃 , 𝑦 ) o 𝑥𝑡+1 = 𝑐𝑙𝑖𝑝(𝑥 ෤𝑡+1, 𝑚𝑖𝑛_𝑣𝑎𝑙, 𝑚𝑎𝑥_𝑣𝑎𝑙)

Our implementation of PGD attack 4

# Attack SOTA pretrained model

# Attack Normal LeNet

Create LeNet Train LeNet

Very low robust accuracy 6

# Adversarial Training to Improve LeNet

Train one batch at an iteration Train in several epochs using adversarial training

Testing robust accuracy

# CNN Visualization

## ❑ Visualize feature maps

## ❑ Visualize raw filters

## ❑ Visualize filters using

## activation map maximization

# GradCam Model Explanability

## ❑ Which part of image that a CNN (e.g., ResNet 50) highly bases on to make

## prediction?

Original image Gradcam Guided Gradcam

# Gradcam Model Explanability

𝑝 = [𝑝1, … , 𝑝𝑦ො , … , 𝑝1000] 𝑦ො is a predicted label

A layer with 3D tensor feature maps

𝐴 = 𝐴1, 𝐴2, … , 𝐴2048 with shape [2048,7,7] 𝐴𝑖 with shape [7,7] is a feature map.

ℎ𝑒𝑎𝑡𝑚𝑎𝑝 =

𝜕𝑝 2048 𝑦ෝ σ 𝑠𝑢𝑚 × 𝐴 has 𝑖=1 𝜕𝐴 𝑖

𝑖 shape [7,7]

Normalize heatmap and resize to [224, 224]

[224, 224]

# Thanks for your attention!