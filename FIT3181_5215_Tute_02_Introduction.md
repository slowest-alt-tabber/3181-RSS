COPYRIGHT WARNING: Copyright in these original lectures is owned by Monash University. You may transcribe, take notes, download or stream lectures for the purpose of your research and study only. If used for any other purpose, (excluding exceptions in the Copyright Act 1969 (Cth)) the University may take legal action for infringement of copyright.

Do not share, redistribute, or upload the lecture to a third party without a written permission!

# FIT3181/5215 Deep Learning

## Tutorial 02: Feed-forward Neural Networks

## Lecturers: Dr Trung Le, Dr Binh Nguyen

## Head TAs: Dr Ruda Nie and Mr Minh Vo

### Email: trunglm@monash.edu

Department of Data Science and AI Faculty of Information Technology, Monash University, Australia

# Introduction

#  The purpose of this tutorial is to practise:

# 1\. Understand the forward propagation

# 2\. How to implement feed-forward NNs in PyTorch

## Shuffling training set and splitting

## to many equal batches

# Mini-batch Forward and Loss

### 32 32 32

## Batch Batch … … Batch

## 1 2 1000

## 1 epoch

#  Optimizer uses batch loss to update model parameters

𝑙 𝑙 𝐿

### o 𝜃 = 𝑊, 𝑏

𝑙=1

# Implementation Feed-forward NNs with PyTorch

## Declare a DL model using Loss and optimizer

## torch.nn.Sequential

## Train model

# Diagnosing the Training

# Fine-tune the Learning Rate Using Valid Set

## DL Pipeline 6

# Another Approach to Implement DL Models with PyTorch

## Declare a class inherited from

## torch.nn.Module

# Thanks for your attention!

# Enjoy the tutorial 2!