FIT3181/5215 Deep Learning

### Knowledge Distillation

Trung Le and Tutor team Department of Data Science and AI Faculty of Information Technology, Monash University

Email: trunglm@monash.edu

## Knowledge Distillation

1. What: transfer knowledge from a complex model (Teacher) to a simpler model (Student)
2. Why: reduce computational complexity
3. How: train a compact deep neural networks to reproduce the output or mimic the behavior of a large complex model (which can be pre-trained or trained concurrently with the student)

## Knowledge Distillation

There are two main families of distillation techniques Output distillation: ❑ Cross-entropy (KD) [1] Feature distillation: mimic the hidden features of teacher network ❑ Feature response (FITNET) [2] ❑ Spatial attention (AT) [3] ❑ Feature distribution (PKT) [4]

[1] Geoffrey Hinton et. al. Distilling the Knowledge in a Neural Network. In NIPS Workshop, 2014 [2] Adriana Romero et. al. Fitnets: Hints for thin deep nets. In ICLR, 2015. [3] Nikos Komodakis and Sergey Zagoruyko. Paying more attention to attention: improving the performance of convolutional neural networks via attention transfer. In ICLR, 2017. [4] Nikolaos Passalis and Anastasios Tefas. Learning deep representations with probabilistic knowledge transfer. In ECCV, 2018

## Resources

1. Geoffrey Hinton et. al. Distilling the Knowledge in a Neural Network. In NIPS Workshop, 2014. https://arxiv.org/abs/1503.02531v1 - 2015

2. Adriana Romero et. al. Fitnets: Hints for thin deep nets. In ICLR, 2015. https://arxiv.org/abs/1412.6550

3. Nikos Komodakis and Sergey Zagoruyko. Paying more attention to attention: improving the performance of convolutional neural networks via attention transfer. In ICLR, 2017. https://arxiv.org/abs/1612.03928

4. Nikolaos Passalis and Anastasios Tefas. Learning deep representations with probabilistic knowledge transfer. In ECCV, 2018. https://arxiv.org/abs/1803.10837

## Resources (Advanced)

1. Yuan, Li, et al. Revisiting knowledge distillation via label smoothing regularization. CVPR. 2020. https://arxiv.org/abs/1909.11723

2. Zhao, Borui, et al. Decoupled knowledge distillation. CVPR. 2022. https://arxiv.org/abs/2203.08679

3. Chen, Pengguang, et al. Distilling knowledge via knowledge review. CVPR.

2021. https://arxiv.org/abs/1503.02531

### Thanks for your attention!