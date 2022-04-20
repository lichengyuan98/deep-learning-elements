[toc]

# deep-learning-elements

## 1. Introduction

> **这个仓库用于储存可能用于反应堆事故诊断的代码片段**。
>
> **This repository is primarily used to store these code blocks, i.e. source material that can be used as diagnostic models for reactor accidents.** 

反应堆事故诊断任务是一个对安全性要求较高的任务，因为这将直接决定操纵员进入哪种特定事故的处置规程。在近30余年的发展中，基于数据（ML & DL）的诊断模型发展迅速，并且取得了在速度和精度上显著超越传统基于专家知识的诊断系统，因此成为了主要的诊断方案。

事故诊断任务存在以下特征：

1. 对诊断**速度**和**精度**的要求较高；
2. 对**事故类型**和**严重程度**均需诊断，不同事故对应的严重程度度量方式亦不同；
3. 智能化的事故诊断系统不能取代操纵员，因此需要给操纵员**提供充分的辅助信息**；
4. 需要考虑保守性假设，即用于诊断的原始数据**需要假设存在一定缺失**。

事故诊断任务所依据的监测系统的原始数据具备以下特征：

1. 数据拥有**时间序列特征**；
2. 数据量庞大，可供监测的热工水力**参数非常多**；
3. 数据**冗余性大**，热工水力参数之间的非线性耦合程度高，且存在潜在的相关性；
4. 数据**来源局限**，目前用于训练模型的数据大都来源于系统分析程序，鲜有实验支撑。

因此，在将深度学习技术应用于事故诊断任务时，首先定义：反应堆事故诊断是个**针对时序数据的不平衡样本多标签模式识别问题**。在问题已被定义的基础上，若干先进的深度学习算法可以应用于此。然而，尽管若干算法的代码已经开源，但是存在以下问题，导致工程研究人员不能快速聚焦于算法本身：

1. **寻找核心代码**：算法核心代码隐藏于CV、NLP领域具体问题之中，需要读通完整的代码后才能定位核心代码的位置；
2. **阅读核心代码**：源代码缺少解说性的注释，对于初学者来说不容易理解算法中复杂的张量变换；
3. **使用核心代码**：纵使读懂了算法的基本原理，但算法的应用之处仍受制于个人知识和经验的局限性，对于应如何植入算法可能会比较迷茫。

为了解决上述问题，本仓库致力于提供开箱即用的算法核心代码（`Pytorch`版），并且通过文本注释和MD文档讲解，让初次涉足此领域的工程、科研人员能够快速上手。请各位同仁在[Issue页](https://github.com/lichengyuan98/deep-learning-elements/issues)提出自己的见解，欢迎`Star`、`Fork`、`Pull`三连！！！

## 2. 单点——网络结构

+ Residual Block

+ Attention

+ PreNorm

+ PE: Positional Encoding

+ SE: Squeeze and Excitation

## 3. 套餐——网络成品

### 模式识别

+ **ViT**: Vision Transformer
+ **CoAtNet**: Convolution + Attention

### 表征学习

+ **MAE**: Masked Auto-Encoder
+ **VAE**: Variational Auto-Encoder
