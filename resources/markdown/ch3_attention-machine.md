---
title: 实现注意力机制 | LLM
description: 从零构建大语言模型第3章:本章讲解注意力机制
pubDate: 2025 09 22
categories:
  - tech
tags:
  - llm
---

![](https://skindhu.github.io/Build-A-Large-Language-Model-CN/Image/chapter3/figure3.1.png)

本章我们将实现四种不同的注意力机制.

![](https://skindhu.github.io/Build-A-Large-Language-Model-CN/Image/chapter3/figure3.2.png)

## 长序列建模的问题

### 循环神经网络(RNN)

在编码器-解码器架构的 RNN 网络中，输入文本被输入到编码器中，编码器按顺序处理文本内容。在每个步骤中，编码器会更新其隐状态（即隐藏层的内部值），试图在最终的隐状态中捕捉整个输入句子的含义，如图所示。随后，解码器使用该最终隐状态来开始逐词生成翻译句子。解码器在每一步也会更新其隐状态，用于携带生成下一个词所需的上下文信息。

![](https://skindhu.github.io/Build-A-Large-Language-Model-CN/Image/chapter3/figure3.4.png)

这里的关键思想在于，编码器部分将整个输入文本处理为==一个==隐藏状态（记忆单元）。解码器随后使用该隐藏状态生成输出.可以将这个隐藏状态视为一个嵌入向量.

编码器-解码器架构的 RNN 的一个重大问题和限制在于，**在解码阶段 RNN 无法直接访问编码器的早期隐藏状态**。因此，它只能依赖当前隐藏状态来封装所有相关信息。这种设计可能导致上下文信息的丢失，特别是在依赖关系较长的复杂句子中，这一问题尤为突出。

> RNN用最后一个隐藏层的信息进入解码器,它聚合了以前的所有编码信息,所有可能导致上下文信息的丢失,这点很好理解.

正是这一缺点促成了注意力机制的设计.

### 注意力机制的解决方法

它的关键思想是**在处理每个词时，不仅依赖于最后的隐藏状态，而是允许模型直接关注序列中的所有词**。这样，即使是较远的词也能在模型计算当前词的语义时直接参与。

## 通过注意力机制捕捉数据依赖关系

![](https://skindhu.github.io/Build-A-Large-Language-Model-CN/Image/chapter3/figure3.6.png)

## 通过自注意力机制关注输入的不同部分

### 一种不含可训练权重的简化自注意力机制

![](https://skindhu.github.io/Build-A-Large-Language-Model-CN/Image/chapter3/figure3.7.png)

在自注意力机制中，上下文向量起着关键作用。它们的目的是通过整合序列中所有其他元素的信息（如同一个句子中的其他词），为输入序列中的每个元素创建丰富的表示.

![](https://skindhu.github.io/Build-A-Large-Language-Model-CN/Image/chapter3/figure3.8.png)

实现自注意力机制的第一步是计算中间值 **ω**，即注意力得分.

每个输入token会先通过权重矩阵$W$ 分别计算出它的Q,K,V三个向量:

- **Q向量（查询向量）**：查询向量代表了这个词在寻找相关信息时提出的问题
- **K向量（键向量）**：键向量代表了一个单词的特征，或者说是这个单词如何"展示"自己，以便其它单词可以与它进行匹配
- **V向量（值向量）**：值向量携带的是这个单词的具体信息，也就是当一个单词被"注意到"时，它提供给关注者的内容

具体生成Q、K、V向量的方式主要通过线性变换：

```
Q1 = W_Q * (E1 + Pos1)
K1 = W_K * (E1 + Pos1)
V1 = W_V * (E1 + Pos1)
```

依次类推，为所有token生成`Q`，`K`，`V`向量，其中`W_Q`，`W_K`和`W_V`是Transformer训练出的权重（每一层不同）

针对每一个目标token，Transformer会计算它的 `Q向量` 与其它所有的token的 `K向量` 的==点积==，以确定每个词对当前词的重要性（即注意力分数）

> 为什么是K与Q呢?V做什么?

接下来，我们对先前计算的每个注意力分数进行归一化.

![](https://skindhu.github.io/Build-A-Large-Language-Model-CN/Image/chapter3/figure3.9.png)

> 我知道,点积可以衡量相似度,但最好先标准化(?我忘记准确的名词了,就是使其长度为1,上过高中都知道).这里算出来诸多注意力分数后再归一化.

归一化采用的是`softmax`函数:

$$ softmax(z_{i}) = \frac{e^{z_{i}}}{\sum_{j}e^{z_{j}}}$$

> `softmax`是神经网络常用的激活函数,有一些优点

最后,计算上下文向量的方法是每个输入向量与对应的注意力权重的加权和.

### 为所有输入的token计算注意力权重

```python
attn_scores = inputs @ inputs.T
print(attn_scores)
```

`inputs`是[token_num, embedding_dim],结果的形状是[token_num, token_num].即所有输入对的注意力得分.

进一步计算注意力权重:

```python
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
```

在使用 PyTorch 时，像 `torch.softmax` 这样的函数中的 `dim` 参数指定了将在输入张量中的哪个维度上进行归一化计算。
`dim=-1`表示表示沿着最后一个维度进行归一化操作.

## 实现带有可训练权重的自注意力机制

![](https://skindhu.github.io/Build-A-Large-Language-Model-CN/Image/chapter3/figure3.13.png)
> 我发现按以上方法没有任何可训练权重.原始Transformer架构通过设置可训练的`W_Q`/`W_K`/`W_V`矩阵来实现这一点.

这种自注意力机制也被称为放缩点积注意力.

### 逐步计算注意力权重

我们通过引入三个可训练的权重矩阵：Wq、Wk 和 Wv 来逐步实现自注意力机制。这三个矩阵用于将嵌入后的输入 token $x^{(i)}$ 映射为查询向量、键向量和值向量.

![](https://skindhu.github.io/Build-A-Large-Language-Model-CN/Image/chapter3/figure3.14.png)

```python
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out))
W_value = torch.nn.Parameter(torch.rand(d_in, d_out))
```

在 GPT 类模型中，输入维度和输出维度通常是相同的。

我们可以通过矩阵乘法获取所有元素的key和value向量：

```python
keys = inputs @ W_key
values = inputs @ W_value
```

> 在标准的自注意力机制中，W、K、V向量都是固定的，然而，由于 GPT 模型是由多层自注意力模块堆叠而成，每一层都会根据当前输入和上下文信息，动态调整查询、键和值向量的**权重矩阵**。因此，即使初始的词嵌入和权重矩阵是固定的，经过多层处理后，模型能够生成与当前上下文相关的 Q、K、V 向量权重矩阵，最终计算出的Q、K、V 向量也就能反映出上下文的语义了。

![](https://skindhu.github.io/Build-A-Large-Language-Model-CN/Image/chapter3/figure3.16.png)
接下来的注意力权重计算,使用点积缩放注意力机制,将注意力得分除以嵌入维度的平方根来进行缩放:

```python
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)
```

> 在自注意力机制中，查询向量（Query）与键向量（Key）之间的点积用于计算注意力权重。然而，当嵌入维度较大时，点积的结果可能会非常大。那么大的点积对接下来的计算有哪些具体影响呢？
>
> - **Softmax函数的特性**：在计算注意力权重时，点积结果会通过Softmax函数转换为概率分布。而Softmax函数对输入值的差异非常敏感，当输入值较大时，Softmax的输出会趋近于0或1，表现得类似于阶跃函数（step function）。
> - **梯度消失问题**：当Softmax的输出接近0或1时，其梯度会非常小，接近于零（可以通过3.3.1小节中提到的Softmax公式推断）。这意味着在反向传播过程中，梯度更新幅度会很小，导致模型学习速度减慢，甚至训练停滞。
> 为了解决上述问题，在计算点积后，将结果除以嵌入维度的平方根。这样可以将点积结果缩放到适当的范围，避免Softmax函数进入梯度平缓区，从而保持梯度的有效性，促进模型的正常训练。

![](https://skindhu.github.io/Build-A-Large-Language-Model-CN/Image/chapter3/figure3.17.png)

现在我们通过值向量的加权和来计算上下文向量。这里，注意力权重作为加权因子，用于衡量每个值向量的重要性。

```python
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)
```

这就是一个上下文向量$z^{(2)}$.我们将计算输入序列中的所有上下文向量,从$z^{(1)}$到$z^{(T)}$.

### 实现一个简洁的子注意力机制Python类

`torch.nn.Module`是PyTorch模型的基础组件,提供了创建和管理模型层所需的必要功能.

![](https://skindhu.github.io/Build-A-Large-Language-Model-CN/Image/chapter3/figure3.18.png)

## 使用因果注意力机制来屏蔽后续词

屏蔽注意力是一种特殊的自注意力形式.它限制模型在处理任何给定的 token 时，只能考虑序列中的前一个和当前输入，而不能看到后续的内容。这与标准的自注意力机制形成对比，后者允许模型同时访问整个输入序列。

![](https://skindhu.github.io/Build-A-Large-Language-Model-CN/Image/chapter3/figure3.19.png)

### 应用因果注意力掩码

![](https://skindhu.github.io/Build-A-Large-Language-Model-CN/Image/chapter3/figure3.20.png)

我们可以使用 PyTorch 的 `tril` 函数生成一个掩码矩阵，使对角线以上的值为零：

```python
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)
```

生成的掩码如下所示：

```python
tensor([[1., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1.]])
```

现在，我们可以将这个掩码矩阵与注意力权重相乘，从而将对角线以上的值置零。

```python
masked_simple = attn_weights*mask_simple
```

第三步是将注意力权重重新归一化,使得每一行的权重和再次等于1.

> 我一开始觉得,两次softmax混入了后面的信息,实则不然,证明如下:
> 第一次softmax:
> $$ \frac{e^{z_{1}}}{\sum e^{z_{j}}}, \frac{e^{z_{2}}}{\sum e^{z_{j}}},\frac{e^{z_{3}}}{\sum e^{z_{j}}},\dots$$
> 第二次softmax:
> $$ \frac{e^{(1)}}{e^{(1)}+e^{(2)}+e^{(3)}},\frac{e^{(2)}}{e^{(1)}+e^{(2)}+e^{(3)}},\frac{e^{(3)}}{e^{(1)}+e^{(2)}+e^{(3)}} $$
> 化简得$$\frac{e^{z_{1}}}{e^{z_{1}}+e^{z_{2}}+e^{z_{3}}},\frac{e^{z_{2}}}{e^{z_{1}}+e^{z_{2}}+e^{z_{3}}},\frac{e^{z_{3}}}{e^{z_{1}}+e^{z_{2}}+e^{z_{3}}}$$
> 不包含后面的信息

![](https://skindhu.github.io/Build-A-Large-Language-Model-CN/Image/chapter3/figure3.21.png)

```python
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1) print(attn_weights)
```

### 使用`dropout`遮掩额外的注意力权重

Dropout 在深度学习中是一种技术，即在训练过程中随机忽略一些隐藏层单元，实际上将它们“丢弃”。这种方法有助于防止过拟合，确保模型不会过于依赖任何特定的隐藏层单元组合。

![](https://skindhu.github.io/Build-A-Large-Language-Model-CN/Image/chapter3/figure3.22.png)

当对注意力权重矩阵应用 50% 的 dropout 时，矩阵中一半的元素会被随机设置为零。为了补偿有效元素的减少，矩阵中剩余元素的值会被放大 1/0.5 = 2 倍。这个缩放操作至关重要，可以在训练和推理阶段保持注意力机制的整体权重平衡，确保注意力机制在这两个阶段的平均影响保持一致。

### 实现一个简洁的因果注意力类

`CausalAttention`类添加了`dropout`和因果掩码组件.

## 从单头注意力拓展到多头注意力

多头’一词指的是将注意力机制划分为多个‘头’，每个头独立运作。在这种情况下，单个因果注意力模块可以视为单头注意力，即只有一组注意力权重用于按顺序处理输入。

### 堆叠多层单头注意力

每个实例都具有独立的权重，然后将它们的输出合并。
它对于识别复杂模式至关重要，这是基于 Transformer 的大语言模型所擅长的能力之一。

![](https://skindhu.github.io/Build-A-Large-Language-Model-CN/Image/chapter3/figure3.24.png)

![](https://skindhu.github.io/Build-A-Large-Language-Model-CN/Image/chapter3/figure3.25.png)

### 通过权重分割实现多头注意力机制

![](https://skindhu.github.io/Build-A-Large-Language-Model-CN/Image/chapter3/figure3.26.png)

Q,K,V张量的拆分是通过张量的重塑和转置操作实现的.

关键操作是将 `d_out` 维度拆分成 `num_heads` 和 `head_dim`，其中 `head_dim = d_out / num_heads`。这种拆分通过 `.view` 方法实现：将形状为 `(b, num_tokens, d_out)` 的张量重塑为 `(b, num_tokens, num_heads, head_dim)`。

接下来对张量进行转置操作，将 `num_heads` 维度移动到 `num_tokens` 维度之前，使其形状变为 `(b, num_heads, num_tokens, head_dim)`。这种转置对于在不同注意力头之间正确对齐查询（queries）、键（keys）和值（values），并高效执行批量矩阵乘法至关重要。

在多头注意力机制中，计算完注意力权重和上下文向量之后，将所有头的上下文向量转置回形状 `(b, num_tokens, num_heads, head_dim)`。然后将这些向量重新塑形（展平）为 `(b, num_tokens, d_out)` 的形状，从而有效地将所有头的输出组合在一起。

此外，我们在多头注意力机制中添加了一个称为输出投影层（self.out_proj）的模块，用于在组合多个头的输出后进行投影。而在因果注意力类中并没有这个投影层。这个输出投影层并非绝对必要（详见附录 B 的参考部分），但由于它在许多 LLM 架构中被广泛使用，因此我们在这里加上以保持完整性。

> 这个线性层的作用是什么?

最小的 GPT-2 模型（1.17 亿参数）具有 12 个注意力头和 768 的上下文向量嵌入大小。而最大的 GPT-2 模型（15 亿参数）则具有 25 个注意力头和 1600 的上下文向量嵌入大小。请注意，在 GPT 模型中，token 输入的嵌入大小与上下文嵌入大小是相同的（`d_in = d_out`）。

## 本章摘要

> 本章的注意力机制是整本书中最重要的内容
