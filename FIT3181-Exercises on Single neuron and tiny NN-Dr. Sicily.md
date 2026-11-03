FIT3181- Deep Learning (Malaysia Cohort) Prepared by Dr. Sicily Ting

## FIT3181-Week 4: Additional exercises’ Calculation

## steps:

### Exercises 1: The Single Neuron

Zone 1: Step 1: Forward pass The neuron first computes a weighted sum plus bias:

𝑧 = 𝑥1𝑤1 + 𝑥2𝑤2 + 𝑏

Insert the given values:

𝑧 = 2(0.5) + 1(−0.5) − 0.5

FIT3181- Deep Learning (Malaysia Cohort) Prepared by Dr. Sicily Ting

Multiply each input by its weight:

𝑧 = 1 − 0.5 − 0.5

Combine the terms:

𝑧 = 0

Now use the sigmoid activation:

𝑦̂ = 𝜎(𝑧)

1 𝜎(𝑧) =

−𝑧 1 + 𝑒

Substitute 𝑧 = 0:

1 𝑦̂ =

0 1 + 𝑒

1 𝑦̂ =

2

𝑦̂ = 0.5

Zone 2: Loss: The squared-error loss is:

1

2 𝐿 = (𝑦̂ − 𝑦)

2 Substitute prediction 0.5 and target 1:

1

2 𝐿 = (0.5 − 1)

2

1

2 𝐿 = (−0.5)

2

𝐿 = 0.125

Zone 3: Backward pass: Start from the loss and move backward. The derivative of squared error with respect to prediction is:

𝜕𝐿

= 𝑦̂ − 𝑦 𝜕𝑦̂

FIT3181- Deep Learning (Malaysia Cohort) Prepared by Dr. Sicily Ting

Substitute the known prediction and target:

𝜕𝐿

= 0.5 − 1 𝜕𝑦̂

𝜕𝐿

= −0.5 𝜕𝑦̂

The derivative of sigmoid is:

𝜕𝑦̂

= 𝑦̂ (1 − 𝑦̂ ) 𝜕𝑧

Substitute 𝑦̂ = 0.5:

𝜕𝑦̂

= 0.5(1 − 0.5) 𝜕𝑧

𝜕𝑦̂

= 0.25 𝜕𝑧

Now apply the chain rule:

𝜕𝐿 𝜕𝐿 𝜕𝑦̂

\= 𝜕𝑧 𝜕𝑦̂ 𝜕𝑧

𝜕𝐿

= (−0.5)(0.25) 𝜕𝑧

𝜕𝐿

= −0.125 𝜕𝑧

This is the neuron’s delta: the error signal that will be distributed to its parameters.

Zone 4: Updates: Because:

𝑧 = 𝑥1𝑤1 + 𝑥2𝑤2 + 𝑏

the local derivatives are:

𝜕𝑧

= 𝑥1 𝜕𝑤1

𝜕𝑧

= 𝑥2 𝜕𝑤2

𝜕𝑧

= 1 𝜕𝑏

FIT3181- Deep Learning (Malaysia Cohort) Prepared by Dr. Sicily Ting

Apply the chain rule to 𝑤1:

𝜕𝐿 𝜕𝐿 𝜕𝑧

\= 𝜕𝑤1 𝜕𝑧 𝜕𝑤1

𝜕𝐿

= (−0.125)(2) 𝜕𝑤1

𝜕𝐿

= −0.25 𝜕𝑤1

For 𝑤2:

𝜕𝐿

= (−0.125)(1) 𝜕𝑤2

𝜕𝐿

= −0.125 𝜕𝑤2

For bias:

𝜕𝐿

= (−0.125)(1) 𝜕𝑏

𝜕𝐿

= −0.125 𝜕𝑏

Gradient-descent update Use the update rule:

𝜕𝐿 𝜃new = 𝜃old − 𝜂

𝜕𝜃

The learning rate is:

𝜂 = 0.1

Update 𝑤1:

𝑤 = 0.5 − 0.1(−0.25) 1,new

𝑤1,new = 0.525

Update 𝑤2:

𝑤 = −0.5 − 0.1(−0.125)

2,new

𝑤2,new = −0.4875

FIT3181- Deep Learning (Malaysia Cohort) Prepared by Dr. Sicily Ting

Update 𝑏:

𝑏 = −0.5 − 0.1(−0.125) new

𝑏new = −0.4875 Because all gradients are negative, subtracting the gradient increases every parameter.

### Exercise 2: A Tiny 2-Layer Network

The network has this direction:

(1) (1) (2) 𝑋 → 𝑧 → ℎ → 𝑧 → 𝑦̂ → 𝐿

The given quantities are:

𝑋 = [1, −1]

(1) 0.5 −0.5 𝑊 = [ ]

0.2 0.8

(1) 𝑏 = [0.1, −0.1]

(2) 0.6 𝑊 = [ ]

−0.4

(2) 𝑏 = 0.2

𝑦 = 1

FIT3181- Deep Learning (Malaysia Cohort) Prepared by Dr. Sicily Ting

### Zone 1: Hidden pre-activation

Use the hidden-layer linear transformation:

(1) (1) (1) 𝑧 = 𝑋𝑊 + 𝑏

Compute the first hidden neuron:

(1) 𝑧 = 1(0.5) + (−1)(0.2) + 0.1 1

(1) 𝑧 = 0.5 − 0.2 + 0.1 1

(1) 𝑧 = 0.4 1

FIT3181- Deep Learning (Malaysia Cohort) Prepared by Dr. Sicily Ting

Compute the second hidden neuron:

(1) 𝑧 = 1(−0.5) + (−1)(0.8) − 0.1 2

(1) 𝑧 = −0.5 − 0.8 − 0.1 2

(1) 𝑧 = −1.4 2

Therefore:

(1) 𝑧 = [0.4, −1.4]

### Zone 2: Hidden-layer output

Apply sigmoid separately to both entries:

(1) (1) ℎ = 𝜎(𝑧)

(1) ℎ = 𝜎(0.4) 1

(1) ℎ ≈ 0.5987 1

(1) ℎ = 𝜎(−1.4) 2

(1) ℎ ≈ 0.1978 2

Thus:

(1) ℎ ≈ [0.5987, 0.1978]

The slide’s solution had simplified these to:

[0.60 0.20]

### Zone 3: Output and loss:

Use the output layer equation:

(2) (1) (2) (2) 𝑧 = ℎ 𝑊 + 𝑏

Using the more precise hidden values:

(2) 𝑧 = 0.5987(0.6) + 0.1978(−0.4) + 0.2

(2) 𝑧 = 0.3592 − 0.0791 + 0.2

(2) 𝑧 ≈ 0.4801

FIT3181- Deep Learning (Malaysia Cohort) Prepared by Dr. Sicily Ting

Compute the sigmoid prediction:

(2) 𝑦̂ = 𝜎(𝑧)

𝑦̂ = 𝜎(0.4801)

𝑦̂ ≈ 0.6178

Compute the loss:

1

2 𝐿 = (𝑦̂ − 𝑦)

2

1

2 𝐿 = (0.6178 − 1)

2

1

2 𝐿 = (−0.3822)

2

𝐿 ≈ 0.0730

FIT3181- Deep Learning (Malaysia Cohort) Prepared by Dr. Sicily Ting

### Backward Pass: Zone 1:

For sigmoid plus squared error:

𝜕𝐿 (2) ′ (2) 𝛿 = = (𝑦̂ − 𝑦)𝜎 (𝑧)

(2) 𝜕𝑧

The sigmoid derivative is:

′ 𝜎 (𝑧) = 𝜎(𝑧)(1 − 𝜎(𝑧))

Therefore:

′ (2) 𝜎 (𝑧) = 𝑦̂ (1 − 𝑦̂ )

Using the precise prediction:

′ (2) 𝜎 (𝑧) = 0.6178(1 − 0.6178)

′ (2) 𝜎 (𝑧) ≈ 0.2361

Now multiply the loss error by this derivative:

(2) 𝛿 = (0.6178 − 1)(0.2361)

(2) 𝛿 ≈ −0.0902

(2) 𝛿 ≈ −0.0902

Zone 1: Output-layer gradients The weight-gradient formula is:

⊤ (1) (2) ∇ (2) 𝐿 = (ℎ) 𝛿 𝑊

FIT3181- Deep Learning (Malaysia Cohort) Prepared by Dr. Sicily Ting

For the first output weight:

𝜕𝐿

= 0.5987(−0.0902) (2) 𝜕𝑊

1

𝜕𝐿

≈ −0.0540 (2) 𝜕𝑊

1

For the second output weight:

𝜕𝐿

= 0.1978(−0.0902) (2) 𝜕𝑊

2

𝜕𝐿

≈ −0.0178 (2) 𝜕𝑊

2

Thus:

−0.0540 ∇ (2) 𝐿 ≈ [ ] 𝑊

−0.0178

(2) The output bias directly adds to 𝑧, so its local derivative is 1:

(2) ∇ (2) 𝐿 = 𝛿 𝑏

∇ (2) 𝐿 ≈ −0.0902 𝑏

### Zone 2: Hidden-layer delta

Here is the most important backpropagation idea. Each hidden neuron affects the loss through its outgoing output-layer weight. First route the output delta backward:

⊤ (2) (2) 𝛿 (𝑊)

−0.0902[0.6 −0.4]

= [−0.0541 0.0361]

Now calculate the derivative of each hidden sigmoid:

′ (1) (1) (1) 𝜎 (𝑧) = ℎ ⊙( 1 − ℎ)

For hidden neuron 1:

0.5987(1 − 0.5987)

≈ 0.2403

FIT3181- Deep Learning (Malaysia Cohort) Prepared by Dr. Sicily Ting

For hidden neuron 2:

0.1978(1 − 0.1978)

≈ 0.1587

So:

′ (1) 𝜎 (𝑧) ≈ [0.2403 0.1587]

Use element-wise multiplication, denoted by ⊙ :

⊤ (1) (2) (2) ′ (1) 𝛿 = (𝛿 (𝑊)) ⊙ 𝜎 (𝑧)

(1) 𝛿 = [−0.0541 0.0361] ⊙[ 0.2403 0.1587]

(1) 𝛿 ≈ [−0.0130 0.0057]

Hidden-layer gradients Use the matrix formula:

⊤ (1) ∇ (1) 𝐿 = 𝑋 𝛿 𝑊

Since:

⊤ 1 𝑋 = [ ]

−1

the outer product is:

1 ∇ (1) 𝐿 = [ ] [−0.0130 0.0057] 𝑊

−1

−0.0130 0.0057 ∇ (1) 𝐿 ≈ [ ] 𝑊

0.0130 −0.0057

The hidden bias gradient is simply the hidden delta:

(1) ∇ (1) 𝐿 = 𝛿 𝑏

∇ (1) 𝐿 ≈ [−0.0130 0.0057] 𝑏

Zone 3: Update every parameter The intended learning rate is:

𝜂 = 0.1

FIT3181- Deep Learning (Malaysia Cohort) Prepared by Dr. Sicily Ting

Use:

𝜃new = 𝜃old − 𝜂∇𝜃𝐿

For the output weights:

(2) 0.6 −0.0540 𝑊 = [ ] − 0.1 [ ]

new

−0.4 −0.0178

(2) 0.6054 𝑊 ≈ [ ]

new

−0.3982

For the output bias:

(2) 𝑏 = 0.2 − 0.1(−0.0902) new

(2) 𝑏 ≈ 0.2090 new

For the hidden weights:

(1) 0.5 −0.5 −0.0130 0.0057 𝑊 = [ ] − 0.1 [ ]

new

0.2 0.8 0.0130 −0.0057

(1) 0.5013 −0.5006 𝑊 ≈ [ ]

new

0.1987 0.8006

For the hidden biases:

(1) 𝑏 = [0.1 − 0.1] − 0.1[−0.0130 0.0057] new

(1) 𝑏 ≈ [0.1013 − 0.1006] new