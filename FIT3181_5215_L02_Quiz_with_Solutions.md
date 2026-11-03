FIT5215 Deep Learning

Quiz for: Feed-forward Neural Nets with PyTorch

Trung Le and Teaching Team

Department of Data Science and AI Faculty of Information Technology, Monash University

Email: trunglm@monash.edu

### Question 1

❑ Assume that we have 4 classes in {cat = 1,dog = 2,lion = 3, monkey = 4}. Given a data example 𝑥 with ground-truth label “dog”, assume that a feed-forward NN gives discriminative scores to this 𝑥 as ℎ1 = 7, ℎ2 = 10, ℎ3 = 5, ℎ4 = −2. Choose all correct answers. (MC)

❑ A. The model predicts 𝑥 as cat ❑ B. The model predicts 𝑥 as dog ❑ C. This is a correct prediction ❑ D. This is an incorrect prediction

### Question 1

❑ Assume that we have 4 classes in {cat = 1,dog = 2,lion = 3, monkey = 4}. Given a data example 𝑥 with ground-truth label “dog”, assume that a feed-forward NN gives discriminative scores to this 𝑥 as ℎ1 = 7, ℎ2 = 10, ℎ3 = 5, ℎ4 = −2. Choose all correct answers. (MC)

❑ A. The model predicts 𝑥 as cat ❑ B. The model predicts 𝑥 as dog [x] ❑ C. This is a correct prediction [x] ❑ D. This is an incorrect prediction

### Question 2

❑ Assume that we have 4 classes in {cat = 1,dog = 2,lion = 3, monkey = 4}. What is one-hot label of categorical label “dog”?

❑ A. [1,0,0,0] ❑ B. [0,1,0,0] ❑ C. [0,0,1,0] ❑ D. [0,0,0,1]

### Question 2

❑ Assume that we have 4 classes in {cat = 1,dog = 2,lion = 3, monkey = 4}. What is one-hot label of categorical label “dog”?

❑ A. [1,0,0,0] ❑ B. [0,1,0,0] [x] ❑ C. [0,0,1,0] ❑ D. [0,0,0,1]

### Question 3

❑ Assume that we have 4 classes in {cat = 1,dog = 2,lion = 3, monkey = 4}. Given a data example 𝑥 with ground-truth label “dog”, assume that a FFN model gives discriminative scores to this 𝑥 as ℎ1 = −3, ℎ2 = 10, ℎ3 = 5, ℎ4 = 0. What is the probability to predict 𝑥 as dog or 𝑝(𝑦 = 𝑑𝑜𝑔 ∣ 𝑥)?

5 𝑒 ❑ A.

−3 10 5 0 𝑒 +𝑒 +𝑒 +𝑒 ❑ B. 1

0 𝑒 ❑ C.

−3 10 5 0 𝑒 +𝑒 +𝑒 +𝑒

10 𝑒 ❑ D.

−3 10 5 0 𝑒 +𝑒 +𝑒 +𝑒

### Question 3

❑ Assume that we have 4 classes in {cat = 1,dog = 2,lion = 3, monkey = 4}. Given a data example 𝑥 with ground-truth label “dog”, assume that a FFN model gives discriminative scores to this 𝑥 as ℎ1 = −3, ℎ2 = 10, ℎ3 = 5, ℎ4 = 0. What is the probability to predict 𝑥 as dog or 𝑝(𝑦 = 𝑑𝑜𝑔 ∣ 𝑥)?

5 𝑒 ❑ A.

−3 10 5 0 𝑒 +𝑒 +𝑒 +𝑒 ❑ B. 1

0 𝑒 ❑ C.

−3 10 5 0 𝑒 +𝑒 +𝑒 +𝑒

10 𝑒 ❑ D. [x]

−3 10 5 0 𝑒 +𝑒 +𝑒 +𝑒

### Question 4

❑ Assume that we have 4 classes in {cat = 1,dog = 2,lion = 3, monkey = 4}. Given a data example 𝑥 with ground-truth label “dog”, assume that a feed-forward NN gives discriminative scores to this 𝑥 as ℎ1 = −3, ℎ2 = 10, ℎ3 = 5, ℎ4 = 0. What is the CE loss suffered by this prediction?

−3 ❑ A. −log

−3 10 5 0 𝑒 +𝑒 +𝑒 +𝑒 ℎ3 𝑒 ❑ B. log

4 ℎ

𝑗 σ 𝑒 𝑗=1

10 𝑒 ❑ C. −log

−3 10 5 0 𝑒 +𝑒 +𝑒 +𝑒 −ℎ3 𝑒 ❑ D. log

4 ℎ

𝑗 σ 𝑒 𝑗=1

### Question 4

❑ Assume that we have 4 classes in {cat = 1,dog = 2,lion = 3, monkey = 4}. Given a data example 𝑥 with ground-truth label “dog”, assume that a feed-forward NN gives discriminative scores to this 𝑥 as ℎ1 = −3, ℎ2 = 10, ℎ3 = 5, ℎ4 = 0. What is the CE loss suffered by this prediction?

−3 ❑ A. −log

−3 10 5 0 𝑒 +𝑒 +𝑒 +𝑒 ℎ3 𝑒 ❑ B. log

4 ℎ

𝑗 σ 𝑒 𝑗=1

10 𝑒 ❑ C. −log [x]

−3 10 5 0 𝑒 +𝑒 +𝑒 +𝑒 −ℎ3 𝑒 ❑ D. log

4 ℎ

𝑗 σ 𝑒 𝑗=1

### Question 5

Input Hidden Hidden Output ❑ Given the following feed-forward neural

Layer Layer 1 Layer 2 Layer

1 2 3 0 ℎ 𝑥 ℎ 𝑥 = 𝑥 ℎ 𝑥 ℎ 𝑥 network. What are the shapes of weight 3

𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) matrices?

1 2 3 𝑊 𝑊 𝑊

1 2 3 𝑏 𝑏 𝑏

1 3×4 2 4×3 3 3×2 ❑ A. 𝑊 ∈ ℝ, 𝑊 ∈ ℝ, 𝑊 ∈ ℝ

1 4×3 2 3×4 3 2×3 ❑ B. 𝑊 ∈ ℝ, 𝑊 ∈ ℝ, 𝑊 ∈ ℝ

1 3×3 2 4×4 3 3×3 ❑ C. 𝑊 ∈ ℝ, 𝑊 ∈ ℝ, 𝑊 ∈ ℝ

1 4×4 2 3×3 3 2×2 ❑ D. 𝑊 ∈ ℝ, 𝑊 ∈ ℝ, 𝑊 ∈ ℝ

### Question 5

Input Hidden Hidden Output ❑ Given the following feed-forward neural

Layer Layer 1 Layer 2 Layer

1 2 3 0 ℎ 𝑥 ℎ 𝑥 = 𝑥 ℎ 𝑥 ℎ 𝑥 network. What are the shapes of weight 3

𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) matrices?

1 2 3 𝑊 𝑊 𝑊

1 2 3 𝑏 𝑏 𝑏

1 3×4 2 4×3 3 3×2 ❑ A. 𝑊 ∈ ℝ, 𝑊 ∈ ℝ, 𝑊 ∈ ℝ [x]

1 4×3 2 3×4 3 2×3 ❑ B. 𝑊 ∈ ℝ, 𝑊 ∈ ℝ, 𝑊 ∈ ℝ

1 3×3 2 4×4 3 3×3 ❑ C. 𝑊 ∈ ℝ, 𝑊 ∈ ℝ, 𝑊 ∈ ℝ

1 4×4 2 3×3 3 2×2 ❑ D. 𝑊 ∈ ℝ, 𝑊 ∈ ℝ, 𝑊 ∈ ℝ

### Question 6

Input Hidden Hidden Output ❑ Given the following feed-forward neural

Layer Layer 1 Layer 2 Layer

1 2 3 0 ℎ 𝑥 ℎ 𝑥 = 𝑥 ℎ 𝑥 ℎ 𝑥 network. We feed the mini-batch 𝑥 ∈ 3

𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) 16×3 ℝ to the network. What is the shape

2 of the hidden values ℎ?

1 2 3 𝑊 𝑊 𝑊

1 2 3 𝑏 𝑏 𝑏

❑ A. [16, 4] ❑ B. [16, 3] ❑ C. [3, 16] ❑ D. [4, 16]

### Question 6

Input Hidden Hidden Output ❑ Given the following feed-forward neural

Layer Layer 1 Layer 2 Layer

1 2 3 0 ℎ 𝑥 ℎ 𝑥 = 𝑥 ℎ 𝑥 ℎ 𝑥 network. We feed the mini-batch 𝑥 ∈ 3

𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) 16×3 ℝ to the network. What is the shape

2 of the hidden values ℎ?

1 2 3 𝑊 𝑊 𝑊

1 2 3 𝑏 𝑏 𝑏

❑ A. [16, 4] ❑ B. [16, 3] [x] ❑ C. [3, 16] ❑ D. [4, 16]

### Question 7

Input Hidden Hidden Output ❑ Given the following feed-forward neural

Layer Layer 1 Layer 2 Layer

1 2 3 0 ℎ 𝑥 ℎ 𝑥 = 𝑥 ℎ 𝑥 ℎ 𝑥 network. Assume that we input to the 3

𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) network feature vector 𝑥 = 1 1 1 .

𝑥 = [1 1 1] ത1 What is the values of pre-activations ℎ?

2 3 𝑊 𝑊 1 −1 1 −3 2 3

𝑏 𝑏 1 𝑊 = 2 1 2 1

1 2 −2 −2 1 1 𝑏 = [0 1 0 1] ❑ A. ℎത = [4 2 1 − 4]

ത1 ❑ B. ℎ = [4 3 1 − 3]

ത1 𝑇 ❑ C. ℎ = 4 3 1 − 3

ത1 𝑇 ❑ D. ℎ = 4 2 1 − 4

### Question 7

Input Hidden Hidden Output ❑ Given the following feed-forward neural

Layer Layer 1 Layer 2 Layer

1 2 3 0 ℎ 𝑥 ℎ 𝑥 = 𝑥 ℎ 𝑥 ℎ 𝑥 network. Assume that we input to the 3

𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) network feature vector 𝑥 = 1 1 1 .

𝑥 = [1 1 1] ത1 What is the values of pre-activations ℎ?

2 3 𝑊 𝑊 1 −1 1 −3 2 3

𝑏 𝑏 1 𝑊 = 2 1 2 1

1 2 −2 −2 1 1 𝑏 = [0 1 0 1] ❑ A. ℎത = [4 2 1 − 4]

ത1 ❑ B. ℎ = [4 3 1 − 3] [x]

ത1 𝑇 ❑ C. ℎ = 4 3 1 − 3

ത1 𝑇 ❑ D. ℎ = 4 2 1 − 4

### Question 8

Input Hidden Hidden Output ❑ Given the following feed-forward neural

Layer Layer 1 Layer 2 Layer

1 2 3 0 ℎ 𝑥 ℎ 𝑥 = 𝑥 ℎ 𝑥 ℎ 𝑥 network. Assume that we input to the 3

𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) network feature vector 𝑥 = 1 1 1 .

𝑥 = [1 1 1] 1 What is the values of hidden values ℎ if

2 3 𝑊 𝑊 1 −1 1 −3

2 3 we use activation ReLU? 1 𝑏 𝑏

𝑊 = 2 1 2 1

1 2 −2 −2 1 ❑ A. ℎ = [4 2 1 0] 1

𝑏 = [0 1 0 1] 1 ❑ B. ℎ = [4 3 1 0]

1 𝑇 ❑ C. ℎ = 4 3 1 0

1 𝑇 ❑ D. ℎ = 4 2 1 0

### Question 8

Input Hidden Hidden Output ❑ Given the following feed-forward neural

Layer Layer 1 Layer 2 Layer

1 2 3 0 ℎ 𝑥 ℎ 𝑥 = 𝑥 ℎ 𝑥 ℎ 𝑥 network. Assume that we input to the 3

𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) network feature vector 𝑥 = 1 1 1 .

𝑥 = [1 1 1] 1 What is the values of hidden values ℎ if

2 3 𝑊 𝑊 1 −1 1 −3

2 3 we use activation ReLU? 1 𝑏 𝑏

𝑊 = 2 1 2 1

1 2 −2 −2 1 ❑ A. ℎ = [4 2 1 0] 1

𝑏 = [0 1 0 1] 1 ❑ B. ℎ = [4 3 1 0] [x]

1 𝑇 ❑ C. ℎ = 4 3 1 0

1 𝑇 ❑ D. ℎ = 4 2 1 0

### Question 9

Input Hidden Hidden Output ❑ Given the following feed-forward neural

Layer Layer 1 Layer 2 Layer

1 2 3 0 ℎ 𝑥 ℎ 𝑥 = 𝑥 ℎ 𝑥 ℎ 𝑥 network. Assume that we input to the 3

𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) network feature vector 𝑥 = 1 1 1 .

𝑥 = [1 1 1] 1 2 What are the values of ℎ and ℎ if we

2 3 𝑊 𝑊

2 3 use activation ReLU? 𝑏 𝑏

−1 1 1 1 −1 −1 −1 1 1 1 −1 1

𝑊 = −1 0 0 −1 2 ❑ A. h = 1 1 1 0 𝑊 =

1 1 −1 1 1 2 1

2 −2 2 1 2

1 𝑏 = [−1 1 − 2] ❑ B. h = 1 1 1 − 2 𝑏 = [0 1 0 − 1]

2 ❑ C. h = 0 2 − 1

2 ❑ D. h = 0 2 0

### Question 9

Input Hidden Hidden Output ❑ Given the following feed-forward neural

Layer Layer 1 Layer 2 Layer

1 2 3 0 ℎ 𝑥 ℎ 𝑥 = 𝑥 ℎ 𝑥 ℎ 𝑥 network. Assume that we input to the 3

𝑝 𝑥 = 𝑠𝑜𝑓𝑡𝑚𝑎𝑥(ℎ (𝑥)) network feature vector 𝑥 = 1 1 1 .

𝑥 = [1 1 1] 1 2 What are the values of ℎ and ℎ if we

2 3 𝑊 𝑊

2 3 use activation ReLU? 𝑏 𝑏

−1 1 1 1 −1 −1 −1 1 1 1 2 1 −1 1

𝑥𝑊 = [1 1 1] −1 0 0 −1 ℎ 𝑊 = [1 1 1 0] ❑ A. h = 1 1 1 0 [x]

1 1 −1 1 1 2 1

2 −2 2 1 2

1 𝑏 = [−1 1 − 2] ❑ B. h = 1 1 1 − 2 𝑏 = [0 1 0 − 1]

ഥ𝟐 𝟏 𝒉 = [𝟎 𝟐 − 𝟏 ] 𝒉ഥ = [𝟏 𝟏 𝟏 − 𝟐] 2 ❑ C. h = 0 2 − 1

ReLU ReLU 2 ❑ D. h = 0 2 0 [x] 𝟐

𝟏 𝒉 = [𝟎 𝟐 𝟎 ] 𝒉 = [𝟏 𝟏 𝟏 𝟎]

### Question 10

𝑝(𝑥) ❑ Given the following feed-forward neural

softmax network. Assume that we input to the

Output layer

3 network a mini-batch 𝑥 with the batch size ℎ (𝑥)

3 3 32 What is the shape of the input 𝑥 if we 𝑊, 𝑏 use activation ReLU?

Hidden layer

2 ℎ (𝑥) ❑ A. [32,5] 2 2

𝑊, 𝑏 ❑ B. [5,32]

Hidden layer

1 ℎ (𝑥) ❑ C. [32,4]

1 1 𝑊, 𝑏 ❑ D. [5,4]

Input layer 0 ℎ (𝑥) = 𝑥

### Question 10

𝑝(𝑥) ❑ Given the following feed-forward neural

softmax network. Assume that we input to the

Output layer

3 network a mini-batch 𝑥 with the batch size ℎ (𝑥)

3 3 32 What is the shape of the input 𝑥 if we 𝑊, 𝑏 use activation ReLU?

Hidden layer

2 ℎ (𝑥) ❑ A. [32,5] [x] 2 2

𝑊, 𝑏 ❑ B. [5,32]

Hidden layer

1 ℎ (𝑥) ❑ C. [32,4]

1 1 𝑊, 𝑏 ❑ D. [5,4]

Input layer 0 ℎ (𝑥) = 𝑥

### Question 11

𝑝(𝑥) ❑ Given the following feed-forward neural

softmax network. Assume that we input to the

Output layer 32×5 3

ℎ (𝑥) network mini-batch 𝑥 ∈ ℝ. What is

3 3 3

𝑊, 𝑏 the shape of the logits ℎ if we use activation ReLU? Hidden layer

2 ℎ (𝑥) ❑ A. [32,5] 2 2

𝑊, 𝑏 ❑ B. [5,32]

Hidden layer

1 ℎ (𝑥) ❑ C. [32,4]

1 1 𝑊, 𝑏 ❑ D. [5,4]

Input layer 0 ℎ (𝑥) = 𝑥

### Question 11

𝑝(𝑥) ❑ Given the following feed-forward neural

softmax network. Assume that we input to the

Output layer 32×5 3

ℎ (𝑥) network mini-batch 𝑥 ∈ ℝ. What is

3 3 3

𝑊, 𝑏 the shape of the logits ℎ if we use activation ReLU? Hidden layer

2 ℎ (𝑥) ❑ A. [32,5] 2 2

𝑊, 𝑏 ❑ B. [5,32]

Hidden layer

1 ℎ (𝑥) ❑ C. [32,4] [x]

1 1 𝑊, 𝑏 ❑ D. [5,4]

Input layer 0 ℎ (𝑥) = 𝑥

### Question 12

𝑝(𝑥) ❑ Given the following feed-forward neural

softmax network. Assume that we input to the

Output layer 32×5 3

ℎ (𝑥) network mini-batch 𝑥 ∈ ℝ. What is the

3 3 3 th 𝑊, 𝑏 meaning of the 5 row in the logits ℎ?

Hidden layer

2 ℎ (𝑥) ❑ A. It has no meaning 2 2

𝑊, 𝑏 th ❑ B. It is the 5 logit values of all data points in the

Hidden layer

1 ℎ (𝑥) mini-batch.

1 1 𝑊, 𝑏 th ❑ C. It is the logits of the 5 data point in our batch.

Input layer 0 ℎ (𝑥) = 𝑥 ❑ D. None of above.

### Question 12

𝑝(𝑥) ❑ Given the following feed-forward neural

softmax network. Assume that we input to the

Output layer 32×5 3

ℎ (𝑥) network mini-batch 𝑥 ∈ ℝ. What is

3 3 3 th 𝑊, 𝑏 the meaning of the 5 row in the logits ℎ?

Hidden layer

2 ℎ (𝑥) ❑ A. It has no meaning 2 2

𝑊, 𝑏 th ❑ B. It is the 5 logit values of all data points in the

Hidden layer

1 ℎ (𝑥) mini-batch.

1 1 𝑊, 𝑏 th ❑ C. It is the logits of the 5 data point in our batch. [x]

Input layer 0 ℎ (𝑥) = 𝑥 ❑ D. None of above.

### Question 13

𝑝(𝑥) ❑ Given the following feed-forward neural

softmax network. Assume that we input to the

Output layer 32×5 3

ℎ (𝑥) network mini-batch 𝑥 ∈ ℝ. What is

3 3 𝑊, 𝑏 the shape of the prediction probabilities 𝑝 if we use activation ReLU? Hidden layer

2 ℎ (𝑥) ❑ A. [32,5] 2 2

𝑊, 𝑏 ❑ B. [5,32]

Hidden layer

1 ℎ (𝑥) ❑ C. [32,4]

1 1 𝑊, 𝑏 ❑ D. [5,4]

Input layer 0 ℎ (𝑥) = 𝑥

### Question 13

𝑝(𝑥) ❑ Given the following feed-forward neural

softmax network. Assume that we input to the

Output layer 32×5 3

ℎ (𝑥) network mini-batch 𝑥 ∈ ℝ. What is

3 3 𝑊, 𝑏 the shape of the prediction probabilities 𝑝 if we use activation ReLU? Hidden layer

2 ℎ (𝑥) ❑ A. [32,5] 2 2

𝑊, 𝑏 ❑ B. [5,32]

Hidden layer

1 ℎ (𝑥) ❑ C. [32,4] [x]

1 1 𝑊, 𝑏 ❑ D. [5,4]

Input layer 0 ℎ (𝑥) = 𝑥

### Question 14

𝑝(𝑥) ❑ Given the following feed-forward neural

softmax network. Assume that we input to the network

Output layer 32×5

3 mini-batch 𝑥 ∈ ℝ. What is the meaning ℎ (𝑥)

th

3 3 of the 4 row in the prediction probabilities 𝑝? 𝑊, 𝑏

Hidden layer

2 ℎ (𝑥) ❑ A. It has no meaning

2 2 𝑊, 𝑏 th ❑ B. It is the 4 probabilities values of all data points in

Hidden layer

1 the mini-batch. ℎ (𝑥)

1 1 th 𝑊, 𝑏 ❑ C. It is the prediction probabilities of the 4 data

Input layer point in our batch. 0

ℎ (𝑥) = 𝑥 ❑ D. None of above.

### Question 14

𝑝(𝑥) ❑ Given the following feed-forward neural

softmax network. Assume that we input to the network

Output layer 32×5

3 mini-batch 𝑥 ∈ ℝ. What is the meaning of ℎ (𝑥)

th

3 3 the 4 row in the prediction probabilities 𝑝? 𝑊, 𝑏

Hidden layer

2 ℎ (𝑥) ❑ A. It has no meaning

2 2 𝑊, 𝑏 th ❑ B. It is the 4 probabilities values of all data points in

Hidden layer

1 the mini-batch. ℎ (𝑥)

1 1 th 𝑊, 𝑏 ❑ C. It is the prediction probabilities of the 4 data

Input layer point in our batch. [x] 0

ℎ (𝑥) = 𝑥 ❑ D. None of above.

### Question 15

❑ Given an implementation as below (assume that n_features =16). Which of following architecture is correct (SC).

❑ A. 16→10(ReLU)→20(ReLU)→15(ReLU)→26(ReLU) ❑ B. 16→10(ReLU)→20(ReLU)→15(ReLU)→26(ReLU) (Softmax) ❑ C. 16→10(ReLU)→20(ReLU)→15(ReLU)→26 ❑ D. 16→10(sigmoid)→20(sigmoid)→15(sigmoid)→26(sigmoid)

### Question 15

❑ Given an implementation as below (assume that n_features =16). Which of following architecture is correct (SC).

❑ A. 16→10(ReLU)→20(ReLU)→15(ReLU)→26(ReLU) ❑ B. 16→10(ReLU)→20(ReLU)→15(ReLU)→26(ReLU) (Softmax) ❑ C. 16→10(ReLU)→20(ReLU)→15(ReLU)→26 [x] ❑ D. 16→10(sigmoid)→20(sigmoid)→15(sigmoid)→26(sigmoid)

### Question 16

❑ Given an implementation as below (assume that n_features =16). What is the total number parameters of this FFN (SC)?

❑ A. 16x10 + 10x20 + 20x15 + 15x26 ❑ B. 16x10 +10 + 10x20 + 20 + 20x15 +15 + 15x26 +26 ❑ C. 16x10 +16 + 10x20 + 10 + 20x15 +20 + 15x26 +15 ❑ D. None of above.

### Question 16

❑ Given an implementation as below (assume that n_features =16). What is the total number parameters of this FFN (SC)?

❑ A. 16x10 + 10x20 + 20x15 + 15x26 ❑ B. 16x10 +10 + 10x20 + 20 + 20x15 +15 + 15x26 +26 [x] ❑ C. 16x10 +16 + 10x20 + 10 + 20x15 +20 + 15x26 +15 ❑ D. None of above.

### Question 17

❑ Given the code as below (assume that n_features =16 and n_classes=26). What is the shape of the logits h (SC)?

❑ A. [32,16] ❑ B. [32,26] ❑ C. [32,15] ❑ D. None of above.

### Question 17

❑ Given the code as below (assume that n_features =16 and n_classes=26). What is the shape of the logits h (SC)?

❑ A. [32,16] ❑ B. [32,26] [x] ❑ C. [32,15] ❑ D. None of above.