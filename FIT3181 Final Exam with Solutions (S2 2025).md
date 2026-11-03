# FIT3181 Final Exam with

# Solutions (S2 2025)

Trung Le

## Part B

## Question 13

## Solution

13a) Shape [2,3] [0.25 points]

[0.125 points] (formula)

[0.125 points] (correct answer)

[0.125 points] (formula) [0.125 points] (correct answer) [0.5 points] (formula)

[0.125 points] (know how to stack) [0.5 points] (results)

[0.125 points] (correct answer)

Shape [2, 3] [0.25 points]

[0.25 points (know how to apply ReLU)] [0.25 points (know how to apply ReLU)]

## Solution

13b) Shape [2, 2] [0.5 points]

[0.5 points] [0.5 points]

[0.5 points]

[0.5 points]

[0.5 points] (no need to get decimal numbers)

## Solution

13c)

[1 point]

[1 point]

1 correct and 1 incorrect → batch accuracy = 0.5 [1 point] 13d) The batch loss is the average of − log 𝑝𝑦 where 0 < 𝑝𝑦 < 1. Hence it is positive. [1 point]

The range is (0, +∞) [1 point] Note: Student can explain in different ways. As long as, it is correct, we should give full marks.

## Question 14

[2 points] [1 point]

[2 points] [2 points]

[2 points] [1 point]

## Solution

14a) A1 = 256 (batch size) [0.5 points], B1= 50 [0.5 points], C1 = D1 = floor((64- 4+2\*1)/2)+1=32 [0.5 points]

[A1, B1, C1, D1] = [256, 50, 32, 32] [0.5 points] 14b) A2 = 256 [0.5 points], B2 = 50 [0.5 points], C2 = D2 = floor((32-2+2\*0)/2)+1=16 [0.5 points]

[A2, B2, C2, D2] = [256, 50, 16, 16] [0.5 points] 14c) C = 256 [0.5 points], D = 50 [0.5 points]

The tensor [C, D] stores 50-dim vector representations of images in the batch. [1 point] 14d) W connects 50 neurons and 100 → W has the shape of [50, 100] [1 point] 14e) The shape of [E,F] is [256, 100] [1 point]. This stores the logits (or prediction probabilities of images in the batch). [1 point] 14f) We apply softmax to logits [E,F] to turn them to prediction probabilities. We then compute losses for each image and average losses to gain the batch loss. [1 point]

## Question 15

## Solution

15a) 𝑠2 1 =< 𝑞2, ℎ1 >= 1 × 1 + −1 × 3 = −2 [0.5 point]

𝑠2 2 =< 𝑞2, ℎ2 ≥= 1 × (−1) + 3 × 1 = 2 [0.5 point] 𝑠2 3 =< 𝑞2, ℎ3 ≥= 1 × 1 + 3 × 1 = 4 [0.5 point] They stand for the alignment levels of 𝑞2 to ℎ1, ℎ2, ℎ3. [1.5 point]

−2 exp(𝑠2(1)) 𝑒 15b) 𝑎2 1 = = = 0.0022 [0.5

−2 2 4 exp 𝑠2 1 +exp 𝑠2 2 +exp(𝑠2(3)) 𝑒 +𝑒 +𝑒 point]

2 exp(𝑠2(2)) 𝑒 𝑎2 2 = = = 0.12 [0.5 point]

−2 2 4 exp 𝑠2 1 +exp 𝑠2 2 +exp(𝑠2(3)) 𝑒 +𝑒 +𝑒

4 exp(𝑠2(3)) 𝑒 𝑎2 3 = = = 0.8778 [0.5 point]

−2 2 4 exp 𝑠2 1 +exp 𝑠2 2 +exp(𝑠2(3)) 𝑒 +𝑒 +𝑒 They stand for alignment probabilities of 𝑞2 to ℎ1, ℎ2, ℎ3. [1.5 point]

## Solution

15c) 𝑐2 = 𝑎2 1 ℎ1 + 𝑎2 2 ℎ2 + 𝑎2 3 ℎ3 ≈ [0.7578,0.9978] [1 point]

- It is called time-varied context vector because its value depends on the time

step of the decoder hidden state. [1 point]

- The objective of time-varied context vector is to specify what words in the input sentence should be paid more attention when predicting the next word at the current hidden state. [1 point]
- Students can answer differently, please give full marks if their answers are correct for example
- Time-varied context vector maintains the influence of some early words in

the context vectors because in time-invariant context vector, we might forget some early words. 15d) We concatenate 𝑞2 and 𝑐2 [1 point]. On top of this concatenation, we build up some dense layers and the output layer to predict the next word. [1 point]

Students can propose their own solutions. As long as, they utilize the information 𝑞2 and 𝑐2 to predict next word, we should give full marks.

## Question 16

## Solution

16a) The shape of one image patch is [3, 20, 20] [2 points] 16b) [A, B, C] is [1, 10, 1024] or [batch_size, 10, 1024] [3 points] 16c) [D, E, F] is [1, 10, 1024] or [batch_size, 10, 1024] [2 points] 16d) Due to the characteristic of the self-attention mechanism, the class token of one layer aggregates the information of all image-patch tokens. Therefore, the class token contains the global information of the entire image. [3 points]

- Students might answer in different ways. Give marks if their answers

make sense.

## Question 17 + Solution

- x [64, 40] [1 point]
- x [64, 40, 100] [1 point]
- h1 [64, 40, 128] [1 point]
- h2 [64, 40, 256] [1 point]
- h3 [64, 40, 128] [1 point]

## Part C

## Question 18

- In CNNs, low-level layers capture low-level features such as edges, pixels. When going up, higher-level layers combine the features of their low-level layers and the context is expanded locally. Therefore, higher layers can capture more abstract or higher-level features. [5 points]
- Students can answer in different ways. Give marks if the answers make sense.

## Question 19

19a) Overfitting is the phenomenon when the models are too complex to data. The models tend to memorize every data points. As a consequence, they perform very well on training sets, but perform poorly on testing sets. [1 point] 19b) We use a deep net with many layers to learn data with a linear correlation between labels and data. [2 points] 19c) 1 point for one solution . Each solution [0.5 for technique name and 0.5 for a brief explanation]

- Weight regularization
- Batch norm
- Data augmentation
- Dropout
- Early stopping
- Data mix-up, cut mix
- ………………

## Question

## 20

20a) The forget gate 𝑓𝑡 controls what information in the previous long-term memory 𝑐𝑡−1 should be erased before carrying out to the next long-term memory 𝑐𝑡. [3 points] 20b) The input gate 𝑖𝑡 controls what information from the internal short-term state 𝑔𝑡 should be carried to the next long-term memory 𝑐𝑡. [2 points]

## Questi

## on 21

21a) RNN encoder aims to summarize a source sentence to a context vector 𝑐 = ℎ𝑇 [2 points].

𝑥 We can use ℎ𝑇 as a context vector because

𝑥

- ℎ𝑇 depends on ℎ𝑇 −1 and 𝑥𝑇, ℎ𝑇 −1 depends on ℎ𝑇 −2 and 𝑥𝑇 −1,…, ℎ2 depends on ℎ1 and 𝑥2, 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 ℎ1 depends on 𝑥1. Therefore, ℎ𝑇 can be viewed as a lossy summary of 𝑥1, … . , 𝑥𝑇. [3 points]. 𝑥 𝑥

21b) RNN decoder reads out the context vector 𝑐 to generate the target sequence 𝑦1, … . , 𝑦𝑇.

𝑦

- We compute 𝑞0 based on 𝑐 = ℎ𝑇 and BOS. We then learn and generate 𝑦1 based on 𝑞0. 𝑥
- We compute 𝑞1 based on 𝑞0 and 𝑦1. We then learn and generate 𝑦2 based on 𝑞1.
- …………….
- We compute 𝑞𝑇 based on 𝑞𝑇 −1 and 𝑦𝑇. We then learn and generate EOS based on 𝑞𝑇. 𝑦 𝑦 𝑦 𝑦

[5 points]