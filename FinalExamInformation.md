### Final Exam Information

# Final Exam

### • Contributes 40% to the total mark

### • Hurdle to pass the unit:

- At least 45% in in-semester assessments (2 quizzes + 2 assignments)
- AND, at least 45% in final exam
- AND, at least 50% overall mark

### • Format: 10 mins reading + 2 hours

### • Closed book

- CALCULATOR is allowed, highly recommended
- TWO pages of notes (one A4 sheet)

# Exam Information

### • Exam difficulty

- 80%: easy and medium questions (directly from lectures, tutorials, and quizzes)
- 10%: quite challenging questions (deducting the answers from lectures, tutorials, and quizzes)
- 10%: challenging questions (connecting the dots and high level understanding knowledge)

# Exam

### • Contents:

- Part A: multiple-choice questions
- Part B: short workout questions
- Part C: knowledge and written answer questions

### • What might be tested?

- Lecture contents from week 01 to week 11 (refer to the revision slides)
- Tutorial contents (code reading) and assignments
- Will focus on the understanding of subjects, concepts and mechanism (e.g., how attention work, etc)

### • What will NOT be tested?

- Appendices from the lectures
- Memorization of existing network specifications such as AlexNet, VGGNet, ResNet, Transformer, BERT, and etc. Moreover, when questions asked are relevant to these architectures, the relevant codes will be provided.

# Exam

### • Tips:

- Revising questions from assignment 1 and two quizzes
- No coding question, read the code snippet and answer the questions (should be straightforward if you’ve done the tutorials)
- Understand essential model expressions, i.e., mathematical expressions for variables in FNN, CNN, RNN, LSTM, etc
- Materials on appendices won’t be tested, revise the lectures

# Part A: multiple-choice questions

1

### • Similar to the quizzes, difficulty level ≈ (Test 1 + Test 2)

2

# Part B: workout questions

### • Calculation based on specific architecture knowledge

- Question: What is the result of 2d convolution using a filter xxx on an image yyy with zero padding size = 1, stride = 2 in all directions?

# Part C: short-answer questions

- Assessing mid-high level knowledge

- Question: given the following architecture, write down the mathematical expression for xxx , its advantages compared with Y, etc.

# eExam

eExam rules: https://www.monash.edu/exams/electronic-exams/rules Need help during exam: https://www.monash.edu/exams/electronic-exams/help

# Hybrid question

# Reading and Explaining Codes

### ❑ For CNNs and vision data

❑ Conv2d, Conv2dTranspose, MaxPool2d, AdaptiveAvg2d, ❑ Flatten, Unflatten, BatchNorm2d, BatchNorm1d, Linear

### ❑ For RNNs and sequential data

❑ nn.Embedding, nn.GRU, nn.LSTM, nn.RNN

# nn.Conv2D

### ❑ What it does:

❑ nn.Conv2d performs a 2D convolution operation, essential for extracting features from images in CNNs. Imagine a

filter sliding across your image, detecting patterns like edges, corners, and textures.

### ❑ Output shape

𝑖𝑛𝑝𝑢𝑡_𝑠𝑖𝑧𝑒+2∗𝑝𝑎𝑑𝑑𝑖𝑛𝑔−𝑘𝑒𝑟𝑛𝑒𝑙_𝑠𝑖𝑧𝑒 ❑ 𝑜𝑢𝑡𝑝𝑢𝑡_𝑠𝑖𝑧𝑒 = + 1

𝑠𝑡𝑟𝑖𝑑𝑒

# nn.Conv2dTranspose

❑ What it does:

❑ nn.ConvTranspose2d performs a transposed convolution, often called "deconvolution" (though not technically accurate). ❑ It's used to upsample feature maps, effectively increasing their spatial dimensions. This is crucial in tasks like image generation for example in generators

of DCGAN. ❑ Think of it as the reverse of nn.Conv2d: While nn.Conv2d extracts features and downsamples, nn.ConvTranspose2d takes those features and reconstructs

a higher-resolution output. ❑ Output size

❑ 𝑜𝑢𝑡𝑝𝑢𝑡_𝑠𝑖𝑧𝑒 = 𝑖𝑛𝑝𝑢𝑡_𝑠𝑖𝑧𝑒 − 1 ∗ 𝑠𝑡𝑟𝑖𝑑𝑒𝑠 − 2 ∗ 𝑝𝑎𝑑𝑑𝑖𝑛𝑔 + 𝑘𝑒𝑟𝑛𝑒𝑙_𝑠𝑖𝑧𝑒

𝑖𝑛𝑝𝑢𝑡_𝑠𝑖𝑧𝑒+2∗𝑝𝑎𝑑𝑑𝑖𝑛𝑔−𝑘𝑒𝑟𝑛𝑒𝑙_𝑠𝑖𝑧𝑒 ❑ You can obtain the above formula by reversing the Conv2d formula: 𝑜𝑢𝑡𝑝𝑢𝑡_𝑠𝑖𝑧𝑒 = + 1

𝑠𝑡𝑟𝑖𝑑𝑒

# nn.Linear, nn.Flatten, nn.Unflatten

❑ What it does:

❑ nn.Linear implements a fully connected layer,

also known as a linear transformation. It connects every neuron in one layer to every neuron in the next layer. This is a fundamental building block for most neural networks. ❑ Think of it as a matrix multiplication: It takes an

input vector and multiplies it by a weight matrix, then adds a bias vector. This allows the network to learn complex relationships between features.

# nn.Flatten and nn.AdaptiveAvg2d

# BatchNorm2d and MaxPool2D

# Example 1 of CNN

❑ If we input random_input [128, 3, 28, 28] to the CNN, what are the shapes of x,h1,h2,h3,h4,h5?

# Example 2 of CNN

❑ If we input random_input [128, 3, 28, 28] to the CNN, what are the shapes of x,h1,h2,h3,h4,h5?

# Example of Conv2dTranspose

### ❑ What is the shape of the output?

# nn.Embedding, nn.GRU, nn.LSTM, nn.RNN

# nn.Embedding, nn.GRU, nn.LSTM, nn.RNN

## Example of LSTM

# Example of GRU

## Thank you and acknowledgement!

Please give us feedback via SETU!

25

### Thanks for your attention!

Question time