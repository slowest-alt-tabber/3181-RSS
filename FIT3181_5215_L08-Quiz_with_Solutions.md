FIT3181/5215 Deep Learning

Quiz for: Word Embedding with Word2Vec

Teaching team

Department of Data Science and AI Faculty of Information Technology, Monash University

Email: trunglm@monash.edu

### Question 1

In Word2Vec with the skip-gram, consider the sentence: “the mouse runs away from the cat”. Which of the following statements is correct if we use a window size of 7?

❑ A. ‘The’, ‘mouse’, ‘runs’, ‘from’, ‘the’, ‘cat’ are used to predict ‘away’ ❑ B. ‘Runs’ is used to predict ‘The’, ‘mouse’, ‘away’, ‘from’, ‘the’, ‘cat’ ❑ C. ‘Runs’, ‘away’ are used to predict ‘The’, ‘mouse’, ‘from’, ‘the’, ‘cat’ ❑ D. ‘Away’ is used to predict ‘the’, ‘mouse’, ‘runs’, ‘from’, ‘the’, ‘cat’

### Question 1

In Word2Vec with the skip-gram, consider the sentence: “the mouse runs away from the cat”. Which of the following statements is correct if we use a window size of 7?

❑ A. ‘The’, ‘mouse’, ‘runs’, ‘from’, ‘the’, ‘cat’ are used to predict ‘away’ ❑ B. ‘Runs’ is used to predict ‘The’, ‘mouse’, ‘away’, ‘from’, ‘the’, ‘cat’ ❑ C. ‘Runs’, ‘away’ are used to predict ‘The’, ‘mouse’, ‘from’, ‘the’, ‘cat’ ❑ D. ‘Away’ is used to predict ‘the’, ‘mouse’, ‘runs’, ‘from’, ‘the’, ‘cat’ [x]

### Question 2

In Word2Vec with the CBOW, consider the sentence: “the mouse runs away from the cat”. Which of the following statements is correct?

❑ A. ‘The’, ‘mouse’, ‘runs’, ‘from’, ‘the’, ‘cat’ are used to predict ‘away’ ❑ B. ‘Runs’ is used to predict ‘The’, ‘mouse’, ‘away’, ‘from’, ‘the’, ‘cat’ ❑ C. ‘Runs’, ‘away’ are used to predict ‘The’, ‘mouse’, ‘from’, ‘the’, ‘cat’ ❑ D. ‘Away’ is used to predict ‘the’, ‘mouse’, ‘runs’, ‘from’, ‘the’, ‘cat’

### Question 2

In Word2Vec with the CBOW, consider the sentence: “the mouse runs away from the cat”. Which of the following statements is correct?

❑ A. ‘The’, ‘mouse’, ‘runs’, ‘from’, ‘the’, ‘cat’ are used to predict ‘away’ [x] ❑ B. ‘Runs’ is used to predict ‘The’, ‘mouse’, ‘away’, ‘from’, ‘the’, ‘cat’ ❑ C. ‘Runs’, ‘away’ are used to predict ‘The’, ‘mouse’, ‘from’, ‘the’, ‘cat’ ❑ D. ‘Away’ is used to predict ‘the’, ‘mouse’, ‘runs’, ‘from’, ‘the’, ‘cat’

### Question 3

You were given a corpus of texts which is assumed to be sufficiently large to learn semantic meaning of words. After applying word2vec embedding on this corpus, each vocabulary had now learned and associated with a vector. As you have learned from the lecture, this now allows you to perform analogical reasoning. Which answer do you expect if we reason “? – prince = queen - princess”

❑A. King ❑B. Royal ❑C. Father ❑D. Dad

### Question 3

You were given a corpus of texts which is assumed to be sufficiently large to learn semantic meaning of words. After applying word2vec embedding on this corpus, each vocabulary had now learned and associated with a vector. As you have learned from the lecture, this now allows you to perform analogical reasoning. Which answer do you expect if we reason “? – prince = queen - princess”

❑A. King [x] ❑B. Royal ❑C. Father ❑D. Dad

### Question 4

Consider an embedding approach in text analytics using the word2vec algorithm with a simple text corpus consisting of the following 9 words:

the quick brown fox jumps over the lazy dog Tokenizing this corpus give us a vocabulary (i.e., unique words in the corpus) as well as their indices. Assume the above corpus returns the following

word-to-indices dictionary = {'brown': 0, 'lazy': 1, 'over': 2, 'fox': 3, 'dog': 4, 'quick': 5, 'the': 6, 'jumps': 7}. What is the one-hot encoding vector for the word dog? ❑A. [0,0,0,0,1,0,0,0] ❑B. [0,0,0,0,0,1,0,0] ❑C. [0,0,0,0,1,1,0,0] ❑D. [0,0,0,0,0,0,0,1]

### Question 4

Consider an embedding approach in text analytics using the word2vec algorithm with a simple text corpus consisting of the following 9 words:

the quick brown fox jumps over the lazy dog Tokenizing this corpus give us a vocabulary (i.e., unique words in the corpus) as well as their indices. Assume the above corpus returns the following

word-to-indices dictionary = {'brown': 0, 'lazy': 1, 'over': 2, 'fox': 3, 'dog': 4, 'quick': 5, 'the': 6, 'jumps': 7}. What is the one-hot encoding vector for the word dog? ❑A. [0,0,0,0,1,0,0,0] [x] ❑B. [0,0,0,0,0,1,0,0] ❑C. [0,0,0,0,1,1,0,0] ❑D. [0,0,0,0,0,0,0,1]

### Question 5

Given a Wikipedia text corpus dataset with 1 billion sentences. We filtered the dataset and found 10k unique words. We train the word2vec model using the dataset, with the context window size 5. Which are correct statements? (MC)

❑ A. Dictionary size N=1 billion ❑ B. Dictionary size N=10k ❑ C. Each word will be represented by a vector size [5, 1] ❑ D. The embedding size is unknown with given information

### Question 5

Given a Wikipedia text corpus dataset with 1 billion sentences. We filtered the dataset and found 10k unique words. We train the word2vec model using the dataset, with the context window size 5. Which are correct statements? (MC)

❑ A. Dictionary size N=1 billion ❑ B. Dictionary size N=10k [x] ❑ C. Each word will be represented by a vector size [5, 1] ❑ D. The embedding size is unknown with given information [x]

### Question 6

### Which are the drawbacks of skip-gram algorithm. (MC)

❑ A. The softmax output layer is computationally expensive. ❑ B. The values of prediction probabilities are small and hard to distinguish among classes. ❑ C. Skip-gram cannot capture the dependency between words. ❑ D. Skip-gram depends on the context window size.

### Question 6

### Which are the drawbacks of skip-gram algorithm. (MC)

❑ A. The softmax output layer is computationally expensive. [x] ❑ B. The values of prediction probabilities are small and hard to distinguish among classes. [x] ❑ C. Skip-gram cannot capture the dependency between words. ❑ D. Skip-gram depends on the context window size. [x]

### Question 7

Which are correct statements about skip-gram with negative sampling. (MC) ❑ A. The output layer represents the probability over classes and sum of output equal to 1. ❑ B. We apply softmax activation function at the output layer. ❑ C. We apply sigmoid activation function at the output layer. ❑ D. Given a pair target word tw and context word cw, we need to randomly sample some

negative words and maximize the probabilities to predict those negative words from the target word. ❑ E. Given a pair target word tw and context word cw, we need to randomly sample some

negative words and minimize the probabilities to predict the pairs of target word and negative words as a positive pair. ❑ F. Given a pair target word tw and context word cw, we need to maximize the probability to

predict the pair of target word and context word as a positive pair.

### Question 7

Which are correct statements about skip-gram with negative sampling. (MC) ❑ A. The output layer represents the probability over classes and sum of output equal to 1. ❑ B. We apply softmax activation function at the output layer. ❑ C. We apply sigmoid activation function at the output layer. [x] ❑ D. Given a pair target word tw and context word cw, we need to randomly sample some

negative words and maximize the probabilities to predict those negative words from the target word. ❑ E. Given a pair target word tw and context word cw, we need to randomly sample some

negative words and minimize the probabilities to predict the pairs of target word and negative words as a positive pair. [x] ❑ F. Given a pair target word tw and context word cw, we need to maximize the probability to

predict the pair of target word and context word as a positive pair. [x]

### Question 8

Given an objective function below, 𝑝(𝑢|𝑣) represents for the probability to base on the word 𝑣 to predict the word 𝑢. Which are correct statements (MC)

𝑇 𝐽 𝜃 = ෑ ෑ 𝑝(𝑤𝑡+𝑗 ∣ 𝑤𝑡; 𝜃)

𝑡=1 −𝐶≤𝑗≤𝐶,𝑗≠0

❑ A. It is the objective function of CBOW algorithm ❑ B. It is the objective function of skip-gram algorithm ❑ C. j represents for the index of target (centre) word ❑ D. t represents for the index of target (centre) word ❑ E. We need to maximize this objective function to find parameter theta

### Question 8

Given an objective function below, 𝑝(𝑢|𝑣) represents for the probability to base on the word 𝑣 to predict the word 𝑢. Which are correct statements (MC)

𝑇 𝐽 𝜃 = ෑ ෑ 𝑝(𝑤𝑡+𝑗 ∣ 𝑤𝑡; 𝜃)

𝑡=1 −𝐶≤𝑗≤𝐶,𝑗≠0

❑ A. It is the objective function of CBOW algorithm ❑ B. It is the objective function of skip-gram algorithm [x] ❑ C. j represents for the index of target (centre) word ❑ D. t represents for the index of target (centre) word [x] ❑ E. We need to maximize this objective function to find parameter theta [x]

### Question 9

Given an objective function below, 𝑝(𝑢|𝑣) represents for the probability to base on the word 𝑣 to predict the word 𝑢. Which are correct statements (MC)

𝑇 𝐽 𝜃 = ෑ 𝑝(𝑤𝑡 ∣ 𝑤𝑡−𝐶 , … , 𝑤𝑡−1, 𝑤𝑡+1, … , 𝑤𝑡+𝐶 ; 𝜃)

𝑡=1

❑ A. It is the objective function of CBOW algorithm ❑ B. It is the objective function of skip-gram algorithm ❑ C. 𝑇 is the window size ❑ D. 𝐶 is the window size ❑ E. 2𝐶 + 1 is the window size

### Question 9

Given an objective function below, 𝑝(𝑢|𝑣) represents for the probability to base on the word 𝑣 to predict the word 𝑢. Which are correct statements (MC)

𝑇 𝐽 𝜃 = ෑ 𝑝(𝑤𝑡 ∣ 𝑤𝑡−𝐶 , … , 𝑤𝑡−1, 𝑤𝑡+1, … , 𝑤𝑡+𝐶 ; 𝜃)

𝑡=1

❑ A. It is the objective function of CBOW algorithm [x] ❑ B. It is the objective function of skip-gram algorithm ❑ C. 𝑇 is the window size ❑ D. 𝐶 is the window size ❑ E. 2𝐶 + 1 is the window size [x]

### Question 10

Given a skip-gram model with vocabulary size 200 and embedding size 100, we consider a pair of target and context words with indices 5 and 10 respectively. Let 𝑈 and 𝑉 be two weight matrices connecting input to hidden layers and hidden to output layers. What statements are correct? (MC)

❑ A. Shape of 𝑈 is [200,100] and shape of 𝑉 is [100,200] ❑ B. Shape of 𝑈 is [100,200] and shape of 𝑉 is [200,100] ❑ C. Input to the network is one-hot vector 15. ❑ D. Input to the network is one-hot vector 110. ❑ E. The hidden value ℎ is the row 10 of the matrix 𝑈 ❑ F. The hidden value ℎ is the row 5 of the matrix 𝑈

### Question 10

Given a skip-gram model with vocabulary size 200 and embedding size 100, we consider a pair of target and context words with indices 5 and 10 respectively. Let 𝑈 and 𝑉 be two weight matrices connecting input to hidden layers and hidden to output layers. What statements are correct? (MC)

❑ A. Shape of 𝑈 is [200,100] and shape of 𝑉 is [100,200] [x] ❑ B. Shape of 𝑈 is [100,200] and shape of 𝑉 is [200,100] ❑ C. Input to the network is one-hot vector 15. [x] ❑ D. Input to the network is one-hot vector 110. ❑ E. The hidden value ℎ is the row 10 of the matrix 𝑈 ❑ F. The hidden value ℎ is the row 5 of the matrix 𝑈 [x]

### Question 11

Given a CBOW model with vocabulary size 100 and embedding size 150, we consider a target word with index 5 and context words with indices 10, 20, 30, 40 respectively. Let 𝑈 and 𝑉 be two weight matrices connecting input to hidden layers and hidden to output layers. What statements are correct? (MC)

❑ A. Shape of 𝑈 is [100,100] and shape of 𝑉 is [150,150] ❑ B. Shape of 𝑈 is [100,150] and shape of 𝑉 is [150,100] ❑ C. Input to the network is one-hot vector 15.

110+120+130+140 ❑ D. Input to the network is .

4 ❑ E. The hidden value ℎ is the average of rows 10,20,30,40 of the matrix 𝑈 ❑ F. The hidden value ℎ row 5 of the matrix 𝑈

### Question 11

Given a CBOW model with vocabulary size 100 and embedding size 150, we consider a target word with index 5 and context words with indices 10, 20, 30, 40 respectively. Let 𝑈 and 𝑉 be two weight matrices connecting input to hidden layers and hidden to output layers. What statements are correct? (MC)

❑ A. Shape of 𝑈 is [100,100] and shape of 𝑉 is [150,150] ❑ B. Shape of 𝑈 is [100,150] and shape of 𝑉 is [150,100] [x] ❑ C. Input to the network is one-hot vector 15.

110+120+130+140 ❑ D. Input to the network is . [x]

4 ❑ E. The hidden value ℎ is the average of rows 10,20,30,40 of the matrix 𝑈 [x] ❑ F. The hidden value ℎ row 5 of the matrix 𝑈

### Question 12

1xN Nxd 1xd dxN 1xN 1xN

𝑇 0 U 𝒉 = 𝟏 𝑼 𝑝(𝑐𝑤|𝑡𝑤) 0 quick 𝒕 𝒐 = 𝑽𝒉

0 V 𝑇.4 .3 .0 𝑇 𝑇 0 dog 0 0.45.09

.3 .5 .0 1 0 0.53.10

.2 .3 .8 0 0.5 .0 .3 .3 .7 .6 .3 .5 0.89.14

.1 .3 .3 0 0.2 .4 .7 .6 .8 .7 .1 .6 0.48.09 𝑤

𝑡+1 ℎ = 0.1 0.9 0.9 0 0.3 .7 .0 0.67 softmax.11

.7 .7 .8 .1 .5 .7 .8 .9 1.1 .9 .9 1.32.21 𝑇

0 0.0 .4 .7 0.77.12

0 0.3 .2 .7 0.81.13

0 𝑤𝑡 = 1𝑡

0 0

the 'fox’: 4, 'dog’: 5, 'quick’: 6, 'the’: 7, 'jumps’: 8 0

1 0 𝑤𝑡−1 Consider the above skip-gram model, let denote 𝑝1 = 𝑝(𝑡ℎ𝑒 ∣ 𝑞𝑢𝑖𝑐𝑘) and 𝑝2 = 𝑝(𝑑𝑜𝑔 ∣ 𝑞𝑢𝑖𝑐𝑘). What statements are correct? (MC) ❑ A. Vocabulary size is 8 and embedding size is 3. ❑ C. 𝑝1 = 0.09, 𝑝2 = 0.10. ❑ B. Vocabulary size is 3 and embedding size is 8. ❑ D. 𝑝1 = 0.45, 𝑝2 = 0.77.

❑ E. 𝑝1 = 0.12, 𝑝2 = 0.11.

### Question 12

1xN Nxd 1xd dxN 1xN 1xN

𝑇 0 U 𝒉 = 𝟏 𝑼 𝑝(𝑐𝑤|𝑡𝑤) 0 quick 𝒕 𝒐 = 𝑽𝒉

0 V 𝑇.4 .3 .0 𝑇 𝑇 0 dog 0 0.45.09

.3 .5 .0 1 0 0.53.10

.2 .3 .8 0 0.5 .0 .3 .3 .7 .6 .3 .5 0.89.14

.1 .3 .3 0 0.2 .4 .7 .6 .8 .7 .1 .6 0.48.09 𝑤

𝑡+1 ℎ = 0.1 0.9 0.9 0 0.3 .7 .0 0.67 softmax.11

.7 .7 .8 .1 .5 .7 .8 .9 1.1 .9 .9 1.32.21 𝑇

0 0.0 .4 .7 0.77.12

0 0.3 .2 .7 0.81.13

0 𝑤𝑡 = 1𝑡

0 0

the 'fox’: 4, 'dog’: 5, 'quick’: 6, 'the’: 7, 'jumps’: 8 0

1 0 𝑤𝑡−1 Consider the above skip-gram model, let denote 𝑝1 = 𝑝(𝑡ℎ𝑒 ∣ 𝑞𝑢𝑖𝑐𝑘) and 𝑝2 = 𝑝(𝑏𝑟𝑜𝑤𝑛 ∣ 𝑞𝑢𝑖𝑐𝑘). What statements are correct? (MC)

❑ C. 𝑝1 = 0.09, 𝑝2 = 0.10. ❑ A. Vocabulary size is 8 and embedding size is 3. [x]

❑ D. 𝑝1 = 0.45, 𝑝2 = 0.77. ❑ B. Vocabulary size is 3 and embedding size is 8.

❑ E. 𝑝1 = 0.12, 𝑝2 = 0.11. [x]