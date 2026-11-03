FIT3181/5215 Deep Learning

Quiz for:

Machine Learning and Mathematical

Background Revisited

Teaching Team

Department of Data Science and AI

Faculty of Information Technology, Monash University

Email: trunglm@monash.edu


## Question 1

Which of the following tasks can be classified as a typical machine learning task (MC):

- A. Clustering gene expression profiles

- B. Detecting spam email

- C. Storing and transferring data to a secure place

- D. Recognising handwritten digit numbers.


## Question 1

Which of the following tasks can be classified as a typical machine learning task (MC): (highlight is correct answer)

- A. Clustering gene expression profiles [x]

- B. Detecting spam email [x]

- C. Storing and transferring data to a secure place

- D. Recognising handwritten digit numbers [x]


## Question 2

Consider a general setting for a machine learning problem, which of the following statements is best at defining the concept of ‘label’ (SC)

- A. Items or instances of data used for training and evaluating ML models.

- B. A set of attributes, each is usually in form of a vector, associated with data points

- C. Values, categories or classes, assigned to data points.

- D. None of above


## Question 2

Consider a general setting for a machine learning problem, which of the following statements is best at defining the concept of ‘label’ (SC)

- A. Items or instances of data used for training and evaluating ML models.

- B. A set of attributes, each is usually in form of a vector, associated with data points

- C. Values, categories or classes, assigned to data points. [x]

- D. None of above


## Question 3

John wants to build a machine learning model which helps to recognize students’ face and automatically records their attendance in his class. Which of the following machine learning approaches would you recommend for this problem (SC)

- A. A regression model

- B. A classification model

- C. A generative model

- D. None of above


## Question 3

John wants to build a machine learning model which helps to recognize students’ face and automatically records their attendance in his class. Which of the following machine learning approaches would you recommend for this problem (SC)

- A. A regression model

- B. A classification model [x]

- C. A generative model

- D. None of above


## Question 4

- Assume that we have 4 classes in {cat = 1,dog = 2,lion = 3, monkey = 4}. Given a data example 𝑥 with ground-truth label “dog”, assume that a feed-forward NN gives discriminative scores to this 𝑥 as ℎ1 = 7, ℎ2 = 3, ℎ3 = 5, ℎ4 = −2. Choose all correct answers. (MC)

- A. The model predicts e as cat

- B. The model predicts e as dog

- C. This is a correct prediction

- D. This is an incorrect prediction


## Question 4

- Assume that we have 4 classes in {cat = 1,dog = 2,lion = 3, monkey = 4}. Given a data example 𝑥 with ground-truth label “dog”, assume that a feed-forward NN gives discriminative scores to this 𝑥 as ℎ1 = 7, ℎ2 = 3, ℎ3 = 5, ℎ4 = −2. Choose all correct answers. (MC)

- A. The model predicts e as cat [x]

- B. The model predicts e as dog

- C. This is a correct prediction

- D. This is an incorrect prediction [x]


## Question 5

- Assume that we have 4 classes in {cat = 1,dog = 2,lion = 3, monkey = 4}. What is one-hot label of categorical label “lion”?

A. [1,0,0,0]

B. [0,1,0,0]

C. [0,0,1,0]

D. [0,0,0,1]


## Question 5

- Assume that we have 4 classes in {cat = 1,dog = 2,lion = 3, monkey = 4}. What is one-hot label of categorical label “lion”?

A. [1,0,0,0]

B. [0,1,0,0]

C. [0,0,1,0] [x]

D. [0,0,0,1]


## Question 6

- Assume that we have 4 classes in {cat = 1,dog = 2,lion = 3, monkey = 4}. Given a data example 𝑥 with ground-truth label “dog”, assume that a machine learning model gives discriminative scores to this 𝑥 as ℎ1 = −3, ℎ2 = 10, ℎ3 = 5, ℎ4 = 0. What is the probability to predict 𝑥 as lion or 𝑝(𝑦 = 𝑙𝑖𝑜𝑛 ∣ 𝑥)?


## Question 6

- Assume that we have 4 classes in {cat = 1,dog = 2,lion = 3, monkey = 4}. Given a data example 𝑥 with ground-truth label “dog”, assume that a machine learning model gives discriminative scores to this 𝑥 as ℎ1 = −3, ℎ2 = 10, ℎ3 = 5, ℎ4 = 0. What is the probability to predict 𝑥 as lion or 𝑝(𝑦 = 𝑙𝑖𝑜𝑛 ∣ 𝑥)?

B. 1


## Question 7

- Assume that we have 4 classes in {cat = 1,dog = 2,lion = 3, monkey = 4}. Given a data example 𝑥 with ground-truth label “lion”, assume that a feed-forward NN gives discriminative scores to this 𝑥 as ℎ1 = −3, ℎ2 = 10, ℎ3 = 5, ℎ4 = 0. What is the CE loss suffered by this prediction?


## Question 7

- Assume that we have 4 classes in {cat = 1,dog = 2,lion = 3, monkey = 4}. Given a data example 𝑥 with ground-truth label “dog”, assume that a feed-forward NN gives discriminative scores to this 𝑥 as ℎ1 = −3, ℎ2 = 10, ℎ3 = 5, ℎ4 = 0. What is the CE loss suffered by this prediction?


## Question 8

- Given a deep net that offers the prediction probabilities ](e) for a data example e - ℝ𝑑 with the ground-truth label 𝑦 ∈ {1,2, … , 𝑀}, what are correct statements about the cross-entropy loss incurred by (x,y)? (MC)

- A. It is 68(1f, ](e)).

- B. It is 68(](e), 1f).

- C. It is log ]f(e)

- D. It is ‒log ]f(e)


## Question 8

- Given a deep net that offers the prediction probabilities ](e) for a data example e - ℝ𝑑 with the ground-truth label 𝑦 ∈ {1,2, … , 𝑀}, what are correct statements about the cross-entropy loss incurred by (x,y)? (MC)

- A. It is 68(1f, ](e)). [x]

- B. It is 68(](e), 1f).

- C. It is log ]f(e)

- D. It is ‒log ]f(e) [x]


## Question 9

- Given a deep net that offers the prediction probabilities ](e) for a data example e - ℝ𝑑 with the ground-truth label 𝑦 ∈ {1,2, … , 𝑀}, when does the cross-entropy loss incurred by (x,y) becomes 0? (MC)


## Question 9

- Given a deep net that offers the prediction probabilities ](e) for a data example e - ℝ𝑑 with the ground-truth label 𝑦 ∈ {1,2, … , 𝑀}, when does the cross-entropy loss incurred by (x,y) becomes 0? (MC)


## Question 10

- Given a deep net that offers the prediction probabilities ](e) for a data example e - ℝ𝑑 with the ground-truth label 𝑦 ∈ {1,2, … , 𝑀}, what is the range of the cross- entropy loss? (SC)

A. (
, +
).

B. (
, 0).

C. (
, 0].

D. (0, +‒).

E. [0, +‐).


## Question 10

- Given a deep net that offers the prediction probabilities ](e) for a data example e - ℝ𝑑 with the ground-truth label 𝑦 ∈ {1,2, … , 𝑀}, what is the range of the cross- entropy loss? (SC)

A. (
, +
).

B. (
, 0).

C. (
, 0].

D. (0, +‒).


## Question 11

- Which ones are a discrete distribution over 4 classes/items? (MC)

- A. [0,1,0,0].

- B. [0,1.2,0, ‒0.2].

- C. [0.3,0.4,0.1,0.2].

- D. [0.25,0.25,0.25,0.25].

- E. [0,0,0,0].


## Question 11

- Which ones are a discrete distribution over 4 classes/items? (MC)

- A. [0,1,0,0]. [x]

- B. [0,1.2,0, ‒0.2].

- C. [0.3,0.4,0.1,0.2]. [x]

- D. [0.25,0.25,0.25,0.25]. [x]

- E. [0,0,0,0].


## Question 12

- Given two discrete distributions over @ classes/items ] = ]Z 𝑀 Z=1 and q = entropy. 𝑞𝑚 𝑚=1 , matching the formulas of KL divergence, Cross-entropy divergence, and 𝑀


## Question 12

- Given two discrete distributions over @ classes/items ] = ]Z 𝑀 Z=1 and q = entropy. 𝑞𝑚 𝑚=1 , matching the formulas of KL divergence, Cross-entropy divergence, and 𝑀
