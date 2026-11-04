#### Question 1 

Consider an image classification task with six classes {cat=1, dog=2, lion=3, flower=4, cow=5, car=6}. Consider an image x. Assume that a Convolutional Neural Network gives a prediction probabilities f(x)=[0.2, 0.15, 0.05, 0.1, 0.3, 0.2] and categorial ground-truth label of x is cow. What is the cross-entropy loss suffered by this prediction? 

Ci) Mark 

Select one: Oa. -log0.2 Ob. -log0.1 Oc. log03 Od. log0.1 Oc. -log03 

Assume that we have5 classes in {cat = 1, dog = 2, lion = 3, monkey = 4, car = 5}. Givena data example x with ground-truth label dog, assume that a feed-forward NN gives discriminative scores to this x as hy= -3, hg=6, h3= -2, h4= -1, hs=3. What is the probability to predict x as lion or p(y=lion | x)? 



<!-- Start of picture text -->
Select one:<br>Oa et<br>eS4eS+e-2+e-1t+e3<br>On ___eé<br>e~Steo+e-2+e-1+63<br>Oc 1<br>Od et<br>eStelte2te 1463<br>Oe e?<br>eS +eete2+et+e8<br><!-- End of picture text -->

Question 3 Given an implementation of the ResNet as below. Assume that we are feeding a batch [16, 3, 64, 64] to our ResNet. What are the shape of A,B,C,D, E, and F? 

C1) Mark 



<!-- Start of picture text -->
class create_Restet(nn.Module):<br>def _ init__(self):<br>super().__init_()<br>self. layers = nn.Moduletist(<br>nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),<br>nn. LazyBatchworm2d(),<br>nn. ReLU(),<br>A + Mn.taxPool2d(kernel_size=3, stride=2, padding-1),<br>B + Resnetalock(64, 2, first_block=true),<br>Ct Resnetslock(128, 2),<br>Ds ResnetBlock(256, 2),<br>E *~Mm-Adaptiveavgpool2d((1, 1)),<br>nn.Flatten(1),<br>F t-M.LazyLinear(10),<br># nn.Softmax(dim=-1)<br>)<br>def forward(self, x):<br>for _, layer in enumerate(self. layers):<br>x = layer(x)<br>return X<br><!-- End of picture text -->

Select one: 

oO a. [16,64,16,16], [16,64,16,16], [16,128,8,8], [16,256,4,4], [16,256,1,1], [16,10] O b. [16,64,16,16], [16,64,16,16}, [16,128,8,8], [16,256,4,4], [16,256], [32,10] Oc. [16,64,16,16}, [16,64,8,8], [16,128,4,4], [16,256,2,2], [16,256,1,1], [16,10] O 4. Raise an error. B Mock exam @ Summary % Quiz 

# Question 4 

Given the loss function J(w) =so Siw —x;)? wherew is the model parameter and x;=i? for i=1,....,1000 are the data points. We need to solve minJ(w) using stochastic gradient descent with the learning rate n=0.2. Assume we sample a batch of indices i,=1, i.=2, i3=3, ig=4 and at the iteration t, we have w,=20. What is the value of w,,; at the next iteration? 

Select one: 

Oa. 15 Ob. 17 Oc. 19 Od. 18 Oc. 16 

## Question 5 

### Matching RNN topologies and applications 



<!-- Start of picture text -->
y Oe Come, ya @®<br>Ow 64 rth Ftd<br>rt | h (hh (hk oh nh (hk (hk<br>bos rr oe<br>many fo one ‘one to many oc a o. a)<br>A y to many<br>B ci D<br><!-- End of picture text -->

Sentiment analysis C-D-A-B Image captioning -C-D-A-B Machine translation C-D-A-B Video classification at frame level C-D-A-B 

## Question 6 

Given an 3D input tensor with shape [10, 64, 64] over which we apply a conv2D with 20 filters each of which has shape [10 , 5, 5], strides [2,2], and padding=1. What is the shape of the output tensor? 

Select one: 

Oa. [20,32,32] Ob. [20,31,31] Oc. [81,31, 10} O 4. [81,31, 20} Oc. (82,32, 20} 

Assume that the tensor before the last tensor of a CNN has shape [128, 10 , 32, 32] and we apply 8 filters each of which has the shape [10, 5, 5] and strides= [2,2] with padding = 0 to obtain the last tensor. What is the shape of the last tensor? 

Select one: Oa. [128,8, 14,14] oO b. [128, 8,15, 15] Oc. [128,8, 16, 16] Od. [128, 8,17, 17] 

Given a DL model f(x;6) parameterized by 6 where f(x;8) represents the prediction probabilities of x associated with a ground-truth label y € {1,...,M}, we find an adversarial example by 

Xadv = ATGMAX,'cp, lf’; 9),y) 

Which statements are correct? 

Select one or more: 

- © a. We maximally increase the chance to predict x with label y. 

- O b. Itis a targeted attack. © c. We maximally increase the chance to predict x with any else label y’zy. Oj d. We maximally decrease the chance to predict x with label y. 

> 0 e. Itis an untargeted attack. 

Given an RNN with three hidden layers of GRU cells, assume that the sizes of hidden states on the hidden layers are 20, 30, and 40 respectively and sequence length of input and hidden layers is 5, and the embedding size of embedding layer is 10. We feed to our network a mini-batch of shape [2,5]. What are the shape of the output tensor of second hidden layer (i.e., hidden layer 2) if we consider all hidden states of this layer? 



<!-- Start of picture text -->
Output layer (sigmoid) (GN)<br>. . [8 6 8 4 2<br>[25] mew | | 2 3 5 4<br><!-- End of picture text -->

Select one: 

Oa. [25,10] Ob. [25,40] & Mock exam @ Summary % Quiz Oc. [25,20] O« 25,30) Ask a question about this document 

Given a skip-gram model with vocabulary size 200 and embedding size 100, we consider a pair of target and context words with indices 8 and 10 respectively. Let U and V be two weight matrices connecting input to hidden layers and hidden to output layers. What statements are correct? 

Select one or more: 

© a. The hidden valueh is the row 10 of the matrix U 

- ©) b. Input to the network is one-hot vector 110. 

- OC c. Input to the network is one-hot vector 18 

- © d. The hidden valueh is the row 8 of the matrix U 

- © e. Shape of U is [200,100] and shape of V is [100,200] 

- © f. Shape of U is [100,200] and shape of V is [200,100] 

In seq2seq for machine translation in the following figure, we derive the log-likelihood as follows: 



<!-- Start of picture text -->
Pote6)ty =P (yx, | ur9) = P( rie, |6.9) PCr 16.9)?v2 L768) PCy |Yig-60) ~P (a7, | ¥ua,-168)<br>oe<br>= | en | yiy-ne, ee | [Poorly 2e8)<br>i=l ra<br>cap BAY Y2 ayj <EOS><br>x, Xt, <BOS> Vn Yj Yt,<br><!-- End of picture text -->

Which statements are correct? 

Select one or more: 

- © a. Inthe derivation (2), q).1 is viewed as a summary of the sequence Yury © b. Inthe derivation (1), c is viewed as a summary of the sequence Yury © c. Inthe derivation (1), c is viewed as a summary of the sequence Xar, Od. P(y; | qj-1,¢,4) means that on top of qj.1, c, we can build up some dense layers to predict y, © e. Inthe derivation (2), qj. is viewed as a summary of the sequence Y1y-1 

Consider LSTM cell which is shown in the following figure. What statements are correct for the forget gate? 



<!-- Start of picture text -->
Forget gate<br>Cea oQ— D > Ce<br>Input gate Output gate<br>Gy Gah)<br>fi It<br>FC FC Fc Fc<br>Rena ania fe<br>Xe<br><!-- End of picture text -->

Select one or more: 

© a. Itcontrols the proportion of information in long-term memory to be carried forward 

Cb. We apply sigmoid activation when computing forget gate 

Oc. Weapply tanh activation when computing forget gate 

- O d. Itcontrols the proportion of information in short-term memory to be carried forward 

Consider a feed-forward neural network as shown in the figure for the spam email detection with two labels (spam=1 and non-spam=2). Assume that we feed a feature vector x=|-1,1] with true label y=1 to the network. 



<!-- Start of picture text -->
spam non-spam<br>Q Q h?,p = softmax(h?)<br>.<br>\ 1 0<br>w2=|1 1],b?= [-6 —7]<br>L ad<br>2 2 © fh = ReLu(h)<br>Rs 1 0 -1<br>. wi =| 11 1!].ot=[-201[ ]<br>fo) @® x=[-1,1],y=1<br><!-- End of picture text -->

a) What are the formulas and the values of 1,1? [2 marks] 

b) What are the formulas and the values for the logit h? and the prediction probability p? [2 marks] 

c) What is the predicted label J and the cross-entropy loss @ for this prediction? Is it a correct or incorrect prediction? [2 marks] 

oe 

d) What is the derivative Sn2 ?[1 mark] 

ot 

e) What is the derivative S72 ?[1 mark] 

f) Assume that we are using Stochastic Gradient Descent with the learning rate n=0.1 to update W2, what is the new weight matrix W2? [2 marks] 

fel Please answer the question on your blank piece of paper. 

Assume that we conduct a Convolution Neural Network (CNN) with the configuration as shown in the below figure to predict the image dataset Cifar100 with 100 classes. We feed a batch of images with the shape [64, 3, 32, 32] to our CNN. Answer the following questions. 



<!-- Start of picture text -->
FiltersFitter} [30, 3, 5,5] tEioD .<br>ire e re volume 7<br>1355] 1a1, 81, C1, D4) 142,82, 2, 02) Qh toftmex<br>Fiter2 Ds *<br>13.3232) asteh hing LI ¥<br>itB55] rewu layer : bd3<br>= oe iter= 30 woes2.2 PfQf cetPs<br>l 4° layer<br>(94,3,32,32] BS) FC layer<br>Conv2D pain 0<br><!-- End of picture text -->

14a) 

What is the shape of the feature maps [A1, B1, C1, D1]? Show the steps of your answe 

14b) 

What is the shape of the feature maps [A2, B2, C2, D2]? Show the steps of your answe 

14c) 

What is the shape of the 2D tensor [C,D] if we follow 1: Flatten? What does this 2D ten store? Moreover, what is the shape of the weight matrix W in this case? 

14d) 

What is the shape of the 2D tensor [C,D] if we follow 2: Global Max Pooling? What doe this 2D tensor store? Moreover, what is the shape of the weight matrix W in this case? 

14e) 

What is the shape of the 2D tensor [E,F]? What does this 2D tensor store? 

Assume that we have [2, 5, 5] input tensor as shown below. We first apply a Conv2D layer with one filter [2, 3, 3] (i.e., filter 1.1 and filter 1.2), strides = (1,1), and padding = 0 to obtain the output 1. We then apply max pooling with kernel size = (2,2), strides = (2,2), and padding=0 to obtain the output 2. 



<!-- Start of picture text -->
Input [2, 5, 5]<br>Feature<br>map 1<br>Conv2D Max Pooling<br>> 1p: LA output 2<br>Strides = (1, 1), padding =0 Kernel size = (2,2), strides = (2,2),<br>[2] Padding =O<br>Feat<br>Filter 1.1 Filter 1.2<br><!-- End of picture text -->

15a) 

e Marks 

What are the shape and the values of the feature maps in Output 1? 

156) 

°e Marks 

What are the shape and the values of the feature maps in Output 2? 

Assume that we have a RNN with two time-steps as in the figure. We feed to this RNN a sentence “I like” to predict a sentiment label in the label set {1:positive, 2: negative}. 



<!-- Start of picture text -->
: ae |<br>os ™ lv v=|0 1<br>- os 110<br>uf of uf1 tOaojandw=lo 1 4<br>2. oe on<br>Xo \%<br>I like<br><!-- End of picture text -->



<!-- Start of picture text -->
Build up vocabulary Embedding matrix (£ [6 x 2])<br>Like (index: 1)<br>LoveBad  (index:(index: 3)2) peat foe:<br>Fantastic (index: 4) |# [1 [3.3 |<br>Recommend (index: 6) | zs |-1_ [10 |<br>[ Be | -1.7 |-1.3|<br><!-- End of picture text -->

a) What is the value of the hidden state ho if we do not use the bias and apply the ReLU activation function? [3 marks] 

b) What is the value of the hidden state h; if we do not use the bias and apply the ReLU activation function? [3 marks] 

c) On the top of hi, we conduct a dense layer with the weight matrix V to predict the label > . What are the prediction probabilities and the predicted label $ ? [4 marks] 

{ce], Please answer the question on your blank piece of paper. 

Question 17 Read the following code and provide the shape of the tensors h1, h2, h3, h4, h5, h6, h7, h8 if we input x [128, 3, 64, 64] to our CNN 

6 Mar 

class MyCNN(nn.Module): def __init_(self): super(MyCNN, self).__init_() self.convi = nn.Conv2d(3, 20, kernel_size= 3, stride= 2, padding=1) self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding =8) self.conv2 = nn.Conv2d(2@, 32, kernel_size= 3, stride= 1, padding=1) self.conv3= nn.Conv2d(32, 64, kernel_size= 3, stride= 1, padding=1) self.avg_pool = nn.AdaptiveavgPool2d((1, 1)) self.flatten = nn.Flatten() self.fcl = nn.Linear(64, 10) def forward(self, x): hi = torch.relu(self.convi(x)) h2 = self.pool(hi1) h3 = torch.relu(self.conv2(h2)) h4 = self.pool(h3) hS = torch.relu(self.conv3(h4)) h6 = self.avg_pool(hS) h7 = self.flatten(h6) h8 = self.fc1(h7) return h8 myCNN = MyCNN() x = torch.randn(128, 3, 64, 64) y = myCNN(x) 

#### Question 18 

Discuss THREE (3) differences and ONE (1) similarity between a convolutional layer (i.e., Conv2d) and a pooling layer (i.e., MaxPool2d or AvgPool2d). 

e Marks 

Consider a Convolution Neural Network (CNN) with the model parameter 6. Specifically, given an image ~ with the ground-truth label y, the CNN returns the prediction probabilities /(x; #) over M classes and suffers the loss | (f(x; @) , y) where /is a loss function (e.g., the cross-entropy loss). 

® Marks 

a) Adversarial examples are a serious issue for CNNs. Give a definition of adversarial examples. Give a practical example to explain why adversarial examples circumvent the applications of CNNs in reality. [3 marks] 

b) Given a benign example x and the «-ball B, = {x': ||x' - Xlloo < «}, describe and give the formula to find out a targeted adversarial example for +. [3 marks] 

c) Given a benign examplex and the e-ball B, = {x’: ||x' - X|log < «}, describe and give the formula to find out a untargeted adversarial example for <r. [3 marks] 

d) Given a mini-batch B = { (x, y; ), -~ , (Xp, Yp)} at an iteration, describe to perform adversarial training for this mini-batch to improve model robustness. [3 marks] 

e) How adversarial training is similar to data augmentation? How adversarial training is different from data augmentation? [3 marks] 

fel Please answer the auestion on vour blank piece of paper. 

Consider a seq2seq model using the encoder-decoder architecture shown in the figure below. Let 6. and 64 be the parameters for the encoder and the decoder, respectively. 

Answer the following questions: 

* % yj <EOS> hy [+ - —e= hr, + qo | nf aa ft ay xy xr, <BOS> MN Yj-a Vy 

a) Explain the role of the BOS and EOS symbols? [2 marks] 

b) Let 6= [6,,64] be the parameter vector to be learned andD be the training set consists of multiple pairs of sequences (x, y), write down the maximum log-likelihood objective function to learn 6. [2 marks] 

c) Use the product rule, expand the log-likelihood term log P(y | x,8) where y=[y;,...yry] and x=[x;,....X7J SO that it can be computed based on the local probability P(y;,| qj. , c, 8). What is the meaning of the local probability P(y;| qj-1 , c, 8)? How can we realize this local probability in an implementation? [3 marks] 

d) During inference, given an input sequence x , what are TWO (2) common strategies to infer corresponding output sequence y? According to you, which strategy is better and explain why? [3 marks] 

e) What is the main drawback of this time-invariant context vector c? Present the global attention mechanism and how this can help improve seq2seq models. [5 marks] 

fel Please answer the question on your blank piece of paper. 

