COPYRIGHT WARNING: Copyright in these original lectures is owned by Monash University. You may transcribe, take notes, download or stream lectures for the purpose of your research and study only. If used for any other purpose, (excluding exceptions in the Copyright Act 1969 (Cth)) the University may take legal action for infringement of copyright.

Do not share, redistribute, or upload the lecture to a third party without a written permission!

## FIT 3181 Deep Learning

Week 01: Unit Logistic

CE/Lecturer: Dr Trung Le (Clayton)

Lecturer: Dr Binh Nguyen (Clayton) Lecturer: Dr Sicily Ting Fung (Malaysia) Email: trunglm@monash.edu

Department of Data Science and AI Faculty of Information Technology, Monash University, Australia

2

# Outline

### Unit Logistics

### Mathematical Background Revisit

### Machine Learning Revisit

### Logistic Regression

### Summary

## Unit Logistics

# Unit logistics

### FIT3181, 2026

Mr Minh Vo Dr. Trung Le Dr Binh Nguyen Dr Sicily Ting Fung

[Head Tutor 3181]

Anh Dongwon Simin

◻ FIT 3181 Team

- Lecturers: Dr Trung Le (trunglm@monash.edu), Dr Binh Nguyen (binh.nguyen1@monash.edu), and Dr Sicily Ting Fung (ting.fung@monash.edu)
- Head Tutor: Mr Minh Vo (minh.vo@monash.edu)

# Lecture and consultation times

### ❑ Lectures for FIT 5215 will be conducted online

- Time for FIT5215: Monday 12:00 pm – 2:00 pm
- Zoom link: https://monash.zoom.us/j/88033513605?pwd=CeKjxJegBzqFBDOpiTUrNlouusEp5a.1

- Time for FIT3181: Friday 8:00 am – 10:00 am
- Zoom link: https://monash.zoom.us/j/88033513605?pwd=CeKjxJegBzqFBDOpiTUrNlouusEp5a.1

❑ Important notes:

- There will be 12 lectures in total, each is 2 hours long.
- There will be 22 consultations during teaching (1 hour long).

# Tutorials

## ◻ Make sure you have been allocated to one tutorial.

## ◻ They are on-campus

◻ Make sure you know the tutorial room

## ◻ If you miss a tutorial and would like to join the next (same)

## tutorial, you must email the tutor for permission and detail to

## attend.

# Assessments

### Tests, Assignments and Exam

## ◻ Two in-semester online tests: 10% each

▪ During week 7 and week 12 ▪ Approximately 1 hour long, online test, questions will be randomised each time.

## ◻ Two assignments: 25% (A1) and 20% (A2)

▪ (Tentatively) due in week 7 and week 12 (Friday)

## ◻ One final exam (closed book): 35%

▪ 10 minutes reading ▪ Duration: 2 hours

## ◻ To pass this unit:

- 40% or more in the final exam
- An overall unit mark of 50% or more

# Assessments

### Tests, Assignments and Exam

## ◻ Late submission

- Subject to marking penalty
- 5% deduction per day

## ◻ Apply for extension:

- Applied for short extensions

# Communication

## ◻ Contact your CE/lecturer, tutor for your concerns as soon as possible.

## ◻ Strongly recommend to use Ed discussion as much as possible!

# Learning objectives

## ◻ Fundamental elements of deep learning

## ◻ Basic and advanced concepts of machine learning, AI and

## deep learning

## ◻ Hand-on experience with practical deep learning models and

## framework

# Course structure

Week Lectures Assessment

1 Machine learning and mathematical background revisited Assignment 1 released

2 Prelude to deep learning and feed-forward neural networks

3 Deep Learning for Vision (I): convolutional neural networks (CNNs)

4 Stochastic Gradient Descent and Optimization for Deep Learning

5 Training (very) deep networks: optimizers and practical skills

6 Deep Learning for Vision (II): network architectures, visualization, interpretability and robustness

Online Quiz Test 1 (10%) 7 Deep learning for time-series and temporal data: RNNs and LSTMs/Exam Information

Assignment 1 due (20%) Assignment 2 released 8 Neural embeddings: Word2Vec and Auto-encoders

9 Seq2Seq and End-to-End Deep Models and Transformers

10 Vision Transformers, Model Fine-tuning with Prompts, Adaptors, and LoRA

11 Deep generative models: GANs and Diffusion Models Online Quiz Test 2 (10%)

Assignment 2 due (20%)

12 Revision and Exam Information

# Unit Logistics

### Textbook and References

Intermediate and hand-on Advanced and Introductory and hand-on Intermediate and hand-on, both

st nd (but don’t use tensorflow) theoretical 1 and 2 edition

### ◻ No single textbook, some tutorial/lectures parts might use these books

### ◻ All of these books are available as online resources from Monash library.

# In-class Kaggle Competition

◻ This year, to increase student engagement in the labs/tutorials, we are applying a new

teaching mode in which:

- Students in each tutorial/lab class will register and work in groups of 3 to 6.
- In several tutorials/labs, we will organize in-class Kaggle competitions. You and your group mates will rely on the code provided in the tutorials/labs to develop and train specific deep learning models and submit the results to Kaggle for your class.
- If you surpass our quite trivial baseline ("beat the first boss"), you will win 1 star (1 ).
- If you surpass our second baseline ("beat the second boss"), you will win 1 star (2 ).
- If you are in the top 1 of your class, you will win 3 stars.
- If you are in the top 1 of entire cohort, you will win 4 stars.
- The stars you earn can be converted to bonus points in assignments, with 2 stars equaling 1 point, up to a maximum of 10 bonus points. Any remaining stars will be converted to extra time for in-semester quizzes, with 1 star equaling 1 extra minute.
- Data and tasks are sufficiently simple so that you can achieve good results in less than one hour. Additionally, in the first half of the tutorials/labs, we will introduce and discuss the lab/tutorial materials with you to provide a solid background for the Kaggle tasks.

◻ In the first tutorial/lab, please identify and connect with others to form your group.

# Unit Logistics

## ◻ How to succeed in this unit?

- Work on the hand-on tutorials and master them as quickly as you can
- Where applicable, attempt all exercises and questions during lectures and tutorials
- Revise key points after each lesson to prepare for the in-semester test
- Start on your assignments early!
- Build your knowledge systematically from lectures and recommended readings ▪ If it helps, use some mindmap tool such as xmind

## ◻ How to become extra … ordinary?

- Read and learn beyond what is being taught
- Work on project/ideas that inspire you
- Be proactive, github and build DL profile

# Plagiarism Notice

### ◻ Plagiarism:

“the practice of taking someone else's work or ideas and passing them off as one's own” (Oxford Dictionary)

### ◻ It is a serious academic offence

### and you can be expelled from

### the University!