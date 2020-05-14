---
layout: article
title: A Short Introduction to Gradient Boosting & XGBoost
---

$$
\DeclareMathOperator{\diag}{diag}
$$

{% newthought 'In this article' %}, we present a very influential and powerful algorithm called *Extreme Gradient Boosting*. This technique, more commonly known as XGBoost, has been the winning algorithm for many recent data science challenges. It is an implementation of Gradient Boosting machines which exploits various hardware and software optimizations to train powerful predictive models very quickly. 

As such, we will first explain *Gradient Boosting* to set readers in context. Then, we walk through the workings of XGBoost qualitatively, drawing connections to gradient boosting concepts as necessary. Finally, we talk about the various optimizations implemented and the ideas behind them. 

In writing this article, I have made it a personal goal to be as concise and as qualitative as possible, bringing in equations only if it aids in the explanation. The goal is to provide readers with an intuition of how Gradient Boosting and XGBoost works. 



## Gradient Boosting

Gradient Boosting involves building an ensemble of weak learners. It builds upon 2 key insights. Here's insight one.

>*If we can account for our model's errors, we will be able to improve our model's performance.*

Let's use a simple example to back this idea. Let's say we have a regressive model which predicts 3 for a test case whose actual outcome is 1. If we know the error (2 in the given example), we can fine-tune the prediction by subtracting the error, 2 ,from the original prediction, 3 and obtain a more accurate prediction of 1. This begs the question, *"How do we know the error made by our model for any given input?"*, which leads us to our second insight.

>*We can train a new model to predict the errors made by the original model.*

Now, given any predictive model, we can improve its accuracy by first, training a new model to predict its current errors. Then, forming an improved model whose output is the fine-tuned version of the original prediction. The improved model, which requires the outputs of both the *original model* and the *error-predicting model*, is now considered an ensemble of the two. 

### An Ensemble of Weak Learners

When trainining a new model to predict a model's errors, we regularize its complexity to prevent *overfitting* {% marginnote 'sn-one' 'A model which memorizes the errors for all of its training samples will have no use in the practical scenario.'%}. As a result, the new model will have *'errors'* in predicting the original model's *'errors'*. 

To mitigate this, we 







