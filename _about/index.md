---
layout: article
title: A Short Introduction to XGBoost
---

IN THIS ARTICLE, we use simple concepts to present a very influential and powerful algorithm called *Extreme Gradient Boosting* (insert reference and link here). This technique, more commonly known as XGBoost, has been the winning algorithm for many recent data science challenges.

Extreme Gradient Boosting (XGBoost) is an implementation of Gradient Boosting (insert reference or link) machines which focuses on exploiting various hardware and software optimizations to train powerful predictive models quickly. 

As such, we will first explain gradient boosting to set readers in context. Then, we walk through the workings of XGBoost qualitatively, drawing connections to the gradient boosting concepts as necessary. Finally, we talk about the various optimizations implemented and the ideas behind them. 

In writing this article, I have made it a personal goal to keep this as concise and as qualitative as possible, bringing in equations only if it aids in the explanation. The goal is to provide readers with an intuition of how Gradient Boosting and XGBoost works. 



## Gradient Boosting

Gradient Boosting involves building an ensemble of weak learners. It builds upon 2 key ideas. The first idea is that *we can improve a model's prediction if we are able to account for its error*. Let's use a simple example to back this idea. Let's say we have a regressive model which predicts 3 for a test case whose actual outcome is 1. If we know the error, which is 2 in the given example, we can fine-tune the prediction by subtracting the error, 2 ,from the original prediction, 3. This yields a more accurate prediction of 1. This begs the question, "How do we know the error?". This leads us to our second key idea. We can train a new model to predict the errors made by the model. 