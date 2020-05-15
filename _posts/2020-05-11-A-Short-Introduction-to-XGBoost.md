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

>If we can account for our model's errors, we will be able to improve our model's performance.

<p id="example">
Let's use a simple example to back this idea. Let's say we have a regressive model which predicts 3 for a test case whose actual outcome is 1. If we know the error (2 in the given example), we can fine-tune the prediction by subtracting the error, 2 ,from the original prediction, 3 and obtain a more accurate prediction of 1. This begs the question, *"How do we know the error made by our model for any given input?"*, which leads us to our second insight.
</p>

>We can train a new model to predict the errors made by the original model.

Now, given any predictive model, we can improve its accuracy by first, training a new model to predict its current errors. Then, forming a new improved model whose output is the fine-tuned version of the original prediction. The improved model, which requires the outputs of both the *original model* and the *error-predicting model*, is now considered an ensemble of the two. In gradient boosting, this is repeated arbitrary number of times to continually improve the model's accuracy. This repeated process forms the crux of gradient boosting.

### An Ensemble of Weak Learners

When trainining a new error-predicting model to predict a model's current errors, we regularize its complexity to prevent *overfitting* {% marginnote 'sn-one' 'A model which memorizes the errors for all of its training samples will have no use in the practical scenario.'%}. As a result, it will have *'errors'* in predicting the original model's *'errors'*. With reference to the <a href="#example">above example</a>, it might not necessarily predict 2. Since the new improved model's prediction depends on that (new error-predicting model's) prediction, it will still have errors albeit lower.

To mitigate this, we perform 2 measures. First, we reduce our reliance or trust on any single error-predicting model by applying a small weight, *$$ \alpha $$* (typically between 0 to 0.1) to its output. Then, instead of stopping after 1 iteration of improvement, we repeat the process multiple times, learning new error-prediction models for newly formed improved models till the accuracy or error is satisfactory. This can be summed up using the equations below.

{% marginnote 'sn-two' 'Typically, the error-predicting model predicts the current negative error and so, we use an addition instead of deduction.'%}

$$
\begin{align*}
&improved\_model(x) = current\_model(x) + \alpha \times error\_pred\_model(x)\\~\\
&current\_model(x) = improved\_model(x)\\~\\
&Repeat \ above \ 2 \ steps \ till \ satisfactory.
\end{align*}
$$

Everytime we improve the overall model, a new model will be learned and added into the ensemble. In the end, we get an ensemble. The number of new models to add and *$$ \alpha $$* are hyperparameters.

### "Gradient" Boosting

To end it off, we explore why this is called *"gradient"* boosting. It turns out that the error which we are talking about earlier is the gradient of the loss function with respect to model prediction prediction, $$ \frac{\partial loss}{\partial pred} $$. Think about the squared error loss function, $$ 0.5 (y_{true}-y_{pred})^2 $$. When we differentiate that, we get $$ y_{pred}-y_{true} $$ which uncoincidentally happens to be the "error" which we train our new error-predicting models to predict. Similarly, errors for other types of predictive problems such as classification problems can be expressed via the gradient.

Mathematically, the derivative of the loss function, $$ \frac{\partial loss}{\partial pred} $$, gives the direction in which the predictions should be adjusted to maximize the loss. In gradient boosting, we predict and adjust our predictions in the opposite (negative gradient) direction. This achieves the opposite (minimize the loss). Since, the loss of a model inversely relates to its performance and accuracy, doing so improves its performance. {% marginnote 'sn-three' 'You can also think about this as a form of Gradient Descend.'%} 

Intuitively, we are shifting our model predictions in small steps towards directions which improve the overall performance of our model. 

## XGBoost

XGBoost is a flavour of gradient boosting machines.





