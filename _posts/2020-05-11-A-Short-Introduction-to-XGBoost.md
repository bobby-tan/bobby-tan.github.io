---
layout: article
title: A Short Introduction to Gradient Boosting & XGBoost
---

$$
\DeclareMathOperator{\diag}{diag}
$$

{% newthought 'In this article' %}, we present a very influential and powerful algorithm called *Extreme Gradient Boosting* or XGBoost. It is an implementation of Gradient Boosting machines which exploits various optimizations to train powerful predictive models very quickly. 

As such, we will first explain *Gradient Boosting* to set readers in context. Then, we walk through the workings of XGBoost qualitatively, drawing connections to gradient boosting concepts as necessary. Finally, we talk about the various optimizations implemented and the ideas behind them. 

In writing this article, I have made it a personal goal to be as concise and as qualitative as possible, bringing in equations only if it aids in the explanation. The goal is to provide readers with an intuition of how Gradient Boosting and XGBoost works. 



## Gradient Boosting

Gradient Boosting involves building an ensemble of weak learners. It builds upon 2 key insights. Here's insight one.

>If we can account for our model's errors, we will be able to improve our model's performance.

<p id="example">
Let's use a simple example to back this idea. Let's say we have a regressive model which predicts 3 for a test case whose actual outcome is 1. If we know the error (2 in the given example), we can fine-tune the prediction by subtracting the error, 2 ,from the original prediction, 3 and obtain a more accurate prediction of 1. This begs the question, *"How do we know the error made by our model for any given input?"*, which leads us to our second insight.
</p>

>We can train a new model to predict the errors made by the original model.

Now, given any predictive model, we can improve its accuracy by first, training a new model to predict its current errors.{% marginnote 'sn-zero' 'The model used to predict the error can be any function approximator.'%}Then, forming a new improved model whose output is the fine-tuned version of the original prediction. The improved model, which requires the outputs of both the *original model* and the *error-predicting model*, is now considered an ensemble of the two. In gradient boosting, this is repeated arbitrary number of times to continually improve the model's accuracy. This repeated process forms the crux of gradient boosting.

### An Ensemble of Weak Learners

When trainining a new error-predicting model to predict a model's current errors, we regularize its complexity to prevent *overfitting* {% marginnote 'sn-one' 'A model which memorizes the errors for all of its training samples will have no use in the practical scenario.'%}. As a result, it will have *'errors'* in predicting the original model's *'errors'*. With reference to the <a href="#example">above example</a>, it might not necessarily predict 2. Since the new improved model's prediction depends on that (new error-predicting model's) prediction, it will still have errors albeit lower.

To mitigate this, we perform 2 measures. First, we reduce our reliance or trust on any single error-predicting model by applying a small weight, *$$ \alpha $$* (typically between 0 to 0.1) to its output. Then, instead of stopping after 1 iteration of improvement, we repeat the process multiple times, learning new error-prediction models for newly formed improved models till the accuracy or error is satisfactory. This can be summed up using the equations below.

{% marginnote 'sn-two' 'Typically, the error-predicting model predicts the current negative error and so, we use an addition instead of deduction.'%}
<p id="steps"></p>>

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

Mathematically, the derivative of the loss function, $$ \frac{\partial loss}{\partial pred} $$, gives the direction in which the predictions can be adjusted to maximize loss. In gradient boosting, we predict and adjust our predictions in the opposite (negative gradient) direction. This achieves the opposite (minimize the loss). Since, the loss of a model inversely relates to its performance and accuracy, doing so improves its performance. {% marginnote 'sn-three' 'You can also think about this as a form of Gradient Descend.'%} 

Intuitively, we are shifting our model predictions in small steps towards directions which improve the overall performance of our model. 

## XGBoost

XGBoost is a flavour of gradient boosting machines and Gradient Boosting Trees (gbtree) is the recommended function approximator. 

We first start with a simple predictor, one that predicts an arbitrary number for all values (usually 0.5). Then, we apply what <a href="#steps">we've learnt above</a>. {% marginnote 'sn-four' 'In XGBoost, gbtrees are trained in a slightly different manner. It does not involve directly predicting gradients. You will see that later on.'%}In the next section, we explain in further detail, how the trees are learnt. 

### Gradient Boosting Tree

In XGBoost, we learn a tree whose output can be added to our current prediction such that the overall loss of the new model is minimized while keeping in mind not to *overfit the model*. Note that in this article, we will be talking about the addition of a single tree to improve model.

To understand it better, let's start from the simplest possble tree which makes no split and predicts the same value regardless of the input. This tree is extremely simple, is independent of the input and is definitely underfitted. Nonetheless, it can still help in decreasing the loss. The problem above can be represented by this equation. 

$$

~\\ ~ \\

Loss(o) = \min_{o}  \sum_{i = 1}^N loss(y_i, f(x_i)+o) + \frac{1}{2}\lambda o^2 \\ ~ \\

\begin{align*}
&where\ N\ is\ the\ number\ of\ samples,\ f\ is\ the\ original\ model,\\
&\lambda \ is\ the\ L2\ regularization\ parameter\ and\ o\ is\ the\ value\ which\ we\ want\ to\ find.
\end{align*}

$$

This can be solved by differentiating the above expression with respect to $$ o $$, setting the the derivative to 0 and then finding the corresponding $$ o $$.{% marginnote 'sn-five' 'Refer to this <a href="https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261">article</a> for the intuition behind L2 reularization.'%} The $$ \frac{1}{2}\lambda o^2 $$ term is the L2 regularization parameter which has been shown experimentally to be effective in preventing overfitting. While not useful in this already underfitted model, it will come into relevance as we increase the tree complexity.

Now, try to solve for $$ o $$. It turns out that there's no trivial solution as the expression above is hard to differentiate. To resolve this problem, we simplify the expression using the Sterling's approximation. We now represent the above objective function as $$ Loss $$, with the capital L.

$$

~ \\ ~ \\ 

Loss(o) \approx  \sum_{i = 1}^N [loss(y_i, f(x_i)) + \frac{\partial loss}{\partial \widehat{y}}(y_i, f(x_i))o + \frac{1}{2} \frac{\partial^2 loss}{\partial \widehat{y}^2}(y_i, f(x_i)) o^2] + \frac{1}{2}\lambda o^2 \\ ~ \\

\approx \sum_{i = 1}^N [loss(y_i, f(x_i)) + g_io + \frac{1}{2} h_i o^2] + \frac{1}{2}\lambda o^2 \\~ \\

where\ g_i = \frac{\partial loss}{\partial \widehat{y}}(y_i, f(x_i)) \ and\ h_i=\frac{\partial^2 loss}{\partial \widehat{y}^2}(y_i, f(x_i)).

$$

This simplified expression can be differentiated easily and after setting the derivative to 0, we can solve and obtain $$ o $$. It turns out that $$ o $$ is the following. We will not solve it here for the sake of conciseness. 

$$

o = \frac{\sum_{i = 1}^N g_i}{\sum_{i = 1}^N h_i + \lambda}

$$

Keep in mind that, now, given a model $$ f $$ and a set of samples, we can find a single adjustment $$ o $$ which can improve our model. Note that $$ o $$ can also be substituted back into the equation to compute the value of $$ Loss $$. The remaining of this section talks about how we can further improve (decrease the loss) by making the simple model more complex (growing the tree). Here's the overall idea. 

>  By cleverly dividing the samples into subgroups and then finding $$ o $$ for each subgroup (using the method above), the performance of the model can be further improved (loss can be brought lower).

The samples can be divided using split conditions. For example, if a split condition is *"feature x less than 10"*, samples whose feature x has value less than 10 will go into 1 subgroup and the rest, to the other group. Each subgroup can be further divided iteratively if necessary (like a decision tree). {% marginnote 'sn-six' 'The minimum loss for a subgroup can be computed by substituting optimal $$ o $$ into $$ Loss $$.'%} For each subgroup, its optimal $$ o $$ and loss can be solved using the above technique. The overall loss, $$ Loss $$, is the summation of the loss of each subgroup (leaves in the decision tree).

At each group or subgroup, the decision of whether to split and if so, which split to use depends on whether a split can reduce the loss of that group and how much each split decreases loss. We choose the split which minimizes $$ Loss $$. 

Let's describe what's happening intuitively. The current model has different levels of error for different parts of the feature space. It overpredict for some samples, underpredicts others and by varying magnitudes. *By segmenting the feature space such that the errors in each subgroup is similar, the errors can be predicted more accurately, enhancing model performance.*

### Overfitting

To prevent model overfitting, the height of trees are limited, limiting the number of subgroups which can be formed. Also, the decrease in loss from a split must exceed a certain threshold for XGBoost to allow it. This is modelled into the $$ Loss $$ via an additional regualarization term, $$ \gamma T\ where\ T\ is\ the\ number\ of\ leaves $$ which was ommited earlier on to prevent confusion. 










