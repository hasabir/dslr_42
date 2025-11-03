
# Logistic Regression Fundamentals


## What **logistic regression** is and how it differs from **linear regression**


Logistic regression is a supervised machine learning algorithm in data science. It is a type of <u>classification algorithm</u> that predicts a discrete or categorical outcome.


> 💡 For example, we can use a classification model to determine whether a loan is approved or not based on predictors such as savings amount, income and credit score.


Like linear regression, it is a type of linear model that examines the relationship between predictor variables and an output variable. The key difference is that linear regression is used when the output is a continuous value - for _example, predicting someone’s credit score._


**Logistic regression is used when the outcome is categorical, such as where a loan is approved or not.** <u>**In Logistic regression, the model predicts the probability that a specific outcome occurs**</u> 


**What is classification in machine learning?**

- Classification in machine learning is a supervised learning task that assigns input data to predefined categories or classes. I_t works by training input data to predefined categories or classes on a labeled dataset._

### **Sigmoid function:**

- A sigmoid function is any mathematical function whose graph has a characteristic S-shaped or sigmoid curve.
- It is defined by the equation $ \sigma(z) = \frac{1}{1 + e^{-z}}$.

![image.png](https://storage.googleapis.com/dashboard-51ba6.appspot.com/087bdc581a7d6c67ac25feaab71cf465.png?GoogleAccessId=firebase-adminsdk-jd298%40dashboard-51ba6.iam.gserviceaccount.com&Expires=16725225600&Signature=Swbsrly%2Bvl%2FH0cbCproMKZiVfzwEu5VhXGSc5CvwC2XrtNIt6RY%2BVhHpvZ0EOiRQxj7Rxk2rSKO91lQHYM3n%2FWywNVlTWOIgqJYg5cbK6Qw2v1LScsS6CFzTRDf84Sf6foCevYmMq1UmJt%2FfClqEqlYBcQS8CVZ3NSB7QmujdW%2FKMEU5aHa3P7%2BGrI53FY%2F%2FeAm1JzV1pB7Kci8Ay7LcoEDj%2BVN7%2BWyd%2FH5GdAKjrS47KTnDH08W%2F89aLL7131EfHQzMVE9qOxqqEUUnVUzMK98CXYOjPPUhGg6GLFvbRwCodvmTRMgYliLlVYn358yfG9vISjPFBsSRA47lpPXAkg%3D%3D)

- It transforms any real-valued input into a value between 0 and 1, which makes it useful for interpreting outputs as probabilities in fields like machine learning and binary classification

---


**Logistic regression under the hood**
    


Like linear regression, logistic regression is a type of linear 
model that falls under the generalized linear models (GLM) family. As in
 the previous example, if we want to represent the probability of 
approve or not approve, we apply the linear function.


$Y\text approval=β0+β1Xsavings$


Because the linear function assumes a linear relationship, as the 
values of $X$ changes, $Y$ can take on a value from $(-inf, inf)$. 
Probabilities, as we know, are confined to $[0,1]$. Using this principle 
of linear model, we cannot directly model the probabilities for a binary
 outcome. Instead, we need a logistic model to make sense of the 
probabilities. Therefore, we want to apply a transformation to the input
 so the outcome can be confined. This transformation is known as the 
logistic regression equation. This equation might look complex, but we 
will break it down step by step how it is derived in the following 
section.

- The sigmoid transformation allows us to make a binary prediction for the preceding use case. After applying the transformation, the value of x can range from $(-inf, inf)$, and y will be confined to [0, 1].

> 💡 To understand the logistic regression function (or the sigmoid function), we need a solid foundation on the following concepts:  
> - Odds, log-odds and odds ratio  
>   
> - Coefficients of the logistic regression  
>   
> - Maximum likelihood estimates (MLE)


###  Types of Classification

- **Binary classification** → two possible classes
- **One-vs-All (OvA)** or **One-vs-Rest (OvR)** → While logistic regression is typically defined for binary classification _(where there are only two possible outcomes, 0 or 1)_, we can extend it to handle multiple classes _(determining whether the input belongs to a specific category or not)_.

# Cost (Loss) Function


**The cost function summarizes how well the model is behaving.** In other words, the cost function tells us how far off our model's guesses are from the real answers. In linear regression, we use something called mean squared error (MSE) to measure this. But in [logistic regression](https://www.baeldung.com/cs/cost-function-logistic-regression-logarithmic-expr#cost-function-of-the-logistic-regression), if we use the average of squared differences, we run into a problem. The result would be bumpy and uneven, with many peaks and valleys that make it hard to find the best solution.


![image.png](https://storage.googleapis.com/dashboard-51ba6.appspot.com/972b89a9b10ae91a8e5369a72ace3658.png?GoogleAccessId=firebase-adminsdk-jd298%40dashboard-51ba6.iam.gserviceaccount.com&Expires=16725225600&Signature=jHc4AUCm8tc28O%2Bqs6ZkOhlLuljsp3A1LR7rd6w4inLPX1kD1Ci4BubAW0xADj9eLoISoOPITMKg2xnfOwnrWprYxX2TITdI%2FAo%2Bz0V9GHX3iUDaTjjw4nWAvakgq2jOrEtv6K9RYwd3dAkQ%2B0Lq%2FCSPh1Hxmcvq%2FwK9w1Yez7vawu%2Fv1Qnv6lZClv67nt%2FeKMKsVzA%2FyN7rflXnQ4m7jO572tBQUBSGf2GAw9m9kitfdDB2eWdvh6SzHW0vG%2B0RWyDteVbBd3bZ%2BEJCNIJ7Mr0hugH4y7uIrVfxwxmSw7Q%2BGeHpcB%2FtRGg4EC%2FXpFqe%2BIzL3AZQ4OiL71HHoCBzRw%3D%3D)


**✅ The Theory Behind Logistic Regression Cost Function:**


### <u>**Logistic Regression Likelihood**</u>


> 💡 $P(y∣x;θ)=[hθ(x)]y1−hθ(x)$


_How did we get there?_


We want to model how likely it is to observe the true label (y) for a given feature vector (x).


---


## 🎯 The Goal of Logistic Regression


We want to compute:


$P(y \mid x; \theta)$


Which means:


> Given feature values $(x)$, what is the probability that the true class is $(y)$?


---


## 📌 Binary Classification Case


$(y)$ can only be:

- $(1)$ → positive class
- $(0)$ → negative class

So:


$P(y=1 \mid x;\theta) = h_\theta(x)$


$P(y=0 \mid x;\theta) = 1 - h_\theta(x)$


Where:


$h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1+e^{-\theta^Tx}}$


🧠 Because the $sigmoid$ **forces output into [0, 1]**, we can interpret it as a probability.


## ✅ Combining Both Cases Into One Formula


We want a formula that handles both (y=0) and (y=1) **automatically**, without writing two cases every time.


Here is the trick:


| y value | Probability formula                                       | Why                      |
| ------- | --------------------------------------------------------- | ------------------------ |
| $(y=1)$ | $(h_\theta(x)^1 \cdot (1-h_\theta(x))^0 = h_\theta(x))$   | because $anything^0 = 1$ |
| $(y=0)$ | $(h_\theta(x)^0 \cdot (1-h_\theta(x))^1 = 1-h_\theta(x))$ | because $anything^0 = 1$ |


![image.png](https://storage.googleapis.com/dashboard-51ba6.appspot.com/09ba80b60ff47e210940ffd20dc7e741.png?GoogleAccessId=firebase-adminsdk-jd298%40dashboard-51ba6.iam.gserviceaccount.com&Expires=16725225600&Signature=Q4uElcmSnYnRVbxm3nB1hWr0ggWEgCnzdwF0ILcfYZ1ASP%2Bj%2FsUOdMYbr5dHRdtR877xSMa0nKKj1pcnv%2BI01Pn6WaF0g50yw%2F2ltqLhJS7IhG893zSbNyo%2FqOpo2HO3Gzwo8CaUQNlYWQJ1r5FGqP6kVjC3e55jxnhylmZJIfZIfnx6uw%2BUtLM3NUeNaRmn9aa6tA%2FyWYLbuekU4b%2BaYRJRdbjiijnKw7AsMXWdurb8Fe7OhqX0o%2Bkc0azMcpN3gKGuLoKiZm4FAPq1CoJgNyYdbrhVAEhS3cZMvoqjKYtAEiz4X9bjhPHaguzC5xY7YAmb6aSnf7Yf%2Fz0iLx1gDA%3D%3D)


So we combine both cases into this elegant formula:


$\boxed{P(y\mid x;\theta)= [h_\theta(x)]^{y} [1 - h_\theta(x)]^{(1-y)}}$


---


### <u>Maximum Likelihood Principle</u>


### What is the Maximum Likelihood Principle?


> We choose the parameters θ that make the observed data the most probable.


Imagine you have a dataset of **inputs** $x^{(i)}$ and **labels** $y^{(i)}$.


We want to find $θ$ $(weights)$ such that:


$\theta^* = \arg\max_\theta P(\text{data} \mid \theta)$


This means:


**Pick the θ that gives the highest probability for the labels we actually observed.**


> We already have this compact probability expression:


	$P(y \mid x;\theta)= [h_\theta(x)]^{y} \, [1 - h_\theta(x)]^{(1-y)}$


## Extending to the Entire Dataset


Assuming training samples are **independent**, the likelihood across all $m$ samples is:


$L(\theta) = \prod_{i=1}^{m} P(y^{(i)} \mid x^{(i)};\theta)$


Substituting our probability formula:


$\boxed{L(\theta)=\prod_{i=1}^m [h_\theta(x^{(i)})]^{y^{(i)}} [1-h_\theta(x^{(i)})]^{(1-y^{(i)})}}$


This is the **likelihood of θ**.


Our goal is to **maximize** it.


<u>_Why Take the Logarithm?_</u>


Multiplying many tiny numbers leads to **underflow**—values become extremely close to zero.


To avoid this, we use the **log-likelihood**:


$\log L(\theta) = \sum_{i=1}^m \Big( y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)})) \Big)$


✔️ Logs convert products into sums, making optimization easier


✔️ Still equivalent, since log is strictly increasing


✔️ Sums are numerically more stable


## From Maximization to Minimization


Machine learning optimization typically **minimizes** functions rather than maximizes them.


So we take the **negative** of the log-likelihood:


$-\log L(\theta) = \sum_{i=1}^{m} \bigg[ -y^{(i)}\log(h_\theta(x^{(i)})) - (1-y^{(i)})\log(1 - h_\theta(x^{(i)})) \bigg]$


Divide by $m$ to get the mean, giving us the final **cost function**:


$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \Big( -y^{(i)}\log(h_\theta(x^{(i)})) - (1-y^{(i)})\log(1 - h_\theta(x^{(i)})) \Big)$


## **Minimizing the Cost with Gradient Descent**


Gradient descent is an iterative optimization algorithm used to find parameters $\theta$ that **minimize** a differentiable cost function. In logistic regression, the cost function is the **log-loss** (negative log-likelihood):


$\min_\theta \, J(\theta)$


The hypothesis function is the sigmoid applied to a linear combination of features:


$h_\theta(x) = \sigma(\theta^T x)$


Since our model contains $n$ features, we must learn $n+1$ parameters $\theta_0, \theta_1, ..., \theta_n$. Gradient descent updates each parameter in the direction that **reduces** the cost:


$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$


where $\alpha$ is the learning rate.


The partial derivative of the cost function with respect to each parameter is:


$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left( h_{\theta}(x^{(i)}) - y^{(i)} \right) x_j^{(i)}$


Substituting this into the update rule gives:


$\boxed{\theta_j := \theta_j - \alpha \frac{1}{m}\sum_{i=1}^{m}\left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}}$   **Batch Gradient Descent**: → $\boxed{\theta := \theta - \alpha \frac{1}{m} X^T(h - y)}$


All parameters $\theta_j$ must be updated **simultaneously** at every iteration.


Interestingly, this update rule has the **same form** as the gradient descent rule for linear regression—but the hypothesis $h_\theta(x)$ and the cost function are different. This is why logistic regression works correctly without squared errors.


By repeatedly applying this update until convergence, we obtain the parameters $\theta$ that minimize the cost and produce the best classifier.


![image.png](https://storage.googleapis.com/dashboard-51ba6.appspot.com/87ef62dc98ffe31c5153dc4471751508.png?GoogleAccessId=firebase-adminsdk-jd298%40dashboard-51ba6.iam.gserviceaccount.com&Expires=16725225600&Signature=Q3MxnzYOlKGfmjIWe3VhIeZPLozJCGconprd6sMK9Qunlv8pOc93dr50%2FTm7D4M3ji1NL2pJSUHjMvLNLLsv3vhS3VYxRgnpRtw2Mbx0EfDEAulRQEPBFX0da8XBC4AfJbrcNQCeRBCg0G%2B3QpTQ7Qo64N%2FRJMTmvMSLLBObYUcS3d7CNQ%2FKS5DoVcR%2BKk8XpvHtaKbXjhtVpN2q3kmei42XzNhZjGDCojotZA74InBRNp7Cgpagy0H2IpDkyQZB6J233V6PYJo4bL1RClbj8SEcDvJPZV%2BxuOuuoBkTrgC%2FFlUMuuWJ0r9Th4Gb9xgVz0dUrsHFwdUMTPg%2BdYbRCw%3D%3D)

