# DataStructures-Algorithms-ML-concepts-using-Python
Here, you will find the explanation of Data Structures and Algorithms for Machine Learning Models using Python.

# What is Gradient Descent?

 - Iterative Algorithm (An iterative algorithm executes steps in iterations. It aims to find successive approximations in sequence to reach a solution. They are most commonly used in linear programs where large numbers of variables are involved.) https://web.stanford.edu/group/sisl/k12/optimization/MO-unit1-pdfs/1.8iterativeloops.pdf
 
## Objective : Gradient descent algorithm is an iterative process that takes us to the minimum of a function.
 
## Mathematically, 
 - used to find a minimum of a differentiable function. https://en.wikipedia.org/wiki/Differentiable_function
 
## Why do we need this in ML? 
 - to minimize the "Cost Function" and to find the corresponding "OPTIMAL PARAMETERS' (Optimization)!

## What is a Cost Function then?
 - A function to measure the deviation of our Model's Prediction from the ground truth!
 
## Which ML models implement Gradient Descent?
 - Linear Regression Models (Simple, Multiple)
 - Logistic Regression Models

## Types of Gradient Descent in ML?
 - Batch Gradient Descent
 - Stochastic Gradient Descent
 
 
# Gradient Descent Procedure:
 
The procedure starts off with initial values for the coefficient or coefficients for the function. These could be 0.0 or a small random value.

coefficient = 0.0

The cost of the coefficients is evaluated by plugging them into the function and calculating the cost.

cost = f(coefficient)

or

cost = evaluate(f(coefficient))

The derivative of the cost is calculated. The derivative is a concept from calculus and refers to the slope of the function at a given point. We need to know the slope so that we know the direction (sign) to move the coefficient values in order to get a lower cost on the next iteration.

delta = derivative(cost)

Now that we know from the derivative which direction is downhill, we can now update the coefficient values. A learning rate parameter (alpha) must be specified that controls how much the coefficients can change on each update.

coefficient = coefficient – (alpha * delta)

This process is repeated until the cost of the coefficients (cost) is 0.0 or close enough to zero to be good enough.

You can see how simple gradient descent is. It does require you to know the gradient of your cost function or the function you are optimizing, but besides that, it’s very straightforward. Next we will see how we can use this in machine learning algorithms.

# Batch Gradient Descent for Machine Learning:

The goal of all supervised machine learning algorithms is to best estimate a target function (f) that maps input data (X) onto output variables (Y). This describes all classification and regression problems.

Some machine learning algorithms have coefficients that characterize the algorithm's estimate for the target function (f). Different algorithms have different representations and different coefficients, but many of them require a process of optimization to find the set of coefficients that result in the best estimate of the target function.

Common examples of algorithms with coefficients that can be optimized using gradient descent are Linear Regression and Logistic Regression.

The evaluation of how close a fit a machine learning model is to estimates the target function which can be calculated a number of different ways, often specific to the machine learning algorithm. The cost function involves evaluating the coefficients in the machine learning model by calculating a prediction for the model for each training instance in the dataset and comparing the predictions to the actual output values and calculating a sum or average error (such as the Sum of Squared Residuals or SSR in the case of linear regression).

From the cost function, a derivative can be calculated for each coefficient so that it can be updated using exactly the update equation described above.

The cost is calculated for a machine learning algorithm over the entire training dataset for each iteration of the gradient descent algorithm. One iteration of the algorithm is called one batch and this form of gradient descent is referred to as batch gradient descent.

Batch gradient descent is the most common form of gradient descent described in machine learning.

# Stochastic Gradient Descent for Machine Learning :

Gradient descent can be slow to run on very large datasets.

Because one iteration of the gradient descent algorithm requires a prediction for each instance in the training dataset, it can take a long time when you have many millions of instances.

In situations when you have large amounts of data, you can use a variation of gradient descent called stochastic gradient descent.

In this variation, the gradient descent procedure described above is run but the update to the coefficients is performed for each training instance, rather than at the end of the batch of instances.

The first step of the procedure requires that the order of the training dataset is randomized. This is to mix up the order that updates are made to the coefficients. Because the coefficients are updated after every training instance, the updates will be noisy jumping all over the place, and so will the corresponding cost function. By mixing up the order for the updates to the coefficients, it harnesses this random walk and avoids it getting distracted or stuck.

The update procedure for the coefficients is the same as that above, except the cost is not summed over all training patterns, but instead calculated for one training pattern.

The learning can be much faster with stochastic gradient descent for very large training datasets and often you only need a small number of passes through the dataset to reach a good or good enough set of coefficients, e.g. 1-to-10 passes through the dataset.

# Tips for Gradient Descent :

This section lists some tips and tricks for getting the most out of the gradient descent algorithm for machine learning.

Plot Cost versus Time: Collect and plot the cost values calculated by the algorithm each iteration. The expectation for a well performing gradient descent run is a decrease in cost each iteration. If it does not decrease, try reducing your learning rate.
Learning Rate: The learning rate value is a small real value such as 0.1, 0.001 or 0.0001. Try different values for your problem and see which works best.
Rescale Inputs: The algorithm will reach the minimum cost faster if the shape of the cost function is not skewed and distorted. You can achieve this by rescaling all of the input variables (X) to the same range, such as [0, 1] or [-1, 1].
Few Passes: Stochastic gradient descent often does not need more than 1-to-10 passes through the training dataset to converge on good or good enough coefficients.
Plot Mean Cost: The updates for each training dataset instance can result in a noisy plot of cost over time when using stochastic gradient descent. Taking the average over 10, 100, or 1000 updates can give you a better idea of the learning trend for the algorithm.


# Summary:

In this file you discovered gradient descent for machine learning. You learned that:

Optimization is a big part of machine learning.
Gradient descent is a simple optimization procedure that you can use with many machine learning algorithms.
Batch gradient descent refers to calculating the derivative from all training data before calculating an update.
Stochastic gradient descent refers to calculating the derivative from each training data instance and calculating the update immediately.

Example of a Real-world application: Agile (SDLC) is a pretty well-known term in the software development process. The basic idea behind it is simple: build something quickly, ➡️ get it out there, ➡️ get some feedback ➡️ make changes depending upon the feedback ➡️ repeat the process. The goal is to get the product near the user and guide you with feedback to obtain the best possible product with the least error. Also, the steps taken for improvement need to be small and should constantly involve the user. In a way, an Agile software development process involves rapid iterations. The idea of — start with a solution as soon as possible, measure and iterate as frequently as possible, is Gradient descent under the hood.
