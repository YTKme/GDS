---
title: Machine Learning Foundations A Case Study Approach <small class="text-muted d-block">Week 2</small>
permalink: /docs/ml/coursera/machine-learning-foundations-a-case-study-approach/week-2/
---

## Regression: Predicting House Price

One of the most widely used statistical tools.

The idea is really simple. Using some sets of **features**, model how the **observations** associated with these **features** change, as the values of the **features** change. In this case, house values.

A house have some set of features:
* Size of the house
* Number of bedrooms
* Number of bathrooms
* Etc.

The observation is what's the value of the house, or the house sales price.

But regression go much beyond doing prediction tasks. It can also used for classification, as well as analyze the importance of the features themselves.

## Predicting House Price

* Problem arise when predicting the value of a house

### How Much The House Worth?

* Unsure how much to list the house for
* Unsure what value the house has
* How to estimate the value of the house

### Check Recent Neighborhood Sales

* Look at the recent sales that occurred in the neighborhood (locally at the neighboring regions)
    * How much are the other houses sold for
    * What do these houses look like
* Record recent sales
    * What was the sales price
    * What was the size of the house sold

### Plot Recent House Sales

* X-Axis: size of the house measured in square feet (sq. ft.)
* Y-Axis: sales price of the house in dollar ($)
* Point: represent some individual house sale

![Figure 1: Recent House Sale]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-2/recent-house-sale.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-50"}

#### Terminology
* X: feature, covariate, or predictor (independent variable)
* Y: observation or response (dependent variable)

### Predict House By Similar House

How to use these observations to estimate house value?
* Take a look how big is the house
* Look for other sales of houses of that size

Though most likely there are going to be exactly zero house sales of the exact same square footage.
Need to be a bit more flexible and look at some neighborhood (not geographic neighborhood) of range around the actual square footage.

![Figure 2: Predict Similar House]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-2/predict-similar-house.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-50"}

But even this approach only have two house sales to base estimate. (throwing away many other observations)
* Look at average price in range
* Still only 2 house
* Throwing out information from all other sales

![Figure 3: Predict Similar House v2]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-2/predict-similar-house-v2.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-50"}

## Linear Regression

Think about modeling the relationship between the square footage of the house and the house sales price.

### Use Linear Regression Model

Leverage all the observations collected, understand the relationship between the square foot of the house and the house sales price.
The simplest model, is just fitting a line through the data.
* Define by an intercept <math>w<sub>0</sub></math> and a slope of <math>w<sub>1</sub></math>
* Often <math>w<sub>1</sub></math> the weight on the feature <math>X</math> or it's called the regression coefficient
* As <math>X</math>, the square footage is varied, how much of an impact does it have on changes in the observed house sales price

The intercept and slope are the parameters of the model.
To be very explicit, the linear function with the subscript <math>w</math> indicates the function is specified by parameters.

The <b><math>f<sub>w</sub> = w<sub>0</sub> + w<sub>1</sub>x</math></b> function parameterized by <b><math>w = (w<sub>0</sub>, w<sub>1</sub>)</math></b>

![Figure 4: Use Linear Regression Model]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-2/linear-regression-model.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### Which Line?

Which line is the right line or a good line to ue for a given data set?
Each one is given by a different set of parameters <math>w</math>.
But which <math>w</math> to choose for the model?

![Figure 5: Use Linear Regression Model Multiple Line]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-2/linear-regression-model-line.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### Cost Using Given Line

One very common cost associated with a specific fit to the data is something called the Residual Sum of Squares (<abbr title="Residual Sum of Squares">RSS</abbr>).
* Take the fitted line and look at each of the observations
* Look at how far is that observation from what the model would predict (point on the line)
* Look at all the distance, and look at the square of these distances
    * Called the residual, which is the difference of the prediction to the actual observation
* Look at the square and sum over them

More explicit equation
* Variable <math>$<sub>house1</sub></math> is the **observed** house sales price for the first house
* Variable <math>sq.ft.<sub>house1</sub></math> is where the <math>$<sub>house1</sub></math> point intercept with the x-axis
* Variable <math>w<sub>0</sub> + w<sub>1</sub> sq.ft.<sub>house1</sub></math> is the **predicted** house sales price for the first house
* The difference between the **observed** house price versus **predicted** (model) for a given house square foot is <math>$<sub>house1</sub> - [w<sub>0</sub> + w<sub>1</sub>sq.ft.<sub>house1</sub>]</math>
* Squaring the result, and summing over all the different houses in the data set

![Figure 6: Residual Sum of Squares]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-2/residual-sum-of-squares.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### Find Best Line

Trying to find the best line according to the metric define, the residual sum of squares.
* Search all possible <math>w<sub>0</sub>, w<sub>1</sub></math>
* Search all possible lines
* Choose the one that **minimize** the residual sum of squares
* Denote the resulting <math>w</math> as <math>w (hat)</math>
    * It is the set of <math>w<sub>0</sub> (hat)</math> and <math>w<sub>1</sub> (hat)</math>, which is the intercept and the slope
    * Variable <math>w<sub>0</sub> (hat)</math> represent the set of all <math>w<sub>0</sub></math>
    * Variable <math>w<sub>1</sub> (hat)</math> represent the set of all <math>w<sub>1</sub></math>

There exists really nie and efficient algorithms for computing <math>w (hat)</math> (search over all possible <math>w</math>)

![Figure 6: Best Line]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-2/best-line.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### Predict House Price

How to take the estimated model parameters and use them to predict the value of the house.
* Plotted line associated with estimate <math>w<sub>0</sub> (hat)</math> and <math>w<sub>1</sub> (hat)</math>
* Best guess of the house price is simply what the line predicts
* Computer the value for this square footage of the house, which is <math>y (hat) = w<sub>0</sub> (hat) + w<sub>1</sub> (hat) sq.ft.<sub>house</sub></math>

![Figure 6: Predict House Price]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-2/predict-house-price.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

## Add Higher Order Effect

### Fit Data With Line?

### What About Quadratic Function?

### Even Higher Order Polynomial

### Believe This Fit?

## Evaluate Overfit Via Train / Test Split

## Appendix

## Reference
* [[PDF] Regression]({{ "/res/misc/ml/coursera/machine-learning-foundations-a-case-study-approach/week-2/regression-intro-annotated.pdf" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:target="_blank"}
* [[Wikipedia] Residual Sum of Squares](https://en.wikipedia.org/wiki/Residual_sum_of_squares){:target="_blank"}