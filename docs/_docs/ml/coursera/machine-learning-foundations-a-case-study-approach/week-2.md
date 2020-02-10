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

### Use Linear Regression Model

![Figure 4: Use Linear Regression Model]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-2/linear-regression-model.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### Which Line?

### Cost Using Given Line

### Find Best Line

### Predict House Price

## Add Higher Order Effect

### Fit Data With Line?

### What About Quadratic Function?

### Even Higher Order Polynomial

### Believe This Fit?

## Evaluate Overfit Via Train / Test Split

## Appendix

## Reference
* [Regression [PDF]]({{ "/res/misc/ml/coursera/machine-learning-foundations-a-case-study-approach/week-2/regression-intro-annotated.pdf" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:target="_blank"}
* [](){:target="_blank"}