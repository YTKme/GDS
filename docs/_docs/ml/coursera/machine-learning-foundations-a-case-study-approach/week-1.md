---
title: Machine Learning Foundations A Case Study Approach <small class="text-muted d-block">Week 1</small>
permalink: /docs/ml/coursera/machine-learning-foundations-a-case-study-approach/week-1/
---

## Welcome

<p class="lead">Machine Learning Specialization</p>
<p class="lead">University of Washington</p>
<blockquote class="blockquote">Machine learning is changing the world</blockquote>

## Old View of Machine Learning

1. Data
    * Start with some data set or data.
2. Machine Learning Algorithm
    * Fed it to some magical machine learning algorithm.
3. My Curve Is Better Than Your Curve
    * Show that my curve is better than your curve.
4. Write A Paper
    * Write the paper to machine learning conference.

![Figure 1: Old View of Machine Learning]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-1/old-view-of-machine-learning.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

## Industry

Disruptive companies differentiated by INTELLIGENT APPLICATIONS using MACHINE LEARNING.

* Google PageRank Search
* Amazon Retail
* Netflix Movie Distribution
* Pandora Music
* Google Adsense Advertising
* Glassdoor Human Resource
* Harmony Dating
* Uber Taxis
* RelateIQ CRM
* Fitbit Wearables
* Avvo Legal Advice
* Zillow Real Estate
* Obama'08 Campaigning
* LinkedIn Networking
* LivingSocial Coupons

## The Machine Learning Pipeline

1. Data
    * Start from data.
2. Machine Learning Method  
    * Bring in a machine learning method that provide a new kind of analysis of the data.
3. Intelligence
    * The analysis will give intelligence such as product likely to buy right now.

Taking this pipeline and working through it in a wide range of settings and a wide range of applications, with a wide range of algorithms. With the understanding how they connect together, be able to really build smart, intelligent applications.

![Figure 2: The Machine Learing Pipeline]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-1/the-machine-learning-pipeline.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

## Machine Learning Case Studies

### Case Study 1: Predicting House Prices

The intelligence derived is a value associated with some house that's not on the market. The value is unknown and learned from the data.

* Data
    * Look at other houses and look at their house sales price to inform the house value of the unknown house.
    * In addition to the sales prices, also look at other features of the houses. (such as how many bedrooms, how many bathrooms, number of square feet, etc.)
* Method
    * The machine learning method is going to relate the house attributes (features) to the sales price.
    * If the model is learned, it can take the house attribute(s) and predict the unknown sales price. (**regression method**)

### Case Study 2: Sentiment Analysis

Explore a sentiment analysis task for reviews of restaurants. By taking the reviews and be able to classify whether it had positive sentiment (thumbs up) or negative sentiment (thumbs down).

* Data
    * Look at a lot of other reviews.
    * Look at the text of the review and rating of the reivew.
    * In order to understand what's the relationship, for classification of the sentiment.
* Method
    * Analyze the text of the review in terms of how many time it uses certain words.
    * From the reviews, learn some decision boundary based on the balance of positive and negative word usage, whether it's a positive or negative reivew.
    * Learn from other reviews based on the rating associated with the text. (**classification method**)

### Case Study 3: Document Retrieval

Document retrieval task where the intelligence derived is an article or a book or something of interest to the reader.

* Data
    * Huge collection of possible articles used for recommendation.
* Method
    * Find structure in this data based on groups of related articles.
    * Maybe there's a collection of articles about sports, world news, entertainment, science, etc.
    * Find a structure and annotate the [corpus](https://en.wikipedia.org/wiki/Text_corpus){:target="_blank"} (a collection of documents with labels).
    * Don't have corpus ahead of time, trying to infer from the data.
    * Use for very rapid document retrieval to retrieve similar articles. (**clustering**)

### Case Study 4: Product Recommendation

Collaborative filtering of product recommendation, where take past purchases and try to recommend some set of other product might be of interest in purchasing. Derive intelligence for production recommendation in relationship to products bought and products likely to buy in the future.

* Data
    * Other users' purchase histories.
    * Possibly features of the users.
* Method
    * Take the data and arrange it into customers by products matrix where the squares indicate products that a customer actually purchased (products liked by that customer).
    * From the matrix, learn features about the users and features about the products.
    * Use the features to see how much agreement there is between:
        * What the user likes.
        * Different attributes the user likes.
        * Whether the product is actually about those attributes.
    * Going from customers by products matrix into learned features about users and products. (**matrix factorization**)

### Case Study 5: Visual Product Recommender

A visual product recommender based on user input of images, which resulting in other products / images of interest.

* Data
    * User input, not text, but image.
* Method
    * Go from an image to a set of related images.
    * Need to have very good features about the input image to find other images that are similar.
    * Derive really detailed features using something called deep learning.
    * Look at neural networks where every layer provide more and more descriptive features.
    * Deeper and deeper layers to get more intricate features. (**deep learning**)

## A Unique Machine Learning Specialization

From use cases to models and algorithms.
* See machine learning through the lens of a wide range of case studies in different areas that really ground the concepts behind them.
* Other machine learning approach can be a laundry list of algorithm and methods.
* The problem is starting with algorithms, and end up having really simplistic use cases with applications, that are disconnected from reality.
* Start with use cases, grasp the key concepts and the techniques, then build and measure the quality to understand whether the intelligent applications are working well or not.

### Introduction

First course is about building, evaluating, and deploying *intelligence* in each case study...

Figure out what task to solve, what machine learning methods make sense, and how to measure them. (using algorithms as black boxes, for now)
* Which machine learning model(s) to use.
* What methods is used to optimize the parameters of the model(s)?
* Is it providing the intelligence needed?
* How to measure the quality of the system?

![Figure 3: The Machine Learing Pipeline Version 2]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-1/the-machine-learning-pipeline-v2.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

Subsequent courses provide depth in models & algorithms, but still use case studies.
1. Introduction
2. Regression
3. Classification
4. Clustering & Retrieval
5. Matrix Factorization & Dimensionality Reduction
6. Capstone: Build an Intelligent Application with Deep Learning

### Regression

Case Study: Predict House Prices (from features of the house)

* Models
    * Linear Regression
    * Regularization
        * Ridge (L2)
        * Lasso (L1)
* Algorithms (Optimization)
    * Gradient Descent
    * Coordinate Descent
* Concepts
    * Loss Functions
    * Bias-Variance
    * Tradeoff
    * Cross-Validation
    * Sparsity
    * Overfitting
    * Model Selection

### Classification

Case Study: Analyzing Sentiment

* Models
    * Linear Classifiers (Logistic Regression, <abbr title="Support Vector Machines">SVMs</abbr> (Support Vector Machines), Perceptron)
    * Kernels
    * Decision Trees
    * (Kernels and Decision Tress to deal with non-linear complex features)
* Algorithms
    * Stochastic Gradient Descent
    * Boosting
* Concepts
    * Decision Boundaries
    * <abbr title="Maximum Likelihood Estimation">MLE</abbr> (Maximum Likelihood Estimation)
    * Ensemble Methods
    * Random Forests
    * <abbr title="Classification And Regression Tree">CART</abbr> (Classification And Regression Tree)
    * Online Learning

### Clustering & Retrieval

Case Study: Finding Documents

* Models
    * Nearest Neighbors
    * Clustering, Mixtures of Gaussians
    * Latent Dirichlet Allocation (<abbr title="Latent Dirichlet Allocation">LDA</abbr>) (advance text analysis clustering technique)
* Algorithms
    * <abbr title="K-Dimensional">KD</abbr>-Trees (K-Dimensional Tree), Locality-Sensitive Hashing (<abbr title="Locality-Sensitive Hashing">LSH</abbr>)
    * K-Means
    * Expectation-Maximization (<abbr title="Expectation-Maximization">EM</abbr>)
* Concepts
    * Distance Metrics
    * Approximation Algorithms
    * Hashing
    * Sampling Algorithms
    * Scaling Up With Map-Reduce
    * (how to scale things up and write them as distributed algorithms)

### Matrix Factorization & Dimensionality Reduction

Case Study: Recommending Products

* Models
    * Collaborative Filtering
    * Matrix Factorization
    * (dimensionality reduction)
    * <abbr title="Principal Component Analysis">PCA</abbr> (Principal Component Analysis)
* Algorithms
    * Coordinate Descent
    * Eigen Decomposition
    * <abbr title="Singular Value Decomposition">SVD</abbr> (Singular Value Decomposition)
* Concepts
    * Matrix Completion
    * Eigenvalues
    * Random Projects
    * Cold-Start Problem
    * Diversity
    * Scaling Up

### Capstone

An intelligent application using deep learning

Build & deploy a recommender using product images and text sentiment.

## Level of the Specialization

<blockquote class="blockquote">
    <b>Motto:</b><br />
    though concepts made intuitive and applicable
</blockquote>

It is not going to be about theorem proving, but understanding at a very intuitive and practical level some very important machine learning algorithms. Thinking about ways in which to deploy them in new problems.

* minimize prerequisite knowledge
* maximize ability to develop and deploy
* learn concepts through case studies

## Target Audience
* Software Engineer
    * Interested in machine learning
* Scientist
    * Become data scientist
* Data Enthusiast
    * Some math, some programming experience
    * Be able to analyze data and do fun things with it
    * Learn more about machine learning and how to derive intelligence from data

## Math Background
* Basic Calculus
    * Concept of Derivatives
* Basic Linear Algebra
    * Vectors
    * Matrices
    * Matrix Multiply

## Programming Experience
* Basic Python Used
    * Can Pick Up Along The Way
    * Knowledge of Other Language

## Computing Need
* Basic Desktop or Laptop
* Access To Internet
* Ability To:
    * Install and run Python
    * Store a few <abbr title="Gigabyte">GB</abbr> of data

## You'll Be Able To Do Amazing Things!

## Journey

![Figure 4: Journey]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-1/journey.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

* Course 1: Build Intelligent Applications
* Course 2-5: Formulate, Implement & Evaluate Machine Learning Methods
* Course 6: Design & Deploy Exciting Application

## The Capstone Project
Build and deploy an intelligent application with deep learning.

An intelligent recommender using images & text

## We Will Do Something Even More Exciting!

![Figure 5: Capstone Project]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-1/capstone-project.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

* Recommenders
    * Recommender system
* Text Sentiment Analysis
    * What people are saying in the review about different products and extract those reviews to analyze them
* Computer Vision
    * Visually understand images
* Deep Learning
    * Take computer vision techniques and make them extremely accurate
* Deploy Intelligent Web Application
    * Intelligent web service for products to interact with
    * Show intelligence behind the service

## Getting Started With Python

There are two really exciting tools for programming machine learning algorithms, and also useful in data science.
* [Python](https://www.python.org/){:target="_blank"} Programming Language
* [iPython Notebook](https://jupyter.readthedocs.io/en/latest/install.html){:target="_blank"} ([Project Jupyter](https://jupyter.org/){:target="_blank"})


**Notebook:** [Getting Started With iPython Notebook](https://github.com/YTKme/GDS/blob/development/ml/coursera/machine-learning-foundations-a-case-study-approach/week-1/work/Getting-Started-With-iPython-Notebook.ipynb){:target="_blank"}

## Getting Started With SFrame (DateFrame) For Data Engineering and Analysis

The specialization will use a number of Python tools to help get started with machine learning. But there are many more.
* [pandas](https://pandas.pydata.org/){:target="_blank"}
* [scikit-learn](https://scikit-learn.org/){:target="_blank"}
    * [NumPy](https://numpy.org/){:target="_blank"}
    * [SciPy](https://www.scipy.org/){:target="_blank"}
* [matplotlib](https://matplotlib.org/){:target="_blank"}
* [TensorFlow](https://www.tensorflow.org/){:target="_blank"}

**Notebook:** [Getting Started With SFrames](https://github.com/YTKme/GDS/blob/development/ml/coursera/machine-learning-foundations-a-case-study-approach/week-1/work/Getting-Started-With-SFrames.ipynb){:target="_blank"}

## Appendix

## Reference
* [[PDF] Introduction]({{ "/res/misc/ml/coursera/machine-learning-foundations-a-case-study-approach/week-1/intro.pdf" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:target="_blank"}
* [[GitHub] Repository Starter Code](https://github.com/learnml/machine-learning-specialization){:target="_blank"}
* [[GitHub] Data Set](https://github.com/KarthiAru/Machine-Learning-Specialization-Coursera)