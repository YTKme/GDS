---
title: Machine Learning Foundations A Case Study Approach
permalink: /docs/ml/coursera/machine-learning-foundations-a-case-study-approach/
---

## Week 1

### Welcome

<p class="lead">Machine Learning Specialization</p>
<p class="lead">University of Washington</p>
<blockquote class="blockquote">Machine learning is changing the world</blockquote>

### Old View of Machine Learning

1. Data
    * Start with some data set or data.
2. Machine Learning Algorithm
    * Fed it to some magical machine learning algorithm.
3. My Curve Is Better Than Your Curve
    * Show that my curve is better than your curve.
4. Write A Paper
    * Write the paper to machine learning conference.

![Figure 1: Old View of Machine Learning]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/old_view_of_machine_learning.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### Industry

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

### The Machine Learning Pipeline

1. Data
    * Start from data.
2. Machine Learning Method
    * Bring in a machine learning method that provide a new kind of analysis of the data.
3. Intelligence
    * The analysis will give intelligence such as product likely to buy right now.

Taking this pipeline and working through it in a wide range of settings and a wide range of applications, with a wide range of algorithms. With the understanding how they connect together, be able to really build smart, intelligent applications.

![Figure 2: The Machine Learing Pipeline]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/the_machine_learning_pipeline.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### Machine Learning Case Studies

#### Case Study 1: Predicting House Prices

The intelligence derived is a value associated with some house that's not on the market. The value is unknown and learned from the data.

* Data
    * Look at other houses and look at their house sales price to inform the house value of the unknown house.
    * In addition to the sales prices, also look at other features of the houses. (such as how many bedrooms, how many bathrooms, number of square feet, etc.)
* Method
    * The machine learning method is going to relate the house attributes (features) to the sales price.
    * If the model is learned, it can take the house attribute(s) and predict the unknown sales price. (**regression method**)

#### Case Study 2: Sentiment Analysis

Explore a sentiment analysis task for reviews of restaurants. By taking the reviews and be able to classify whether it had positive sentiment (thumbs up) or negative sentiment (thumbs down).

* Data
    * Look at a lot of other reviews.
    * Look at the text of the review and rating of the reivew.
    * In order to understand what's the relationship, for classification of the sentiment.
* Method
    * Analyze the text of the review in terms of how many time it uses certain words.
    * From the reviews, learn some decision boundary based on the balance of positive and negative word usage, whether it's a positive or negative reivew.
    * Learn from other reviews based on the rating associated with the text. (**classification method**)

#### Case Study 3: Document Retrieval

Document retrieval task where the intelligence derived is an article or a book or something of interest to the reader.

* Data
    * Huge collection of possible articles used for recommendation.
* Method
    * Find structure in this data based on groups of related articles.
    * Maybe there's a collection of articles about sports, world news, entertainment, science, etc.
    * Find a structure and annotate the [corpus](https://en.wikipedia.org/wiki/Text_corpus){:target="_blank"} (a collection of documents with labels).
    * Don't have corpus ahead of time, trying to infer from the data.
    * Use for very rapid document retrieval to retrieve similar articles. (**clustering**)

#### Case Study 4: Product Recommendation

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

#### Case Study 5: Visual Product Recommender

A visual product recommender based on user input of images, which resulting in other products / images of interest.

* Data
    * User input, not text, but image.
* Method
    * Go from an image to a set of related images.
    * Need to have very good features about the input image to find other images that are similar.
    * Derive really detailed features using something called deep learning.
    * Look at neural networks where every layer provide more and more descriptive features.
    * Deeper and deeper layers to get more intricate features. (**deep learning**)

### A Unique Machine Learning Specialization

From use cases to models and algorithms.

#### Introduction

First course is about building, evaluating, and deploying *intelligence* in each case study...

![Figure 3: The Machine Learing Pipeline Version 2]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/the_machine_learning_pipeline_v2.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

Subsequent courses provide depth in models & algorithms, but still use case studies.
1. Introduction
2. Regression
3. Classification
4. Clustering & Retrieval
5. Matrix Factorization & Dimensionality Reduction
6. Capstone: Build an Intelligent Application with Deep Learning

#### Regression