---
title: Machine Learning Foundations A Case Study Approach <small class="text-muted d-block">Week 6</small>
permalink: /docs/ml/coursera/machine-learning-foundations-a-case-study-approach/week-6/
---

## Deep Learning: Searching For Images

Going to talk about on of the most exciting thing to be happening with machine learning over the last few years.
It's a new area called deep learning.
In particular, going to talk about a particular use case of this area related to shopping for products just based on image similarity.

## Visual Product Recommender

There are many ways to shop for products today.
Typically use what's called keyword search.
Type a query on a search engine and try to find products of interest.

### I Want To Buy New Shoes, But...

* Too many options online...

So for example...want to buy a pair of shoes.
* Cool black pair of shoes
* Another black pair of shoes (might look the same, but totally different style)
* Crazy dress shoe (might look transparent, but actually a really cool shade of blue)
* Crazy two colored sneaker
* Another really interesting pair of sneakers
* Purple boots

There are a lot of shoes online.
It's really hard to find the ones that are interesting, stylish, and different.

Using keyword search doesn't really help. By typing in **dress shoes**, it just find a bunch of boring usual shoes.
But still want to find something different, and don't know what keywords to type or how to search for it.

### Visual Production Search

There are something even more complicated than buying shoes.
Want to buy a really interesting **dress**.
* Just use textual keyword search for **dress**, going to find a bunch of dresses
* But really don't know how to choose a dress
* Don't know how to describe it
* Even if specifying **floral dress**, still presented with number of options

Maybe there's something that will catch the eye.
* Instead of finding dresses based on keyword search
* Want to use image similarity to find similar dresses
* Find dresses that look similar based on image quality
* This is much easier to find something

## Features are Key to Machine Learning

Talked about an application of finding cool shoes or dresses just based on image features.
The technique used is called deep learning, and in particular, it's based on something called neural networks.

But before that, need to talk about data representation.
Discussed things like <abbr data-bs-toggle="tooltip" title="term frequency-inverse document frequency">tf-idf</abbr>, and bag of word models.
But how to really represent data when it comes to images?
These are called **features**, and is a key part of machine learning.

### Goal: Revisit Classifiers, But Using More Complex, Non-Linear Features

![Figure 1: Revisit Classifier]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-6/revisit-classifier.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

So typically when talking about machine learning, it is given some **input**.
For classification, talked about sentimental analysis.
* Given a sentence
* It goes through a classifier model
* Decided if that sentence has positive or negative sentiment

### Image Classification

In image classification, the goal is to go from an image, this is the **input**, the pixel of the images, to a classification.

## Neural Networks &rarr; Learning \*Very\* Non-Linear Features

So as discussed, features are the representation of the data that's used to feed into the classifier.
There are many representations...
* Text
    * Bag of words
    * <abbr data-bs-toggle="tooltip" title="term frequency-inverse document frequency">tf-idf</abbr>
* Image
    * There's a lot of other representations
    * Discuss a few more of those

Focus on neural networks, which provide the non-linear representation for the data.

### Linear Classifier

Going back to classification for a little review.

Discussed linear classifier
* Which create this line or linear decision boundary between say the positive class and the negative class
* The boundary is stated by the **Score**, **<math>w<sub>0</sub> + w<sub>1</sub>x<sub>1</sub> (first feature)</math>**, **<math>w<sub>2</sub>x<sub>2</sub> (second feature)</math>** so on...
    * On the positive side, the **Score** is greater than zero
    * On the negative side, the **Score** is less than zero
* Having a nice **Score** function, can separate the positives from the negatives

![Figure 2: Linear Classifier]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-6/linear-classifier.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-75"}

### Graph Representation of Classifier: Useful For Defining Neural Networks

In neural network, classifiers are represented using graphs.
* Have a node for each feature **<math>x<sub>1</sub></math>**, **<math>x<sub>2</sub></math>**, all the way to the **<math>d<sup>th</sup></math>** feature **<math>x<sub>d</sub></math>**
* A node for output **<math>y</math>**, what is trying to predict
* The first feature **<math>x<sub>1</sub></math>** is multiplied by the weight **<math>w<sub>1</sub></math>**, putting that weight on the edge
* The second feature **<math>x<sub>2</sub></math>** is multiplied by the second weight **<math>w<sub>2</sub></math>**, going to put it on that edge
* All the way to **<math>x<sub>d</sub></math>**, which is multiplied by weight **<math>w<sub>d</sub></math>**, to put in the last edge
* The last weight, **<math>w<sub>0</sub></math>** doesn't get multiplied by any feature, but it gets multiplied by **<math>1</math>**

Imagine multiplying the weights **<math>w<sub>0</sub></math>** through **<math>w<sub>d</sub></math>** with the features **<math>x<sub>1</sub></math>** through **<math>x<sub>d</sub></math>** and the coefficient **<math>1</math>**, to get the **Score**.
* When the **Score** is greater than **<math>0</math>**, it is declared the output to be **<math>1</math>**
* When the **Score** is less than **<math>0</math>**, it is declared the output to be **<math>0</math>**

This is an example of a small, one layer, neural network.

**Note:** If the perceptron takes an input of exactly **<math>0</math>**, what should it output?
An input of **<math>0</math>** (zero) is an edge case: there is not hard and fast rule as to whether the perceptron should output **<math>0</math>** or **<math>1</math>**.
Each implementation should pick one way and output the same value for all inputs of **<math>0</math>** (zero).

![Figure 3: Graph Representation of Classifier]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-6/graph-representation-classifier.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-75"}

### What Can a Linear Classifier Represent?

It was described the small linear classifiers is a neural network, a one layer neural network.
What can this one layer neural network represent?

Take the function **<math>x<sub>1</sub> OR x<sub>2</sub></math>**
* Can it be represented using a small neural network
* Define the function a little bit more formally
* Have variable **<math>x<sub>1</sub></math>**, **<math>x<sub>2</sub></math>**, and the output **<math>y</math>**
* There are some possibilities...
    * When **<math>x<sub>1</sub></math>** is **<math>0</math>**, and **<math>x<sub>2</sub></math>** is **<math>0</math>**, the output **<math>y</math>** would be **<math>0</math>**
    * When **<math>x<sub>1</sub></math>** is **<math>1</math>**, and **<math>x<sub>2</sub></math>** is **<math>0</math>**, the output **<math>y</math>** would be **<math>1</math>**
    * When **<math>x<sub>1</sub></math>** is **<math>0</math>**, and **<math>x<sub>2</sub></math>** is **<math>1</math>**, the output **<math>y</math>** would be **<math>1</math>**
    * Similarly, when they are both **<math>1</math>**, the output is **<math>1</math>**

{:.table}
| **<math>x<sub>1</sub></math>** | **<math>x<sub>2</sub></math>** | **y** | **Score** |
|--------------------------------|--------------------------------|-------|-----------|
| 0                              | 0                              | 0     | -0.5      |
| 1                              | 0                              | 1     | 0.5       |
| 0                              | 1                              | 1     | 0.5       |
| 1                              | 1                              | 1     | 1.5       |

* Define a **Score** function such that the value is greater than **<math>0</math>** for the last three (3) rows, but it is less than **<math>0</math>** for the first row
* How to do that (there are many ways of doing it actually)
    * Put a weight of **<math>1</math>**, one each of the edges **<math>x<sub>1</sub></math>** and **<math>x<sub>2</sub></math>**
    * Think about the **Score**
    * The **Score** of the first row is **<math>0</math>**, and the **Score** of the other rows are greater than **<math>0</math>**
* Might want to add a little bit of separation, might put a negative value on the first edge (**<math>-0.5</math>**)
    * When **<math>x<sub>1</sub></math>** is **<math>0</math>**, and **<math>x<sub>2</sub></math>** is **<math>0</math>**, then the **Score** becomes **<math>-0.5</math>**
    * When **<math>x<sub>1</sub></math>** is **<math>1</math>**, and **<math>x<sub>2</sub></math>** is **<math>0</math>**, then the **Score** becomes **<math>0.5</math>**
    * When **<math>x<sub>1</sub></math>** is **<math>0</math>**, and **<math>x<sub>2</sub></math>** is **<math>1</math>**, then the **Score** becomes **<math>0.5</math>**
    * When they are both **<math>1</math>**, the **Score** is **<math>1.5</math>**

With this simple weights on the edges, it represents the function of **<math>x<sub>1</sub> OR x<sub>2</sub></math>**

Now can represent the function **<math>x<sub>1</sub> AND x<sub>2</sub></math>**

![Figure 4: What Can a Linear Classifier Represent]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-6/linear-classifier-represent.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

## Reference
* [[PDF] Deep Learning]({{ "/res/misc/ml/coursera/machine-learning-foundations-a-case-study-approach/week-6/deeplearning-annotated.pdf" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:target="_blank"}
* [Object Recognition From Local Scale-Invariant Features](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf){:target="_blank"}
* [Spin-Images: A Representation For 3-D Surface Matching](https://www.ri.cmu.edu/pub_files/pub2/johnson_andrew_1997_3/johnson_andrew_1997_3.pdf){:target="_blank"}
* [Representing and Recognizing the Visual Appearance of Materialsusing Three-dimensional Textons](https://people.eecs.berkeley.edu/~malik/papers/LM-3dtexton.pdf){:target="_blank"}
* [A Sparse Texture Representation Using Local Affine Regions](https://hal.inria.fr/inria-00548530/document){:target="_blank"}
* [A Performance Evaluation of Local Descriptors](http://lear.inrialpes.fr/pubs/2005/MS05/mikolajczyk_pami05.pdf){:target="_blank"}
* [Histograms of Oriented Gradients for Human Detection](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf){:target="_blank"}
* [SURF: Speeded Up Robust Features](https://people.ee.ethz.ch/~surf/eccv06.pdf){:target="_blank"}
* [ImageNet Classification with Deep Convolutional Neural Networks](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf){:target="_blank"}