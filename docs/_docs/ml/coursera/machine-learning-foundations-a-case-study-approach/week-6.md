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

## Linear Classifier

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