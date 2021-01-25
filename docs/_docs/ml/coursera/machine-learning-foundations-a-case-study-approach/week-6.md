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
* Similarly can put weights **<math>1</math>** and **<math>1</math>** on the edges **<math>x<sub>1</sub></math>** and **<math>x<sub>2</sub></math>**
* But in this case, only want to turn it on when both **<math>x<sub>1</sub></math>** and **<math>x<sub>2</sub></math>** have value **<math>1</math>**
* So instead of putting **<math>-0.5</math>** on the top edge, put **<math>-1.5</math>**
* If fill out the table just like with the first example, notice that it is represent the function **<math>x<sub>1</sub></math>** and **<math>x<sub>2</sub></math>** using a simple neural network

![Figure 4: What Can a Linear Classifier Represent]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-6/linear-classifier-represent.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-75"}

### What Can't a Simple Linear Classifier Represent?

A one layer neural network is basically the same as the standard linear classifiers.
So what can linear classifier not represent?
It can represent **<math>x<sub>1</sub> OR x<sub>2</sub></math>**.
It can represent **<math>x<sub>1</sub> AND x<sub>2</sub></math>**.
But what's a function, a very simple function it cannot represent?

Well, here is an example...
* There is no line that separate the **pluses** and **minuses**
* This function is called the **XOR**
* It is a counter example to (almost) everything
* Whenever a counter example is needed, first thing to try is **XOR**
* For this case, the linear features described are not enough and need some kind of non-linear features
* This is when the neural networks come to play for real

![Figure 5: What Can't a Linear Classifier Represent]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-6/simple-linear-classifier.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-75"}

### Solving The XOR Problem: Adding a Layer

So **XOR** has value **<math>1</math>** either...
* Value of **<math>x<sub>1</sub></math>** is true AND **<math>x<sub>2</sub></math>** is false, so NOT **<math>x<sub>2</sub></math>**
* Value of **<math>x<sub>1</sub></math>** is false AND **<math>x<sub>2</sub></math>** is true, so NOT **<math>x<sub>1</sub></math>**

How can this be represented with a neural network?
* Call the first term **<math>z<sub>1</sub></math>**
* Call the second term **<math>z<sub>2</sub></math>**

Going to build a neural network to represent not directly the inputs of **<math>x<sub>1</sub></math>** and **<math>x<sub>2</sub></math>** to predict **<math>y</math>**.
But they predict intermediate values **<math>z<sub>1</sub></math>** and **<math>z<sub>2</sub></math>**, and then those are going to predict **<math>y</math>**.

Take **<math>z<sub>1</sub></math>**
* How to represent only a neural network that can predict **<math>z<sub>1</sub></math>**
* Since it have to negate, it is NOT **<math>x<sub>2</sub></math>**
    * Put a **<math>-1</math>** on that edge **<math>x<sub>2</sub></math>**
    * Put a **<math>+1</math>** on **<math>x<sub>1</sub></math>**
    * Put a **<math>-0.5</math>** on **<math>1</math>** edge
* Now have the representation for **<math>z<sub>1</sub></math>**

Similarly for **<math>z<sub>2</sub></math>**
* Put a **<math>-1</math>** on edge **<math>x<sub>1</sub></math>**
* Put a **<math>+1</math>** on edge **<math>x<sub>2</sub></math>**
* Put a **<math>-0.5</math>** on **<math>1</math>**, the constant edge
* Now it represents  **<math>z<sub>2</sub></math>**

The last step...
* If **<math>z<sub>1</sub></math>** and **<math>z<sub>2</sub></math>** exist, just have to **OR** them
* Already know how to **OR** the boolean variables
* It is just **<math>1</math>** and **<math>1</math>** on the **<math>z<sub>1</sub></math>** and **<math>z<sub>2</sub></math>** edge, and **<math>-0.5</math>** on the constant edge

Now it has built out the first deep neural network, not super deep, but it has two (2) layers.

![Figure 6: Solving The XOR Problem]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-6/solving-the-xor-problem.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-75"}

### A Neural Network

* Layers and layers and layers of linear models and non-linear transformation
* Around for about 50 years
    * Fell in "disfavor" in 90s
* In last few years, big resurgence
    * Impressive accuracy on several benchmark problems
    * Powered by huge datasets, GPUs, and modeling / learning algorithm improvements

In general...
* Neural networks is about this layers and layers of transformations of the data
    * Use these transformations to create these non-linear features (more example in computer vision)
* Neural network has been around for about 50 years (about as long as machine learning's been around)
    * However, they fell in disfavor around the 90's
    * Because folks are having a hard time getting good accuracy in neural networks
* But everything changed about 10 years ago (because of two things that came about)
    * First, it was a lot more data
        * Because neural networks have so many, many more layers
        * So many layers that it need a lot of data to be able to train all those layers
        * They have a lot of parameters
        * Recently have came about lots and lots and lots of data from a variety of sources, especially the web
    * Second, it was computing resources
        * Because have to deal with bigger neural networks, and more data
        * Need faster computers and GPUs which were originally design for accelerating graphics for computer games
        * Turns out to be exactly the right tool to build and use neural network with lots of data
        * So because of GPUs and because of these deep neural networks, everything changed

![Figure 7: A Neural Network]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-6/neural-network.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-50"}

## Application of Deep Learning To Computer Vision

The first place where neural networks made a tremendous difference, is in an area called computer vision. (Analyzing images and videos)
In order to understand how deep learning, or these big neural networks, can be applied to computer vision, is good to understand what **image features** are.

### Image Features

* Features = local detectors
    * Combined to make prediction
    * (in reality, features are more low-level)

In computer vision, **image features** are kind of like local detectors that get combined to make a prediction.
Take a particular image, want to predict whether this is a face image or not a face image.

Run the neural detector, if all these fire, using a little neural network, then can say this is a face.
* Nose detector
* Eye detector
* Another eye detector
* Mouth detector

This is a simple example of how it can build a classifier for images, but in reality they don't explicitly have a nose detector or eye detector.

![Figure 8: Image Feature]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-6/image-feature.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-50"}

### Typical Local Detectors Look For Locally "Interest Points" in Image

* **Image features:** collections of locally interesting points
    * Combined to build classifiers

What happens is these called **image features**, or interest points (there are various names for it), they really tried to final local image segments, patches, that are really distinctive.
So maybe they'll find the corner around the eye, maybe the corner around the nose.
So if there are lots of these corner detectors (a face is comprised of corners), corner detector firings at places around the eyes, the mouth, and nose.
If enough of these fire in a particular pattern, a face is discovered.
This is how computer vision typically works, how classification works.
Of course, there're more general models and more complex ones, but this is the basic idea.

### Many Hand Created Features Exist For Finding Interest Points...

* Spin Images [Johnson and Herbert '99]
* Textons [Malik et al. '99]
* RIFT [Lazebnik '04]
* GLOH [Mikolajczyk and Schmid '05]
* HoG [Dalal and Triggs '05]
* <abbr data-bs-toggle="tooltip" title="Scale-Invariant Feature Transform">SIFT</abbr> [Lowe '99]

For years, these types of detectors of local features are built by hand.
A very popular one was called SIFT features, and this retransformed their computer vision because they were really quite applicable and quite cool.
There are many other that can improve accuracy.
Other kinds of features that can be used.

### Standard Image Classification Approach

Talked about this hand created **image features** like <abbr data-bs-toggle="tooltip" title="Scale-Invariant Feature Transform">SIFT</abbr> feature.
Now talk about how they can be typically used for classification.
* Run the sifted textures over the image and they fire in various places
* (For example the corners of the eyes and the mouth)
* Create a vector that describe the image based on the firings, the locations where those SIFT features fired
    * Might have some firings in some locations, no firings in other locations
    * Can be viewed similarly to the words in a document
    * Does the word **messy** appear
    * Does the word **football** appear
    * Similarly, does a corner appear in a particular place in the image
* Once have the description of the image, can feed it to a classifier (for example, a simple linear classifier)
    * Logistic regression
    * Support Vector Machine
    * ...and more
* From there, get a detection as to whether this image is a face or not

Now that sounds pretty exciting and it had a real significant impact in their computer vision.

![Figure 9: Standard Image Classification Approach]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-6/standard-image-classification-approach.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### Many Hand Created Feature Exist For Finding Interest Points...

...but very painful to design

The challenge though, is that creating these hand built **image features** was a really complicated process and require several PhD thesis to be done well.

### Deep Learning: Implicitly Learns Features

Neural networks are going to discover and learn those features automatically.

Example, supposed given an input image, and they run it through a three layer neural network before making a prediction
* Typically what happens, is that it learn local feature detectors (they're like <abbr data-bs-toggle="tooltip" title="Scale-Invariant Feature Transform">SIFT</abbr>)
* But at different levels and different layers
* These detectors that is learned, they detect different things, different properties of the the image at different levels
* The first layer
    * Might learn detectors that look kind of like little patches
    * Which really react to things like diagonal edges
    * All about capturing diagonal edges
        * The first one is about capturing diagonal edges
        * The center one is about capturing diagonal edges in the other direction
        * The last one is about capturing transitions and color from dark to green
* The next (second) layer
    * Combining the diagonal edge detection into some kind of more complex detector
    * For example, discovered this wiggly line and pattern detectors in the layer
    * Also discovered this kind of detectors that react to and detect corners in the image
* The final (third) layer
    * Come up with detectors that are even more complicated
    * For a variety of images, might end up with things that react to torsos and faces
    * Maybe with a bigger data set, it can even fire up with images of corals

So neural networks capture different types of **image features** at different layers, and then they get learned automatically.

![Figure 10: Deep Learning Implicitly Learns Features]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-6/deep-learning-implicitly-learn-feature.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

## Reference
* [[PDF] Deep Learning]({{ "/res/misc/ml/coursera/machine-learning-foundations-a-case-study-approach/week-6/deeplearning-annotated.pdf" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:target="_blank"}
* [Object Recognition From Local Scale-Invariant Features](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf){:target="_blank"}
* [Spin-Images: A Representation For 3-D Surface Matching](https://www.ri.cmu.edu/pub_files/pub2/johnson_andrew_1997_3/johnson_andrew_1997_3.pdf){:target="_blank"}
* [Representing and Recognizing the Visual Appearance of Material Three-dimensional Textons](https://people.eecs.berkeley.edu/~malik/papers/LM-3dtexton.pdf){:target="_blank"}
* [A Sparse Texture Representation Using Local Affine Regions](https://hal.inria.fr/inria-00548530/document){:target="_blank"}
* [A Performance Evaluation of Local Descriptors](http://lear.inrialpes.fr/pubs/2005/MS05/mikolajczyk_pami05.pdf){:target="_blank"}
* [Histograms of Oriented Gradients for Human Detection](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf){:target="_blank"}
* [SURF: Speeded Up Robust Features](https://people.ee.ethz.ch/~surf/eccv06.pdf){:target="_blank"}
* [ImageNet Classification with Deep Convolutional Neural Networks](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf){:target="_blank"}
* [Computer Vision](https://en.wikipedia.org/wiki/Computer_vision){:target="_blank}
* [Scale-Invariant Feature Transform](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform){:target="_blank}