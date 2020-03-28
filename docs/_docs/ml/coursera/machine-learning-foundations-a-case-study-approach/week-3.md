---
title: Machine Learning Foundations A Case Study Approach <small class="text-muted d-block">Week 3</small>
permalink: /docs/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/
---

## Classification: Analyzing Sentiment

Perhaps one of the most common areas of machine learning. It is also one of the most useful, and perhaps one of the most intuitive as well.

It can figure out:
* Whether email is spam or not
* Whether a document comes from a sports topic, about politics, or about entertainment

Starting with a use case, which is an exciting one, around restaurant reviews.

## Predicting Sentiment By Topic

### An Intelligent Restaurant Review System

There are many system for restaurant reviews out there, but going to talk about a pretty new and exciting one that can be built.

### It's Big Day & Want To Book Table For Nice Japanese Restaurant

In Seattle there are really awesome places for Japanese food.
There are many, many places for sushi, and there are many that have really good ratings (4 stars).
What are people saying about the food? The ambiance?

Need to think about different aspects:
* In terms of the food
* In terms of the ambiance
* In particular, in terms of the sushi

Want to get the best, freshest fish, and the most awesome, innovative sushi.

### Positive Reviews Not Positive About Everything

**Sample Review:**
* Watching the chefs create incredible edible art made the <u>experience</u> very unique.
    * Experience was positive (👍🏻)
* My wife tried their <u>ramen</u> and it was pretty forgettable.
    * Ramen was negative (👎🏻)
* All the <u>sushi</u> was delicious! Easily best <u>sushi</u> in Seattle.
    * Sushi was really positive (👍🏻👍🏻)

Do not care much about the ramen because not going to this place for ramen. But care about the good experience and amazing sushi.

### From Reviews To Topic Sentiments

When thinking about restaurant reviews, want to understand the aspect of the restaurant.
Their positive or negative, and really think about which one of those really affect the interest.

Look at all the restaurant reviews, feed them to this new, really exciting, new type of restaurant recommendation system that is going to tell:
* The experience was good, four stars
* The ramen was so-so, but who cares
* The sushi, the most important, was five stars

It is going to give some interesting feedback, such as this is easily the best sushi in Seattle.

![Figure 1: Review To Topic Sentiment]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/review-to-topic-sentiment.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### Intelligent Restaurant Review System

How to build such an intelligent restaurant review system?
* Start with all the reviews
* Break them up into sentences
    * Each review is composed of multiple sentences
    * Some sentences cover different aspects of the restaurant

![Figure 2: Intelligent Restaurant Review System]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/intelligent-restaurant-review-system.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### Core Building Block

For example, a sentence can say: "**Easily best sushi in Seattle**".

Feed the sentence through a sentiment classifier.
The sentiment classifier will return this sentence as positive, or as negative?

![Figure 3: Core Building Block]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/core-building-block.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### Intelligent Restaurant Review System

With the sentiment classifier, which can take a sentence and say it is positive or negative.

How to build this new kind of cool restaurant review experience?
* Take all the reviews
* Break them into sentences that will be discussed
* Select the sentences (not all of them, but a subset of those) that talk about the important aspect (in this case sushi)
* Feed those sentences through the sentiment classifier (each one of them)
* Average the results

In this case all of the sentences were positive, so this is a five star restaurant review in terms of the sushi.

Can also look at the sentences about sushi and look at which ones are most positive.

![Figure 4: Intelligent Restaurant Review System Extra]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/intelligent-restaurant-review-system-v2.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

## Classifier Applications

Build a while new kind of restaurant experience, or restaurant review experience, using classifiers.
Dig in and really understand a little bit more what a classifier is, and some other applications of classifier.

### Classifier

A classifier:
* Takes some input **x** (for example, a sentence from the review or other input)
* It pushes it through what's called a model
* To output some value **y** (trying to predict)
    * Here it is a class (for example, positive or negative)
    * Positive in the case of sentiment analysis corresponds with thumbs up reviews
    * Negative corresponds with thumbs down

But that's just one example of classification.

![Figure 5: Classifier]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/classifier.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-75"}

### Example Multiclass Classifier

Output **y** has more than 2 categories.

Look at text of a web page(s), figure out which web page(s) is interesting, align them to categories
* Education
* Finance
* Technology
* So on...

There's not just two categories. There can be three, four, or even thousands of categories to predict from.

![Figure 6: Example Multiclass Classifier]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/example-multiclass-classifier.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-50"}

### Spam Filtering

Now, another example of classification, which really has impacted lives, is in spam filtering.

Perhaps in the early 2000s, quality of samp filters were not very good.
* Were all hand tuned
* Spammers kept changing the words a little bit
* Adding numbers instead of letters
* Beating the spam filters

What really changed the world of spam filtering is machine learning, it is classifiers.

They took input **x** of the email and they fed it through a classifier that predicted whether it is spam or not.
It did that very well, by not just looking at the text of the email, but it looks at other characteristics.
* Who sent it
    * Somebody's whose a close friend
    * Somebody with a lot of communication
    * It is less likely to be spam
* The <abbr title="Internet Protocol">IP</abbr> address (a person sending from the usual computer)
* So on...

![Figure 7: Spam Filtering]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/spam-filtering.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-50"}

### Image Classification

In computer vision, there is a lot of classification.
Take an image, and figure out what is in that image.
* Input **x**: are the pixels of the image
* Feed it to a classifier
* Predict things like is this a dog
    * Labrador retriever
    * Golden retriever
    * Different kind of dog

### Personalized Medical Diagnosis

Can also use classification in medical diagnosis systems.
This is what doctors do.
* Take temperature
* Look at x-ray
* Look at some medical test
* Make prediction about what's ailing somebody (variable **y** being predicted)
    * Healthy
    * Have a cold
    * Have the flu
    * Have pneumonia
    * ...

Now these days, there is really interesting new things around personalized medicine.
The prediction doesn't just depend on the standard measurement, but can be really personalized.
* Can depend on particular <abbr title="DeoxyriboNucleic acid">[DNA](https://en.wikipedia.org/wiki/DNA)</abbr> sequencing
* Lifestyle

### Reading Mind

This idea of classification in machine learning has really gone much further, even to be able to read minds.

Take an image of the brain using technology called <abbr title="Functional Magnetic Resonance Imaging">[FMRI](https://en.wikipedia.org/wiki/Functional_magnetic_resonance_imaging)</abbr>, which is a brain scan, which predict:
* When reading a word of text
* Whether you are reading the word **hammer** or the word **house**

But in fact, there are many interesting things
* Look at a picture of a **hammer** or a **house**
* Train classifier on reading words **hammer** and **house**
* Still able to read mind and figure out what picture looking at

## Linear Classifier

One of the most common types of classifiers is also called linear classifier.

### Representing Classifiers

The question now is how to represent the classifier?
* Start with some sentences
* Feed it through a classifier
* Get a prediction whether it is a positive or a negative sentence

![Figure 8: Representing Classifier]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/representing-classifier.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-75"}

How does the classifier work? In the sentimental analysis, imagine a simple kind of threshold classifier.
* Take a sentence
* Identify all of the positive words and negative words
* Count how many positive word(s) and how many negative word(s)
* If the number of positive words is higher than the number of negative words
    * Have a positive sentence
* But if there is more negative words
    * Have a negative sentence

![Figure 9: Representing Classifier v2]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/representing-classifier-v2.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-50"}

If the input sentence is, "Sushi was great, food was awesome, but the service was terrible":
* Sushi was great (positive one)
* Food was awesome (positive two)
* But the service was terrible (negative one)

There is two positive and one negative, and in the end positive wins, so it is a positive prediction.

![Figure 10: Representing Classifier v3]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/representing-classifier-v3.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-50"}

### Problems With Threshold Classifier

Now threshold classifiers have some limitations.

* Addressed by learning a classifier
    * How do we get list of positive/negative words?
    * Words have different degrees of sentiment:
        * Great > good (tune and figure out what is better)
        * How do we weigh different words?
* Addressed by more elaborate features
    * Single words are not enough:
        * Good &rarr; Positive
        * Not good &rarr; Negative

### A (Linear) Classifier

So a linear classifier, instead of having a list of positive and negative words, actually takes all of the words and adds weight to them.
Words that really don't matter (restaurant, the, we, where, ...) to sentiment get weight of zero.

Will use training data to learn a weight for each word.

{:.table .table-responsive}
| Word                            | Weight |
|---------------------------------|--------|
| good                            | 1.0    |
| great                           | 1.5    |
| awesome                         | 2.7    |
| bad                             | -1.0   |
| terrible                        | -2.1   |
| awful                           | -3.3   |
| restaurant, the, we, where, ... | 0.0    |
| ...                             | ...    |

### Scoring Sentence

Given those weight, how to figure out if the sentence is positive or negative?
Using the idea of scoring.

Take this sentence, "sushi was great, the food was awesome, but the service was terrible".
To score the sentence, compute the score of the input sentence x.

{:.table .table-responsive}
| Word                            | Weight |
|---------------------------------|--------|
| good                            | 1.0    |
| great                           | 1.5    |
| awesome                         | 2.7    |
| bad                             | -1.0   |
| terrible                        | -2.1   |
| awful                           | -3.3   |
| restaurant, the, we, where, ... | 0.0    |
| ...                             | ...    |

**Input x:**
* Sushi was <u>great</u>,
* The food was <u>awesome</u>,
* But the service was <u>terrible</u>.

**Calculation:**
<math>Score(x) = 1.2 + 1.7 - 2.1 = 0.8</math>

```
if score(x) > 0: + (positive)
if score(x) < 0: - (negative)
```

So this is how linear classifier works, if the weight of each word is known.

Called a <u>linear classifier</u>, because output is <u>weighted sum of input</u>.

Just weight, what features appears and what words appear in the input.

So working for simple linear classifier to start out with.
* Given a sentence and given the weights for the sentence
* Compute the score, which is the weighted count of the words that appear in the sentence
* If the score is greater than zero, predict **<math>&ycirc;</math> (<math>y (hat)</math>)** to be positive
* If the score is less than zero, predict it to be negative

![Figure 11: Scoring Sentence]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/scoring-sentence.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-50"}

## Decision Boundaries

So classifiers are really trying to make decisions.
* Whether a sentence is positive or negative
* Whether a set of lab tests plus x-ray plus measurements lead to a certain disease like flu or cold

How classifiers, especially linear classifiers make decisions?

### Suppose Only Two Words Had Non-Zero Weight

To understand decision boundaries, suppose there is only two words with non-zero weight.

{:.table .table-responsive}
| Word    | Weight |
|---------|--------|
| awesome | 1.0    |
| awful   | -1.5   |

**Calculation**
<math>Score(x) = 1.0 #awesome - 1.5 #awful</math>

![Figure 12: Non-Zero Weight]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/non-zero-weight.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-50"}

Understand a little better how to score the sentences and what does that imply about the decision.
* Three (3) awesomes gives a positive prediction because the score is greater than zero (that is true for every point on the bottom right of the axis)
* While points on the top left all have score less than zero (for example the point three (3) awfuls, one (1) awesome) are label negative

In fact, what separates the negative predictions from the positive predictions is the line that defines the places unknown for positive and negative.
That's the line where <b><math>1.0 #awesome - 1.5 #awful = 0</math></b>.
That's the line when the prediction is uncertain, and it is called the decision boundary.
Everything on one side predicted as positive, everything on the other is predicted as negative.

Notice that the decision boundary is a line. That's why it's called a linear classifier (linear decision boundary).

![Figure 13: Non-Zero Weight v2]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/non-zero-weight-v2.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-50"}

### Decision Boundary Separates Positive & Negative Predictions

* For linear classifiers
    * When 2 weights are non-zero
        * line
    * When 3 weights are non-zero
        * plain (somehow inclined in space)
    * When many weights are non-zero
        * hyperplane (really high dimensional separators)
* For more general classifiers
    * more complicated shapes (squiggly separations)

## Appendix

## Reference
* [[PDF] Classification]({{ "/res/misc/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/classification-annotated.pdf" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:target="_blank"}
* [[Wikipedia] Statistical Classification](https://en.wikipedia.org/wiki/Statistical_classification){:target="_blank"}