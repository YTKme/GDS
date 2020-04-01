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
That's the line where **<math>1.0 #awesome - 1.5 #awful = 0</math>**.
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

## Training + Evaluating Classifier

Here in classification, the errors are a little different because it is talking about which inputs are correct and which inputs are wrong.

### Training Classifier = Learning Weights

Talk a little bit about measuring error in classification.

Learn a classifier:
* Given a set of input data
    * These are sentences that have been marked to say positive or negative sentiment
* Split it into a training set and a testing set
* Feed the training set to the classifier to learn
    * The algorithm is actually going to learn the weights for words
* These weights are going to be used to score every element in the testing set
    * Evaluate how good in terms of classification

What does that evaluation looks like?

![Figure 14: Training Classifier Learning Weight]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/training-classifier-learning-weight.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### Classification Error

How to measure (classification) error?

Given a set of test examples in the form, **Sushi was great**, is a *positive* sentence.
Trying to figure out how many of these test get correct and how many get mistake.

Take the sentence, **Sushi was great**, and feed it through the learned classifier.
But don't want the learned classifier to actually see the true label.
Want to see if it gets the true label right, so going to hide that true label.
The sentence gets fed to the learned classifier while the true label is hidden.

Now given the sentence, going to predict **<math>&ycirc;</math>** as being positive.
Leaving this as a positive sentence, made a correct prediction.
The number of correct sentences goes up by one (1).

![Figure 15: Classification Error]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/classification-error.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

Now take another sentence, another test example, **Food was OK**, as a *negative* sentence.
So that's a bit of an ambiguous sentence, but it's been labeled as negative in the training set.

So now feed the sentence to the classifier, hide the label.
In this case, because the **Food was OK** can be revealed as positive.
Maybe it makes a prediction that this is a positive sentence, then it is a mistake, because the true label is negative.

Do this for every sentence in the [corpus](https://en.wikipedia.org/wiki/Text_corpus){:target="_blank"}.

![Figure 16: Classification Error v2]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/classification-error-v2.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### Classification Error & Accuracy

There are two common measures of quality in classification.

* Error measures fraction of mistakes
    * Best possible value is **<math>0.0</math>**

![Figure 17: Classification Error Formula]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/classification-error-formula.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-50"}

Now, it's common to instead of talk about error, to also talk about accuracy of the classifier.
Accuracy is exactly opposite of that.

* Often, measure **accuracy**
    * Fraction of correct predictions
    * Best possible value is **<math>1.0</math>**

In fact, there's a really natural relationship between error and accuracy, **<math>error = 1 - accuracy</math>**, and vise versa.

![Figure 18: Classification Accuracy Formula]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/classification-accuracy-formula.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-50"}

## What's Good Accuracy?

Discussed **error** and **accuracy** as ways to evaluate a classifier.
Now, it's very important to understand the **error** and **accuracy** actually getting from classifier.
Really think deeply about whether those are good **error** or good level of **accuracy**.

### What If You Ignore The Sentence, Just Guess?

One common mistake might make is to say how good is the classification at all?
When building a classifier, the first baseline comparison it should do is against random guessing.

* For binary classification:
    * Half the time, you'll get it right! (on average)
        * The <math>accuracy = 0.5</math>, or <math>50%</math>

* For <math>k</math> classes, the <math>accuracy = 1 / k</math>
    * It is <math>0.333</math> for 3 classes, <math>0.25</math> for 4 classes, ...

So at the very least, it should beat random guessing really well.
If not, then the approach is basically pointless.

### Is Classifier With 90% Accuracy Good? Depends...

Now, even beyond beating random guessing, truly think deeply about whether the classifier, even if it looks really good, is it really meaningfully good?

For example, suppose there is a spam predictor that gets <math>90%</math> accuracy.
Should go brag about it?
It that awesome?
Well, it really depends.
So the case of spam, not so good, because in 2010, data shows that <math>90%</math> of the emails ever sent were spam.
So if just guess that every email is spam, accuracy would be <math>90%</math>.

This is a problem where this is what's called **majority class prediction**, so it's just predicted classes is most common.
It can have amazing performance in cases where there's what's called **class imbalance**.
* One class has much more representation than the others
* Spam is much more representative than regular good emails

Be very cautious and really look at whether is **class imbalance** when try to figure out whether accuracy is good.
This approach also beats random guessing if the majority class is known.

![Figure 19: Classifier Accuracy]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/classifier-accuracy.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### So, Always Be Digging, Asking The Hard Question About Reported Accuracy

So should always be digging into problem, and understanding, really thinking about the predictions and whether that accuracy is really meaningfully good for the problem.

* Is there class imbalance?
* How does it compare to a simple, baseline approach?
    * Random guessing
    * Majority class
    * ...
* Most importantly: *what accuracy does my application need*?
    * What is good enough for my user's experience?
    * What is the impact of the mistake we make?

## False Positive, False Negative, Confusion Matrices

Talked about **error** and **accuracy** that a classifier might make.
But there are different kind of **error**.

### Mistake Type

This kind of **error** is called **type of mistake**.
It's important to look at the **type of mistake** a classifier might make.
One way to do that is through what's called a confusion matrix.

So talk about the relationship between the true label and whatever classifier predicts, the predicted label.
* If the true label is positive, and predicted a positive value for the sentence, call that a **true positive** because it is right
* Similarly, if the true label is negative and predicted negative, that's a **true negative**, because got that right

Now, there is two kinds of mistakes
* If the true label is positive, but predicted negative, call that a **false negative**
* If the true label is negative, when predicted positive, call that a **false positive**

The **false negative** and **false positive** can have different impacts on what can happen in practice with the classifier.

![Figure 20: Mistake Type]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/mistake-type.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### Cost Different Mistake Type Can Be Different (&High) For Some Applications

Look at two applications, and what the cost has of **false negative** versus **false positive**.
* If consider spam filtering, a **false negative** is an email that ws spam but went into the folder it thought it was not spam
* If looking at a **false positive** that's an email that was not spam that go labeled as spam (went to spam folder), never saw it, lost that email forever (high cost)

Now can also look at medical diagnosis
* A **false negative** is there's a disease but didn't get detected, so the classifier said it was negative
    * Don't have disease
    * Disease goes untreated, which can be really bad thing
* A **false positive** can also be a bad thing
    * Classify as having the disease when never had the disease
    * In this case, get treated potentially with a really bad drug or false side effect for disease that never had

In medical complications it really depends on the cost of the treatment and how many side effects it had versus how bad the disease can be.

![Figure 21: Mistake Type Cost]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/mistake-type-cost.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### Confusion Matrix - Binary Classification

Now, this relationship between the true label and the predicted label, **false negative**, **false positive**, is called the **confusion matrix**.

So for example, have a setting with 100 test examples, 60 positive, and 40 negative.
So there's a little bit of class imbalance but not too much.
* 60 positive
    * So of those 60 positives, if got 50 of them correct
* 40 negative
    * So of those 45 negatives, if got 35 of them correct

So out of the 100 examples, got 85 correct.
The **accuracy** is <math>85</math> correct over <math>100</math>, which is <math>0.85</math>, or <math>85%</math>.

Can also discuss the **false negative** and the **false positive**.
* The positive, got labeled as negative, that's a **false negative** (10)
* The negative, got labeled as positive, that's a **false positive** (5)

So in this example, got <math>85%</math> accuracy.
Got higher **false negative** rate, than **false positive** rate.
Now those words, **false negative**, **false positive**, apply only for minor classification for two classes.
But the ideal **confusion matrix** works well even when have more classes.

![Figure 22: Confusion Matrix Binary]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/confusion-matrix-binary.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### Confusion Matrix - Multiclass Classification

So say there are 100 test examples and this is for medical diagnosis, so there are 3 classes, **healthy**, **cold**, **flu**.

The 100 test subjects:
* Had 70 with that were healthy
    * Got 60 correct for healthy
    * Got 8 were confused with cold
    * Got 2 were confused with flu
* Had 20 that had cold
    * Got 12 correct for cold
    * Half got confused with healthy (4)
    * Half got diagnosis with something stronger, the flu (4)
* Had 10 that had the flu
    * Got 8 correct for flu
    * Made no mistake, nobody that came in for flu was thought healthy (0)
    * But 2 of those 10 were thought to have just a cold and not the flu

So, the total, the **accuracy**, here was <math>80</math> (<math>60 + 12 + 8</math>), divided by <math>100</math>.
So that is <math>0.8</math>, or <math>80%</math> **accuracy**.

But can talk about false predictions.
Can say it's more common to confuse healthy with having a cold than it is with having the flu.
The flu is a more complex disease so might have those mistake.

So this is an example of a confusion matrix.
Can really understand the types of mistakes made, and can interpret those.
Really important thing to do in classification.

![Figure 23: Confusion Matrix Multiclass]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/confusion-matrix-multiclass.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

## Learning Curves: How Much Data Needed?

In the regression, talked about the relationship between error and accuracy in the complexity of the model.
Talk a little bit about the relationship in terms of the amount of data to learn.

### How Much Data Does Model Need To Learn?

Explore the question of how much data is needed to learn.
This is a really difficult and complex question in machine learning.

* The more the merrier
    * But data quality is most important factor
    * (Having bad data, lots of bad data, is much worse than having much less, much fewer data point with really good, clean, high-quality data point)

* Theoretical techniques sometimes can bound how much data is needed
    * Typically too loose for practical application
    * (In practice, there's some empirical techniques to really try to understand how much error is made, and what that kind of error looks like)
    * But provide guidance

* In practice:
    * More complex models require more data
    * Empirical analysis can provide guidance

### Learning Curves

Now the important representation for this relationship between data and quality is what's called the learning curve.
* A learning curve relates the amount of data for training with the error.
    * Here talking about testing error
* Very little data for training, the test error is going to be high
* But a lot od data for training, the test error is going to be low

The curve is going to get better and better with more and more data.

![Figure 24: Learning Curve]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/learning-curve.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-50"}

### Limit? Yes, For Most Models...

Is there a limit?
Is this quality just going to get better and better forever by adding more data?
The test error is going to decrease by adding more data.

However, there is some gap here.
The question is whether that gap can go to zero?
The answer is...in general, no.
This gap is called the bias.

So intuitively, it says even with infinite data, the test error will not go to zero.

![Figure 25: Learning Curve Limit]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/learning-curve-limit.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-50"}

### More Complex Models Tend To Have Less Bias...

More complex models tend to have less bias.
Look at the sentiment analysis classifier, if just use single words like awesome, good, great, terrible, awful, it can do ok.
Maybe it do really well, maybe just does okay.

But even with infinite data, even with all the data in the world, it never going to get his sentence right, **the sushi was not good**.
This is because it is not looking at pairs of words, but just looking at the words **good** and not individually.

So more complex models, that deals with combinations of words, or simply called the [bigram](https://en.wikipedia.org/wiki/Bigram){:target="_blank"} model, where looking at pairs of secret words like **not good**.

Those models require more parameters, because there's more possibilities.
They can do better.
* They may have a parameter for **good**, say **<math>1.5</math>**
* But **not good**, say **<math>-2.1</math>**

It gets that sentence, **the sushi was not good**, correct.
So they have less bias.

They can represent sentences that couldn't be represented as words, so they're potentially more accurate.
But they need more data to learn, because there's more parameters.
There's not just a parameter for **good**, there's now a parameter for **not good**, and all possible combinations of words.

The more parameters the model has, in general, the more data it need to learn.

![Figure 26: Complex Model]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/complex-model.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-50"}

### Models With Less Bias Tend To Need More Data To Learn, But Do Better With Sufficient Data

Talked about the fact of an amount of training data on the test error.
* Building a classifier using single word
* Question is, how does that relate to a classifier, based on pairs of words

Now for a classifier based on bigrams, when there is less data, it's not going to do as well, because it has more parameters to fit.
* But when there is more data, it's going to do better
* It's going to be able to capture settings like, **the sushi was not good**

At some point, there's a crossover where it starts doing better than the classifier with single word.

But notice the background model still has some bias here.
Although the bias is small, it still has some bias.

![Figure 27: Model With Less Bias]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/model-with-less-bias.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block width-50"}

## Class Probabilities

Talked about classification in terms of just predicting
* Is this a positive sentence or a negative sentence
* Is email spam or not spam

But in general, want to go a little bit beyond that and ask about, what is the probability that his is email spam?

### How Confident For The Prediction?

* Thus far, we've outputted a prediction of
    * Positive (**+**)
    * Negative (**-**)

* But, how sure are you about the prediction?
    * "The sushi & everything else were awesome!" &larr; **<math>P(y = + | x) = 0.99</math>**
        * Definite positive (**+**)
    * "The sushi was good, the service was OK." &larr; **<math>P(y = + | x) = 0.55</math>**
        * Not sure, it's not as definite

So what a classifier will often do, is not just output positive or negative, but output how confident, how sure it is.
One way to do that is probabilities.
* So have to play the probability of being a positive or negative sentence, given the input sentence **<math>x</math>**
* So the output label, what's the probability output label, given the input sentence
* So instead of saying that's definite positive (**+**), say the probability that it's a positive (**+**) given **<math>x</math>** in is <math>0.99</math>
* The probability of being a positive (**+**), given **<math>x</math>**, is only <math>0.55</math>, because uncertain

Predicting probabilities or level of confidence is extremely important, and it allow to do many things.
For example, when the probability is known, it can make decisions like
* What is a good decision boundary that trades off false positive and false negative
* Balance between the two

![Figure 28: How Confident For The Prediction]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/how-confident-for-the-prediction.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

## Summary Classification

Seen classification in a wide variety of setting and how can it really use to predict a class like positive or negative sentiment from data.

In regression, talked about this block diagram that really describe how a machine learning algorithm iterates through its data.
Take this same block diagram and work through it and describe how it works out in the case of classification with sentiment analysis.

Classification for sentiment
1. The data is the text of the reviews, so for each review, the text of review is associated with a particular labeled sentiment
2. From that text of the review, feed it through a feature extraction phase which give **<math>x</math>**, the input to the algorithm
    * This **<math>x</math>** here is going to be the **word counts**
    * So word counts for every data point, for every review
3. Now the machine learning model is going to take that input data, so the **word counts**, as well as some several parameters
    * Which calling here **<math>&wcirc;</math> (<math>w (hat)</math>)**
    * Which are the **weights for each word**
4. Combining these two (**word counts** and **weights for each word**), going to output the prediction
    * If the score is greater than zero, it's going to be positive
    * If the score is less than zero, it's going to be negative
    * So this output here is the **predicted sentiment**
5. If just using the model, it would be done here
    * But really, in machine learning algorithm phase, it is going to evaluate that result, and then feed it back into the algorithm to improve the parameters
    * So going to take the **predicted sentiment**, **<math>&ycirc;</math> (<math>y (hat)</math>)**, and compare it with the **true label** for the **sentiment** (sentiment label for each data point)
    * That's going to fit in and the **quality measure** is going to be **classification accuracy**
6. The machine learning algorithm is going to take that **accuracy** and try to improve it
    * The way the improvement works, is by updating the parameter **<math>&wcirc;</math>**
    * That's what the cycle for machine learning algorithm classification would look like

![Figure 29: Detail Machine Learning Pipeline Sentiment]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/detail-machine-learning-pipeline-sentiment.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### What You Can Do Now...

Looked at how to do classification
* Looked at various examples of where it can be applied
* Talked about a few models for building classification, especially in the context of sentiment analysis
* Built a notebook of a classifier from data and analyzed it
* With this knowledge, ready to build an intelligent application that uses a classifier at its core

* Identify a classification problem and some common applications
* Describe decision boundaries and linear classifiers
* Train a classifier
* Measure its error
    * Some rules of thumb of good accuracy
* Interpret the types of error associated with classification
* Describe the tradeoffs between model bias and data set size
* Use class probability to express degree of confidence in prediction

## Appendix

## Reference
* [[PDF] Classification]({{ "/res/misc/ml/coursera/machine-learning-foundations-a-case-study-approach/week-3/classification-annotated.pdf" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:target="_blank"}
* [[Wikipedia] Statistical Classification](https://en.wikipedia.org/wiki/Statistical_classification){:target="_blank"}