---
title: Machine Learning Foundations A Case Study Approach <small class="text-muted d-block">Week 4</small>
permalink: /docs/ml/coursera/machine-learning-foundations-a-case-study-approach/week-4/
---

## Clustering and Similarity: Retrieving Documents

Often there are a lot of observations, and want to infer some kind of structure underlying these observations.
The structure gonna talk about is groups of related observations or clusters.
In this case, it is about a task of retrieving documents of interest.

## Retrieving Documents With Interest

Start with describing the task in a little bit more detail.

### Document Retrieval

* Currently reading article

The article is about soccer, and like to retrieve another article that might be of interest in reading.
But the question is how to do this?
There are lots and lots of articles out there and can't expect to go and reach each of them.
Like to think of a way to automatically retrieve a document that might be of interest.

### Challenges

* How to measure similarity?
    * Need to have that in order to say that this article is similar to the one reading now
    * Might also be of interest
    * Large set of article that are very different and probably are not of interest
* How to search over articles?
    * Retrieve the next article to recommend

## Word Count Representation For Measuring Similarity

The first thing need to describe is how to represent documents being looked at.

### Word Count Document Representation

* Bag of words model
    * Ignore order of words
    * Count **# of instances** of each word in vocabulary

Perhaps the most popular model to represent a document is something called the bag of words model.
Simply ignore the order or words that are present in the document.

The reason it's called a bag of words model is
* Think of taking a bag
* Throwing all the words from that document into the bag
* Shaking it up
* The new document created with the words all jumbled up has exactly the same representation as the original document where the words were ordered

Instead of considering the structure or order of the words, simply gonna count the number of instances of every word in the document.

So in this document, there's just one sentence:

<blockquote class="blockquote text-center">
    <p class="mb-0">Vargas calls the sport futbol, Selena calls the sport soccer.</p>
</blockquote>

To count the number of instances of words in this very short document, gonna look at a vector.
This vector is defined over the vocabulary.
* So maybe one word in the vocabulary is the name, **Vargas** (1 instance)
* Another place in this vector is the index for the word **sport** (2 instances)
* Somewhere else have the word **futbol** (1 instance)
* Then say **Selena** is the last entry (1 instance)
* There is also the word **calls** (2 instances)
* Of course the word **the** (2 instances)
* Also the word **soccer** (1 instance)

For this document, every other entry would be zero (0 instance).
All these other entries represent all the other words that are out there in the vocabulary.
These can be like the word **cat**, and **dog**, and **tree**, and every other word.

So it's a very, very long and sparse vector that counts the number of words seen in this document.

![Figure 1: Word Count Document Representation]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-4/word-count-document-representation.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### Measuring Similarity

Talked about this representation of documents in terms of just these raw word counts, this bag of words model.

Want to talk about how to measure the similarity between different documents.
Gonna use that in order to find documents that are related to one another and so on.
Vargas is reading an article, so what's another article of interest?
* So imagine that this is the count factor for this article on soccer, with this famous Argentinian player, Messi
* Then there's another article showing in blue and the associated word counts, this article is about another famous soccer player, Pele

So when thinking about measuring similarity, simply look at an element wise product over this vector.
* So for every element in the vector, gonna multiply the two elements appearing in these two different count vector
* Add up over all the different elements in this vector

So that measures the similarity between these two articles on soccer.

![Figure 2: Measuring Similarity]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-4/measuring-similarity.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

But now compare to another article, which happens to be something about a conflict in Africa.
* Providing the example of word counts that appear in this article
* When go to measure the similarities between these articles, using the method described of element wise product, and then adding
* That the similarity here in this case is 0

![Figure 3: Measuring Similarity v2]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-4/measuring-similarity-v2.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### Issues With Word Counts - Document Length

Talk about an issue that arises when using these raw word counts to measure the similarity between documents.

So to do this, look at these green and blue articles.
Repeating the word count vectors, what is calculated before was the fact that the similarity between these two articles that are both about soccer is <math>13</math>.

But now look at what happens if simply double the length of the documents.
* So now every word that appeared in that original document appears twice in this twice as long document
* So the word count vector is simply two times the word count vector had before
* So when go to calculate the similarity here, now the similarity is calculated to be <math>52</math>.

Think about this.
* What it is saying is that two documents that are related to each other in the same way as before
* They're both talking about the same two sports
* But one just is replicated twice is a lot more similar

It could be said, that someone is a lot more interested in this longer document.
Then what happened when reading the shorter documents?
So this doesn't make a lot of sense when trying to do document retrieval.
It biases very strongly towards long documents.
So think about how to cope with this.

![Figure 4: Issue With Word Count]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-4/issue-with-word-count.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

### Solution - Normalize

So one solution is very straightforward where simply gonna normalize this vector.
* Take this word count vector and gonna compute the **norm** of the vector
    * Computing the **norm** of a vector, simply add the square of every entry in the vector
    * Then take the square root

So in this case, that happens to be the number <math>6</math>.
The resulting normalized word count vector is shown on the bottom.
What this does is it allow to place all articles considering, regardless of length, on equal footing.
Then use this normalized vector when doing retrieval.

![Figure 5: Solution Normalize]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-4/solution-normalize.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

## Prioritizing Important Words With <abbr title="Term Frequency – Inverse Document Frequency">TF–IDF</abbr>

So the normalized count vector just described helped address some of the issues with the original proposal of just using raw word count as the representation of the document.
But there's another issue, which is often, like to really emphasize the important words in a document.
So gonna discuss a little bit about what it means to be an important word and how to think about emphasizing them.

### Issue With Word Counts - Rare Words

**Common words** in document:
* the
* player
* field
* goal

Dominate **rare words** like:
* futbol
* Messi

Before talking about important words, first talk about rare words.
Imagine reading an article about soccer, and in this article there are lots and lots of common words.
What it means by a common word, it's a word that appears frequently in the corpus.
In the corpus, that's just terminology for all the documents out there when doing the task of retrieval.

What happens though is that these common words dominate the similarity metric talked about when compare against other documents.
These words appear in lots and lots of documents.
Whereas in contrast, there are some very rare words in this document being looked at.
Words like **futbol**, **Messi**, the specific player, that get completely swamped by all these common words.
They get swamped because they just appear very infrequently in the corpus.

So think about how to increase the importance of these rare words.
Those often are the ones that are really relevant in describing what's unique about the article, and what might indicate which other article out there is interested in reading.

### Document Frequency

* What characterizes a **rare word**?
    * Appears infrequently in the corpus
        * Want to up weight these rare words

* Emphasize words appearing in **few docs**
    * Equivalently, discount word **<math>w</math>** based on **# of docs containing <math>w</math> in corpus**
        * Think about taking each word
        * Discounting the weight on that word based on how many document that word appears in the entire corpus

### Important Words

* Do we want only rare words to dominate??
    * Clearly don't wanna emphasize the very common words
    * The words like **the**, **a**, **it**, **etc.**
    * Wanna include things that are very relevant for that document
    * So not just words like **Messi**, but also probably **soccer**, **field**, and **goal**

Things that describe the sport looking at, which might be different tha ever another article about sport.
An article about basketball that takes about **basketball**, **hoop**, **court**, and words like that.
So in this case, wanna emphasize what is call **important word**.

* What characterizes an **important word**?
    * Appears frequently in document (**common locally**)
    * Appears rarely in corpus (**rare globally**)

* Trade off between **local frequently** and **global rarity**

### TF-IDF Document Representation

* Term Frequency - Inverse Document Frequency (tf-idf)

One way to represent this trade off between something that's **common locally** but **rare globally** is something that's called <abbr title="Term Frequency – Inverse Document Frequency">TF–IDF</abbr>.

So first describe what **term frequency** is.
* Gonna look locally
* Looking just at the document that the person is currently reading
* Simply count the number of words
* So this is just the word count factor

* Term Frequency
    * Same as word counts
    * But then, gonna down weight this factor based on something called the Inverse Document Frequency

For this, gonna look at all documents in the corpus, and gonna compute the following.

* Inverse Document Frequency
    * **# docs** is the number of document in the corpus
    * **# docs using word** is the number of document that contain the word looking for

So think about why have this form?
* So first, think about a very common word (a word appearing in many documents)
    * So here, gonna be very, very strongly down weighting all the way to zero
    * The counts of any word that appears extremely frequently
    * Where that appears in all of the documents
* But in contrast, if have a rare word
    * A large number, assuming have a large number of documents
    * This is gonna be, or to say some largeish number, or a not a zero, or a small number
    * The reason to have this **<math>1</math>** here, is the fact that can't assume every word appears in any documents in the corpus
    * So there might be some word in the vocabulary that doesn't appear anywhere in the corpus
    * Also wanna avoid dividing by **0**

![Figure 6: TF-IDF Document Representation]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-4/tfidf-document-representation.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

So looking at an example
* Where there's the index for the word **the**.
    * Say that appears something like **<math>1000</math>** times in the document looking at
* Then there's the word **Messi**
    * Appears **<math>5</math>** times
* Now gonna look at computing the inverse document frequency for that word
* So the word **the**
    * Assume that the word **the** appears in every document in the corpus, except one
    * So when looking at this entry, gonna compute **<math>log</math>** of the number of documents in the corpus
    * Assume there are **<math>64</math>** documents in this corpus
    * Assume that the word **the** didn't appear in one of these **<math>64</math>** documents
    * So what his gives is the **<math>0</math>** that talked about before
    * So the word **the** gets down weighted completed by **<math>0</math>**
* In contrast, when looking at **Messi**
    * Assume to have some **<math>64</math>** total documents
    * Assume that the word **Messi** appears only in **<math>3</math>** of these documents
* So this is the term frequency, and the inverse document frequency of these two words

When go to compute the term frequency, inverse document frequency, for specific document which is gonna be the new representation of this document, simply multiple these two factors together.
* So theres some numbers here, where the word **the**, turns into a **<math>0</math>**
* Then there are some other number and then the word **Messi** is gonna be up weighted, so a weight of **<math>20</math>**.

The point wanna make here is the fact that these very common words like **the**, get down weighted and the rare and potentially very important words like **Messi** are getting up weighted.

![Figure 7: TF-IDF Document Representation v2]({{ "/res/img/ml/coursera/machine-learning-foundations-a-case-study-approach/week-4/tfidf-document-representation-v2.svg" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:class="img-fluid rounded mx-auto d-block"}

## Retrieving Similar Documents

So talked about how to represent documents and also talked about how to measure similarity between documents.
So now turn to the actual task of interest which is retrieving a document.
So someone is reading an article, assuming they like that article, and like to present them with another article to read.

### Nearest Neighbor Search

So in particular one of the most popular ways of doing this is something called nearest neighbor search.

* Query article
    * Article being read
* Corpus
    * All of the documents out there to search over to recommend some new article
* **Specify:** Distance metric
    * So this is gonna be the measure of similarity that talked about earlier
* **Output:** Set of most similar articles
    * Then what algorithm is gonna output is a collection of related articles

### 1-Nearest Neighbor

So one example of nearest neighbor search is something called **1-Nearest Neighbor**.

* **Input:** Query article
* **Output:** *Most* similar article
    * What gonna return is just simply the most related article, out of all those articles out there, to the query article

* Algorithm:
    * Search over each article in corpus
        * Compute **<math>s = similarity(query article, corpus)</math>**
        * Gonna compute the similarity using the methods described earlier between the **query article** and **this article examining in the corpus**
        * If **<math>s > Best_s, record best article = examine article</math>** and set **<math>Best_s = s</math>**
        * If that similarity is better than the best similarity that is found so far, then gonna keep this article as the best article that is found so far
* Return **best article**
    * Then at the end after iterated through every article in the corpus
    * Gonna look at what is the best article found
    * Gonna recommend this to the reader

### K-Nearest Neighbor

A very straightforward variant of this is something called **K-Nearest Neighbor** search.
Instead of just outputting the most relevant article found, gonna present the reader with a collection of **<math>K</math>** different article that are the top **<math>K</math>** most relevant articles.

* **Input:** Query article
* **Output:** *List of K* similar article

The algorithm is nearly identical except instead of keeping just the most related article, gonna keep a priority queue of the top **<math>K</math>** article found so far.

## Reference
* [[PDF] Classification]({{ "/res/misc/ml/coursera/machine-learning-foundations-a-case-study-approach/week-4/clustering-intro-annotated.pdf" | prepend : "/" | prepend : site.baseurl | prepend : site.url }}){:target="_blank"}
* [[Wikipedia] Corpus](https://en.wikipedia.org/wiki/Text_corpus){:target="_blank"}