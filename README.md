# Sentimental-Analysis
Sentimental Analysis of Movie Reviews Using Pytorch

------------------------

## Preparing Data
* One of the main concepts of TorchText is the Field. These define how your data should be processed. In our sentiment classification task the data consists of both the raw string of the review and the sentiment, either "pos" or "neg".

* The parameters of a Field specify how the data should be processed.

* We use the TEXT field to define how the review should be processed, and the LABEL field to process the sentiment.

* TEXT field has tokenize='spacy' as an argument. This defines that the "tokenization" (the act of splitting the string into discrete "tokens") should be done using the spaCy tokenizer. If no tokenize argument is passed, the default is simply splitting the string on spaces.

* LABEL is defined by a LabelField.

### Building Vocabulary
* The number of unique words in our training set is over 100,000 , which means that our one-hot vectors will have over 100,000 dimensions</br>
* To reduce the dimensions we only keep 25,000 Most Common Words

---------------

## Model
* We'll be using LSTMs, as they don't Suffer from vanishing gradient problem</br>
* They overcome this by Using Cell state </br>
* To overcome the problem of Overfitting we'll use - Dropout </br>

## Optimizer
* Adam Optimizer is used
