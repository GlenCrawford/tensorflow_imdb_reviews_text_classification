# IMDB movie review sentiment classification with a Tensorflow and Keras neural network

Tensorflow/Keras neural network to train on the [IMDB dataset of 50,000 movie reviews](http://ai.stanford.edu/%7Eamaas/data/sentiment/) and classify reviews as positive or negative using binary sentiment classification with 87% accuracy.

The dataset is a collection of reviews (25,000 for training and 25,000 for testing), each one being a movie review as an array of "words", each word represented as an integer which maps to a word in the word index. The word index is a dictionary of nearly a hundred thousand words.

Each review has an associated label, which is a binary integer representing whether the review is positive or negative.

## Requirements

Python version: 3.7.4
See dependencies.txt for packages and versions (and below to install).

## Architecture of the neural network

Each review input, after preprocessing, is an array of "words", represented as integers that map to a word in the word index, truncated/padded as necessary to 250 words.

__Embedding and GlobalAveragePooling1D layers:__ Groups similar words in the word index together, based on the context that they are used in.

__Hidden layer:__ 16 neurons.

__Output layer:__ 1 neuron with a value between 0 and 1 (squashed using the sigmoid function) denoting whether the review is positive or negative.

For more details of the model's architecture, refer to the comment annotations in the code.

## Setup

Clone the Git repo.

Install the dependencies:

```bash
pip install -r dependencies.txt
```

## Run

```bash
python main.py
```
