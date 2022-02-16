# Sentiment-Analysis

This repository contains the project for the Deep Natural Language Processing course.

Tha goal is making interpretable Sentimental Analysis on a datased of Amazon reviews. The implementation is Pytorch-based.

The available files are:

- Dataset: class Dataset, contains the preprocessing step, the word embedding phase and the classic "__get_item__" function that allowe to retrieve an element. In order to improve the efficiency, the dataset is created only the first time, then it is saved as a Pickle file.
- lstm_cell: our implementation of an LSTM cell with 4 gates
- attention_layer: our implementation of a self-attention layer
- net: the overall BiLSTM model, in which there are the forward, backward and merging steps and the final self-attention call
- training: is the main file, in which the execution is managed. To execute the experiment is necessary to execute this file
- params: contains all the parameters used during the training; specifically:
  - embedding_type: allows to choose among 2 types of word embedding: "concat" or "avg", which respectively takes the concatenation [average] of the 4 last hidden layers froma BERT model
  - loss[1][2]: allow to choose which loss is desidered for both the regression and the classification part
- ranking: it contains, for each of the 160 most common words in the dataset, the average attention value among the whole training step
  
 
