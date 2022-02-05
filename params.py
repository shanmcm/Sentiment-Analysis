# -*- coding: utf-8 -*-
"""params

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12c7msOeu3U-_K5rDirc0dimwBYtiljBp
"""

import random
import torch
import nltk


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


path_ds = "/content/"

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('corpus')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('sentiwordnet')
nltk.download('averaged_perceptron_tagger')

NUM_CLASSES = 5  #1,2,3,4,5 stars

BATCH_SIZE = 32
NUM_EPOCHS = 8
#WEIGHT_DECAY = 0.00001
LR = 0.025
DROPOUT_RATE = 0.4
SEED = 42

#NUM_WORKERS = 2 if torch.cuda.is_available() else 0
# if 4 colab gives warning : UserWarning: This DataLoader will create 4 worker processes in total.
# Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader
# is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze,
# lower the worker number to avoid potential slowness/freeze if necessary.
#   cpuset_checked))
