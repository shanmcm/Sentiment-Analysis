import nltk
import os
import torch
path_ds = os.getcwd()
pickled_name = "amazonDataset_avg.pkl"
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('corpus')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('sentiwordnet')
nltk.download('averaged_perceptron_tagger')

NUM_CLASSES = 5
BATCH_SIZE = 128
NUM_EPOCHS = 8
LR = 1e-3
DROPOUT_RATE = 0.4
SEED = 42
MAX_SENT_LEN = 500
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBEDDING_TYPE = 'avg' # 'concat'
LOSS1 = 'ce' # 'bce'
LOSS2 = 'mse' # '1HEMSE'
NUM_FEATURES = 768 if EMBEDDING_TYPE == 'avg' else 3072
