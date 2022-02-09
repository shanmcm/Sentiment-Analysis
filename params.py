import nltk

path_ds = "./"
pickled_name = "amazonDataset.pkl"
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('corpus')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('sentiwordnet')
nltk.download('averaged_perceptron_tagger')

NUM_CLASSES = 5
BATCH_SIZE = 32
NUM_EPOCHS = 8
WEIGHT_DECAY = 0.00001
LR = 0.025
DROPOUT_RATE = 0.4
SEED = 42
MAX_SENT_LEN = 500
NUM_FEATURES = 3072