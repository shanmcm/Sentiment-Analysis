import os

import nltk
import numpy as np
import pandas as pd
import pickle
import torch
from joblib.numpy_pickle_utils import xrange
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer
import torch
import params


def splitter(s, n):
    if not isinstance(s, list):
        s = s.split()
        for i in xrange(0, len(s), n):
            yield ' '.join(s[i:i + n])
    else:
        for i in range(0, len(s), n):
            yield s[i:i + n]


def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def get_sentiment(word: str) -> float:
    tag = nltk.pos_tag([word])[0][1]
    # print(tag)
    ss = wn.synsets(word)
    if ss:
        score = swn.senti_synset(ss.__getitem__(0).name())
        word_score = (score.pos_score() + score.neg_score()) / 2
        return word_score
    else:
        return 1.

'''
original
idx_to_remove = [226, 371, 412, 1884, 1887, 3613, 4906, 5010,
                 5056, 5118, 5738, 5973, 6576, 8068, 8086, 8500,
                 8636, 8713, 9120, 9151, 9182, 9303, 9355, 9571, 9790,
                 9824, 10026, 10039, 10043, 10184]

#aggiornato con andre
idx_to_remove = [225, 370, 411, 1883, 1886, 3612, 4905, 5009,
                 5055, 5117, 5737, 5972, 6575, 8067, 8085, 8499,
                 8635, 8712, 9119, 9150, 9181, 9302, 9354, 9570, 9789,
                 9823, 10025, 10038, 10042, 10183]
'''


class AmazonDataset(Dataset):
    def __init__(self):
        self.maximum_embedding_len = None
        self.tokenizer = None
        self.embedded_words_dict = None
        self.ranking = None
        self.tfidf = None
        self.labels = None
        self.data = None
        self.path = params.path_ds
        #print(self.path)
        assert os.path.exists(self.path), "Please insert a valid dataset path"
        self.loaded = os.path.exists(os.path.join(self.path, params.pickled_name))

    def clean_strings(self):
        to_remove = []
        for i, sentence in self.data.iteritems():
            if isinstance(sentence, str):
                filtered_chars = ''.join(x for x in sentence if x.isalpha() or x.isspace())
                filtered_chars = ' '.join(filtered_chars.split())
                filtered_chars = filtered_chars.lower()
                self.data.iloc[i] = filtered_chars
            else:
                to_remove.append(i)
        return to_remove

    def sw_removal(self):
        stop_words = set(stopwords.words('english'))
        to_remove = []
        for i, sentence in self.data.iteritems():
            if isinstance(sentence, str):
                word_tokens = word_tokenize(sentence)
                filtered_sentence = ' '.join(w for w in word_tokens if not w.lower() in stop_words)
                if len(filtered_sentence) > 0:
                    self.data.iloc[i] = filtered_sentence
                else:
                    to_remove.append(i)
        return to_remove

    def get_tfidf(self):
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(self.data.values)
        terms = vectorizer.get_feature_names()
        # sum tfidf frequency of each term through documents
        sums = tfidf.sum(axis=0)
        # connecting term to its sums frequency
        data = []
        for col, term in enumerate(terms):
            data.append((term, sums[0, col]))

        ranking = pd.DataFrame(data, columns=['term', 'rank'])
        ranking.sort_values(by=['rank'], inplace=True)
        ranking = ranking[:160]
        self.ranking = ranking

    def bert_embedding(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        model.eval()
        embedded_data = {}
        len_max = 0
        for i, sent in self.data.iteritems():
            tokenized_text = self.tokenizer.tokenize(sent)
            if len(tokenized_text) <= params.MAX_SENT_LEN:
                if len(tokenized_text) > len_max:
                    len_max = len(tokenized_text)
                input_id = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                attention_mask = [1] * len(tokenized_text)
                input_ids = list(splitter(input_id, 510))
                attention_masks = list(splitter(attention_mask, 510))
                tokenized_texts = list(splitter(tokenized_text, 510))
                for id_chunk, mask_chunk, test_chunk in zip(input_ids, attention_masks, tokenized_texts):
                    id_chunk = torch.cat([torch.tensor([101]), torch.tensor(id_chunk), torch.tensor([102])])
                    mask_chunk = torch.cat([torch.tensor([1]), torch.tensor(mask_chunk), torch.tensor([1])])
                    test_chunk = ["[CLS]"] + test_chunk + ["[SEP]"]
                    with torch.no_grad():
                        assert len(id_chunk) == len(mask_chunk), \
                            f"len(token_tensor) expected to be {len(id_chunk)}, but got {len(mask_chunk)}"
                        id_chunk = torch.unsqueeze(id_chunk, 0)
                        mask_chunk = torch.unsqueeze(mask_chunk, 0)
                        outputs = model(id_chunk, mask_chunk)
                        hidden_states = outputs[2]
                    token_embeddings = torch.stack(hidden_states, dim=0)
                    token_embeddings = torch.squeeze(token_embeddings, dim=1)
                    token_embeddings = token_embeddings.permute(1, 0, 2)
                    for j, token in enumerate(token_embeddings):
                        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
                        embedded_data[test_chunk[j]] = cat_vec
        self.embedded_words_dict = embedded_data
        self.maximum_embedding_len = len_max

    def load_dataset(self):
        if not self.loaded:
            amazon_ds = pd.read_csv(f"{self.path}/reviews.csv", sep =';')
            self.data = amazon_ds["reviewText"].copy(deep=True)
            self.labels = amazon_ds["overall"].copy(deep=True)
            tmp1 = self.clean_strings()
            tmp2 = self.sw_removal()
            to_remove = np.append(tmp1, tmp2)
            print(f"Dropped elements = {to_remove}")
            self.data.drop(to_remove, inplace=True)
            self.labels.drop(to_remove, inplace=True)
            self.bert_embedding()
            with open(f'{self.path}/{params.pickled_name}', 'wb') as outp:
                pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
        else:
            with open(f'{self.path}/{params.pickled_name}', 'rb') as inp:
                amazon_ds = pickle.load(inp)
                self.tokenizer = amazon_ds.tokenizer
                self.embedded_words_dict = amazon_ds.embedded_words_dict
                self.ranking = amazon_ds.ranking
                self.tfidf = amazon_ds.tfidf
                self.labels = amazon_ds.labels
                self.data = amazon_ds.data
                self.path = amazon_ds.path

    def filter(self, idx_to_remove):
        self.data.drop(idx_to_remove, inplace=True)
        self.labels.drop(idx_to_remove, inplace=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sent = self.data[idx]
        lab = self.labels[idx]
        embedding = [torch.Tensor([0]*params.NUM_FEATURES)]*500
        tokenized_text = self.tokenizer.tokenize(sent)
        for i, token in enumerate(tokenized_text):
            sent = get_sentiment(token)
            #tmp = torch.Tensor(self.embedded_words_dict[token] * sent)
            #embedding[i] = tmp
            #'''
            try:
                tmp = torch.Tensor(self.embedded_words_dict[token] * sent)
            except KeyError:
                print(idx, token, self.data[idx][:150])
            try:
                embedding[i] = tmp
            except IndexError:
                print(i, len(tokenized_text))
            #'''
        return embedding, lab


#  to check
class Collator(object):
    def __init__(self, test=False, percentile=100):
        self.test = test
        self.percentile = percentile

    def __call__(self, batch):
        global MAX_LEN
        if self.test:
            texts, lens = zip(*batch)
        else:
            texts, lens, target = zip(*batch)
        lens = np.array(lens)
        max_len = min(int(np.percentile(lens, self.percentile)), MAX_LEN)
        texts = torch.tensor(sequence.pad_sequences(texts, maxlen=max_len), dtype=torch.long)
        if self.test:
            return texts

        return texts, torch.tensor(target, dtype=torch.float32)
