import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.autograd import Variable
import dataset
import net
import params
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_weights_for_balanced_classes(data, weight_per_class):
    weight = [0] * len(data)
    for idx, val in enumerate(data):
        weight[idx] = weight_per_class[val - 1]
    return weight


def train_val_dataset(d, val_split=0.25):
    train_idx, val_idx = train_test_split(list(d.data.index), test_size=val_split)
    len_train = [(i, len(d.data[i].split(" "))) for i in train_idx]
    len_train = sorted(len_train, key=lambda el: el[1], reverse=True)
    train_idx = [el[0] for el in len_train]
    len_val = [(i, len(d.data[i].split(" "))) for i in val_idx]
    len_val = sorted(len_val, key=lambda el: el[1], reverse=True)
    val_idx = [el[0] for el in len_val]
    return Subset(d, train_idx), Subset(d, val_idx)


print(f"Preparing...")
ds = dataset.AmazonDataset()
ds.load_dataset()
ds.filter()
print("Filtered dataset")
len_ds = ds.__len__()
weights = class_weight.compute_class_weight('balanced', classes=np.unique(ds.labels), y=ds.labels)
weights = torch.Tensor(weights)
train_ds, test_ds = train_val_dataset(ds)
ds.get_tfidf()
# Define parameters
batch_size = params.BATCH_SIZE
hidden_dim = 256
embedding_size = params.NUM_FEATURES
dropout_rate = params.DROPOUT_RATE
lr = params.LR
epochs = params.NUM_EPOCHS
best_loss = float('inf')
train_labels = torch.Tensor(ds.labels[train_ds.indices].to_numpy())
train_data = ds.data[train_ds.indices]

loader_weights = make_weights_for_balanced_classes(ds.labels[train_ds.indices].to_numpy(), weights)
loader_weights = torch.DoubleTensor(loader_weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(loader_weights, len(loader_weights))

train_loader = DataLoader(train_ds, num_workers=4, collate_fn=dataset.collate_batch,
                          batch_size=batch_size, drop_last=True)
# Build the model
lstm_model = net.SentimentAnalysis(batch_size, hidden_dim, embedding_size, dropout_rate)
lstm_model.to(device)

# optimization algorithm
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)
scheduler = ExponentialLR(optimizer, gamma=0.9)
ce = nn.CrossEntropyLoss(weight=weights)
bce = nn.BCEWithLogitsLoss(pos_weight=weights)
mse = nn.MSELoss()
softmax = torch.nn.Softmax(dim=1)

lstm_model = lstm_model.train()

# train and validate
print("Start running")
with torch.enable_grad():
    for epoch in range(epochs):
        # training
        epoch_loss = 0; epoch_acc = 0; epoch_f1 = 0; epoch_precision = 0; epoch_recall = 0
        for idxs, (batch, labels, sentences_list) in enumerate(train_loader):
            important_words = [[(i, cnt, x) for cnt, x in enumerate(sentences_list[i]) if
                                x.replace("#", "") in ds.ranking["term"].to_numpy()] for i in
                               range(len(sentences_list))]
            batch = batch.to(device)
            predictions, attention = lstm_model(batch)
            attention = attention.detach()
            scores_by_w = [[(word, attention[j][i]) for i, j, word in s] for s in important_words]
            ds.update_ranking(scores_by_w)
            predictions = predictions.to('cpu')
            long_labels = labels.type(torch.LongTensor)
            assert params.LOSS1 in ['ce', 'bce'], "Insert valid LOSS1"
            if params.LOSS1 == 'ce':
                loss1 = ce(predictions, long_labels) * 0.5
            elif params.LOSS1 == 'bce':
                loss1 = bce(predictions, nn.functional.one_hot(long_labels).float())  * 0.5
            float_preds = torch.argmax(softmax(predictions), 1)
            float_preds = float_preds.type(torch.FloatTensor)
            float_preds.requires_grad = True
            labels = labels.type(torch.FloatTensor)
            assert params.LOSS2 in ['mse', '1HEMSE'], "Insert valid LOSS2"
            if params.LOSS2 == 'mse':
                loss2 = torch.sqrt(mse(float_preds, labels)) * 0.5
            elif params.LOSS2 == '1HEMSE':
                ohe_labels = nn.functional.one_hot(long_labels).float()
                sm_preds = softmax(predictions)
                loss2 = torch.sqrt(mse(sm_preds, ohe_labels)) * 0.5
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = loss.item()
            del loss1
            del loss2
            accuracy = accuracy_score(float_preds.data, long_labels)
            f1 = f1_score(float_preds.data, long_labels, average='weighted', labels=np.unique(long_labels))
            precision = precision_score(float_preds.data, long_labels, average='weighted',
                                        labels=np.unique(long_labels))
            recall = recall_score(float_preds.data, long_labels, average='weighted',
                                  labels=np.unique(long_labels))
            epoch_loss = epoch_loss + loss
            epoch_acc = epoch_acc + accuracy
            epoch_f1 = epoch_f1 + f1
            epoch_precision = epoch_precision + precision
            epoch_recall = epoch_recall + recall
            if not idxs % 10:
                print(f"Loss: {loss}, Accuracy: {accuracy}, F1: {f1}")
                cm = confusion_matrix(float_preds.data, long_labels, labels=[0,1,2,3,4])
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1,2,3,4])
                disp.plot()
                plt.show()
        scheduler.step()
        train_loss, train_acc = epoch_loss / len(train_loader), epoch_acc / len(train_loader)
        train_f1, train_precision = epoch_f1 / len(train_loader), epoch_precision / len(train_loader)
        train_recall = epoch_recall / len(train_loader)
        # save best model
        if train_loss < best_loss:
            best_valid_loss = train_loss
            torch.save(lstm_model.state_dict(), 'saved_weights_BiLSTM.pt')
        print(f'Epoch; {epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f"\tF1 = {train_f1}, precision = {train_precision}, recall = {train_recall}")


# load weights and make predictions
lstm_model.load_state_dict(torch.load("saved_weights_BiLSTM.pt"))
epoch_loss = 0
epoch_acc = 0

lstm_model.eval()

test_loader = DataLoader(test_ds, num_workers=4, collate_fn=dataset.collate_batch,
                         batch_size=batch_size, drop_last=True)

with torch.no_grad():
    for idxs, (batch, labels, _) in enumerate(test_loader):
        predictions, _ = lstm_model(batch)
        predictions = predictions.to("cpu")
        long_labels = labels.type(torch.LongTensor)
        loss1 = 0.5 * ce(predictions, long_labels)
        float_preds = torch.argmax(softmax(predictions), 1)
        float_preds = float_preds.type(torch.FloatTensor)
        loss2 = 0.5 * mse(float_preds, labels)
        loss = loss1.clone() + loss2.clone()
        loss = Variable(loss, requires_grad=True)
        accuracy = accuracy_score(float_preds.data, long_labels)
        epoch_loss += loss
        epoch_acc += accuracy
        epoch_f1 = epoch_f1 + f1
        epoch_precision = epoch_precision + precision
        epoch_recall = epoch_recall + recall

    test_loss, test_acc = epoch_loss / len(test_loader), epoch_acc / len(test_loader)
    test_f1, test_precision = epoch_f1 / len(test_loader), epoch_precision / len(test_loader)
    test_recall = epoch_recall / len(test_loader)

    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
    print(f"\tF1 = {test_f1}, precision = {test_precision}, recall = {test_recall}")

