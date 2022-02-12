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


def train_val_dataset(d, val_split=0.25):
    train_idx, val_idx = train_test_split(list(d.data.index), test_size=val_split)
    return Subset(d, train_idx), Subset(d, val_idx)


ds = dataset.AmazonDataset()  # fare prova con anche dataset non caricato
ds.load_dataset()
ds.filter()
ds.maximum_embedding_len = 500

train_ds, test_ds = train_val_dataset(ds)

# Define parameters
batch_size = params.BATCH_SIZE
hidden_dim = 128
embedding_size = params.NUM_FEATURES  # lunghezza embedding (selezionato uno a caso perchè tanto tutti hanno la stessa dimensione)
dropout_rate = params.DROPOUT_RATE
lr = params.LR
epochs = params.NUM_EPOCHS
best_loss = float('inf')

train_loader = DataLoader(train_ds, batch_size=32, collate_fn=dataset.collate_batch, shuffle=True, num_workers=0)
# Build the model
lstm_model = net.SentimentAnalysis(batch_size,
                                   hidden_dim,
                                   embedding_size,
                                   dropout_rate)  # modificato LSTM con SentimentAnalysis (nome rete)

# optimization algorithm
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)

to_train = True
ce = nn.CrossEntropyLoss()
mse = nn.MSELoss()
softmax = torch.nn.Softmax(dim=1)
# train and validate
if to_train:
    for epoch in range(epochs):
        # training
        epoch_loss = 0
        epoch_acc = 0
        for idxs, (batch, labels) in enumerate(train_loader):
            print(f"idxs = {idxs}")
            predictions = lstm_model(batch)
            labels = torch.Tensor([x-1 for x in labels.data.numpy()]) #mapping classes 1-5 in 0-4
            long_labels = labels.type(torch.LongTensor)
            loss1 = 0.5 * ce(predictions, long_labels)
            float_preds = torch.argmax(softmax(predictions), 1)
            float_preds = float_preds.type(torch.FloatTensor)
            
            loss2 = 0.5 * mse(float_preds, labels)
            loss = loss1.clone() + loss2.clone()
            loss = Variable(loss, requires_grad=True)
            accuracy = accuracy_score(float_preds.data, long_labels)
            
            # perform backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss + loss
            epoch_acc = epoch_acc + accuracy
        
        train_loss, train_acc = epoch_loss / len(train_loader), epoch_acc / len(train_loader)
        # save best model
        if train_loss < best_loss:
            best_valid_loss = train_loss
            torch.save(lstm_model.state_dict(), 'saved_weights_BiLSTM.pt')
        print(f'\tEpoch; {epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')

# load weights and make predictions
lstm_model.load_state_dict(torch.load("saved_weights_BiLSTM.pt"))
epoch_loss = 0
epoch_acc = 0

lstm_model.eval()

test_loader = DataLoader(test_ds, batch_size=32, shuffle=True, num_workers=0)  # , collate_fn=None, pin_memory=False)

with torch.no_grad():
    for idxs, (batch, labels) in enumerate(test_loader):
        predictions = lstm_model(batch)
        labels = torch.Tensor([x - 1 for x in labels.data.numpy()])  # mapping classes 1-5 in 0-4
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

test_loss, test_acc = epoch_loss / len(test_loader), epoch_acc / len(test_loader)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')
