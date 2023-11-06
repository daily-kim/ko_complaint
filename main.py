
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from sklearn.metrics import  auc, mean_squared_error
from sklearn.preprocessing import label_binarize

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

import random
seed = 2023
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(seed)

import warnings
warnings.filterwarnings('ignore')

from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import logging

import matplotlib.pyplot as plt
import os
import sys
import seaborn as sns
os.chdir('/workspace/')
sns.set(font_scale = 1.5,font ='NanumGothic' )
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss

def print_scores(y_true, y_pred, train_set):
    if train_set:
        print("scores for training set:\n")
    else:
        print("scores for validation/test set:\n")
    
    # print scores for each class
    accuracy = sum(y_true == y_pred)/len(y_true)
    print(f"accuracy: {accuracy}")

    f1_micro = f1_score(y_true, y_pred, average='micro')
    print(f"f1 micro: {f1_micro}")
    f1_macro = f1_score(y_true, y_pred, average='macro')
    print(f"f1 macro: {f1_macro}")
    hamming = hamming_loss(y_true, y_pred)
    print(f"hamming loss: {hamming}")

def sentences_to_sequences(sentences, MAX_LENGTH):
  input_ids = np.zeros((len(sentences), MAX_LENGTH)).astype(np.int)
  attention_masks = np.zeros((len(sentences), MAX_LENGTH)).astype(np.int)
  index = 0
  for sentence in sentences:
    # initialize sequence
    sequence = np.zeros(MAX_LENGTH).astype(np.int)             
    # initialize mask
    mask = np.zeros(MAX_LENGTH).astype(np.int)                 
    # add special BERT tokens
    sentence = "[CLS] " + sentence + " [SEP]"                   
    # tokenize each sentence to its words
    tokenized_sentence = tokenizer.tokenize(sentence)          
    # Map the token strings to their vocabulary indices.
    sequence_indices = tokenizer.convert_tokens_to_ids(tokenized_sentence)  
    length = min(MAX_LENGTH, len(sequence_indices))
    # sequence gets padded with [PAD]/zeros or truncated 
    sequence[:length] = np.array(sequence_indices[:length])
    # add [SEP] token if it got truncated    
    if sequence[length-1] != tokenizer.convert_tokens_to_ids('[SEP]'):
      sequence[length-1] = tokenizer.convert_tokens_to_ids('[SEP]')
    # set attention mask to 1 for non padded tokens 
    mask[:length] = np.array([1 for i in range(length)])       
    input_ids[index] = sequence
    attention_masks[index] = mask
    index += 1
  # return input_ids, segment_ids, attention masks as tensors
  return torch.from_numpy(input_ids), torch.from_numpy(attention_masks)

def execute_training(model, train_dataloader, val_dataloader, loss_func, optimizer, batch_size, num_epochs, clip_grad=True, stop_early=True):
    """ function that trains the model in bathes of batch_size, for num_epochs, using given loss function, optimizer
        returns the train_losses, val_losses, train_scores, val_scores and the final trained model
        clip_grad : Flag that indicates if gradient clipping will be used (default is True)
        stop_early : Flag that indicates if early stopping regularization will be used (default is True) """
    
    #Initialize lists we care about
    train_losses = []
    val_losses = []
    train_scores = []
    val_scores = []

    # for early stopping
    min_val_loss = np.Inf   # keeps track of minimum validation loss thus far
    epochs_no_improve = 0   # keeps track of number of consecutive epochs where validation loss did not improve over minimum validation loss
    epoch_tolerance = 3     # if validation loss does not improve over 3 consecutive epochs, we stop training (early stopping)
    epochs_count = 0  

    for epoch in range(num_epochs):
        train_batch_losses = []
        val_batch_losses = []
        val_batch_scores = []
        
        model.train()
        progress_loop = tqdm(train_dataloader,disable= True)

        for train_batch in progress_loop:
            
            x_batch = train_batch[0].to(device)
            masks = train_batch[1].to(device)
            y_batch = train_batch[2].to(device)
            optimizer.zero_grad()
            out = model(x_batch, token_type_ids=None, attention_mask=masks)
            y_pred = out.logits
            # cross entropy loss
            loss = loss_func(y_pred, y_batch)
            train_batch_losses.append(loss.item())
            loss.backward()

            #Update model's weights based on the gradients calculated during backprop
            optimizer.step()
            progress_loop.set_description(f'Epoch {epoch}')
            progress_loop.set_postfix(loss=loss.item())
            
            # append average training loss for epoch
            train_losses.append(sum(train_batch_losses)/len(train_dataloader))
        

        model.eval()
        # validate model for current epoch
        val_preds = []
        val_labels = []
        for val_batch in val_dataloader:
            x_batch = val_batch[0].to(device)
            masks = val_batch[1].to(device)
            y_batch = val_batch[2].to(device)
            with torch.no_grad():
                out = model(x_batch, token_type_ids=None, attention_mask=masks)
                y_pred = out.logits
                val_loss = loss_func(y_pred, y_batch)
                val_batch_losses.append(val_loss.item())
                val_preds.extend(torch.argmax(y_pred, axis=1).cpu().detach().numpy())
                val_labels.extend(y_batch.cpu().detach().numpy())
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_scores.append(f1_score(val_labels, val_preds, average='micro'))
        val_losses.append(sum(val_batch_losses)/len(val_dataloader))

        # early stopping
        if stop_early:
            if val_losses[-1] < min_val_loss:
                min_val_loss = val_losses[-1]
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == epoch_tolerance:
                    print_scores(val_labels, val_preds, train_set=False)
                    print(f'Early stopping after {epoch} epochs')
                    break
        if epoch % 1 == 0:
            print(f'Epoch {epoch}:')
            print_scores(val_labels, val_preds, train_set=False)

    return train_losses, val_losses, train_scores, val_scores, model
           


if __name__ == '__main__': 
    
    # BERT_MODEL = 'bert-base-uncased'
    # BERT_MODEL = 'bert-base-cased'
    # BERT_MODEL = 'bert-large-cased'
    BERT_MODEL = 'beomi/KcELECTRA-base-v2022'
    MAX_LENGTH = 512 

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)


    df = pd.read_csv('./data/sample.csv', )
    essay_col = '본문'
    label_col = 'class'
    # do train test split
    from sklearn.model_selection import train_test_split

    training_df, validation_df = train_test_split(df, test_size=0.2, random_state=2023, stratify=df[label_col])

    # do label encoding
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    num_class = len(df[label_col].unique())
    label_encoder.fit(df[label_col])

    training_df[label_col] = label_encoder.transform(training_df[label_col])
    validation_df[label_col] = label_encoder.transform(validation_df[label_col])


    train_essay_df = pd.DataFrame(training_df, columns=[essay_col])
    train_label_df = pd.DataFrame(training_df, columns=[label_col])

    val_essay_df = pd.DataFrame(validation_df, columns=[essay_col])
    val_label_df = pd.DataFrame(validation_df, columns=[label_col])

    train_sentences = [x for x in train_essay_df[essay_col]]
    val_sentences = [x for x in val_essay_df[essay_col]]
        


    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=num_class,  output_attentions = False, output_hidden_states = False)
    model.to(device)

    x_train = sentences_to_sequences(train_sentences, MAX_LENGTH)
    y_train = torch.tensor(train_label_df.values, dtype= torch.long)
    y_train = y_train.view(-1)

    x_val = sentences_to_sequences(val_sentences, MAX_LENGTH)
    y_val = torch.tensor(val_label_df.values, dtype=torch.long)
    y_val = y_val.view(-1)

    # more hyperparamenter tuning
    batch_size = 8
    num_epochs = 100
    learning_rate = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    #cross entropy loss for sparse class labels
    loss_func  = nn.CrossEntropyLoss()

    #Initialize dataloader for training set
    train_dataset = torch.utils.data.TensorDataset(x_train[0], x_train[1], y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    #Initialize dataloader for validation set
    val_dataset = torch.utils.data.TensorDataset(x_val[0], x_val[1], y_val)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    train_losses, val_losses, train_scores, val_scores, model \
        = execute_training(model, train_dataloader, val_dataloader, loss_func, optimizer, batch_size, num_epochs)


    torch.save(model.state_dict(), './model.pt')
    print('model saved')
