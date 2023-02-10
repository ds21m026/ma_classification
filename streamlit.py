# thing to be done for running:
# in anaconda prompt: streamlit run streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from torch import nn

st.title('Klassifizierung des Risikos für Anforderungen')

st.write("Hier wird jetzt einiges geladen")

def get_pred(model, test_data):

    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:

        model = model.cuda()

    with torch.no_grad():
    
        pred = []

        for test_input, test_label in test_dataloader:

              test_label = test_label.to(device)
              mask = test_input['attention_mask'].to(device)
              input_id = test_input['input_ids'].squeeze(1).to(device)

              output = model(input_id, mask)

              if output.argmax(dim=1)[0].item() == 3:
                pred = np.append(pred, 'gering')  
              if output.argmax(dim=1)[0].item() == 2:
                pred = np.append(pred, 'mittel')  
              if output.argmax(dim=1)[0].item() == 1:
                pred = np.append(pred, 'hoch')            

    test_data['Vorhersage'] = pred
    return test_data
	
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
labels = {'gering':3,
          'mittel':2,
          'hoch':1
          }

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = [labels[label] for label in df['ANF_RISIKO']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['ANF_BESCHREIBUNG']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-german-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


device = torch.device('cpu')
model = BertClassifier()
model.load_state_dict(torch.load('mdl_ds1.pt', map_location=device))

st.write("Jetzt ist alles fertig")

d = {'ANF_BESCHREIBUNG': ["Hier steht ein Beispiel für einen Text!"], 'ANF_RISIKO': ["gering"]}
df = pd.DataFrame(data=d)

pred = get_pred(model, df)
text = pred[['ANF_BESCHREIBUNG']]
label = pred[['Vorhersage']]
st.write(text, label)
