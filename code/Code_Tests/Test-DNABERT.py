import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import random
import matplotlib.pyplot as plt
from transformers.models.bert.configuration_bert import BertConfig
from transformers import AutoTokenizer, AutoModel
import torch

from Bio import SeqIO
import gc

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

device = "mps" if torch.backends.mps.is_available() else "cpu"

def clean_gpu():
    torch.cuda.empty_cache()
    gc.collect()

clean_gpu()


import triton # installed with hombrew

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config) # Requires triton to run

# Tokenizer to device. Makes sure the model is running on the GPU instead of the CPU
model.to(device)


dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"].to(device)
hidden_states = model(inputs)[0] # [1, sequence_length, 768]
embedding_mean = torch.mean(hidden_states[0], dim=0)
hidden_states.shape



from math import floor
data = pd.read_csv("code/RNN/Coding_NonCoding_DNA_Sequences.csv")
data = data[["DNA_sequence", "Target"]]
data = data.rename(columns={"DNA_sequence": "sequence", "Target": "target"})
batch_size = 100
data = data.sample(10000)
data



inputs = tokenizer(data.sequence.values.tolist(), return_tensors="pt", padding=True)["input_ids"].to(device)
inputs.shape

inputs = torch.reshape(inputs, (-1, batch_size, inputs.shape[1]))
inputs.shape


for param in model.parameters():
    param.requires_grad = False


full_embedding = []
for input_i in inputs:
    embedding_data = model(input_i)[0]
    embedding_data = torch.mean(embedding_data, dim=1)
    embedding_data = embedding_data.cpu().detach().numpy()
    full_embedding.append(embedding_data)
embedding_data = np.array(full_embedding).reshape(-1,embedding_data.shape[-1])
embedding_data.shape


def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    metrics = [acc, precision, recall, f1, auc]
    return pd.DataFrame(metrics, ["acc", "precision", "recall", "f1", "auc"])


X_train, X_test, y_train, y_test = train_test_split(embedding_data, data.target)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
get_metrics(y_test, y_pred)


clean_gpu()
import keras
from keras import layers
from keras.metrics import *
from keras.callbacks import *
mlp = keras.Sequential(
    [
        layers.Input((768,)),
        layers.Dense(1000, kernel_regularizer="l2"),
        layers.ReLU(),
        layers.Dropout(0.25),
        layers.Dense(1, activation="sigmoid"),
    ]
)
mlp.compile(optimizer="adam", loss="binary_crossentropy", metrics=[BinaryAccuracy(), Precision(), Recall(), AUC()])
callbacks = [
    EarlyStopping(patience=10),
    ModelCheckpoint("code/RNN//model.weights.h5", save_best_only=True, save_weights_only=True, verbose=True),
    ReduceLROnPlateau(patience=1),
    CSVLogger("code/RNN//history.csv")
]
history = mlp.fit(X_train, y_train, epochs=500, validation_split=0.2, callbacks=callbacks, verbose=0)
mlp.load_weights("code/RNN//model.weights.h5")




plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Cross entropy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim([0, 1])
plt.legend(['train', 'val'], loc='upper left')
plt.show()


pd.DataFrame([mlp.evaluate(X_test, y_test, verbose=0)], columns=["auc", "acc", "loss", "precision", "recall"])

clean_gpu()



from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Create a t-SNE object
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)

# Fit and transform the embeddings
embeddings_2d = tsne.fit_transform(embedding_data)

# Make a scatter plot, color by target class
plt.figure(figsize=(8,6))
plt.scatter(
    embeddings_2d[:,0],
    embeddings_2d[:,1],
    c=data.target,
    cmap='coolwarm',
    s=5,
    alpha=0.7
)
plt.title("t-SNE visualization of DNABERT embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.colorbar(label='Target')
plt.show()
