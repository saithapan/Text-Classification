import pandas as pd
import re, joblib, pickle
from preprocess_data import preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from gensim_embeddings import convert_txt_vctr, wv
from train_ml_models import train_ml_models
from dataloader import convert_to_tensor_dataset, dataloader
from dl_models import ANN, CNN, LSTM
from train_dl_models import train_dl_model
from torch import nn
import torch.optim as optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


def process_df_txt(dfx, text_col, label_col, drop_duplicates=True):
    """
    This function will remove Nan values in text and then preprocess the text,
    then removes the duplicates in the text

    :param dfx:
    :param text_col:
    :param label_col:
    :param drop_duplicates:
    :return:
    """

    dfx[text_col] = dfx[text_col].apply(lambda x: preprocess(x, remove_punct=True, stemming=False, lower=True))

    if drop_duplicates:
        dfx = dfx.drop_duplicates(subset=text_col, keep=False).reset_index(drop=True)

    if label_col:
        dfx[label_col] = dfx[label_col].apply(lambda x: re.sub("\s\s+", " ", x.strip().lower()))

    return dfx

if __name__ == "__main__":

    df = pd.read_csv('input/complaints.csv')
    text_col = 'Consumer complaint narrative'
    label_col = 'Product'

    df1 = df[[text_col, label_col]].copy()
    # Removing Nan Values in 'Consumer complaint narrative'
    df1 = df1[pd.notnull(df1[text_col])]

    df1 = process_df_txt(dfx=df1, text_col=text_col, label_col=label_col, drop_duplicates=True)
    train_data, test_data = train_test_split(df1, test_size=0.15, random_state=0, stratify=df1[label_col])

    le = LabelEncoder()
    le = le.fit(train_data[label_col])

    train_data[label_col] = le.transform(train_data[label_col])
    test_data[label_col] = le.transform(test_data[label_col])

    # Getting the embeddings for the text
    X_train_gen = [convert_txt_vctr(txt.split()) for txt in train_data[text_col]]
    X_val_gen = [convert_txt_vctr(txt.split()) for txt in test_data[text_col]]

    # Here it will train for Random forest, Xgboost, Logistic Regression and save there pickle file in models folder
    train_ml_models(X_train_gen=X_train_gen,X_val_gen=X_val_gen,train_data=train_data,
                    test_data=test_data, label_col=label_col)

    #  converting dataloader
    dl_train_data, dl_val_data = convert_to_tensor_dataset(train_x=np.array(X_train_gen), train_y=np.array(train_data[label_col]),
                                                                        val_x=np.array(X_val_gen), val_y=np.array(test_data[label_col]))

    train_loader, val_loader = dataloader(train_data=dl_train_data, val_data=dl_val_data,batch_size=1000)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    #---- ANN
    learning_rate = 0.001
    num_epochs = 10
    input_dim = 200
    hidden_dim = 120
    output_dim = train_data[label_col].nunique()

    model = ANN(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = train_dl_model(train_loader=train_loader, val_loader=val_loader,
                           device=device,num_epochs=num_epochs, model=model,
                           criterion=criterion, optimizer=optimizer)
    torch.save(model.state_dict(), 'models/ann/ann.pth')

    # ------ CNN
    num_filters = 100  # Number of filters
    filter_sizes = [3, 4, 5]  # Filter sizes for convolutional layers
    input_dim = 1
    output_dim = train_data[label_col].nunique()
    learning_rate = 0.001
    model = CNN(input_dim=input_dim,num_filters=num_filters, filter_sizes=filter_sizes,output_dim=output_dim,
        dropout=0.25).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = train_dl_model(train_loader=train_loader, val_loader=val_loader,
                           device=device,num_epochs=num_epochs, model=model,
                           criterion=criterion, optimizer=optimizer)
    torch.save(model.state_dict(), 'models/cnn/cnn.pth')

    # ---- LSTM
    learning_rate=0.01
    input_size = wv.vector_size
    hidden_size = 256
    num_layers = 2
    output_dim = train_data[label_col].nunique()
    model = LSTM(input_size=input_size, hidden_size=hidden_size,
                  output_size=output_dim, num_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = train_dl_model(train_loader=train_loader, val_loader=val_loader,
                           device=device,num_epochs=num_epochs, model=model,
                           criterion=criterion, optimizer=optimizer)
    torch.save(model.state_dict(), 'models/lstm/lstm.pth')
