# DataGrid Assignment

In this project, a classifier is built to categorize customer complaints. The latest dataset from the Consumer Complaint Database is downloaded and used for training the model. I have train this data with Random Forest, Logistic Regression, XGBoost, ANN, CNN, LSTM. I have created total 6 endpoints for this 6 different models. The endpoint will take the audio file as input and returns the text present along with Predicted categorise Label.

## Prerequisites

- Before going further steps please Download the dataset from [here](https://catalog.data.gov/dataset/consumer-complaint-database). Extract the CSV File and place in it `input` folder. Since this is large dataset it not possible to share it in github

## Installation

- Clone the code from this repo
- Create new environment and requirements.txt file

```commandline
conda create --name <env_name> python=3.9
conda activate <env_name>
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

## Setup

- Run Project in New terminal

```shell
flask run
```
- It will take around 3 to 4 minutes to start the app, since it is loading the word2vec from gensim
## Files explanation
1. `input` - Have test audio files in this folder
2. `models` - For ML Models are save as pickle and Deep Neural Models are saved as .pth in this folder
3. `app.py` - Created total six endpoints for Random forest, Logistic Regression, XGBoost, ANN, CNN, LSTM. whenever we call the endpoint it will load the respective model from models and provde the predictions for it
4. `dataloader.py` - In this we have 3 functions that we are using to train, test data to dataloader for Deep Neural model training
5. `dl_models.py` - In this we created the models classes for ANN, CNN, LSTM
6. `gensim_embeddings.py` - This is file is used to generate the embeddings for the text.
7. `main.py` - This is the main file where all training happen for ML and Deep Neural models and this models are saved in the models folder
8. `ml_models.py` - In this we created the models for Classic ML.
9. `preprocess_data.py` - The module present in this file will preprocess the text like removing the stopwords, lowering the text etc....
10. `stopwords_collection.py` - Instead of using pre-defined stopwords from libraries i used own stopwords collection so that i can't miss the important words
11. `train_dl_models.py` - Here it will train the Neural Network model
12. `train_ml_models.py` - Here it will train the ML Model and save the model the models folder

## Available endpoints
1. http://127.0.0.1:5000/upload/rf        -- Random Forest
2. http://127.0.0.1:5000/upload/logistic  -- Logistic Regression
3. http://127.0.0.1:5000/upload/xgboost   -- XGBOOST
4. http://127.0.0.1:5000/upload/ann       -- ANN
5. http://127.0.0.1:5000/upload/cnn       -- CNN
6. http://127.0.0.1:5000/upload/lstm      -- LSTM

## Testing the api

### In Postman

- Once server started running
- Now Open Postman, select POST Method and provide this url(http://127.0.0.1:5000/upload/lstm) (Any one from available endpoints)
- Now select Body -> form-data -> In key provide key name as `audio` and select type as `file` and upload
  the `audion_1.wav`(Any one from input folder) file and click send button
![image](https://github.com/saithapan/datagrid_assignment/assets/36238978/5d86a754-0aee-442b-b258-c466c7272811)

## Accuracy for each model on test data
1. Random Forest -- 67%
2. Logistic Regression -- 53%
3. XGBOOST -- 73%
4. ANN -- 58%
5. CNN -- 62%
6. LSTM -- 72%

## Further Steps and Missing Things
- Due to limited resource i have, Like system configuration etc.., training with State of the Art model like  BERT, RoBERTa, GPT3 etc.. taking long time. So for now i have skip training with SOTA. And also existing models we need more hyperparameters tuning also
- 
