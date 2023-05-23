import pickle
from ml_models import train_xgboost, train_logistic, train_random_forest


def train_ml_models(X_train_gen,X_val_gen,train_data,test_data,label_col):
    """
    This function will train all 3 ML Models and save them in pickle in models folder
    :param X_train_gen:
    :param X_val_gen:
    :param train_data:
    :param test_data:
    :param label_col:
    :return:
    """

    # Training the logistic model and saving the model in pickle file
    lr_clf = train_logistic(x_train=X_train_gen, y_train=train_data[label_col],
                            x_test=X_val_gen, y_test=test_data[label_col])
    with open("models/lr/model.pkl", 'wb') as f:
        pickle.dump(lr_clf, f)

    # Training the xgboost model and saving the model in pickle file
    xgboost_clf = train_xgboost(x_train=X_train_gen, y_train=train_data[label_col],
                                x_test=X_val_gen, y_test=test_data[label_col])
    with open("models/xgboost/model.pkl", 'wb') as f:
        pickle.dump(xgboost_clf, f)

    # Training the Random Forest model and saving the model in pickle file
    random_forest_clf = train_random_forest(x_train=X_train_gen, y_train=train_data[label_col],
                                            x_test=X_val_gen, y_test=test_data[label_col])
    with open("models/rf/model.pkl", 'wb') as f:
        pickle.dump(random_forest_clf, f)

    return "ML Models training completed"