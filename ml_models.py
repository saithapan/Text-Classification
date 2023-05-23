import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def train_logistic(x_train, y_train, x_test, y_test):
    clf = LogisticRegression(random_state=0).fit(x_train, y_train)
    trans_predictions = clf.predict(x_test)
    print("----- Logistic")
    print(classification_report(y_true=y_test, y_pred=trans_predictions))
    return clf


def train_xgboost(x_train, y_train, x_test, y_test):
    xgb_transformers = xgb.XGBClassifier()

    xgb_transformers.fit(x_train, y_train)
    trans_predictions = xgb_transformers.predict(x_test)
    print("----- XGBoost")
    print(classification_report(y_true=y_test, y_pred=trans_predictions))
    return xgb_transformers


def train_random_forest(x_train, y_train, x_test, y_test):
    rf_clf = RandomForestClassifier(n_estimators=100,
                                    criterion="gini",
                                    max_depth=8,
                                    min_samples_split=2,
                                    max_features="sqrt")

    rf_clf.fit(x_train, y_train)
    trans_predictions = rf_clf.predict(x_test)
    print("----- Random Forest")
    print(classification_report(y_true=y_test, y_pred=trans_predictions))
    return rf_clf
