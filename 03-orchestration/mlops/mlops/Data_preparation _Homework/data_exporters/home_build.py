from typing import List, Tuple

import pandas as pd
from scipy.sparse._csr import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression

from mlops.utils.data_preparation.encoders import vectorize_features
from mlops.utils.data_preparation.feature_selector import select_features
from sklearn.feature_extraction import DictVectorizer

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
import mlflow
import pickle
from pathlib import Path

# mlflow.set_tracking_uri("sqlite:///new_mlflow.db")
# mlflow.set_experiment("nyc-taxi-experiment")

@data_exporter
def export(
    df:pd.DataFrame, **kwargs/ 
):
    print(type(df))
    categorical = ['PULocationID', 'DOLocationID']
    train_dicts = df[categorical].to_dict(orient = "records")

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    target= 'duration'
    y_train = df[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print(lr.intercept_)
    # output_path = "./models/lin_model.bin"
    # output_path = Path(output_path)
    # output_path.parent.mkdir(exist_ok=True, parents=True)
    # print(output_path)
    # with open(output_path, 'wb' ) as f_out :
    #     pickle.dump((dv, lr), f_out)

    #mlflow.log_artifact(local_path="./models/lin_reg.bin", artifact_path="models_pickle")
    return lr, dv