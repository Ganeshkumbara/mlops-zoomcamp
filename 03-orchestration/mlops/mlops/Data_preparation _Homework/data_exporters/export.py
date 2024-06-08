if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter
import mlflow
import pickle
from pathlib import Path
import os
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("nyc-taxi-experiment")


@data_exporter
def export(
    data, **kwargs
):
    lr, dv = data
    
    output_path = "./utils/lin_model.bin"
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    print(output_path)
    with open(output_path, 'wb' ) as f_out :
        pickle.dump(lr, f_out)

    mlflow.log_artifact(local_path=output_path, artifact_path="models_pickle")
    return lr, dv
