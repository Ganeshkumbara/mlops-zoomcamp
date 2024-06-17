import pickle
from flask import Flask, jsonify, request
import mlflow
from mlflow.tracking import MlflowClient


MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
RUN_ID = 'f36a894b060442d28b492636581cad8c'

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logged_model = f's3://ml-model-bucket/1/{RUN_ID}/artifacts/model'
# logged_model=f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)

## manual load from local directory
# with open('model.bin', 'rb' ) as f_in :
#     dv, model = pickle.load(f_in)

app = Flask("duration-prediction")

def prepare_features(ride):
    features = {}
    features['PU_DO'] = f'{ride["PULocationID"]}_{ride["DOLocationID"]}'
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    # X = dv.transform(features)
    # preds = model.predict(X)
    preds = model.predict(features)
    return preds[0]

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)


# import boto3

# # Create an S3 client
# import os


# accesskey = os.getenv("AWS_ACCESS_KEY_ID")
# secretaccesskey = os.getenv("AWS_SECRET_ACCESS_KEY")
# # Specify the file path and bucket name
# file_path = 'new.txt'
# bucket_name = 'ml-model-bucket'
# region = "ap-south-1"
# s3 = boto3.client('s3', aws_access_key_id=accesskey, aws_secret_access_key=secretaccesskey, region_name=region)
# # Upload the file to the specified bucket
# s3.upload_file(file_path, bucket_name, 'text_files/file.txt')