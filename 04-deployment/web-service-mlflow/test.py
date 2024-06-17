# import predict


# ride = {
#     "PULocationID": 10,
#     "DOLocationID": 50,
#     "trip_distance": 40
# }
# features = predict.prepare_features(ride)
# pred = predict.predict(features)
# print(pred)

import requests

ride = {
    "PULocationID": 10,
    "DOLocationID": 20,
    "trip_distance": 30
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=ride)
print(response.json())

# docker build -t duraction-predict:v1 .
# docker run -it --rm -p 9696:9696 duraction-predict:v1
# mlflow server --backend-store-uri=sqlite:///mlflow.db --default-artifact-root=s3://ml-model-bucket/