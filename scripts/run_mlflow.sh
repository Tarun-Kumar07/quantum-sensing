#!/bin/bash

mlflow server \
    --backend-store-uri sqlite:///data/mlflow/mlflow.db \
    --default-artifact-root ./data/mlflow/artifacts \
    --host localhost \
    --port 8080