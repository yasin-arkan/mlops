I used these commands to copy model.bin file to local, then used the model to predict the may 2023 data

docker run -d --name model-container agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

docker cp model-container:/app/model.bin ./model.bin
