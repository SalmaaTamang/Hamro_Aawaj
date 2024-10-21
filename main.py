from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import numpy as np
import cv2
from yolo_predictions import YOLO_Pred
from pipeline import ClassifierPipelineAlexnet
import os
import datetime
import log_predictions
import uvicorn
import pandas as pd


app = FastAPI()

yolo_model = YOLO_Pred('./models/best.onnx', './config/config.yaml')
pipeline = ClassifierPipelineAlexnet()
pipeline.initialize_model()
pipeline.load_model('./models/trained_model_alex01.pth')


image_paths = []
prediction_sentence = []
seconds = 2;

@app.post('/uploadVideos')
async def upload_video(files: UploadFile = File(...)):
    contents = await files.read()
    
    # Save video temporarily to disk for cv2.VideoCapture
    temp_file_path = '/tmp/temp_video.mp4'
    with open(temp_file_path, 'wb') as f:
        f.write(contents)
    
    cap = cv2.VideoCapture(temp_file_path)
    if not cap.isOpened():
        return JSONResponse(content={"error": "Could not open video file"}, status_code=400)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * seconds)  # k seconds interval

    frame_count = 0
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    
    # Remove the temporary video file
    os.remove(temp_file_path)
    frames_dir = '/tmp/frames'
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Process the extracted frames
    predictions = []
    for i, frame in enumerate(frames):
        # Save the frame as a PNG file
        
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(frames_dir, f'frame_{current_time}.png')
        cv2.imwrite(img_path ,frame)
        
        # Process the frame with the modelf
        pred_img = yolo_model.predictions(frame)
        if pred_img is not None and len(pred_img) > 0 :
            prediction = pipeline.predict(pred_img,img_path)
            if prediction is not None and prediction not in predictions:
                predictions.append(prediction)
                image_paths.append(img_path)
                prediction_sentence.append(prediction)
                log_predictions.log_prediction(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') ,prediction ,img_path)
                log_predictions.update_label_count(prediction)
               
            
    log_predictions.log_sentence(image_paths , prediction_sentence)
    predictions_str = ' '.join(predictions)
    return JSONResponse(content = {"data":predictions_str})


@app.post('/uploadImages')
async def upload_images(files: List[UploadFile] = File(...)):
    predictions = []

    for image in files:
        contents = await image.read()
        nparray = np.frombuffer(contents, np.uint8)
        image_data = cv2.imdecode(nparray, cv2.IMREAD_COLOR)

        pred_img = yolo_model.predictions(image_data)
        
        if pred_img is not None:
            img_path = f"./uploads/{image.filename}"
            if not os.path.exists('./uploads'):
                os.makedirs('./uploads')
            with open(img_path, 'wb') as f:
                f.write(contents)


            prediction = pipeline.predict(pred_img, img_path)

            if prediction is not None and prediction not in predictions:
                predictions.append(prediction)
                image_paths.append(img_path)
                prediction_sentence.append(prediction)
                log_predictions.log_prediction(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') ,prediction ,img_path)
                log_predictions.update_label_count(prediction)
               
            
    log_predictions.log_sentence(image_paths , prediction_sentence)
    unique_prediction = " ".join(predictions) 
    return JSONResponse(content=unique_prediction)       

label_count_csv = 'label_count.csv'
log_csv = 'log.csv'
predictions_csv = 'predict_log.csv'

# Function to read CSV file and return JSON data
def read_csv_to_json(file_path):
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path, encoding='utf-8')
        return df.to_dict(orient='records')
    else:
        return("File  not found.")

# Endpoint to return lengths of log.csv and predictions.csv
@app.get("/file_lengths")
async def get_file_lengths():
    try:
        data = read_csv_to_json(label_count_csv)
        log_length = 0
        predictions_length = 0

        if os.path.isfile(log_csv):
            log_df = pd.read_csv(log_csv, encoding='utf-8')
            log_length = len(log_df)

        if os.path.isfile(predictions_csv):
            predictions_df = pd.read_csv(predictions_csv, encoding='utf-8')
            predictions_length = len(predictions_df)

        return {
            "data":data,
            "log_length": log_length,
            "predictions_length": predictions_length
        }
    except Exception as e:
        return str(e)
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)