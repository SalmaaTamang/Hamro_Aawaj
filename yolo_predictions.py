import cv2
import numpy as np
import os
import cv2.dnn
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_yaml

class YOLO_Pred:
    
    def __init__(self, onnx_model, data_yaml):
        self.CLASSES = yaml_load(check_yaml(data_yaml))['names']
        self.yolo = cv2.dnn.readNetFromONNX(onnx_model)
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = f"{self.CLASSES[class_id]} ({confidence:.2f})"
        color = self.generate_color(class_id)
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    def crop_image_from_box(self, img, box, scale):
        x, y, w, h = map(int, box)
        x, y, w, h = int(x * scale), int(y * scale), int(w * scale), int(h * scale)
        cropped_img = img[y:y+h, x:x+w]
        return cropped_img

    def predictions(self, img):  
        row, col, d = img.shape
        max_rc = max(row, col)
        input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
        input_image[0:row, 0:col] = img

        blob = cv2.dnn.blobFromImage(input_image, 1/255, (640, 640), swapRB=True)   
        self.yolo.setInput(blob)
        outputs = self.yolo.forward()
        
        # Non Maximum Suppression
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []
        image_w, image_h = input_image.shape[:2]
        scale = max_rc / 640
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                    outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2],
                    outputs[0][i][3],
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        if len(result_boxes) == 0:
            print("No bounding box detected.")
            return None

        detections = []
        cropped_img = None

        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                "class_id": class_ids[index],
                "class_name": self.CLASSES[class_ids[index]],
                "confidence": scores[index],
                "box": box,
                "scale": scale,
            }
            detections.append(detection)
            
            cropped_img = self.crop_image_from_box(img, box, scale)

            self.draw_bounding_box(
                img,
                class_ids[index],
                scores[index],
                round(box[0] * scale),
                round(box[1] * scale),
                round((box[0] + box[2]) * scale),
                round((box[1] + box[3]) * scale),
            )
        
        return cropped_img
    
    def generate_color(self, ID):
        np.random.seed(10)
        colors = np.random.randint(100, 255, size=(1, 3)).tolist()
        return tuple(colors[ID])