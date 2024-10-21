import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from torchvision.models.alexnet import AlexNet_Weights
import os
import logging
import json
import datetime
from log import setup_csv_logger

class ClassifierPipelineAlexnet:
    def __init__(self, model_name='alexnet', num_classes=12, log_file='./predict_log.csv', predict_img_dir='./predicted_images'):
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_names = ['उनी', 'काम', 'घर', 'छ', 'त्यो', 'नेपाली', 'म', 'मेरो', 'रुख', 'शिक्षक', 'साथी', 'हो']
        self.model = None
        self.log_file = log_file 
        self.prediction_counts = {class_name: 0 for class_name in self.class_names}
        self.predict_img_dir = predict_img_dir

        if not os.path.exists(self.predict_img_dir):
            os.makedirs(self.predict_img_dir)

        # Setup CSV logger
        self.logger = setup_csv_logger(self.log_file)

    def initialize_model(self):
        torch.cuda.empty_cache()
        self.model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        self.model.classifier[6] = nn.Linear(4096, self.num_classes)
        self.model.classifier.add_module('7', nn.LogSoftmax(dim=1))
        self.model = self.model.to(self.device)

    def load_model(self, path='./models/trained_model_alex01.pth'):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model = self.model.to(self.device)

    def predict(self, img, img_path, log_success=True, log_failure=True):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        try:
            image = Image.fromarray(img).convert('L')
            if image.size[0] == 0 or image.size[1] == 0:
                if log_failure:
                    self._log_prediction('ERROR', 'Failed prediction: Input image has zero size.', img_path)
                return None

            image = transform(image).unsqueeze(0)
            image = image.to(self.device)

            with torch.no_grad():
                outputs = self.model(image)
                _, preds = torch.max(outputs, 1)

            predicted_class = self.class_names[preds.item()]
            self.prediction_counts[predicted_class] += 1
            if log_success:
                self._log_prediction('INFO', predicted_class, img_path)

            save_path = os.path.join(self.predict_img_dir, f"{predicted_class}_{os.path.basename(img_path)}")
            Image.fromarray(img).save(save_path)

            return predicted_class
        except Exception as e:
            if log_failure:
                self._log_prediction('ERROR', f"Error in prediction: {str(e)}", img_path)
            return None

    def _log_prediction(self, level, predicted_class, image_path):
            level = logging.getLevelName(level)
            total_count = json.dumps(self.prediction_counts,ensure_ascii=False)
            log_record = logging.LogRecord(
                name='csv_logger',
                level=level,
                pathname='',
                lineno=0,
                msg='',
                args=(),
                exc_info=None
            )
            log_record.asctime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_record.predicted_class = predicted_class
            log_record.image_path = image_path
            log_record.total_count = total_count
            self.logger.handle(log_record)       

if __name__ == "__main__":
    pipeline = ClassifierPipelineAlexnet()
    pipeline.initialize_model()
    pipeline.load_model()
