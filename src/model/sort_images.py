from PIL import Image
import torch
import pickle
from src.data.feature_extraction import get_features

class SortImageModel:
    def __init__(self):
        print('Solver init')
        self.object_detector = torch.hub.load('ultralytics/yolov5', 'custom', path='src/model/model_checkpoints/yolov5s_weights.pt')
        with open('src/model/model_checkpoints/binary_clf_model.pkl', 'rb') as fd:
            self.binary_classifier = pickle.load(fd)
    
    def predict(self, img_path: str):
        # The following pipeline
        # 1. Check if picture is broken. If broken -> returns class [1, 0, 0]
        # 2. Via the binary classifier check if picture has bad quality. If bad quality (class 0) -> return [1, 0, 0]
        # 3. On good quality images try to find any animal via Object Detection model. If animal exists return [0, 0, 1] else [0, 1, 0]
        
        # 1
        try:
            Image.open(img_path).load()
        except Exception as e:
            return [1, 0, 0]
        
        # 2
        image_features = get_features(img_path=img_path)
        prediction = self.binary_classifier.predict(image_features)
        if prediction == 0:
            return [1, 0, 0]

        # 3
        # Object detection
        result = self.object_detector(img_path)
        bboxes = result.pandas().xyxy[0]
        if bboxes.shape[0] == 0:
            return [0, 1, 0]
        elif bboxes.shape[0] == 1 and bboxes['name'][0] == 'car':
            return [0, 1, 0]
        
        return [0, 0, 1]
