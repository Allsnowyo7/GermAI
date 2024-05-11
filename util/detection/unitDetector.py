import numpy as np 
# from ultralytics import YOLO
import cv2 as cv
import onnxruntime
from onnx import numpy_helper
from PIL import Image
from util.detection.onnxDetector import onnxDetector



model = 'util/detection/models/unit.onnx'
class unitDetect(onnxDetector):
    def __init__(self):
        # self.model = YOLO(model)   
        
        self.session = onnxruntime.InferenceSession(model, None)
        
        self.input_name = self.session.get_inputs()[0].name  
        self.output_name = self.session.get_outputs()[0].name
        
    def preproces(self, source):
        # resize image to 416 by 416    
        source = cv.resize(source, (416, 416), interpolation=cv.INTER_CUBIC)
        # convery array to float 32
        source = np.array(source, dtype=np.float32)
        # adds an aditional dimension (not really sure why its required but sure)
        source = np.expand_dims(source.transpose(2, 0, 1), axis=0)
        # converts the rgb values... i think
        source = source / 255
        return source
    
    def _post_process(self, pred, **kwargs):
        # i honestly have no idea what this does
        height, width, image = kwargs['height'], kwargs['width'], kwargs['image']
        pred[:, [1, 3]] *= 0.8 - 0.05
        pred[:, [1, 3]] += 0.8 * height
        
        return pred
    
    def runModel(self, source):
        height, width = source.shape[:2]
        source = self.preproces(source)
    
        result = self.session.run([self.output_name], {self.input_name: source})[0]
        pred = np.array(self.nms(result)[0])
        #scales the image back up
        pred[:, [0,2]] *= width / 416
        pred[:, [1, 3]] *= height / 416
        
        pred = self._post_process(pred, width=width, height=height, image=source)
        
        print(height)
        return pred
    