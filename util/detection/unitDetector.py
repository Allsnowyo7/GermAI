import numpy as np 
# from ultralytics import YOLO
import cv2 as cv
import onnxruntime
from onnx import numpy_helper
from PIL import Image



model = 'util/detection/models/unit.onnx'
class unitDetect():
    def __init__(self):
        # self.model = YOLO(model)   
        
        self.session = onnxruntime.InferenceSession(model, None)
        
        self.input_name = self.session.get_inputs()[0].name  
        self.output_name = self.session.get_outputs()[0].name
    @staticmethod
    def _xywh_to_xyxy(boxes):
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

    @staticmethod
    def _nms(boxes, scores, thresh):
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]  # get boxes with more ious first

        keep = []
        while order.size > 0:
            i = order[0]  # pick maximum iou box
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)  # maximum width
            h = np.maximum(0.0, yy2 - yy1 + 1)  # maximum height
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
        
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
    
    def nms(self, predicton):
        prediction = predicton.transpose((0, 2, 1))
        output = [np.zeros((0, 6))] * len(predicton)
        for i in range(len(predicton)):
            x = predicton[i]
            
            scores = x[:, 4:]
            best_scores_idx = np.argmax(scores, axis=1).reshape(-1, 1)
            best_scores = np.take_along_axis(scores, best_scores_idx, axis=1)
            
            # masks out predictions below conf level
            mask = np.ravel(best_scores > 0.35)
            best_scores = best_scores[mask]
            best_scores_idx = best_scores_idx[mask]
            
            boxes = x[mask, :4]
            self._xywh_to_xyxy(boxes)
            
            #determins which boxes to keep
            keep = self._nms(boxes, np.ravel(best_scores), .45)
            best = np.hstack([boxes[keep], best_scores[keep], best_scores_idx[keep]])
            
            output[i] = best
        return output
    
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
    