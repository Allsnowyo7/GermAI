import onnxruntime
import numpy as np

class OnnxDetector:
    def __init__(self):
        self.sess = onnxruntime.InferenceSession('util/vision/unit.onnx',
                                                 providers=['CPUExecutionProvider'])
        self.output_name = self.sess.get_outputs()[0].name
        self.input_name = self.sess.get_inputs()[0].name
    def scanImage(self, image):
        
        image = np.array(image, dtype=np.float32)
        image = np.expand_dims(image.transpose(2, 0, 1), axis=0)
        image = image / 255
        
        pred = self.sess.run([self.output_name], {self.input_name: image})[0]
        
        return pred