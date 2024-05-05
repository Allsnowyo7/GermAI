import onnxruntime

from util.detection.onnxs.constants import UNIT_CONF, UNIT_IOU


path = "util/detection/onnxs/unit.onnx"


class unitDetection:
    def __init__(self):
        self.conf = UNIT_CONF
        self.iou = UNIT_IOU

        self.session = onnxruntime.InferenceSession(
            path, providers=onnxruntime.get_available_providers())
        self.get_input_details()

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
    
    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def inferance(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs
    
    def detect_objects(self, image):
        input_tensor = image
        outputs = self.inference(input_tensor)
        
        self.boxes, self.scores, self.class_ids = self.process_output(outputs)
        return self.boxes, self.scores, self.class_ids
    