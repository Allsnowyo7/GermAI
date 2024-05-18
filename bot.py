from ultralytics import YOLO
from util.detection.unitDetector import unitDetect

detector = unitDetect()
model = YOLO('best.pt')
class bot():
    def run(self, screen):
        annotated_frame = detector.runModel(screen)
        # detections = model(screen, conf=0.5)
        # annotated_frame = detections[0].plot()
        return annotated_frame
        
        
        
        
        
    