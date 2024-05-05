from util.detector import YOLOv8


detector = YOLOv8(conf_thres=0.3, iou_thres=0.5)

class bot():
    def __init__(self):  
        pass
      
    def run(self, screen):
        annotated_frame = detector(screen)
        return annotated_frame
        
        
        
        
        
    