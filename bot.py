from util.detection.unitDetector import unitDetect

detector = unitDetect()

class bot():
       
    def run(self, screen):
        annotated_frame = detector.runModel(screen)
        return annotated_frame
        
        
        
        
        
    