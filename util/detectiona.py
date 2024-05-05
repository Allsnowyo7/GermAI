import numpy as np
import captureScreen as cs 
import cv2 as cv 
from ultralytics import YOLO
from detector import YOLOv8
from time import time


grabScreen = cs.WindowCapture("Clash Royale")

detector = YOLOv8(conf_thres=0.3, iou_thres=0.5)
# model = YOLO('util/vision/best.pt')

loopTime = time()

while(True):
    source = grabScreen.get_screen()
    # results = model.track(source, conf=0.4, device=0)   
    # annotated_frame = results[0].plot()
    detector(source)
    annotated_frame = detector.draw_detections(source)
    
    # puts fps and shows the detections
    cv.putText(annotated_frame, str(round(1 / (time() - loopTime), 2)), (40,50), cv.FONT_HERSHEY_PLAIN, 2, (46,201,164), 3, 3)
    cv.imshow("YOLOv8 Tracking", annotated_frame)
    
    # resets looptime for fps calculation
    loopTime = time()
    
    # Break the loop if 'q' is pressed
    if cv.waitKey(1) == ord("q"):
        break


