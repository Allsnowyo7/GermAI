from bot import bot
import util.captureScreen as cs 
import cv2 as cv
from time import time

grabScreen = cs.WindowCapture("Clash Royale")
detection = bot()
loopTime = time()

while(True):
    screen = grabScreen.get_screen()
    annotated_frame = detection.run(cv.imread('image.png'))
    
    # puts fps and shows the detections
    cv.putText(annotated_frame, str(round(1 / (time() - loopTime), 2)), (40,50), cv.FONT_HERSHEY_PLAIN, 2, (46,201,164), 3, 3)
    cv.imshow("YOLOv8 Tracking", annotated_frame)
    
    # resets looptime for fps calculation
    loopTime = time()
    
    # Break the loop if 'q' is pressed
    if cv.waitKey(1) == ord("q"):
        break
    