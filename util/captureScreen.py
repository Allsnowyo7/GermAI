import mss
import numpy as np
import win32gui


class WindowCapture():
        def __init__(self, windowName):
            self.sct = mss.mss()
              
            #finds the window
            self.window = win32gui.FindWindow(None, windowName)
            if not self.window:
                raise Exception('Window not found: {}'.format(windowName))
            
            #gets the coordnates of the window
            self.windowPos = win32gui.GetWindowRect(self.window)
            
        def get_screen(self):
            # takes a screenshot of the window
            screen = np.array(self.sct.grab(self.windowPos))
            screen = screen[...,:3]
            return screen