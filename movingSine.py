
import cv2
import numpy as np

h,w=660,1260
def sin2d(x,y):
    """2-d sine function to plot"""
    return np.sin(x) + np.cos(y)

def getFrame():
    """Generate next frame of simulation as numpy array"""

    # Create data on first call only
    if getFrame.z is None:
        xx, yy = np.meshgrid(np.linspace(0,2*np.pi,w), np.linspace(0,2*np.pi,h))
        getFrame.z = sin2d(xx, yy)
        getFrame.z = cv2.normalize(getFrame.z,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Just roll data for subsequent calls
    getFrame.z = np.roll(getFrame.z,(1,2),(0,1))
    return getFrame.z

getFrame.z = None

while True:

    # Get a numpy array to display from the simulation
    npimage=getFrame()

    cv2.imshow('image',npimage)
    cv2.waitKey(1)