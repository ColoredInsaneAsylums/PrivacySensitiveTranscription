import cv2
import numpy

# Scale-Invariant Feature Transform
class SIFT:

    # initialize SIFT descriptor
    def __init__(self):
        self.descriptor = cv2.xfeatures2d.SIFT_create()

    # compute SIFT descriptors
    def compute(self, im):
        return self.descriptor.detectAndCompute(im, None)[1].flatten()

# Speeded-Up Robust Features
class SURF:

    # initialize SURF descriptor
    def __init__(self, hessianThreshold=400):
        self.descriptor = cv2.xfeatures2d.SURF_create(hessianThreshold)

    # compute SURF descriptors
    def compute(self, im):
        return self.descriptor.detectAndCompute(im, None)[1].flatten()

# Histogram of Oriented Gradients
class HOG:

    # initialize HOG descriptor
    def __init__(self, width=128, height=64):
        self.descriptor = cv2.HOGDescriptor(_winSize = (width,height),
                                            _blockSize = (16,16),
                                            _blockStride = (8,8),
                                            _cellSize = (8,8),
                                            _nbins = 9)

    # compute HOG descriptors
    def compute(self, im):
        return self.descriptor.compute(im).flatten()

# Oriented FAST and Rotated BRIEF
class ORB:

    # initialize ORB descriptor
    def __init__(self):
        self.descriptor = cv2.ORB_create()

    # compute ORB descriptors
    def compute(self, im):
        return self.descriptor.compute(im, None)[1].flatten()

