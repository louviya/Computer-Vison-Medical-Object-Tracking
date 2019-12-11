
from scipy import misc
import numpy as np
import utils
import glob
import pandas as pd
import SimpleITK as sitk
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, bina
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border
from skimage import data
from scipy import ndimage as ndi
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom

fullImage = np.int16(im_original)
print(fullImage[0])
fullImage = fullImage - 1024
print(fullImage[0])
print(np.max(im[0]))
print(np.max(fullImage[0]))
masked_im = utils.createMaskedImage(fullImage,results)
print(np.max(masked_im[0]))
print(np.min(arrayori[0])) 
print(np.min(im_original[0])) 
print(np.max(results[0]))
import tifffile as tiff
import cv2
t='AJS044\AJS044contrast_0001.tif'
tif32 =tiff.imread('AJS044\AJS044contrast_0027.tif')
tif16 =tif32.astype('int16')
cv2.imshow('16',tif16)
cv2.waitKey(0)
cv2.DestroyAllWindows()

cwd = os.getcwd()
origin_folder = os.path.join(cwd, 'AJS044')
print(np.min(tif16))
tif16[tif16==-2000]=0
intercept=0
slop=0.07  #323/512=0.6308
tif16change = slop*tif16+intercept
tif16change[tif16change<0]=0
tif16change=tif16change.astype('int16')
print(np.max(tif16change))

slop=0.06
tif32change = slop*tif32+intercept
tif32change=tif32change.astype('uint16')
print(np.max(tif32change))

tif32b=tif32change.copy()
tif32b[tif32b<1284]=0
tif32b[tif32b>=1284]=255
tif32b=tif32b.astype('uint8')
cv2.imshow('b',tif32b)
cv2.waitKey(0)
cv2.destroyAllWindows()
def overlap(contours_a, contours_b, width, height):
        mask_a = np.zeros((width, height, 1), dtype=np.uint8)
        mask_b = np.zeros((width, height, 1), dtype=np.uint8)
        mask_a = cv2.drawContours(mask_a, [contours_a], -1, (255, 255, 255), -1)
        mask_b = cv2.drawContours(mask_b, [contours_b], -1, (255, 255, 255), -1)
        intersection = cv2.bitwise_and(mask_a, mask_b)
        if np.amax(intersection) == 255:
            return True
        return False

def filterContour(contour):
        if cv2.contourArea(contour) > 7500:
            return False
        if cv2.contourArea(contour) < 2:
            return False
        # Filter by convexity
        minConvexity = 0.5
        hull = cv2.convexHull(contour)
        area = cv2.contourArea(contour)
        hullArea = cv2.contourArea(hull)
        if not hullArea > 0:
            return False
        ratio = area / hullArea
        if ratio < minConvexity:
           return False
        return True

def setVesselParams(params):
        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 255

        # Filter by area
        params.filterByArea = True
        params.minArea = 200
        params.maxArea = 4000

        # Filter by circularity
        params.filterByCircularity = True
        params.minCircularity = 0.45  # Might want to try turning this off if it doesn't detect anything

        # Filter by convexity
        params.filterByConvexity = True
        params.minConvexity = 0.8

        # Filter by inertia. Small inertia = line shape, large inertia = circle shape
        params.filterByInertia = True
        params.minInertiaRatio = 0.3

        params.filterByColor = True
        params.blobColor = 255




w,h= im_original[0].shape  # Width, height, and channels
t=im_original[:]

n = im.__len__()  # Number of slices
mask_final = im_original[:]  #copy()
contour_list = [[] for _ in range(n)]
initial_params = cv2.SimpleBlobDetector_Params()
setVesselParams(initial_params)
detector = cv2.SimpleBlobDetector_create(initial_params)
        # Detect initial aorta
keypoints = detector.detect(t[0])

        # Make mask of initial aorta and get contours from it
mask_init = np.zeros((w, h), dtype=np.uint8)
if keypoints:
     for x in range(0, keypoints.__len__()):
         mask_init = cv2.circle(mask_init, (np.int(keypoints[x].pt[0]), np.int(keypoints[x].pt[1])),
              radius=np.int(keypoints[x].size / 2 + 3), color=(255, 255, 255),
                                       thickness=-1)
     mask_init = cv2.bitwise_and(t[0], t[0], mask=mask_init)
     
     
cv2.imshow('t',t[0])
cv2.waitKey(0)
#cv2.destroyAllWindows()
_, contours, hierarchy = cv2.findContours(mask_init, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
_, contours_new, hierarchy_new = cv2.findContours(im[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
img=np.zeros((w,h),dtype=np.uint8)
cv2.drawContours(img,contours_new,-1,(255,255,255),-1)
cv2.imshow('m',intersection)
cv2.waitKey(0)
cv2.destroyAllWindows()

mask_a_con = np.zeros((w, h, 1), dtype=np.uint8)
mask_b_new = np.zeros((w, h, 1), dtype=np.uint8)
mask_a_con = cv2.drawContours(mask_a_con, contours, -1, (255, 255, 255), -1)
mask_b_new = cv2.drawContours(mask_b_new, contours_new, -1, (255, 255, 255), -1)
intersection = cv2.bitwise_and(mask_a_con, mask_b_new)
print(np.amax(intersection))
        if np.amax(intersection) == 255:
            return True
        return False
    
from PIL import Image as ImagePIL,ImageFont,ImageDraw
from PIL import Image
test=ImagePIL.open('')
test2=tiff.imread(r'AJS044\AJS044contrast_0001.tif')
print(np.max(test2))
test2=Image.fromarray(cv2.cvtColor(test2,cv2.COLOR_BGR2RGB))
test2.save('test.tif',dpi=(512.0,512.0))
print(np.max(test2))
test3=cv2.imread('test.tif')
print(np.max(test3))
test3=cv2.cvtColor(test3, cv2.COLOR_RGB2GRAY)
print(np.max(test3))

