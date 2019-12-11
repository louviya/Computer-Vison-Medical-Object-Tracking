import cv2
import numpy as np
import pandas as pd


def calculateOverlap(binary_image, overlap_type):
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
        if cv2.contourArea(contour) > 1400:
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
        params.maxArea = 1400

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


    
    w, h = binary_image[0].shape  # Width, height, and channels
    n = binary_image.__len__()  # Number of slices
    mask_final = binary_image[:]  #copy()
    contour_list = [[] for _ in range(n)]

    if overlap_type == 'vessel':
        initial_params = cv2.SimpleBlobDetector_Params()

        setVesselParams(initial_params)

        # Create detector
        detector = cv2.SimpleBlobDetector_create(initial_params)

        # Detect initial aorta
        keypoints = detector.detect(binary_image[0])

        # Make mask of initial aorta and get contours from it
        mask_init = np.zeros((w, h), dtype=np.uint8)
        if keypoints:
            for x in range(0, keypoints.__len__()):
                mask_init = cv2.circle(mask_init, (np.int(keypoints[x].pt[0]), np.int(keypoints[x].pt[1])),
                                       radius=np.int(keypoints[x].size / 2 + 3), color=(255, 255, 255),
                                       thickness=-1)
            mask_init = cv2.bitwise_and(binary_image[0], binary_image[0], mask=mask_init)
    else:
        mask_init = binary_image[0]

    _, contours, hierarchy = cv2.findContours(mask_init, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    c= []
    # Loop through and find overlapping blobs
    for i in range(n):
        # Create 'checked' list
        checked = []

        mask = np.zeros((w, h), dtype=np.uint8)

        # Look for new contours
        _, contours_new, hierarchy_new = cv2.findContours(binary_image[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours_temp = []
       
        # Check for overlap
        for x in range(0, contours_new.__len__()):
            for y in range(0, contours.__len__()):
                if x not in checked and overlap(contours[y], contours_new[x], w, h) and filterContour(contours_new[x]):
                    contours_temp.append(contours_new[x])
                    checked.append(x)

        # Contours added to list become new array to check against
        contours = contours_temp
        contour_list[i] = contours
        
        c.append(len(contour_list[i])) #number of contours
        
        
        # Draw current contours array
        cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
        mask_final[i] = mask
        #print(mask_final[i].shape) #(512,512)

        


    return mask_final, contour_list
