import cv2
#from . import calculateOverlap
#from . import utils
import calculateOverlap
import utils
import imageio
import tifffile as tiff
import os
import numpy as np


def ask_yes_no(prompt):
    while True:
        try:
            value = input(prompt).lower()
        except ValueError:
            print("Invalid answer format")
            continue
        if value not in ['yes', 'no', 'y', 'n']:
            print("Answer must be yes/no/y/n")
            continue
        else:
            break
    if value in ['yes', 'y']:
        return True
    if value in ['no', 'n']:
        return False


def validate_file(cwdir, prompt):
    while True:
        try:

            value = os.path.join(cwdir, input(prompt))
            _ = open(value)
        except Exception: #FileExistsError:
            print("File not found: ", value)
            continue
        else:
            break
    return value


def validate_filename(prompt):
    while True:
        try:
            value = input(prompt)
        except ValueError:
            print("Invalid formatting")
            continue
        if not value.endswith(".tif"):
            print("File name must end in '.tif'")
            continue
        else:
            break
    return value


def validate_int(prompt):
    while True:
        try:
            value = int(input(prompt))
        except ValueError:
            print("Value must be an integer")
            continue
        if value < 0:
            print("Value must be positive")
            continue
        else:
            break
    return value


if __name__ == '__main__':
    mask = False
    mask_output = ''
    cwd = os.getcwd()
    print("Current directory:", cwd)
    binary_file = validate_file(cwd, "Enter path to binary CT scan: ")
    original_file = validate_file(cwd, "Enter path to original CT scan: ")
    if ask_yes_no("\nWould you like to save the results to a new .tif file? "):
        mask_output = validate_filename("Enter name of output file: ")
        mask = True
    min_value = validate_int("Enter minimum pixel value for vessel: ")
    max_value = validate_int("Enter maximum pixel value for vessel: ")

    # Read binary file
    CTA = []
    _, im = cv2.imreadmulti(mats=CTA, filename=binary_file, flags=cv2.IMREAD_ANYCOLOR)

    # Calculate the mask for the vessel and calcifications
    results, contour_list = calculateOverlap.calculateOverlap(im, 'vessel')
    im_original = tiff.imread(original_file)

    xResolution, yResolution, zResolution = utils.getResolution(original_file)
    utils.pixelCount(im_original, results, xResolution, yResolution, zResolution, min_value, max_value, contour_list)
    if mask:
        imageio.mimwrite(mask_output, results)

    if ask_yes_no("\nWould you like to superimpose results on the original scan?: "):
        rgb_file = validate_file(cwd, "Enter path to RGB CT scan: ")
        output_rgb = validate_filename("Enter name of superimposed output file: ")

        CTA2 = []
        ret2, im2 = cv2.imreadmulti(mats=CTA2, filename=rgb_file, flags=1)#cv2.IMREAD_ANYCOLOR)
        im_original = tiff.imread(original_file) #(446, 512,512)

        sresults = utils.superimpose_color(im2, im_original, results, min_value, max_value)
        change = np.array(sresults)
        change = change+1024         #test3

        
        imageio.mimwrite(output_rgb, change)
