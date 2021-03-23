#!/usr/bin/python3

# B657 Assignment 1
# Code by Andrew Corum, Josep Han, Kenneth Zhang, Zheng Chen
#
# usage: python3 omr.py filename
#
# References used:
# - PIL docs: https://pillow.readthedocs.io/en/stable/
# - 

import sys
import os
import numpy as np

from PIL import Image

TEMPLATE_DIR = "./templates/"
# TODO: Figure out the actual distance of staves in templates
# Hopefully we can just inspect the sample images to figure this out
# If not, we might need to write some code to compute it.... but hopefully not
TEMPLATE_STAVE_DIST = 5 
# 
def convolve(image, kernel, padtype = 'edge'):
    # kernel needs to be flipped horizontally and vertically before applying convolution kernel; else it becomes cross-correlation.
    kernel = np.flipud(np.fliplr(kernel))
    # x,y side kernel length will be used to determine the length of the image patch to convolve with the kernel.
    # e.g. a for a 3x2 kernel, we will apply a 3x2 patch of the padded image to the kernel.
    xklen = kernel.shape[0]
    yklen = kernel.shape[1]
    # image lengths will be used to create an empty canvas with the dimensions increased by the padding.
    ximlen = image.shape[0]
    yimlen=image.shape[1]
    
    # padding lengths are 1 less than the kernel size so that the entire canvas applies convolution within the image bounds.
    xpadding = xklen - 1
    ypadding = yklen - 1

    # begin to create the padded image: zero canvas the size of the image + padding sizes. 
    # Then copy the image onto the padded image.
    padded_image = image
    padded_image = np.pad(padded_image,((xpadding,xpadding),(ypadding,ypadding)), padtype)

    # initialize final output image.
    final_image = np.zeros_like(padded_image)
    # iterate through the padded_image
    for i in range(0, padded_image.shape[0]):
        for j in range(0, padded_image.shape[1]):
            # apply the image patch with the kernel by matrix multiplication and the resulting total sum.
            try:
                final_image[i,j] = (padded_image[i:i+xklen,j:j+yklen] * kernel).sum()
            # when dimensions not equal, then the loop stops.
            except:
                break    
    # trim  out the padding
    final_image = final_image[xpadding:xpadding+ximlen,ypadding:ypadding+yimlen]
    # return final_image
    return final_image



# assume kx will be in its transpose form already when applying convolve_separable.
def convolve_separable(im, kx, ky):    
    '''
    Given grayscale image, convolve with a separable kernel k = kx^T * ky

    Params:
        im (PIL.Image): grayscale image
        kx (np.array): kernel in x direction
        ky (Np.array): kernel in y direction
    
    Returns:
        imOut (PIL.Image): resulting image
    '''
    # kx = np.transpose(kx)
    output = convolve(im,kx)
    output = convolve(output,ky)
    return output 

# TODO
def detect_stave_distance(im):
    '''
    Given grayscale PIL.Image of sheet music, use Hough transform to find
    distance between staves.
    Use 2D voting space (D1 = row of first line, D2 = spacing distance)

    Params:
        im (PIL.Image): grayscale image of sheet music

    Returns:
        staveDist (float): distance between staves
    '''
    pass

# TODO
def scale_from_staves(im, staveDist):
    '''
    Given grayscale PIL.Image of sheet music, and distance between staves,
    scale image to match the assumed stave distance of the templates.

    Params:
        im (PIL.Image): grayscale image of sheet music
        staveDist (int): distance between staves

    Returns:
        imScaled (PIL.Image): scaled grayscale image
        scale (float): image scaling factor
    '''
    pass

# TODO
def detect_notes(imScaled, scale):
    '''
    Given appropriately scaled grayscale image of sheet music, detect notes
    and rests given templates. Adjust note postion/scale to match original
    image scale

    Possible approaches:
      - Hamming distance between region and template (using convolution)
      - Compute Sobel edge maps with different scoring fn (see assignment pdf)
    
    Params:
        imScaled (PIL.Image): scaled grayscale image of sheet music
        scale (float): image scaling factor

    Returns:
        notes (list): List of notes in original image. Each note should include:
            [row, col, height, width, symbol_type, pitch, confidence]
    '''
    noteTemp = Image.open(TEMPLATE_DIR + "template1.png")
    quarterTemp = Image.open(TEMPLATE_DIR + "template2.png")
    eighthTemp = Image.open(TEMPLATE_DIR + "template3.png")

    pass

# TODO
def visualize_notes(im, notes):
    '''
    Given original image and list of notes, create a new RGB image and
    visualize the notes (see assignment pdf Fig 1b for example)

    Params:
        im (PIL.Image): grayscale image of sheet music (original size)
        notes (list): List of notes in original image. Each note should include:
            [row, col, height, width, symbol_type, pitch, confidence]

    Returns:
        imAnnotated (PIL.Image): new RGB image annotated with note/rest labels
    '''
    pass

# TODO
def notes_to_txt(notes):
    '''
    Given list of notes, save them in a .txt file

    Params:
        notes (list): List of notes in original image. Each note should include:
            [row, col, height, width, symbol_type, pitch, confidence]
    '''
    pass

if __name__ == '__main__':
    if len(sys.argv) < 2: exit("Error: missing filename")
    im = Image.open(sys.argv[1]).convert(mode='L')

    staveDist = detect_stave_distance(im)
    imScaled, scale = scale_from_staves(im, staveDist)
    notes = detect_notes(imScaled, scale)
    visualize_notes(im, notes).save("detected.png")
    notes_to_txt(notes)
