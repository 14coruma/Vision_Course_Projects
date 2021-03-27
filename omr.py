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
import math
import numpy as np
from numba import njit

from PIL import Image

TEMPLATE_DIR = "./templates/"
TEMPLATE_STAVE_DIST = 12

#  @njit()
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
    # padded_image = np.zeros((int(ximlen) + xpadding * 2,int(yimlen) +ypadding*2))
    padded_image = image
    # padded_image[xpadding:xpadding+ximlen,ypadding:ypadding+yimlen] = image
    # padded_image[0:xpadding,0]
    padded_image = np.pad(padded_image,((xpadding,xpadding),(ypadding,ypadding)), padtype)
    # initialize final output image.
    final_image = np.zeros_like(padded_image)
    # iterate through the padded_image
    for i in range(0, padded_image.shape[0]):
        for j in range(0, padded_image.shape[1]):
            # apply the image patch with the kernel by matrix multiplication and the resulting total sum.
            try:
                # final_image[i-xpadding,j-ypadding] = (padded_image[i:i+xklen,j:j+yklen] * kernel).sum()
                final_image[i,j] = (padded_image[i:i+xklen,j:j+yklen] * kernel).sum()
            # when dimensions not equal, then the loop stops.
            except:
                break    
    # trim  out the padding
    final_image = final_image[xpadding:xpadding+ximlen,ypadding:ypadding+yimlen]
    # return final_image
    return final_image
    # for i in range(xpadding, padded_image.shape[0]+xpadding):
    #     for j in range(ypadding,padded_image.shape[1]+ypadding):

# @njit()
def convolve_separable(im, kx, ky):    
    '''
    Given grayscale image, convolve with a separable kernel k = kx^T * ky

    Params:
        im (2d np.array): grayscale image
        kx (np.array): kernel in x direction
        ky (Np.array): kernel in y direction
    
    Returns:
        imOut (2d np.array): resulting image
    '''
    output = convolve(im,kx.T)
    output = convolve(output,ky)
    return output 

# Reference (Hough): Principles of Digital Image Processing (Burger, Burge 2009)
#   Pages 50-63
# Reference (Hough): Course slides (from Canvas)
@njit()
def hough_voting(edges):
    '''
    Given edge-detected image, using Sobel operator. Apply Hough transform.
    Use 2D voting space (D1 = row of first line, D2 = spacing distance)

    Params:
        im (2d np.array): edge-detected image of sheet music

    Returns:
        row (int): row where top-voted stave appears
        staveDist (int): spacing between staves
    '''
    height, width = edges.shape

    # Prepare Hough space accumulator (row, spacing)
    # Max starting row -> height
    # Max spacing -> height//4
    acc = np.zeros((height, height//4))
    # Fill accumulator (each edge pixel casts vote for possible (row, spacing)
    for r in range(height):
        if r%25 == 0: print("Iteration "+str(r)+"/"+str(height))
        for c in range(width):
            if edges[r,c] <= 0: continue
            # Only consider points that are part of a horizontal edge
            #  if edges[r,c] <= 0: continue
            # Possible starting staff rows
            for pRow in range(1,r):
                # Possible spacing (between (r-pRow)//4 and (height-pRow)//4)
                # Where minSpace is assuming r is bottom staff, and where
                # maxSpace is assuming bottom staff is at bottom of image
                minSpace, maxSpace = max((r-pRow)//4,4), (height-pRow)//4
                for pSpace in range(minSpace, maxSpace):
                    # If distance between point and pRow is a multiple of pSpace:
                    if (r-pRow)%(pSpace) == 0: acc[pRow,pSpace] += 1

    # Threshold hough space
    thresh = acc.max() * .5 
    acc = np.where(acc<thresh, 0, acc)

    # Find best spacing
    bestIndex = np.argmax(acc)
    row, space = bestIndex // (height//4), bestIndex % (height//4)
    rows = [row]

    # Search for other staves
    acc = acc[:,space]
    while np.count_nonzero(acc) > 0:
        for i in range(row-space*4, row+space*4):
            if i>0 and i<len(acc): acc[i] = 0
        row = np.argmax(acc)
        rows.append(row+3) # Shift row down by 3. Needed because of gauss blur
    # Remove last row (this algorithm always adds 1 extra stave at row=0)
    rows.pop()

    return rows, space
   
# Reference (Canny): https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
@njit()
def non_maximal_supression(im):
    # Apply non-maximal supression, in y direction only
    newIm = np.zeros(im.shape, dtype=np.float64)
    for r in range(1,im.shape[0]-1):
        for c in range(im.shape[1]):
            val = im[r,c]
            valAbove = im[r-1,c]
            valAbove2 = im[r-2,c]
            valBelow = im[r+1,c]
            valBelow2 = im[r+2,c]
            if (val >= valAbove) and (val >= valBelow) and (val >= valAbove2) and (val >= valBelow2):
                newIm[r,c] = val
    return newIm

# Reference (Sobel): https://en.wikipedia.org/wiki/Sobel_operator
def detect_stave_distance(im):
    '''
    Given grayscale PIL.Image of sheet music, use Hough transform to find
    distance between staves.

    Params:
        im (2d np.array): grayscale image of sheet music

    Returns:
        staveDist (float): spacing between staves
        rows (list): list of all stave starting rows
    '''
    # Blur image with 5x5 gauss
    gauss = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/273
    im = convolve(im, gauss)

    # Simple threshold of the image (we only care about black lines)
    thresh = 200
    im = np.array(np.where(im < thresh, 0, 255), dtype=np.float64)

    # Sobel edge detection
    # We only care about horizontal lines, so just use gradient in y direction
    sy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    edges = abs(convolve(im, sy))

    #  edges = non_maximal_supression(edges)
    rows, staveDist = hough_voting(edges)
    print("Found stave distance {} at rows {}".format(staveDist, rows))
    return staveDist, rows

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
    scale = TEMPLATE_STAVE_DIST/staveDist
    return im.resize((int(im.width*scale), int(im.height*scale))), scale

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
        imScaled (2d np.array): scaled grayscale image of sheet music
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

    print("Detecting stave distance...")
    staveDist, rows = detect_stave_distance(np.array(im, dtype=np.float64))
    imScaled, scale = scale_from_staves(im, staveDist)
    print("Detecting notes...")
    notes = detect_notes(np.array(imScaled), scale)
    visualize_notes(im, notes).save("detected.png")
    notes_to_txt(notes)
    print("Done.")
