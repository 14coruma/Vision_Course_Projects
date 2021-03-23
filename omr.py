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
from numba import njit
from tqdm import tqdm

from PIL import Image

TEMPLATE_DIR = "./templates/"
TEMPLATE_STAVE_DIST = 12

# TODO: Currently padding with zeros. Change later if necessary (but this might be fine)
def convolve(image,kernel,padding =0):
    '''
    Given grayscale image, convolve with a kernel

    Params:
        im (PIL.Image): grayscale image
        k (2d np.array): convolution kernel

    Returns:
        imOut (PIL.Image): resulting image
    '''
    image = np.array(image)
    kernel = np.flipud(np.fliplr(kernel))
    xklen = kernel.shape[0]
    yklen = kernel.shape[1]
    ximlen = image.shape[0]
    yimlen = image.shape[1]
    if padding != 0:
        padded_image = np.zeros((int(ximlen) + padding * 2,int(yimlen) +padding*2))
        padded_image[padding:padding+ximlen,padding:padding+yimlen] = image
    else:
        padded_image = image
    final_image = np.zeros((padded_image.shape[0] - padding ,padded_image.shape[1] - padding))
    for i in range(0, padded_image.shape[0]):
        for j in range(0,padded_image.shape[1]):
            try:
                final_image[i,j] = (padded_image[i:i+xklen,j:j+yklen] * kernel).sum()
            except:
                break    
    return Image.fromarray(final_image)

# TODO: FIX
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
    kx = kx.T
    output = convolve(im,kx)
    output = convolve(output,ky)
    return output 

# Reference (Sobel): https://en.wikipedia.org/wiki/Sobel_operator
# Reference (Hough): Principles of Digital Image Processing (Burger, Burge 2009)
#   Pages 50-63
# Reference (Hough): Course slides (from Canvas)
# TODO
def detect_stave_distance(im):
    '''
    Given grayscale PIL.Image of sheet music, use Hough transform to find
    distance between staves.
    Use 2D voting space (D1 = row of first line, D2 = spacing distance)

    Params:
        im (PIL.Image): grayscale image of sheet music

    Returns:
        staveDist (float): spacing between staves
    '''
    # Apply Sobel edge detection
    # TODO: Canny edge detection for better results??
    # TODO: Need to fix convolve_separable() first
    #  sy1, sy2 = np.array([1,0,-1]), np.array([1,2,1])
    #  edges = convolve_separable(im, sy1, sy2)
    # We only care about horizontal lines
    sy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    edges = np.array(convolve(im, sy))
    height, width = edges.shape

    # Prepare Hough space accumulator (row, spacing)
    # Max starting row -> height-5
    # Max spacing -> height//4
    acc = np.zeros((height-5, height//4))
    # Fill accumulator (each edge pixel casts vote for possible (row, spacing)
    print
    for r in tqdm(range(height), desc="Detecting stave distance..."):
        for c in range(width):
            # Only consider points that are part of a horizontal edge
            if edges[r,c] <= 0: continue
            # Possible starting staff rows
            for pRow in range(r):
                # Possible spacing
                minSpace, maxSpace = max((r-pRow)//4,1), (height-pRow)//4
                for pSpace in range(minSpace, maxSpace):
                    # If distance between point and pRow is a multiple of pSpace:
                    if (r-pRow)%pSpace == 0: acc[pRow,pSpace] += 1

    # Find best spacing
    bestIndex = np.argmax(acc)
    row, staveDist = bestIndex // (height//4), bestIndex % (height//4)
    print("Found stave distance {}, starting at row {}.".format(staveDist, row))
    
    return staveDist

# TODO
    print(im.height, height)
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
