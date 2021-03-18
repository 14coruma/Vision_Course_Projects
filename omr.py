#!/usr/bin/python3

# B657 Assignment 1
# Code by Andrew Corum, Josep Han, Kenneth Zhang, Zheng Chen
#
# usage: python3 omr.py filename
#
# References used:
# - PIL docs: https://pillow.readthedocs.io/en/stable/

import sys
import os
import numpy as np

from PIL import Image

TEMPLATE_DIR = "./templates/"
TEMPLATE_STAVE_DIST = 5 # TODO: Figure out the actual distance of staves in templates

# TODO
def convolve(im, k):
    '''
    Given grayscale image, convolve with a kernel

    Params:
        im (PIL.Image): grayscale image
        k (np.array): convolution kernel

    Returns:
        imOut (PIL.Image): resulting image
    '''
    pass

# TODO
def convolve_separable(im, k):
    '''
    Given grayscale image, convolve with a separable kernel

    Params:
        im (PIL.Image): grayscale image
        k (np.array): separable convolution kernel (k = kx^T * ky)
    
    Returns:
        imOut (PIL.Image): resulting image
    '''
    pass

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
