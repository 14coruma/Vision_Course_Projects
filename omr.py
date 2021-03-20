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

# TODO: Currently padding with zeros. Change later if necessary (but this might be fine)
def convolve(image, kernel):
    
    kernel = np.flipud(np.fliplr(kernel))
    xklen = kernel.shape[0]
    yklen = kernel.shape[1]
    ximlen = image.shape[0]
    yimlen=image.shape[1]
    # xpadding
    xpadding = xklen - 1
    ypadding = yklen - 1
    # if padding != 0:
    #     padded_image = np.zeros((int(ximlen) + padding * 2,int(yimlen) +padding*2))
    #     padded_image[padding:padding+ximlen,padding:padding+yimlen] = image
    # else:
        # else:
        # padded_image = image

    
    padded_image = np.zeros((int(ximlen) + xpadding * 2,int(yimlen) +ypadding*2))
    padded_image[xpadding:xpadding+ximlen,ypadding:ypadding+yimlen] = image
    # final_image = np.zeros((padded_image.shape[0] - padding ,padded_image.shape[1] - padding))
    final_image = np.zeros((ximlen,yimlen))
    # final_image = np.zeros_like(image)
    for i in range(xpadding, padded_image.shape[0]+xpadding):
        for j in range(ypadding,padded_image.shape[1]+ypadding):
            try:
                final_image[i-xpadding,j-ypadding] = (padded_image[i:i+xklen,j:j+yklen] * kernel).sum()
            except:
                break    
    return final_image

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
