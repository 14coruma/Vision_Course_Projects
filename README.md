# Computer vision course projects (B657)

These are two computer vision projects that I worked on when taking IU's course.
* `omr/` - Optical music recognition
* `stereo/` - Stereo matching

The goal of the optical music recognition (OMR) project is to find and detect music notes on an image.
I primarily worked on the modified Hough transform to detect staves (groups of 5 lines) and distance between
the lines. I also managed the project and its organziation.

The stereo matching code has two parts. `part1/` was done by a teammate (performs homography to map/warp one image
onto another). I completed `part2/`, which performs basic stereo matching. Given two stereo images, assume
they are representing the same scene but with the camera shifted horizontally. Then we try to match pixels between
the images to compute a depth-map. It would be better to use MRFs, but this naive approach is a good first step.