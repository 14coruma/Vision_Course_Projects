# Assignment 1
Andrew Corum, Josep Han, Kenneth Zhang, Zheng Chen
#### 03/29/2021

 <!-- Your report should explain how to run your code
and any design decisions or other assumptions you made -->

To run the code: in the command line, run: \newline
```python3 omr.py ./images/music1.png```
### Design Decisions and Assumptions
As the assignment states, we assume that the music lines from the image are parallel and straight. This is in order to implement the Hough Transform for line detection. We also assumed that every lines are in treble clef. This is because the assignment did not provide a template model for treble or bass clefs. 

When applying convolutions, we assumed that the padding of the image should repeat the values found at the edges. This prevented the darkened halo effect on both our template images as well as the music sheets when applying Gaussians or Sobel kernels.

When detecting the staff lines, we decided to implement a naive thresholding, where if the pixel value is less than 78% (100% being white), then we convert that pixel to 0. This significantly helped the Hough Transform algorithm as there were less ambiguous pixels when applying the Sobel operator and non-maximal suppression.

### Performance of algorithms
#### Convolution
We have tested with Sobel operators and Gaussians, and the results looked identical to code packages from CV2. Since we made it compatible with separable 1D kernels, it is safe to assume that the runtime performance should be improved over 2D kernels.


### Template Matching
The Hamming Distance calculation was not hard to implement, since it checks every pixel of the template with the given area of the image, much like convolution. However, we noticed that this would result in a less accurate note detection, where things like the circle in the treble clef would count as a note. 

The alternate approach suggested we take the edge maps of the image and templates before calculating the template matching values in a slightly different way. We found that this implementation is much slower compared to the Hamming distance method. 

### Future Work
The first thing we could look at to improve is to provide a way to detect the clefs of each staff. This would make the note detection more consistent to the true values. 