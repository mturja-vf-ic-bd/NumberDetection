# Handwritten Number Recognition

A simple handwriting recognition projects to recognize handwritten numbers.

## Input

Input of this program is an image - any image with digits in it. An example of such image is given below.
![Problem Loading Image](testImages/ex4.jpg?raw=true "Sample Input")

Limitations:
1. Does not work if the intensities of the digit pixels are greater than the background pixels.
2. Does not work if the digits are broken into multiple pieces.

##Output

Output of this program is an image that has bounding boxes around the original digits and a label describing the actual digit associated with it.
The sample output for the image in the Input section is given below.
![Problem Loading Image](output.jpg?raw=true "Sample Input")