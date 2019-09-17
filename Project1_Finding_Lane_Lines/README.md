# Project1 - Finding Lane Lines on the Road
The goal of this project is to use a simple pipeline to detect lane lines from video streams (which is a series of images).

## Pipeline
The pipeline consists of following 6 steps:
1. Convert image to gray scale.
2. Apply Gaussian smoothing on the gray-scaled image
3. Apply Canny edge detection to find lines in the image
4. Construct a polygon to keep the interested region for line detection
5. Apply Hough Transform on the masked image after Canny Transform to create the line image
6. Combine the line image with the original image

This is a very simple pipeline involving only **computer vision** and **manual parameter tuning**.

## Image Output
Following is one example of the lane lines detection on the road.
![test_output](/Project1_Finding_Lane_Lines/test_images_output/solidYellowCurve.jpg) 
