# Project 4 - Advanced Lane Finding
[![video_gif](https://media.giphy.com/media/eIm0Bjq0uswdha7CfE/giphy.gif)](https://youtu.be/HHL2RmPcaG0)

## The Project
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view)

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the code cells under **Camera Calibration and Image Undistortion** of the IPython notebook [Project2_Advanced_Lane_Finding.ipynb](/Project2_Advanced_Lane_Finding/Project2_Advanced_Lane_Finding.ipynb).

First of all, I prepared "object points" `obj_p`, which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. To detect the calibration pattern, I used the function `cv2.findChessboardCorners()` to find all the chessboard corners in [camera_cal](). These corners will then be appended to `img_points`, which are 2D points on image, with the (x, y) pixel position, and every time the corners are detected, `obj_p` will be appended to `obj_points`.

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Original Image            |  Undistorted Image
:-------------------------:|:-------------------------:
![alt_text](/Project2_Advanced_Lane_Finding/output_images/chessboard.jpg)  |  ![alt_text](/Project2_Advanced_Lane_Finding/output_images/undist_chessboard.jpg) 

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

After finding the camera calibration and distortion coefficients, we can apply them to the test images using `cv2.undistort()`. The result is as follow:
Original Test Image            |  Undistorted Test Image
:-------------------------:|:-------------------------:
![alt_text](/Project2_Advanced_Lane_Finding/output_images/test_img.jpg)  |  ![alt_text](/Project2_Advanced_Lane_Finding/output_images/undist_test_img.jpg) 

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

This step is contained in the code cells under **Gradients and Color Transform** of the IPython notebook [Project2_Advanced_Lane_Finding.ipynb](/Project2_Advanced_Lane_Finding/Project2_Advanced_Lane_Finding.ipynb).

I used a combination of color and gradient thresholds to generate a binary image. For the gradient threshold, I use the intersection of the gradient magnitude threshold and gradient direction threshold (eliminate the gradient which is close to horizontal direction). For the color threshold, I use the intersection of the "S" channel in the HSL color space and the "R" channel in the RGB color space, which shows a good identification of yellow and white lines. Finally, I generated the binary image with the union of both gradient and color thresholds.

Here's an example of my output for this step. 

![alt text](/Project2_Advanced_Lane_Finding/output_images/binary_test_img.jpg) 

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

This step is contained in the code cells under **Perspective Transform** of the IPython notebook [Project2_Advanced_Lane_Finding.ipynb](/Project2_Advanced_Lane_Finding/Project2_Advanced_Lane_Finding.ipynb).

First, I used trial and error to find the lines which are parallel to the straight road lines in the original undistorted image like the image below:
![alt text](/Project2_Advanced_Lane_Finding/output_images/find_parallel_straight_lines.jpg) 

Then the four vertices (two for each line), will serve as the hardcoded souce points (`src`). The destination points (`dst`) will just be the same size as the original image. The souce points (`src`). The destination points (`dst`) I chose are as follow:
| Source        | Destination   | 
|:-------------:|:-------------:| 
| 0, 720        | 0, 720        | 
| 1280, 720     | 1280, 720     |
| 555, 460      | 0, 0          |
| 725, 460      | 1280, 0       |

After haveing the source and destination points, we can use the `cv2.getPerspectiveTransform()` function to get the transformation matrix. Then the warped image can be generated by using the `cv2.warpPerspective()` function. We can see that the lines appear to be parallel in the warped image

![alt text](/Project2_Advanced_Lane_Finding/output_images/warped_straight_lines.jpg) 

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/HHL2RmPcaG0)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
