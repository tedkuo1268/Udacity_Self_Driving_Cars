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

Original Image             |  Undistorted Image
:-------------------------:|:-------------------------:
![alt_text](/Project2_Advanced_Lane_Finding/output_images/chessboard.jpg)  |  ![alt_text](/Project2_Advanced_Lane_Finding/output_images/undist_chessboard.jpg) 

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

After finding the camera calibration and distortion coefficients, we can apply them to the test images using `cv2.undistort()`. The result is as follows:

Original Test Image        |  Undistorted Test Image
:-------------------------:|:-------------------------:
![alt_text](/Project2_Advanced_Lane_Finding/output_images/test_img.jpg)  |  ![alt_text](/Project2_Advanced_Lane_Finding/output_images/undist_test_img.jpg) 

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

This step is contained in the code cells under **Gradients and Color Transform** of the IPython notebook [Project2_Advanced_Lane_Finding.ipynb](/Project2_Advanced_Lane_Finding/Project2_Advanced_Lane_Finding.ipynb).

I used a combination of color and gradient thresholds to generate a binary image. For the gradient threshold, I use the intersection of the gradient magnitude threshold and gradient direction threshold (eliminate the gradient which is close to horizontal direction). For the color threshold, I masked yellow line with **H** and **S** channels in HSL color space and masked white line with **L** channel in HSL color space and **R** channel in RGB color space. Finally, I generated the binary image with the union of both gradient and color thresholds.

Here's an example of my output for this step. 

![alt text](/Project2_Advanced_Lane_Finding/output_images/binary_test_img.jpg) 

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

This step is contained in the code cells under **Perspective Transform** of the IPython notebook [Project2_Advanced_Lane_Finding.ipynb](/Project2_Advanced_Lane_Finding/Project2_Advanced_Lane_Finding.ipynb).

First, I used trial and error to find the lines which are parallel to the straight road lines in the original undistorted image like the image below:
![alt text](/Project2_Advanced_Lane_Finding/output_images/find_parallel_straight_lines.jpg) 

Then the four vertices (two for each line), will serve as the hardcoded souce points (`src`). The destination points (`dst`) will just be the same size as the original image. The souce points (`src`) and th destination points (`dst`) I chose are as follows:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 0, 720        | 0, 720        | 
| 1280, 720     | 1280, 720     |
| 555, 460      | 0, 0          |
| 725, 460      | 1280, 0       |

After haveing the source and destination points, we can use the `cv2.getPerspectiveTransform()` function to get the transformation matrix `M`. Then the warped image can be generated by using the `cv2.warpPerspective()` function. We can see that the lines appear to be parallel in the warped image

![alt text](/Project2_Advanced_Lane_Finding/output_images/warped_straight_lines.jpg) 

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

All of steps remaining are contained in the code cells under **Lane Detection** of the IPython notebook [Project2_Advanced_Lane_Finding.ipynb](/Project2_Advanced_Lane_Finding/Project2_Advanced_Lane_Finding.ipynb).

There are two different ways to identify lane-line pixels in a image:

1. **Search by using sliding windows**: Find lines by using sliding windows from the bottom of the image. The starting points of the left and right lines are the highest peaks from the left and right halves of the histogram of the activated binary pixels. Then we slide the search windows toward the top of the image and identify all the lane-line pixels. This method is implemented in `find_lane_pixels_by_sliding_windows()`

2. **Search around the polynomials from the previous frame**: Find lines by searching around the polynomial from the previous frame within a given margin. This method is implemented in `find_lane_pixels_by_searching_around_poly()`

If the frame is brand-new or the lines from previous frame are not detected, we should use sliding windows to search from the bottom to the top of the image, because we don't have any polynomials to search around. However, this is time-consuming, and therefore, if we have the detected polynomials from the previous frame, we can search around them to find lane-line pixels, which would be more efficient.

After the lan-line pixels are detected, I used the `np.polyfit()` function in `fit_polynomial()` to fit the found pixels with a second-order polynomial.

To make the code more organized, I created a class `Line` to keep track of the properties of the line.

```python
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x and y pixel values of the fitted lines
        self.fitx = None
        self.fity = None
        # x values of the last n fits of the line (at the bottom)
        self.recent_xfitted = deque()
        #average x values of the fitted line over the last n iterations (at the bottom)
        self.best_x = None     
        #polynomial coefficients in the last iteration
        self.last_fit_pixel = None  
        self.last_fit_meter = None  
        #polynomial coefficients over the last n iterations
        self.recent_fits_pixel = deque()  
        self.recent_fits_meter = deque()  
        #polynomial coefficients averaged over the last n iterations
        self.best_fit_pixel = None
        self.best_fit_meter = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #x values for detected line pixels
        self.all_x_pixels = None  
        #y values for detected line pixels
        self.all_y_pixels = None  
    
    def update_line(self, poly_fit_pixel, poly_fit_meter, y_eval):
        self.last_fit_pixel = poly_fit_pixel
        self.last_fit_meter = poly_fit_meter
        # Append the polynomial coefficients to the deque
        self.recent_fits_pixel.append(self.last_fit_pixel)
        self.recent_fits_meter.append(self.last_fit_meter)
        # Remove the oldest polynomial coefficients if n is larger than 10
        if len(self.recent_fits_pixel)>10:
            self.recent_fits_pixel.popleft()
            self.recent_fits_meter.popleft()
        # Find the average polynomial coefficients
        self.best_fit_pixel = np.mean(self.recent_fits_pixel, axis=0)
        self.best_fit_meter = np.mean(self.recent_fits_meter, axis=0)
        # Calculate the radius of curvature at the bottom, which is closest to the car position
        self.radius_of_curvature = ((1 + (2*self.best_fit_meter[0]*y_eval*ym_per_pix + self.best_fit_meter[1])**2)**1.5) / np.absolute(2*self.best_fit_meter[0])
```

The method `update_line()` is used to update the detexcted pixels, fitted polynomials, and the radius of curvature, which will be discussed in the next part, of the lines. This line update is implemented in the function `fit_polynomial()` to get the up-to-date informations of the line. 

The result of lane-line pixels finding and polynomials fitting are as follows:

Binary Bird's-eye View     |  Bird's-eye View with Detected Lines and Fitted Polynomials
:-------------------------:|:-------------------------:
![alt_text](/Project2_Advanced_Lane_Finding/output_images/warped_binary_test_img.jpg)  |  ![alt_text](/Project2_Advanced_Lane_Finding/output_images/bird_eye_with_lines.jpg) 

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

1. **Radius of Curvature**: In the `line_update` method in `Line` class, the following line of code does the calculation of the radius of curvature:

```python
self.radius_of_curvature = ((1 + (2*self.best_fit_meter[0]*y_eval*ym_per_pix + self.best_fit_meter[1])**2)**1.5) / np.absolute(2*self.best_fit_meter[0])
```

The parameter `y_eval` is the y-value where we want the radius of curvature. The value I chose here is the maximum y-value which correspond to the position just in front of the car.

The equation of find the radius of curvature can be found[here](https://www.intmath.com/applications-differentiation/8-radius-curvature.php).

2. **Vehicle Offset from the Lane Center**: To obtain the offset value, we can find the middle position of the lane first and then calculate the difference between the middle position of the lane and the middle position of the image, and finally, convert the difference from pixels to meter. This part is implemented in `calculate_offset()` function. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

After doing all the steps above, we can finally transform the lines we found back onto the road image, which is implemented in `draw_onto_road()`. To do this, I used the `cv2.getPerspectiveTransform()` function to get the **inverse** transformation matrix `Minv` by just swapping the source and destination points. Plugging in `Minv` and the lane in bird's-eye view to `cv2.warpPerspective()`, we will get an unwarped lan and can be overlaid on the road image. The result is shown in the following picure:

![alt_text](/Project2_Advanced_Lane_Finding/output_images/test_img_output.jpg) 

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/HHL2RmPcaG0)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

If the thresholding of the image fails, it's definitely that the lane detection will not succeed because we cannot identify the correct lines from the binary image. After some testing, this pipeline is likely to fail when there is an extreme condition, such as too bright or too dark, in the image. To solve this problem, techniques other than thresholding must be used. One example is convolutional neural network (CNN), and we can use it to train a more robust model for lane detection.


