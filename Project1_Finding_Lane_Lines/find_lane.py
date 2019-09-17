import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, y_upper, y_bottom, color=[255, 0, 0], thickness=6):
    
    x1_right_sum = 0
    x2_right_sum = 0
    x1_left_sum = 0
    x2_left_sum = 0
    right_num = 0
    left_num = 0
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            
            # slope
            m = (y2 - y1) / (x2 - x1)
            
            # If the slope is positive, then it's the right lane, otherwise, it's the left lane (here I only accept the slope whose absolute value is in the range of 0.5 to 0.9 to ensure the lines found are road lanes. This step is crucial for the challenge video)
            if 0.9 > m > 0.5:
                
                # Extrapolate the line to the upper bound and lower bound in y-direction
                x_bottom_r = x1 + (y_bottom - y1) / m
                x_upper_r = x1 + (y_upper - y1) / m
                
                # Sum the upper x- and bottom x-positions
                x1_right_sum += x_bottom_r
                x2_right_sum += x_upper_r
                right_num += 1
            elif -0.9 < m < -0.5:
                
                # Extrapolate the line to the upper bound and lower bound in y-direction
                x_bottom_l = x1 + (y_bottom - y1) / m
                x_upper_l = x1 + (y_upper - y1) / m
                
                # Sum the upper x- and bottom x-positions
                x1_left_sum += x_bottom_l
                x2_left_sum += x_upper_l
                left_num += 1

    # Calculate the average of the x-positions (x1 is the bottom vertex and x2 is the upper vertex)
    if right_num > 0:
        x1_right = math.floor(x1_right_sum / right_num)
        x2_right = math.floor(x2_right_sum / right_num)
    
        # Draw right lane
        cv2.line(img, (x1_right, y_bottom), (x2_right, y_upper), color, thickness)
    
    if left_num > 0:
        x1_left = math.floor(x1_left_sum / left_num)
        x2_left = math.floor(x2_left_sum / left_num)
        
        # Draw left lane
        cv2.line(img, (x1_left, y_bottom), (x2_left, y_upper), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
        `img` should be the output of a Canny transform.
        
        Returns an image with hough lines drawn.
        """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, math.floor(0.6 * line_img.shape[0]), line_img.shape[0])
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

def process_image(image):
    # TODO: Build your pipeline that will draw lane lines on the test_images
    # then save them to the test_images_output directory.
    
    # Read in and grayscale the image
    gray = grayscale(image)
    
    img_height = image.shape[0]
    img_width = image.shape[1]
    
    # Apply Gaussian smoothing on the gray-scaled image
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)
    
    # Apply Canny edge detection
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    
    # Define the polygon to keep the intereted region for line detection
    vertices = np.array([[(0.05 * img_width,img_height),
                          (0.45 * img_width, 0.6 * img_height),
                          (0.55 * img_width, 0.6 * img_height),
                          (0.95 * img_width, img_height)]],
                        dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    
    # Define the input arguments for the hough Transform
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 30    # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20 #minimum number of pixels making up a line
    max_line_gap = 5    # maximum gap in pixels between connectable line segments
    
    # Plot lines
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    lines_edges = weighted_img(line_image, image, α=0.8, β=1., γ=0.)
    plt.imshow(lines_edges)

    return lines_edges

