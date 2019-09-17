from find_lane import process_image
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip


if __name__ == "__main__":
    
    # Output a video
    white_output = 'test_videos_output/solidWhiteRight.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
    
    # Output an image
    img = mpimg.imread("test_images/solidYellowLeft.jpg")
    img = process_image(img)
    mpimg.imsave("test_images_output/solidYellowLeft.jpg", img)

   
