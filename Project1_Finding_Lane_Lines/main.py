from find_lane import process_image
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip


if __name__ == "__main__":
    
    # Output a video
    white_output = 'test_videos_output/solidWhiteRight.mp4'
    clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
    
    # Output an image
    img = mpimg.imread("test_images/solidYellowLeft.jpg")
    img = process_image(img)
    mpimg.imsave("test_images_output/solidYellowLeft.jpg", img)

   
