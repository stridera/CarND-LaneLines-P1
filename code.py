#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import math
from pandas import rolling_median

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def hsv(img):
    """ Convert it to hsv for image range selection """
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

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

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def get_median_filtered(signal, threshold=3):
    signal = signal.copy()
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    if median_difference == 0:
        s = 0
    else:
        s = difference / float(median_difference)
    mask = s > threshold
    print (mask)
    print(signal)
    signal[mask] = np.median(signal)
    return signal

def detect_outlier_position_by_fft(signal, threshold_freq=0.1,
                                   frequency_amplitude=.001):
    signal = signal.copy()
    fft_of_signal = np.fft.fft(signal)
    outlier = np.max(signal) if abs(np.max(signal)) > abs(np.min(signal)) else np.min(signal)
    if np.any(np.abs(fft_of_signal[threshold_freq:]) > frequency_amplitude):
        index_of_outlier = np.where(signal == outlier)
        return index_of_outlier[0]
    else:
        return None

slide_length = 5
frame = 0
left_ints_slide = []
right_ints_slide = []
left_slopes_slide = []
right_slopes_slide = []
def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    global frame
    global left_ints_slide
    global right_ints_slide
    global left_slopes_slide
    global right_slopes_slide
    global slide_length

    img_shape_length = .63
    left_ints = []
    right_ints = []
    left_slopes = []
    right_slopes = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            m = ((y2-y1)/(x2-x1))
            intercept = y1 - x1 * m
            # print("{} - {}".format(m, intercept))

            if (m > 0.15 and m < (0.15 * img.shape[1])):
                right_ints.append(intercept)
                right_slopes.append(m)
            elif (m < -0.5 and m > -(0.15 * img.shape[1])):
                left_ints.append(intercept)
                left_slopes.append(m)

    right_average_tmp = np.average(right_slopes)
    right_int_average_tmp = np.average(right_ints)
  
    if (right_average_tmp > 0 and right_int_average_tmp > 0):
        if (len(right_slopes_slide) < slide_length):
            right_slopes_slide.append(right_average_tmp)
            right_ints_slide.append(right_int_average_tmp)
        else:  
            right_slopes_slide[frame % slide_length] = right_average_tmp
            right_ints_slide[frame % slide_length] = right_int_average_tmp

    right_average = np.average(right_slopes_slide)
    right_int_average = np.average(right_ints_slide)

    right_bottom_x = (img.shape[0] - right_int_average) / right_average
    right_top_x = (img.shape[0] * img_shape_length - right_int_average) / right_average

    if (right_bottom_x > 0 and right_top_x < img.shape[1]):
        cv2.line(img, (int(right_bottom_x), img.shape[0]), (int(right_top_x), int(img.shape[0]*img_shape_length)), color, thickness)

    left_average_tmp = np.average(left_slopes)
    left_int_average_tmp = np.average(left_ints)
    print("{} - {}".format(left_average_tmp, left_int_average_tmp))
    if (left_average_tmp < 0 and left_int_average_tmp > 0):
        if (len(left_slopes_slide) < slide_length):
            left_slopes_slide.append(left_average_tmp)
            left_ints_slide.append(left_int_average_tmp)
        else:  
            left_slopes_slide[frame % slide_length] = left_average_tmp
            left_ints_slide[frame % slide_length] = left_int_average_tmp

    left_average = np.average(left_slopes_slide)
    left_int_average = np.average(left_ints_slide)

    left_top_x = (img.shape[0] - left_int_average) / left_average
    left_bottom_x = ((img.shape[0] * img_shape_length) - left_int_average) / left_average
    
    if (left_bottom_x > 0 and left_top_x < img.shape[1]):
        cv2.line(img, (int(left_top_x), img.shape[0]), (int(left_bottom_x), int(img.shape[0]*img_shape_length)), [0,255,255], thickness)

    frame = frame + 1

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def my_canny(image):
    """
    Blur the image slightly and then do canny edge detection.
    """
    kernel_size = 5
    blurred_img = gaussian_blur(image,kernel_size)
    
    canny_low_threshold = 50
    canny_high_threshold = 150
    return canny(blurred_img,canny_low_threshold,canny_high_threshold)   

def my_region_of_interest(image):
    """
    Mask out the bottom left, near middle left, near middle right, bottom right of the image.
    """
    x = image.shape[1]
    y = image.shape[0]
    vertices = np.array([[(0,y),(x*.475, y*.575), (x*.525, y*.575), (x,y)]], dtype=np.int32)
    return region_of_interest(image, vertices)
    
def my_hough_lines(image):
    """
    Finally, lets do hough lines to get the lane lines
    """
    hough_rho = 3
    hough_theta = np.pi/150
    hough_threshold = 30
    hough_min_line_length = 70
    hough_max_line_gap = 155
    return hough_lines(image,hough_rho,hough_theta,hough_threshold,hough_min_line_length,hough_max_line_gap)
    
def process_image(image):
    grayscale_image = grayscale(image)
    subdued_gray = (grayscale_image / 2).astype('uint8')

    hsv_image = hsv(image)

    """
     First get a mask of Yellow or White in the images, bitwise_and them together.
     http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#how-to-find-hsv-values-to-track
    """ 
    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([35,255,255])
    lower_white = np.array([0,0,230])
    upper_white = np.array([255,255,255])
    white_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    yellow_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    yellow_white_mask = cv2.bitwise_and(white_mask, yellow_mask)
    masked_image = cv2.bitwise_or(subdued_gray, yellow_white_mask)
    
    canny_img = my_canny(masked_image)
    
    cropped_image = my_region_of_interest(canny_img)

    hough_img = my_hough_lines(cropped_image)
    
    """
    Last but not least, lets mark the lines on the original image and return it. 
    """
    return weighted_img(hough_img,image)

# challenge_output = 'extra.mp4'
# clip2 = VideoFileClip('challenge.mp4')
# challenge_clip = clip2.fl_image(process_image)
# challenge_clip.write_videofile(challenge_output, audio=False)
white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)