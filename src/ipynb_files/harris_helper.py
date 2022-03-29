import numpy as np
import math
import os
import sys
import cv2
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
from scipy import linalg
import json
import glob
import scipy.signal as sci_signal  # see subtelties of usage here: https://stackoverflow.com/q/41613155


def convolve_2d(img_mat, ker_mat):

    '''
    Performs correlation (instead of the misterm convolution as I am not flipping the kernel here)
    Expects odd size across all dimensions of the kernel
    '''
    img_h, img_w=img_mat.shape
    ker_h, ker_w=ker_mat.shape

    assert(ker_h%2==1)
    assert(ker_w%2==1)
    ans=np.zeros((img_h, img_w))

    # assert(ker_h==ker_w)

    gap_h=ker_h//2
    gap_w=ker_w//2
    for i in range(img_h):
        for j in range(img_w):
            if (i-gap_h>=0) and (i+gap_h<img_h) and (j-gap_w>=0) and (j+gap_w<img_w):
                for idx1, dx in enumerate(range(-gap_h, gap_h+1)):
                    for idx2, dy in enumerate(range(-gap_w, gap_w+1)):
                        ans[i][j]+=(img_mat[i+dx][j+dy])*(ker_mat[idx1][idx2])
    return ans

def apply_gaussian_blur(A_mat, window_size, sigma_val=1):
    h, w=A_mat.shape
    assert(window_size%2==1)

    window_dz=window_size//2
    ans_mat=np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            for dx in range(-window_dz, window_dz+1):
                for dy in range(-window_dz, window_dz+1):
                    x=i+dx
                    y=j+dy
                    if x>=0 and x<h and y>=0 and y<w:
                        add=A_mat[x][y]
                        add*=(     (math.exp(     -(dx**2+dy**2)/ (2*(sigma_val**2))   )))    /    (2*math.pi*(sigma_val**2))
                        ans_mat[i][j]+=add
    return ans_mat


def sum_over_window(A, window_size, mode='avg', sigma_val=1):
    assert(mode in ['avg', 'gaussian'])
    window_h, window_w=window_size[0], window_size[1]
    h, w, = A.shape

    assert((window_h%2==1) and (window_w%2==1))
    gap_h=window_h//2
    gap_w=window_w//2

    ans=np.zeros((h,w))
    if mode=='avg':
        sum_ker=np.ones((window_h,window_w))
        # sum_ker=np.divide(sum_ker , window_h*window_w)
    else:
        sum_ker=np.zeros((window_h,window_w))
        for dx in range(-gap_h, gap_h+1):
            for dy in range(-gap_w, gap_w+1):
                sum_ker[dx+gap_h][dy+gap_w]=(     (math.exp(     -(dx**2+dy**2)/ (2*(sigma_val**2))   )))    /    (2*math.pi*(sigma_val**2))
        # print("sum ker is \n", sum_ker)



    for i in range(gap_h, h-gap_h):
        for j in range(gap_w, w-gap_w):
            if (i-gap_h>=0) and (i+gap_h<h) and (j-gap_w>=0) and (j+gap_w<w):
                ans[i][j]=np.sum(np.multiply(A[i-gap_h:i+gap_h+1, j-gap_w:j+gap_w+1], sum_ker))
            else:
                assert(False)
    return ans

       
def find_eigenvalues(A_mat):
    # print("The function you are using is for SYMMETRIC MATRICES ONLY, DO NOT GENERALIZE")
    return LA.eigvals(A_mat)
    # pass

def compute_harris_val(I_xx, I_yy, I_xy,algo_used="harris", k_val=0.04):
    h, w = I_xx.shape
    ans_mat=np.zeros((h,w))
    tmp_mat=np.zeros((2,2))
    for i in range(h):
        for j in range(w):
            # tmp_mat=np.zeros((2,2))
            if algo_used=='harris':
                xx=I_xx[i][j]
                yy=I_yy[i][j]
                xy=I_xy[i][j]
                # print(xx)
                det_val=xx*yy-(xy**2)
                # print(det_val)
                trace_val=(xx+yy)
                # print(f"{i}:{j} = ",trace_val)
                pixel_val=det_val-k_val*(trace_val**2)
                ans_mat[i][j]=pixel_val
            else:
                tmp_mat[0][0]=I_xx[i][j]
                tmp_mat[1][1]=I_yy[i][j]
                tmp_mat[0][1]=I_xy[i][j]
                tmp_mat[1][0]=I_xy[i][j]

                eigenvals=find_eigenvalues(tmp_mat)
                # make sure all eigen vals are positive
                eigenvals=sorted(eigenvals)
                # print("eigenvals are ", eigenvals )
                assert(eigenvals[0]>=0)
                ans_mat[i][j]=eigenvals[0]
                
    return ans_mat


def find_pixel_winners(A_mat, window_size, eligibility_threshold):
    h, w=A_mat.shape
    assert(window_size%2==1)
    ans=np.zeros((h,w))
    window_dz=window_size//2
    pixel_coordinates=[]

    for i in range(h):
        for j in range(w):
            curr_val=A_mat[i][j]
            # print(curr_val)
            if curr_val<eligibility_threshold:
                # print('threshold not crossed')
                continue
            is_max=True
            for dx in range(-window_dz, window_dz+1):
                for dy in range(-window_dz, window_dz+1):
                    x=i+dx
                    y=j+dy
                    if x>=0 and x<h and y>=0 and y<w:
                        if A_mat[x][y]>curr_val:
                            is_max=False
                            # print("LOST")
                            break
            if is_max:
                pixel_coordinates.append([i,j])
                ans[i][j]=1
    return ans, pixel_coordinates


def perform_suppression(A_mat, window_size):
    '''
    This function takes a matrix which contains the value of the Harris function for each pixel.
    It then performs non-maximal suppression on a square window

    '''

    h, w=A_mat.shape
    assert(window_size%2==1)
    ans=np.array(A_mat)
    window_dz=window_size//2
    pixel_coordinates=[]

    for i in range(h):
        for j in range(w):
            curr_val=A_mat[i][j]
            # print(curr_val)
            # if curr_val<eligibility_threshold:
            #     # print('threshold not crossed')
            #     continue
            is_max=True
            for dx in range(-window_dz, window_dz+1):
                for dy in range(-window_dz, window_dz+1):
                    x=i+dx
                    y=j+dy
                    if x>=0 and x<h and y>=0 and y<w:
                        if A_mat[x][y]>curr_val:
                            is_max=False
                            # print("LOST")
                            break
            if is_max:
                # pixel_coordinates.append([i,j])
                # ans[i][j]=1
                pass
            else:
                ans[i][j]=0
    return ans

def find_corners(img_array, kernel_x, kernel_y, do_gaussian=False, algo="harris",  sigma_val=1, k_val=0.04, window_size_for_aggregating=3, window_size_for_suppression=11):
    '''

    Info: OpenCV implementation can be found here: https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga4055896d9ef77dd3cacf2c5f60e13f1c 

    args:
    img_array: needs to receive a grayscale image matrix ie h x w
    kernel_x= 2D matrix, kernel_y=2 D matrix : used for computing gradients at a pixel
    do_gaussian: True if gaussian weights need to be applied to neighbouring pixels, if False, simple average is taken
    window_size_for_aggregating=window over which gaussian blurring is done
    sigma_val=used only if do_gaussian=True
    k_val = paramter used in the Harris function, unused in case of Shi-Tomasi
    fraction_threshold: determines the cut-off for being a corner and is based on the best obtained score

    '''
    print("window for suppression is ", window_size_for_suppression)
    # cmu slide 42: http://www.cs.cmu.edu/~16385/s17/Slides/6.2_Harris_Corner_Detector.pdf 
    # assert(not do_gaussian)

    img_h, img_w= img_array.shape

    ######################################
    # Normalize pixel intensity on dividing by 255
    img_array=np.divide(img_array, 255)

    ###################################
    print("Shape is ", img_array.shape)
    assert(algo in ['harris', 'shi-tomasi'])

    # find the gradient at each point using the operator in each direction
    I_x_mat=convolve_2d(img_array, kernel_x)
    I_y_mat=convolve_2d(img_array, kernel_y)

    print("Convolving done")

    # Find the initial gradient matrices
    I_xx=np.square(I_x_mat)
    I_yy=np.square(I_y_mat)
    I_xy=np.multiply(I_x_mat, I_y_mat)

    ###################################################################################
    # return I_xx, I_yy, I_xy
    if do_gaussian:
        # This has been suggested in the Harris Corner paper
        print("Applying gaussian")
        I_xx=sum_over_window(I_xx,[window_size_for_aggregating,window_size_for_aggregating],mode='gaussian',  sigma_val=sigma_val)
        I_yy=sum_over_window(I_yy,[window_size_for_aggregating,window_size_for_aggregating],mode='gaussian',  sigma_val=sigma_val)
        I_xy=sum_over_window(I_xy,[window_size_for_aggregating,window_size_for_aggregating],mode='gaussian',  sigma_val=sigma_val)
        print("Gaussian done")
    else:
        print("Doing average")
        I_xx=sum_over_window(I_xx,[window_size_for_aggregating,window_size_for_aggregating],mode='avg')
        I_yy=sum_over_window(I_yy,[window_size_for_aggregating,window_size_for_aggregating],mode='avg')
        I_xy=sum_over_window(I_xy,[window_size_for_aggregating,window_size_for_aggregating],mode='avg')
        print("Average done")

    ##################################################################

    ## Now, we must find a harris value for each pixel
    harris_mat=compute_harris_val(I_xx,I_yy, I_xy, algo , k_val)
    print("Harris value found")

    # make a function to perform suppression
    harris_mat=perform_suppression(harris_mat, window_size_for_suppression)
    print("Suppression done")
    return harris_mat

    
def plot_normalized_version(A):
    min_val=A.min()
    A-=min_val
    A/=A.max()
    A*=255
    return A

def plot_pixel_points(curr_img, coord_details, color_wanted, circle_rad=3):
    # each elem of coord_details should be a pair of <x,y>
    img=curr_img.copy()
    for curr_elem in coord_details:
        img=draw_dots(img, curr_elem[0], curr_elem[1], circle_rad, color_wanted
                     ,-1)
    return img

def fetch_annotated_image(image_mat, harris_mat, gap=10):
    arr=[]
    h,w,c=image_mat.shape
    for i in range(h):
        for j in range(w):
            if harris_mat[i][j]>0:
                arr.append((harris_mat[i][j], (i,j)))
    arr=sorted(arr, key=lambda x:-x[0])
    pts=[x[1] for x in arr[:3*gap]]
    image_mat=plot_pixel_points(image_mat, pts[:gap], (255,0,0))
    image_mat=plot_pixel_points(image_mat, pts[gap:2*gap], (0,255,0))
    image_mat=plot_pixel_points(image_mat, pts[2*gap:3*gap], (0,0,255))
    return image_mat

def fetch_annotated_image_fraction(image_mat, harris_mat, thresholds=[0.7,0.5,0.3,0.1,0.05]):
    COLOR_VALS=[(255,0,0), (255,197,0), (255,212,0),(255,255,0),(231,233,185)]
    arr=[]
    pts=[]
    for i in range(len(thresholds)):
        pts.append([])
    print(pts)
    
    max_val=harris_mat.max()
    h,w,c=image_mat.shape
    for i in range(h):
        for j in range(w):
            if harris_mat[i][j]>0:
                min_idx=-1
                for t_idx, thres in enumerate(thresholds):
                    if harris_mat[i][j]>=thres*max_val:
                        min_idx=t_idx
                        break
                if min_idx!=-1:
                    # print("pts is " , pts)
                    # print("appending to ", min_idx)
                    pts[min_idx].append((i,j))
                    # print("len now is ", len(pts[min_idx]))
    # print(pts)
    for idx, curr_elem in enumerate(pts):
        print(f"len for {COLOR_VALS[idx]} is {len(curr_elem)}")
        image_mat=plot_pixel_points(image_mat, curr_elem, COLOR_VALS[idx])
    return image_mat



def draw_dots(cv_img_obj,centre_x,centre_y, circle_radius, color_tuple, circle_boundary_thinkness ):

    # origin is at leftmost corner of image
    assert(len(color_tuple)==3)
    assert(max(color_tuple)<=255)
    
    img_height, img_width, img_channels = cv_img_obj.shape
    #print("height is ", img_height)
    #print("width is ", img_width)
    #print("cnetre is ", centre_x)
    #print("cnetre y is ", centre_y)
    # print("centre x is ", centre_x)
    # print("img height is ", img_height)
    assert(centre_x<=img_height)
    assert(centre_y<=img_width)
    # cv_img_obj = cv.circle(cv_img_obj, centerOfCircle, radius, color, thickness)
    cv_img_obj = cv2.circle(cv_img_obj, (centre_y,centre_x), radius=circle_radius, color=color_tuple, thickness=circle_boundary_thinkness)
    return cv_img_obj

