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
from harris_helper import *
import scipy.signal as sci_signal  # see subtelties of usage here: https://stackoverflow.com/q/41613155
import plotly.express as px
from copy import deepcopy


UNKNOWN_FLOW_THRESH = 1e7
def show_flow(filename):
    """
    visualize optical flow map using matplotlib
    :param filename: optical flow file
    :return: None
    """
    flow = read_flow(filename)
    img = flow_to_image(flow)
    plt.imshow(img)
    plt.show()

def read_flow(filename):
    """
    read optical flow from Middlebury .flo file
    :param filename: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print ('Magic number incorrect. Invalid .flo file')
    else:
        w = int(np.fromfile(f, np.int32, count=1)[0])
        h = int(np.fromfile(f, np.int32, count=1)[0])
        #print("Reading %d x %d flo file" % (h, w))
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """

    # find the x component
    u = flow[:, :, 0]

    # find the y component
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.



    # as per documentation, the flow calculation for these points is to be ignored
    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    # calculating extremes to find boundary conditions for normalizing while plotting

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    # maybe to make sure that maxrad is NOT ZERO
    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)



def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """

    # find more details about this color wheel here: https://www.researchgate.net/figure/Color-coding-of-the-flow-vectors-Direction-is-coded-by-hue-length-is-coded-by_fig7_226781293 
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel

      # find more details about this color wheel here: https://www.researchgate.net/figure/Color-coding-of-the-flow-vectors-Direction-is-coded-by-hue-length-is-coded-by_fig7_226781293 
      
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

##############################################################################


def fetch_image(img_path, is_colored=False):
    image = cv2.imread(img_path)
        # convert the input image into
        # grayscale color space
    # print(files[0])
    if is_colored==False:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def convolve_lk(img, ker):
    h,w=img.shape
    # print("kernel is ",)
    assert(ker.shape==(2,2))
    ans=np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            if i-1>=0 and j-1>=0:
                ans[i][j]= (ker[0][0]*img[i-1][j-1]) + (ker[0][1]*img[i-1][j]) + (ker[1][0]*img[i][j-1]) + (ker[1][1]*img[i][j])
    return ans




def compute_flow(I_xx,I_yy, I_xy, I_xt, I_yt):
    h,w = I_xx.shape
    u=np.zeros((h,w))
    v=np.zeros((h,w))
    eigens=np.zeros((h,w))
    # eigens=[]
    for i in range(h):
        for j in range(w):

            a_t_a= np.array([
                                [I_xx[i][j], I_xy[i][j]], 
                                [I_xy[i][j], I_yy[i][j]]
                            ])
            # a_t_a=a_mat.T.dot(a_mat)
            a_t_b= np.array([
                                [-I_xt[i][j]],
                                [-I_yt[i][j]]
                            ])
            ######################################################
            # checking if A_t_A is singular
            try:
                A_T_A_inv = np.linalg.inv(a_t_a)
                # check if TAU IS MORE THAN SMALLEST EIGENVALUE
                eigen_vals=find_eigenvalues(a_t_a)
                eigen_vals=sorted(eigen_vals)
                # eigens.append(eigen_vals[0])
                # assert(eigen_vals[0]>=0)
                if eigen_vals[0]>=0:
                    ans=np.dot(A_T_A_inv, a_t_b)
                    u[i][j]=ans[0]
                    v[i][j]=ans[1]
                    eigens[i][j]=eigen_vals[0]
                else:
                    # print("negative smallest eigenvalue")
                    eigens[i][j]=0

                    
            except Exception as err:
                eigens[i][j]=0
                # print("err is ", err)
                # print("Matrix is singular : \n", a_t_a)
                # print("Det is : \n", np.linalg.det(a_t_a))
                pass
            #################################
    return u, v, eigens

def fetch_sobel_dict():
    with open("sobel_choices.json",'r') as fd:
        sobel_dict=json.load(fd)
    for curr_key in sobel_dict:
        for key2 in sobel_dict[curr_key]:
            sobel_dict[curr_key][key2]=np.array(sobel_dict[curr_key][key2])
    return sobel_dict


def apply_lk_algo(img_1, img_2, windowSize, sum_mode='avg', recv_sigma_val=-1):

    # normalize pixel values
    img_1=img_1/255
    img_2=img_2/255

    # first, make sure all sizes are fine
    assert(len(windowSize)==2)
    w, h = windowSize[0], windowSize[1]


    '''

    Let pixels in left image be a1, a2 and a3
    Let pixels in right image be b1, b2 and b3

    Change in val at middle pixel (I_t) = b2-a2

    # now, if we assume 'x' to be in positive right direction, then we can say that b2=a1

    Also, theoretically, I_x= a2-a1 = a2-b2 which calls for a [-1,1] X KERNEL
    '''

    sobel_dict=fetch_sobel_dict()
    # gradient kernel | here, the one direction filter seems to work nicely
    # gradient_ker_x = np.array([[-1, 1], [-1, 1]])
    # gradient_ker_y = np.array([[-1, -1], [1, 1]])

    # gradient_ker_x=np.array([[-1,0,1]])
    # gradient_ker_y=np.array([[-1],[0],[1]])


    # Now, we need to find the gradient at each point
    I_x=convolve_2d(img_1, sobel_dict['3x3']['x'])
    I_y=convolve_2d(img_1, sobel_dict['3x3']['y'])
    # I_x=sci_signal.convolve2d(img_1, gradient_ker_x, boundary='symm', mode='same')
    # I_y=sci_signal.convolve2d(img_1,gradient_ker_y, boundary='symm', mode='same')


    # Find I_t (2 ways, either a single pixelwise difference or weightsum of pixels in a window)
    I_t=img_2-img_1
    # print("I_x is \n", I_x)
    # print("I_y is \n", I_y)
    # print("I_t is \n", I_t)


    # Calculate matrices involved in gradient
    I_xx=np.multiply(I_x, I_x)
    I_yy=np.multiply(I_y, I_y)
    I_xy=np.multiply(I_x, I_y)
    I_xt=np.multiply(I_x, I_t)
    I_yt=np.multiply(I_y, I_t)


    # after that, we need to consider all the pixels in a window size
    I_xx=sum_over_window(I_xx, windowSize,mode=sum_mode, sigma_val=recv_sigma_val)
    I_yy=sum_over_window(I_yy, windowSize,mode=sum_mode, sigma_val=recv_sigma_val)
    I_xy=sum_over_window(I_xy, windowSize,mode=sum_mode, sigma_val=recv_sigma_val)
    I_xt=sum_over_window(I_xt, windowSize,mode=sum_mode, sigma_val=recv_sigma_val)
    I_yt=sum_over_window(I_yt, windowSize,mode=sum_mode, sigma_val=recv_sigma_val)
    # TAU_VAL=0.001

    # print("I x x is \n", I_xx)
    # print("I y y is \n", I_yy)
    # print("I x y is \n", I_xy)

    u, v, eigens = compute_flow(I_xx,I_yy, I_xy, I_xt, I_yt )
    return u,v, eigens


def testing_kernel_modes():
    a=np.array([[1,2,3],[40,50,60],[7,8,9]])
    ker=np.array([[-1,1],[-2,2]])
    ker_x=np.array([[-1,0,1]])
    ker_y=np.array([[-1],[0],[1]])
    print("Shape of kernel x is ", ker_x.shape)
    ans=convolve_2d(a, ker_y)

    print(a)
    # symmetrical will automatically lead to ZEROS AT THE END
    # ans=sci_signal.convolve2d(a, ker, mode='same',boundary='symm')
    print(ans)
    

def fetch_quiver_plot(colored_image, u,v):
    h,w=colored_image.shape[0], colored_image.shape[1]
    f = plt.imshow(colored_image)
    ax=f.axes
    # u=u*2
    # v=v*2
    done=False
    # print(u.shape)
    # print(colored_image.shape)
    for i in range(h):
        for j in range(w):
            if u[i][j]==0 and v[i][j]==0:
                continue
            # print(i)
            # print(j)
            # print(f"{i}:{j}:{u[i][j]}:{v[i][j]}")
            ax.arrow(j,i,v[i,j],u[i,j],head_width = 7, head_length = 7, color = (1,0,0))


    plt.show()

def filter_pts(u, v, eigens, threshold=-1, top_k=-1, fraction_thres=-1):
    u=np.array(u)
    v=np.array(v)
    eigens=np.array(eigens)
    cnt=0
    if top_k!=-1:
        cnt+=1
    if threshold!=-1:
        cnt+=1
    if fraction_thres!=-1:
        cnt+=1
    if cnt!=1:
        print("invalid args")
        return None
    h,w=u.shape
    pts=[]

    if threshold!=-1:
        for i in range(h):
            for j in range(w):
                if eigens[i][j]<threshold:
                    eigens[i][j]=0
                    u[i][j]=0
                    v[i][j]=0
                else:
                    pts.append([i,j])
    elif fraction_thres!=-1:
        return filter_pts(u, v, eigens, threshold=eigen_vals.max()*fraction_thres)
    else:
        flat=eigens.flatten()
        flat.sort()
        return filter_pts(u, v, eigens, threshold=flat[-top_k])
    return u,v,eigens,pts




##############################################################################################################

def iterative_optical_flow(img_1, img_2):
    pass

def down_sample_img(img, times=1):
    ans=deepcopy(img)
    for i in range(times):
        ans=cv2.pyrDown(ans)
    return ans


def up_sample_img(img, times=1):
    ans=deepcopy(img)
    for i in range(times):
        ans=cv2.pyrUp(ans)
    return ans


def of_refine(img1, img_2_received, windowSize, u0, v0):
    '''accepts and refines parameters u0 and v0, which corresponds to the initial
        estimates of the optical flow'''

    print("img shape at level is ", img1.shape)

    # shift image 2 by some pixels
    img2=np.zeros(img1.shape)

    img_h, img_w=img1.shape

    for i in range(img_h):
        for j in range(img_w):

            v1,v2=u0[i][j], v0[i][j]
            v1,v2=int(round(v1)), int(round(v2))
            x_want, y_want= i+v1, j+v2

            ######################
            x_want=max(x_want,0)
            x_want=min(x_want,img_h-1)
            ####################
            ######################
            y_want=max(y_want,0)
            y_want=min(y_want,img_w-1)
            ####################

            img2=img_2_received[x_want][y_want]

    better_u_vals, better_v_vals, eigens=apply_lk_algo(deepcopy(img1), deepcopy(img2), windowSize)
    return better_u_vals, better_v_vals, eigens
    

def fetch_best_estimate(img1, img2, windowSize, curr_level):
    '''Return best estimate so far'''

    print("currently trying to find best estimate at level ", curr_level)

    if curr_level!=0:
        u_now,v_now , earlier_eigens=fetch_best_estimate(    down_sample_img(img1),
                                             down_sample_img(img2),
                                             windowSize, 
                                             curr_level-1
                                             )
        u_now=up_sample_img(u_now)
        v_now=up_sample_img(v_now)
    else:
        u_now=np.zeros(img1.shape)
        v_now=np.zeros(img1.shape)
    print("now refining at level:", curr_level)
    
    u_est, v_est, eigens=of_refine(img1, img2, windowSize, u_now, v_now)
    return u_est+u_now, v_est+v_now, eigens
    


def apply_pyramid_lk(img1, img2, windowSize, numLevels):
    '''
    img_1= full image, 
    img_2=full_image
    numLevels: total levels in pyramid
    '''

    img_h, img_w=img1.shape

    u_ans, v_ans, eigens=fetch_best_estimate(img1, img2, windowSize, numLevels)
    return u_ans, v_ans, eigens
    #step 01. Gaussian smooth and scale Img1 and Img2 by a factor of 2ˆ(1 - numLevels).
    # we are at the top step

    #Step 02. Compute the optical flow at this resolution.


    #Step 03. For each level,
        #a. Scale Img1 and Img2 by a factor of 2ˆ(1 - level)
        #b. Upscale the previous layer‘s optical flow by a factor of 2
        #c. Compute u and v by calling OpticalFlowRefine with the     previous level‘s optical flow

def fetch_normalized_matrix(A, filter_thres=1e3):
    A=np.absolute(A)
    A[A > filter_thres] = 0
    A=np.divide(A, A.max())
    A=np.multiply(A,255)
    return A

def filter_extremas(A, filter_thres=1e4):
    A[A > filter_thres]=0
    A[A < -filter_thres]=0
    return A

def fetch_overall_quiver_plot(colored_image,u,v,SKIP_GAP=10):
    u=filter_extremas(u)
    v=filter_extremas(v)
    x_pos=[]
    y_pos=[]
    dx=[]
    dy=[]
    h,w=u.shape

    f = plt.imshow(colored_image)
    ax=f.axes
    
    max_mag=0
    # ax.quiver([400],[25],[100],[50],color='red')
    for i in range(0,h,SKIP_GAP):
        for j in range(0,w,SKIP_GAP):
            x_pos.append(j)
            y_pos.append(i)
            dx.append(u[i][j])
            dy.append(-v[i][j])
            max_mag=max(math.sqrt(u[i][j]**2+v[i][j]**2),max_mag)
    color_add=[( min(math.sqrt(dx[idx]**2+dy[idx]**2)/max_mag+0.4,1.0), 0, 0) for idx in range(len(x_pos))]
    # ax.quiver(x_pos, y_pos, dx, dy,color=color_add)
    ax.quiver(x_pos, y_pos, dx, dy,color=color_add)
    
    # plt.show()

