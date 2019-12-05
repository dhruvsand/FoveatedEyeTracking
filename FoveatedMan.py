import cv2 # Import relevant libraries
import numpy as np
from pymouse import PyMouse


if __name__ == '__main__':
    # img = cv2.imread('Landscape.jpg', 0) # Read in image
    img1 = cv2.imread('LandscapeGrey.jpg',0) # Read in image
    img2 = cv2.imread('LandscapeGrey2.jpg',0) # Read in image
    dst2 = cv2.resize(img2, None, fx=2, fy=2)

    img4 = cv2.imread('LandscapeGrey4.jpg',0) # Read in image
    dst4 = cv2.resize(img4, None, fx=4, fy=4)

    img8 = cv2.imread('LandscapeGrey8.jpg',0) # Read in image
    dst8 = cv2.resize(img8, None, fx=8, fy=8)

    img16 = cv2.imread('LandscapeGrey16.jpg',0) # Read in image

    m = PyMouse()
    x_dim, y_dim = m.screen_size()
    # img16 = cv2.resize(img, None, fx=0.0625, fy=0.0625)
    #
    # cv2.imwrite("LandscapeGrey16.jpg",img16)


    height = img1.shape[0] # Get the dimensions
    width = img1.shape[1]

    # Define mask
    while True:
        mask = np.ones(img1.shape, dtype='uint8')
        dst = cv2.resize(img16,None,fx=16,fy=16)


        mouse_x,mouse_y = m.position()
        # Draw circle at x = 100, y = 70 of radius 25 and fill this in with 0
        cv2.circle(dst, (int(mouse_x*width/x_dim), int(mouse_y*height/y_dim)), 1000, 8, -1)
        cv2.circle(dst, (int(mouse_x*width/x_dim), int(mouse_y*height/y_dim)), 500, 4, -1)
        cv2.circle(dst, (int(mouse_x*width/x_dim), int(mouse_y*height/y_dim)), 200, 2, -1)
        cv2.circle(dst, (int(mouse_x*width/x_dim), int(mouse_y*height/y_dim)), 50, 1, -1)

        dst[dst == 8] = dst8[dst == 8]
        dst[dst == 4] = dst4[dst == 4]
        dst[dst == 2] = dst2[dst == 2]
        dst[dst == 1] = img1[dst == 1]





        cv2.imshow('image',dst)
        cv2.waitKey(1)

    # Apply distance transform to mask
    # out = cv2.distanceTransform(mask, cv2.DIST_L2, 3)/np.sqrt(height*height+width*width)
    # out = cv2.distanceTransform(mask, cv2.DIST_L1, 3)/ (width+height)
    #
    # for column in out:
    #     for ix, num in enumerate(column):
    #         if num == 0 :
    #             continue
    #         elif num < 0.2:
    #             column[ix] = 20
    #         elif num < 0.4:
    #             column[ix] = 40
    #         elif num < 0.6:
    #             column[ix] = 80
    #         else:
    #             column[ix] = 160
    #

    # Define scale factor
    # scale_factor = 10
    #
    # # Create output image that is the same as the original
    # filtered = img.copy()
    #
    # # Create floating point copy for precision
    # img_float = img.copy().astype('float')
    #
    # # Number of channels
    # if len(img_float.shape) == 3:
    #   num_chan = img_float.shape[2]
    # else:
    #   # If there is a single channel, make the images 3D with a singleton
    #   # dimension to allow for loop to work properly
    #   num_chan = 1
    #   img_float = img_float[:,:,None]
    #   filtered = filtered[:,:,None]
    #
    # # For each pixel in the input...
    # for y in range(height):
    #   for x in range(width):
    #
    #     # If distance transform is 0, skip
    #     if out[y,x] == 0.0:
    #       continue
    #
    #     # Calculate M = d / S
    #     mask_val = np.ceil(out[y,x] / scale_factor)
    #
    #     # If M is too small, set the mask size to the smallest possible value
    #     if mask_val <= 3:
    #       mask_val = 3
    #
    #     # Get beginning and ending x and y coordinates for neighbourhood
    #     # and ensure they are within bounds
    #     beginx = x-int(mask_val/2)
    #     if beginx < 0:
    #       beginx = 0
    #
    #     beginy = y-int(mask_val/2)
    #     if beginy < 0:
    #       beginy = 0
    #
    #     endx = x+int(mask_val/2)
    #     if endx >= width:
    #       endx = width-1
    #
    #     endy = y+int(mask_val/2)
    #     if endy >= height:
    #       endy = height-1
    #
    #     # Get the coordinates of where we need to grab pixels
    #     xvals = np.arange(beginx, endx+1)
    #     yvals = np.arange(beginy, endy+1)
    #     (col_neigh,row_neigh) = np.meshgrid(xvals, yvals)
    #     col_neigh = col_neigh.astype('int')
    #     row_neigh = row_neigh.astype('int')
    #
    #     # Get the pixels now
    #     # For each channel, do the foveation
    #     for ii in range(num_chan):
    #       chan = img_float[:,:,ii]
    #       pix = chan[row_neigh, col_neigh].ravel()
    #
    #       # Calculate the average and set it to be the output
    #       filtered[y,x,ii] = int(np.mean(pix))
    #
    # # Remove singleton dimension if required for display and saving
    # if num_chan == 1:
    #   filtered = filtered[:,:,0]

    # Show the image
    # cv2.imshow('Output', filtered)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

