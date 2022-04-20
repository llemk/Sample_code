import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate


def MatchSIFT(loc1, des1, loc2, des2):
    """
    Find the matches of SIFT features between two images
    
    Parameters
    ----------
    loc1 : ndarray of shape (n1, 2)
        Keypoint locations in image 1
    des1 : ndarray of shape (n1, 128)
        SIFT descriptors of the keypoints image 1
    loc2 : ndarray of shape (n2, 2)
        Keypoint locations in image 2
    des2 : ndarray of shape (n2, 128)
        SIFT descriptors of the keypoints image 2

    Returns
    -------
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    """

    # search the target image for nearest neighbors of the template image.
    n1 = NearestNeighbors(n_neighbors=2).fit(des2)
    dist1, ind1 = n1.kneighbors(des1)

    # initialize empty list for keypoint index
    index_filtered = []
    # for all matched keypoints
    for ii in range(len(dist1) - 1):
        nearest_feat = dist1[ii]
        feat_index = ind1[ii]
        ratio = nearest_feat[0] / nearest_feat[1]
        # if the two nearest distances pass the ratio test then add the keypoint indices to the filtered list
        if ratio < .7:
            index_filtered.append([ii, feat_index[0]])

        # find the matches from image 2 to image 1
        # search the template image for nearest neighbors of the target image.
    n2 = NearestNeighbors(n_neighbors=2).fit(des1)
    dist2, ind2 = n2.kneighbors(des2)

    # initialize empty list for keypoint index
    index_filtered2 = []
    # for all matched keypoints
    for ii in range(len(dist2) - 1):
        nearest_feat = dist2[ii]
        feat_index = ind2[ii]
        ratio = nearest_feat[0] / nearest_feat[1]
        # if the two nearest distances pass the ratio test then add the keypoint indices to the filtered list
        if ratio < .7:
            index_filtered2.append([ii, feat_index[0]])

    # do bidirectional matching between image 1 and image 2
    # initialize temporary variables to hold matching keypoints
    temp1 = []
    temp2 = []

    # for each keypoint pair from image 1 to image 2 that passed the filter, seperate the keypoints into template and target lists
    for ii in range(len(index_filtered) - 1):
        # get the matching pair
        pair_index = index_filtered[ii]
        # separate the template and target index
        template_index = pair_index[0]
        target_index = pair_index[1]
        # add the keypoint to the temporary variables
        template_keypoint = loc1[template_index]
        target_keypoint = loc2[target_index]
        temp1.append(template_keypoint.pt)
        temp2.append(target_keypoint.pt)

    # initialize temporary variables to hold matching keypoints from image 2 to image 1
    temp3 = []
    temp4 = []
    # for each keypoint pair from image 2 to image 1 that passed the filter, separaate the keypoints into templat and target lists
    for ii in range(len(index_filtered2) - 1):
        # get the matching pair
        pair_index = index_filtered2[ii]
        # separate the template and target index
        template_index = pair_index[1]
        target_index = pair_index[0]
        # add the keypoint to the temporary variables
        template_keypoint = loc1[template_index]
        target_keypoint = loc2[target_index]
        temp3.append(template_keypoint.pt)
        temp4.append(target_keypoint.pt)

    # initialize output arrays
    x1 = []
    x2 = []

    # combine remaining points
    i1i2 = np.concatenate((np.asarray(temp1), np.asarray(temp2)), axis=1)
    i2i1 = np.concatenate((np.asarray(temp3), np.asarray(temp4)), axis=1)
    # get rid of duplicate matches
    i1i2 = np.unique(i1i2, axis=0)
    i2i1 = np.unique(i2i1, axis=0)

    # for each pair matching from image 1 to image 2
    for ii in range(len(i1i2)):
        querry = str(i1i2[ii])
        # for each pair matching from image 2 to image 1
        for jj in range(len(i2i1)):
            # check that the pair is also in the other list
            if querry == str(i2i1[jj]):
                # put the pair into the output variable
                x1.append(i1i2[ii][0:2])
                x2.append(i1i2[ii][2:4])

    # set the output to be an array
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)

    return x1, x2


def EstimateH(x1, x2, ransac_n_iter, ransac_thr):
    """
    Estimate the homography between images using RANSAC
    
    Parameters
    ----------
    x1 : ndarray of shape (n, 2)
        Matched keypoint locations in image 1
    x2 : ndarray of shape (n, 2)
        Matched keypoint locations in image 2
    ransac_n_iter : int
        Number of RANSAC iterations
    ransac_thr : float
        Error threshold for RANSAC

    Returns
    -------
    H : ndarray of shape (3, 3)
        The estimated homography
    inlier : ndarray of shape (k,)
        The inlier indices
    """

    # initialize the largest inlier variable
    largest_inlier = 0

    # for each ransac iteration
    for ii in range(ransac_n_iter):
        # randomly select 4 points
        rand_select = np.random.choice(len(x1), 4, replace=False)

        # get the coordinates of the 4 random points
        points1 = x1[rand_select]
        points2 = x2[rand_select]

        # construct the A matrix for homography computation
        a = np.asarray([
            [points1[0, 0], points1[0, 1], 1, 0, 0, 0, -points1[0, 0] * points2[0, 0], -points1[0, 1] * points2[0, 0]],
            [0, 0, 0, points1[0, 1], points1[0, 0], 1, -points1[0, 0] * points2[0, 1], -points1[0, 1] * points2[0, 1]],
            [points1[1, 0], points1[1, 1], 1, 0, 0, 0, -points1[1, 0] * points2[1, 0], -points1[1, 1] * points2[1, 0]],
            [0, 0, 0, points1[1, 1], points1[1, 0], 1, -points1[1, 0] * points2[1, 1], -points1[1, 1] * points2[1, 1]],
            [points1[2, 0], points1[2, 1], 1, 0, 0, 0, -points1[2, 0] * points2[2, 0], -points1[2, 1] * points2[2, 0]],
            [0, 0, 0, points1[2, 1], points1[2, 0], 1, -points1[2, 0] * points2[2, 1], -points1[2, 1] * points2[2, 1]],
            [points1[3, 0], points1[3, 1], 1, 0, 0, 0, -points1[3, 0] * points2[3, 0], -points1[3, 1] * points2[3, 0]],
            [0, 0, 0, points1[3, 1], points1[3, 0], 1, -points1[3, 0] * points2[3, 1], -points1[3, 1] * points2[3, 1]]])

        # construct the b vector for ax=b
        b = points2.reshape((8,1))
        # calculate x from ax=b
        x = np.linalg.inv(a.T @ a) @ a.T @ b
        # reshape x into the homography matrix
        h = np.append(x, 1).reshape((3,3))

        # initialize inlier tracking variables
        inlier = 0
        inlier_ind = []

        # for each sift match
        for jj in range(len(x1)):
            # extract the coordinates of a keypoint in img 1 and append 1 at the end
            point1 = np.append(x1[jj], 1).reshape((3,1))
            # estimate the coordinates of the keypoint in img 2
            estimate = h@point1
            # scale the estimate to have a 1 in the last position
            estimate = estimate / estimate[-1]
            # get the ground truth coordinates of the keypoint in img 2
            truth = x2[jj]
            # calculate the error between the estimated location and ground truth location of the keypoint in img 2
            error = ((truth[0]-estimate[0,0])**2 + (truth[1]-estimate[1,0])**2)**0.5
            # if the error is below the ransac threshold
            if error < ransac_thr:
                # increment the inlier counter
                inlier = inlier + 1
                # append the inlier index to the list
                inlier_ind.append(jj)

        # if the number of inliers is greather than the largest inlier
        if inlier > largest_inlier:
            # set the output variable to be the H matrix with largest number of inliers
            h_out = h
            # update the largest inlier variable
            largest_inlier = inlier
            # update the output of inlier indices
            largest_inlier_ind = np.asarray(inlier_ind)

    return h_out, largest_inlier_ind


def EstimateR(H, K):
    """
    Compute the relative rotation matrix
    
    Parameters
    ----------
    H : ndarray of shape (3, 3)
        The estimated homography
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters

    Returns
    -------
    R : ndarray of shape (3, 3)
        The relative rotation matrix from image 1 to image 2
    """

    # estimate R from the homography and K matrices
    R = np.linalg.inv(K) @ H @ K
    #svd cleanup
    u,d,v = np.linalg.svd(R)
    R_rect = u@v.T
    # check_rect = np.linalg.det(R_rect)
    return R_rect


def ConstructCylindricalCoord(Wc, Hc, K):
    """
    Generate 3D points on the cylindrical surface
    
    Parameters
    ----------
    Wc : int
        The width of the canvas
    Hc : int
        The height of the canvas
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters of the source images

    Returns
    -------
    p : ndarray of shape (Hc, Hc, 3)
        The 3D points corresponding to all pixels in the canvas
    """

    # get the focal length from the K matrix
    f = K[0,0]
    # construct the meshgrid
    w = np.linspace(0, 2*math.pi, Wc)
    h = np.linspace(-(Hc-1)//2, (Hc-1)//2, Hc)
    phi, hh = np.meshgrid(w,h)
    # initialize the p matrix
    p = np.zeros((Hc, Wc, 3))

    # fill in the p matrix
    # XYZ coordinate
    p[:,:,0] = f*np.sin(phi)
    p[:,:,1] = hh
    p[:,:,2] = f*np.cos(phi)

    # cylindrical coordinate
    # p[:,:,0]=np.ones((p.shape[0],p.shape[1]))*f
    # p[:,:,1]=phi
    # p[:,:,2]=hh

    return p


def Projection(p, K, R, W, H):
    """
    Project the 3D points to the camera plane
    
    Parameters
    ----------
    p : ndarray of shape (Hc, Wc, 3)
        A set of 3D points that correspond to every pixel in the canvas image
    K : ndarray of shape (3, 3)
        The camera intrinsic parameters
    R : ndarray of shape (3, 3)
        The rotation matrix
    W : int
        The width of the source image
    H : int
        The height of the source image

    Returns
    -------
    u : ndarray of shape (Hc, Wc, 2)
        The 2D projection of the 3D points
    mask : ndarray of shape (Hc, Wc)
        The corresponding binary mask indicating valid pixels
    """

    # initialize u array
    u = np.zeros((p.shape[0],p.shape[1],2))
    # reshape the p matrix into nx3 matrix corresponding to X,Y, and Z coordinates
    flat_x = p.reshape((p.shape[0]*p.shape[1],3))
    # calculate the mapping from P to the image]
    flat_mapping = K @ R @ flat_x.T
    # reshape the mapping to the shape of the canvas
    mapping = flat_mapping.T.reshape((p.shape[0],p.shape[1],3))
    # normalize u and v slices by the third slice
    u_slice = np.divide(mapping[:,:,0], mapping[:,:,2])
    v_slice = np.divide(mapping[:,:,1], mapping[:,:,2])
    # populate the output matrix
    u[:,:,0] = u_slice
    u[:,:,1] = v_slice

    # get the locations where the map is mapped to a location on the image
    mask_u_up = np.less(u_slice, W)
    mask_v_up = np.less(v_slice, H)
    mask_u_low = np.greater_equal(u_slice,0)
    mask_v_low = np.greater_equal(v_slice,0)

    # combine the mask for u and v slices
    mask_u = np.multiply(mask_u_low*1,mask_u_up*1)
    mask_v = np.multiply(mask_v_low*1,mask_v_up*1)
    mask = np.multiply(mask_u,mask_v)

    # Mask the points behind the camera
    # calculate theta from the rotation matrix
    theta = math.atan(R[2,0]/R[2,2])
    # print(theta)
    # calculate the Wc value corresponding to theta
    camera_front = int(theta / (2*math.pi) * p.shape[1])
    # find the Wc value of the right limit of the part of the cylinder in front of the camera. Modulo to wrap around the cylinder
    right_limit = camera_front + (p.shape[1]//4) % p.shape[1]
    # find the Wc value of the left limit of the part of the cylinder in front of the camera
    left_limit = camera_front - (p.shape[1]//4)
    # if the left limit is negative
    # print(right_limit)
    # print(left_limit)
    if left_limit < 0:
        # find the corresponding possitive point on the cylinder
        left_limit = p.shape[1] + left_limit

    # mask the portions of the cylinder that are behind the camera
    if left_limit > right_limit:
        mask[:,right_limit:left_limit] = np.zeros((mask.shape[0],mask.shape[1]//2))
    else:
        mask[:,0:left_limit] = np.zeros((mask.shape[0], left_limit))
        mask[:,right_limit:-1] = np.zeros((mask.shape[0],mask.shape[1]-right_limit-1))

    return u, mask


def WarpImage2Canvas(image_i, u, mask_i):
    """
    Warp the image to the cylindrical canvas
    
    Parameters
    ----------
    image_i : ndarray of shape (H, W, 3)
        The i-th image with width W and height H
    u : ndarray of shape (Hc, Wc, 2)
        The mapped 2D pixel locations in the source image for pixel transport
    mask_i : ndarray of shape (Hc, Wc)
        The valid pixel indicator

    Returns
    -------
    canvas_i : ndarray of shape (Hc, Wc, 3)
        the canvas image generated by the i-th source image
    """

    # initialize canvas_i
    canvas_i = np.zeros((u.shape[0],u.shape[1],3), dtype=np.uint8)
    # create variables for the image coordinates
    h = np.linspace(0, image_i.shape[0] - 1, image_i.shape[0])
    w = np.linspace(0, image_i.shape[1] - 1, image_i.shape[1])
    # get the indices of non-masked locations of the
    aa = np.where(mask_i == 1)
    ind_list = list(zip(aa[0],aa[1]))
    # for each non-masked location
    for ii in range(len(ind_list)):
        # get the index
        ind = np.asarray(ind_list[ii])
        # get the coordinate of the point in the given image
        img_coor = u[ind[0],ind[1],:]
        # put the corrisponding image value into the canvas
        canvas_i[ind[0],ind[1],:]=image_i[img_coor[1].astype(np.uint16),img_coor[0].astype(np.uint16)].astype(np.uint8)
    return canvas_i


def UpdateCanvas(canvas, canvas_i, mask_i):
    """
    Update the canvas with the new warped image
    
    Parameters
    ----------
    canvas : ndarray of shape (Hc, Wc, 3)
        The previously generated canvas
    canvas_i : ndarray of shape (Hc, Wc, 3)
        The i-th canvas
    mask_i : ndarray of shape (Hc, Wc)
        The mask of the valid pixels on the i-th canvas

    Returns
    -------
    canvas : ndarray of shape (Hc, Wc, 3)
        The updated canvas image
    """
    
    # get the index of the non-masked part of the cylinder
    aa = np.where(mask_i == 1)
    ind_list = list(zip(aa[0],aa[1]))
    # for each non-masked pixel of the cylinder
    for ii in range(len(ind_list)):
        # get the index
        index = ind_list[ii]
        # update the canvas with the value in canvas_i
        canvas[index[0],index[1]]=canvas_i[index[0],index[1]]
    return canvas

# code given in a previous assignment from csci 5561 to visualize matches
def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    ransac_n_iter = 4000
    ransac_thr = 10
    K = np.asarray([
        [320, 0, 480],
        [0, 320, 270],
        [0, 0, 1]
    ])

    # Read all images
    im_list = []
    for i in range(1, 9):
        im_file = '{}.jpg'.format(i)
        im = cv2.imread(im_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_list.append(im)

    rot_list = []
    rot_list.append(np.eye(3))
    for i in range(len(im_list) - 1):
        # Load consecutive images I_i and I_{i+1}
        img1 = im_list[i]
        img2 = im_list[i+1]

        # Extract SIFT features
        # generate the keypoints and features of the image I_i and I_{i+1}
        sift_template = cv2.xfeatures2d.SIFT_create()
        loc1, des1 = sift_template.detectAndCompute(img1, None)
        sift_target = cv2.xfeatures2d.SIFT_create()
        loc2, des2 = sift_target.detectAndCompute(img2, None)

        # Find the matches between two images (x1 <--> x2)
        x1, x2 = MatchSIFT(loc1, des1, loc2, des2)

        # # function to visualize matches found
        # visualize_find_match(img1, img2, x1, x2)

        # Estimate the homography between images using RANSAC
        H, inlier = EstimateH(x1, x2, ransac_n_iter, ransac_thr)

        # # code to visualize homography
        # img1_warp = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))
        # img2_warp = cv2.warpPerspective(img2, np.linalg.inv(H), (img2.shape[1],img2.shape[0]))
        # cv2.imshow('x', img1_warp)
        # cv2.waitKey(0)
        # cv2.imshow('y', img2_warp)
        # cv2.waitKey(0)

        # Compute the relative rotation matrix R
        R = EstimateR(H, K)

		# Compute R_new (or R_i+1)
        R_prev = rot_list[i]
        R_new = R_prev.T @ R

        # add new rotation matrix to the list of rotation matrices
        rot_list.append(R_new)

    Him = im_list[0].shape[0]
    Wim = im_list[0].shape[1]

    Hc = Him
    Wc = len(im_list) * Wim // 2

    canvas = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    p = ConstructCylindricalCoord(Wc, Hc, K)

    fig = plt.figure('HW1')
    plt.axis('off')
    plt.ion()
    plt.show()
    for i, (im_i, rot_i) in enumerate(zip(im_list, rot_list)):
        # Project the 3D points to the i-th camera plane
        u, mask_i = Projection(p, K, rot_i, Wim, Him)
        # Warp the image to the cylindrical canvas
        canvas_i = WarpImage2Canvas(im_i, u, mask_i)
        # Update the canvas with the new warped image
        canvas = UpdateCanvas(canvas, canvas_i, mask_i)
        plt.imshow(canvas)
        plt.savefig('output_{}.png'.format(i+1), dpi=600, bbox_inches = 'tight', pad_inches = 0)
