import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


def find_match(img1, img2):
    # generate the keypoints and features of the template and target image
    sift_template = cv2.xfeatures2d.SIFT_create()
    keypoint_template, feature_template = sift_template.detectAndCompute(img1, None)
    sift_target = cv2.xfeatures2d.SIFT_create()
    keypoint_target, feature_target = sift_target.detectAndCompute(img2, None)

    # search the target image for nearest neighbors of the template image.
    n1 = NearestNeighbors(n_neighbors=2).fit(feature_target)
    dist, ind = n1.kneighbors(feature_template)

    # initialize empty list for keypoint index
    index_filtered = []
    # for all matched keypoints
    for ii in range(len(dist)-1):
        nearest_feat = dist[ii]
        feat_index = ind[ii]
        ratio = nearest_feat[0]/nearest_feat[1]
        # if the two nearest distances pass the ration test then add the keypoint indices to the filtered list
        if ratio < .8:
            index_filtered.append([ii, feat_index[0]])

    temp1 = []
    temp2 = []
    # for each keypoint pair that passed the filter, seperate the keypoints into template and target lists
    for ii in range(len(index_filtered)-1):
        pair_index = index_filtered[ii]
        template_index = pair_index[0]
        target_index = pair_index[1]
        template_keypoint = keypoint_template[template_index]
        target_keypoint = keypoint_target[target_index]
        temp1.append(template_keypoint.pt)
        temp2.append(target_keypoint.pt)

    # initialize output arrays
    x1 = np.zeros((len(temp1)-1, 2))
    x2 = np.zeros((len(temp2)-1, 2))

    # for each keypoint pair, fill in the output arrays
    for ii in range(len(x2)):
        alb = temp2[ii]
        ald = temp1[ii]
        x1[ii, 0] = ald[0]
        x1[ii, 1] = ald[1]
        x2[ii, 0] = alb[0]
        x2[ii, 1] = alb[1]

    return x1, x2


def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
    # To do
    largest_inlier = 0
    # iterate through for the specified number of iterations
    for ii in range(ransac_iter):
        # select a random permutation of 4 keypoint pairs
        rand_set = np.random.choice(range(1, len(x1)), 8, replace=False)

        # separate the coordinates of each keypoint into u and v
        u1x = x1[rand_set[0], 0]
        u1y = x1[rand_set[0], 1]
        u2x = x1[rand_set[1], 0]
        u2y = x1[rand_set[1], 1]
        u3x = x1[rand_set[2], 0]
        u3y = x1[rand_set[2], 1]
        u4x = x1[rand_set[3], 0]
        u4y = x1[rand_set[3], 1]
        u5x = x1[rand_set[4], 0]
        u5y = x1[rand_set[4], 1]
        u6x = x1[rand_set[5], 0]
        u6y = x1[rand_set[5], 1]
        u7x = x1[rand_set[6], 0]
        u7y = x1[rand_set[6], 1]
        u8x = x1[rand_set[7], 0]
        u8y = x1[rand_set[7], 1]

        v1x = x2[rand_set[0], 0]
        v1y = x2[rand_set[0], 1]
        v2x = x2[rand_set[1], 0]
        v2y = x2[rand_set[1], 1]
        v3x = x2[rand_set[2], 0]
        v3y = x2[rand_set[2], 1]
        v4x = x2[rand_set[3], 0]
        v4y = x2[rand_set[3], 1]
        v5x = x2[rand_set[4], 0]
        v5y = x2[rand_set[4], 1]
        v6x = x2[rand_set[5], 0]
        v6y = x2[rand_set[5], 1]
        v7x = x2[rand_set[6], 0]
        v7y = x2[rand_set[6], 1]
        v8x = x2[rand_set[7], 0]
        v8y = x2[rand_set[7], 1]

        # create the 'A' matrix of the Ax=b equation
        big_a = np.array([[u1x*v1x, u1y*v1x, v1x, u1x*v1y, u1y*v1y, v1y, u1x, u1y, 1],
                          [u2x*v2x, u2y*v2x, v2x, u2x*v2y, u2y*v2y, v2y, u2x, u2y, 1],
                          [u3x*v3x, u3y*v3x, v3x, u3x*v3y, u3y*v3y, v3y, u3x, u3y, 1],
                          [u4x*v4x, u4y*v4x, v4x, u4x*v4y, u4y*v4y, v4y, u4x, u4y, 1],
                          [u5x*v5x, u5y*v5x, v5x, u5x*v5y, u5y*v5y, v5y, u5x, u5y, 1],
                          [u6x*v6x, u6y*v6x, v6x, u6x*v6y, u6y*v6y, v6y, u6x, u6y, 1],
                          [u7x*v7x, u7y*v7x, v7x, u7x*v7y, u7y*v7y, v7y, u7x, u7y, 1],
                          [u8x*v8x, u8y*v8x, v8x, u8x*v8y, u8y*v8y, v8y, u8x, u8y, 1]])

        big_a_t = np.transpose(big_a)
        # build the 'b' vector of the Ax=b equation
        b = np.array([[u1p], [v1p], [u2p], [v2p], [u3p], [v3p], [u4p], [v4p]])
        # solve Ax=b equation
        h = np.matmul(np.matmul(np.linalg.inv(np.matmul(big_a_t, big_a)), big_a_t), b)
        # fill out the affine transformation
        h1 = np.append(h, [0, 0, 1])
        A = np.reshape(h1, (3, 3))

        # create an array of keypoints to check that do not include the random samples to create the affine transform
        tempx1 = np.copy(x1)
        tempx2 = np.copy(x2)
        tempx1 = np.delete(tempx1, rand_set, 0)
        tempx2 = np.delete(tempx2, rand_set, 0)
        inliers = 0

        # for each keypoint to be checked, check that the transformed template keypoint is within its expected distance
        # from the target keypoint.
        for jj in range(tempx1.shape[0]):
            query1 = tempx1[jj]
            query1 = np.append(query1, [1])
            query2 = tempx2[jj]
            aas = np.matmul(A, query1)
            dist = ((aas[0]-query2[0])**2 + (aas[1]-query2[1])**2)**0.5
            # count the number of inliers
            if dist < ransac_thr:
                inliers +=1

        # update the output affine transform if a transform with more inliers is found
        if inliers > largest_inlier:
            A_out = A
            largest_inlier = inliers

    return A_out


def warp_image(img, A, output_size):
    # initialze the output image
    img_warped = np.zeros((output_size[0], output_size[1]))
    # for each pixel of the output image, grab the pixel it is mapped to in the target image
    for ii in range(output_size[0]):
        for jj in range(output_size[1]):
            dd = (np.matmul(A, [[jj], [ii], [1]])).astype(int)
            img_warped[ii, jj] = img[dd[1], dd[0]]

    return img_warped


def align_image(template, target, A):
    # put the input affine transorm into a variable for the loop
    p = A

    # calculate the differential images of the template. Professor said it was ok to use cv2.Sobel for this in the
    # canvas discussion section
    im_dx = cv2.Sobel(template, ddepth=cv2.CV_16U, dx=1, dy=0)
    im_dy = cv2.Sobel(template, ddepth=cv2.CV_16U, dx=0, dy=1)

    # initialize the arrays for the steepest descent images
    im_dx_u = np.zeros(im_dx.shape)
    im_dx_v = np.zeros(im_dx.shape)
    im_dy_u = np.zeros(im_dy.shape)
    im_dy_v = np.zeros(im_dy.shape)

    # fill in the steepest descent images. u and v are normalized
    for ii in range(im_dx.shape[1]):
        im_dx_u[:, ii] = im_dx[:, ii] * ii // im_dx.shape[1]
        im_dy_u[:, ii] = im_dy[:, ii] * ii // im_dx.shape[1]

    for ii in range(im_dx.shape[0]):
        im_dx_v[ii, :] = im_dx[ii, :] * ii // im_dx.shape[0]
        im_dy_v[ii, :] = im_dy[ii, :] * ii // im_dx.shape[0]
    # # visualize steepest descent images
    # plt.subplot(231)
    # plt.imshow(im_dx_u, cmap='jet', vmin=1, vmax=255)
    # plt.title('I_dx_u')
    # plt.axis('off')
    # plt.subplot(232)
    # plt.imshow(im_dx_v, cmap='jet', vmin=1, vmax=255)
    # plt.title('I_dx_v')
    # plt.axis('off')
    # plt.subplot(233)
    # plt.imshow(im_dx, cmap='jet', vmin=1, vmax=255)
    # plt.title('I_dx')
    # plt.axis('off')
    # plt.subplot(234)
    # plt.imshow(im_dy_u, cmap='jet', vmin=1, vmax=255)
    # plt.title('I_dy_u')
    # plt.axis('off')
    # plt.subplot(235)
    # plt.imshow(im_dy_v, cmap='jet', vmin=1, vmax=255)
    # plt.title('I_dy_v')
    # plt.axis('off')
    # plt.subplot(236)
    # plt.imshow(im_dy, cmap='jet', vmin=1, vmax=255)
    # plt.title('I_dy')
    # plt.axis('off')
    # plt.show()

    # create the steepest descent
    steepest_descent = np.array([im_dx_u, im_dx_v, im_dx, im_dy_u, im_dy_v, im_dy])

    # initialize the hessian
    H = np.zeros((6, 6))

    # fill in the hessian by summing the pixels of the multiplication of steepest descent images.
    for ii in range(len(steepest_descent)):
        for jj in range(len(steepest_descent)):
            H[ii, jj] = np.sum(np.sum(np.matmul(np.transpose(steepest_descent[ii]), steepest_descent[jj])))

    # set threshold and initialize loop termination variable
    thresh = 0.018
    delp_norm = 999
    # initialize error list
    errors = []

    while delp_norm > thresh:
        # warp the target image
        warped_im = warp_image(target, p, template.shape)
        # calculate the error between the warped image and the template and normalize it
        error_img = (warped_im - template) / 255
        # initialize F 6x1
        F = np.zeros((6, 1))
        # fill each element of F by summing the pixels of the multiplication of steepest descent image with error img
        for ii in range(len(F)):
            F[ii] = np.sum(np.sum(np.matmul(np.transpose(steepest_descent[ii]), error_img)))
        # calculate the delta p
        delp= np.matmul(np.linalg.inv(H), F)
        # update the loop termination variable
        delp_norm = np.linalg.norm(delp)
        delp = np.reshape(delp, (2, 3))
        delp = np.append(delp, [[0, 0, 0]], axis=0)
        # update the affine transform
        p = p-delp
        # add error to the error list
        errors.append(np.linalg.norm(error_img))

    # set the output to be the final affine transform.
    A_refined = p

    return A_refined, errors


def track_multi_frames(template, img_list):
    # initialize the output list
    A_list = []
    # find the initial keypoint matches
    x1, x2 = find_match(template, img_list[0])
    # align image using features
    thresh = 5
    iter = 10000
    A_init = align_image_using_feature(x1, x2, thresh, iter)

    # for each image in the image list
    for ii in range(len(img_list)-1):
        # calculate the refined affine transformation
        A_refined = align_image(template, img_list[ii], A_init)
        # update the template to be the warped image of the current target image using the refined affine transform
        template = warp_image(img_list[ii], A_refined, template.shape)
        # update the affine transformation
        A_init = A_refined
        # add the affine transformation to the list
        A_list.append(A_refined)

    return A_list


def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()


def visualize_align_image(template, target, A, A_refined, errors=None):
    img_warped_init = warp_image(target, A, template.shape)
    img_warped_optim = warp_image(target, A_refined, template.shape)
    err_img_init = np.abs(img_warped_init - template)
    err_img_optim = np.abs(img_warped_optim - template)
    img_warped_init = np.uint8(img_warped_init)
    img_warped_optim = np.uint8(img_warped_optim)
    overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
    overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
    plt.subplot(241)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(242)
    plt.imshow(img_warped_init, cmap='gray')
    plt.title('Initial warp')
    plt.axis('off')
    plt.subplot(243)
    plt.imshow(overlay_init, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(244)
    plt.imshow(err_img_init, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.subplot(245)
    plt.imshow(template, cmap='gray')
    plt.title('Template')
    plt.axis('off')
    plt.subplot(246)
    plt.imshow(img_warped_optim, cmap='gray')
    plt.title('Opt. warp')
    plt.axis('off')
    plt.subplot(247)
    plt.imshow(overlay_optim, cmap='gray')
    plt.title('Overlay')
    plt.axis('off')
    plt.subplot(248)
    plt.imshow(err_img_optim, cmap='jet')
    plt.title('Error map')
    plt.axis('off')
    plt.show()

    if errors is not None:
        plt.plot(errors * 255)
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    template = cv2.imread('./Hyun_Soo_template.jpg', 0)  # read as grey scale image
    target_list = []
    for i in range(4):
        target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i + 1), 0)  # read as grey scale image
        target_list.append(target)

    x1, x2 = find_match(template, target_list[0])
    visualize_find_match(template, target_list[0], x1, x2)

    ransac_thr = 3
    ransac_iter = 5000
    A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)

    img_warped = warp_image(target_list[0], A, template.shape)
    plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()

    A_refined, errors = align_image(template, target_list[0], A)
    visualize_align_image(template, target_list[0], A, A_refined, errors)

    A_list = track_multi_frames(template, target_list)
    visualize_track_multi_frames(template, target_list, A_list)