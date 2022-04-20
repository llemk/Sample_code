import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D


def find_match(img1, img2):
    # generate the keypoints and features of the template and target image
    sift_template = cv2.xfeatures2d.SIFT_create()
    keypoint_template, feature_template = sift_template.detectAndCompute(img1, None)
    sift_target = cv2.xfeatures2d.SIFT_create()
    keypoint_target, feature_target = sift_target.detectAndCompute(img2, None)

    # find the matches from image 1 to image 2
    # search the target image for nearest neighbors of the template image.
    n1 = NearestNeighbors(n_neighbors=2).fit(feature_target)
    dist, ind = n1.kneighbors(feature_template)

    # initialize empty list for keypoint index
    index_filtered = []
    # for all matched keypoints
    for ii in range(len(dist) - 1):
        nearest_feat = dist[ii]
        feat_index = ind[ii]
        ratio = nearest_feat[0] / nearest_feat[1]
        # if the two nearest distances pass the ratio test then add the keypoint indices to the filtered list
        if ratio < .7:
            index_filtered.append([ii, feat_index[0]])

    # find the matches from image 2 to image 1
    # search the template image for nearest neighbors of the target image.
    n2 = NearestNeighbors(n_neighbors=2).fit(feature_template)
    dist2, ind2 = n2.kneighbors(feature_target)

    # initialize empty list for keypoint index
    index_filtered2 = []
    # for all matched keypoints
    for ii in range(len(dist2) - 1):
        nearest_feat = dist2[ii]
        feat_index = ind2[ii]
        ratio = nearest_feat[0] / nearest_feat[1]
        # if the two nearest distances pass the ration test then add the keypoint indices to the filtered list
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
        template_keypoint = keypoint_template[template_index]
        target_keypoint = keypoint_target[target_index]
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
        template_keypoint = keypoint_template[template_index]
        target_keypoint = keypoint_target[target_index]
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

    # for each pair from matching from image 1 to image 2
    for ii in range(len(i1i2)):
        querry = str(i1i2[ii])
        # for each pair from matching image 2 to image 1
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


def compute_F(pts1, pts2):
    # initialize a variable to hold the largest amount of inliers found
    largest_inlier = 0
    # set a random seed for consistency
    np.random.seed(2)
    # iterate through for the specified number of iterations
    for ii in range(200):
        # select a random permutation of 8 keypoint pairs
        rand_set = np.random.choice(range(1, len(pts1)), 8, replace=False)

        # separate the coordinates of each keypoint into u and v
        u1x = pts1[rand_set[0], 0]
        u1y = pts1[rand_set[0], 1]
        u2x = pts1[rand_set[1], 0]
        u2y = pts1[rand_set[1], 1]
        u3x = pts1[rand_set[2], 0]
        u3y = pts1[rand_set[2], 1]
        u4x = pts1[rand_set[3], 0]
        u4y = pts1[rand_set[3], 1]
        u5x = pts1[rand_set[4], 0]
        u5y = pts1[rand_set[4], 1]
        u6x = pts1[rand_set[5], 0]
        u6y = pts1[rand_set[5], 1]
        u7x = pts1[rand_set[6], 0]
        u7y = pts1[rand_set[6], 1]
        u8x = pts1[rand_set[7], 0]
        u8y = pts1[rand_set[7], 1]

        v1x = pts2[rand_set[0], 0]
        v1y = pts2[rand_set[0], 1]
        v2x = pts2[rand_set[1], 0]
        v2y = pts2[rand_set[1], 1]
        v3x = pts2[rand_set[2], 0]
        v3y = pts2[rand_set[2], 1]
        v4x = pts2[rand_set[3], 0]
        v4y = pts2[rand_set[3], 1]
        v5x = pts2[rand_set[4], 0]
        v5y = pts2[rand_set[4], 1]
        v6x = pts2[rand_set[5], 0]
        v6y = pts2[rand_set[5], 1]
        v7x = pts2[rand_set[6], 0]
        v7y = pts2[rand_set[6], 1]
        v8x = pts2[rand_set[7], 0]
        v8y = pts2[rand_set[7], 1]

        # create the 'A' matrix of the Ax=b equation
        big_a = np.array([[u1x * v1x, u1y * v1x, v1x, u1x * v1y, u1y * v1y, v1y, u1x, u1y, 1],
                          [u2x * v2x, u2y * v2x, v2x, u2x * v2y, u2y * v2y, v2y, u2x, u2y, 1],
                          [u3x * v3x, u3y * v3x, v3x, u3x * v3y, u3y * v3y, v3y, u3x, u3y, 1],
                          [u4x * v4x, u4y * v4x, v4x, u4x * v4y, u4y * v4y, v4y, u4x, u4y, 1],
                          [u5x * v5x, u5y * v5x, v5x, u5x * v5y, u5y * v5y, v5y, u5x, u5y, 1],
                          [u6x * v6x, u6y * v6x, v6x, u6x * v6y, u6y * v6y, v6y, u6x, u6y, 1],
                          [u7x * v7x, u7y * v7x, v7x, u7x * v7y, u7y * v7y, v7y, u7x, u7y, 1],
                          [u8x * v8x, u8y * v8x, v8x, u8x * v8y, u8y * v8y, v8y, u8x, u8y, 1]])

        # find the nullspace of the A matrix
        f = null_space(big_a).reshape((3, 3))

        # i tried using this method to calculate the nullspace of the matrix but I was unsuccessful
        # _, _, v1 = np.linalg.svd(big_a)
        # f = v1[:, -1].reshape((3, 3))
        # u2, s2, v2 = np.linalg.svd(f)
        # s2[2] = 0
        # f = u2 @ np.diag(s2) @ v2.T

        # initialize a variable to hold the number of inlier found in the current iteration
        inlier = 0
        # for each point
        for jj in range(len(pts1)):
            # get the u and v vectors of each point
            v_querry = np.array([pts2[jj, 0], pts2[jj, 1], 1]).reshape((1, 3))
            u_querry = np.array([[pts1[jj, 0]], [pts1[jj, 1]], [1]])

            # calculate the error from the fundamental matrix ( v.T * F * u = 0)
            error = abs(v_querry @ f @ u_querry)
            # if the error is bellow a threshold
            if error < 0.1:
                # increment the current iteration inlier count
                inlier = inlier + 1
        # if the number of inliers of this iteration is larger than the previous largest number of inliers
        if inlier > largest_inlier:
            # set the output variable to be the f matrix with largest number of inliers
            f_out = f
            # update the largest inlier variable
            largest_inlier = inlier

    return f_out


def triangulation(P1, P2, pts1, pts2):
    # initialize output array
    pts3D = np.zeros((len(pts1), 3))
    # for each point
    for ii in range(len(pts1)):
        # get the 2D coordinates of u and v
        u = pts1[ii]
        v = pts2[ii]
        # create the skew symetric matrices
        skew_u = np.array([[0, -1, u[1]], [1, 0, -u[0]], [-u[1], u[0], 0]])
        skew_v = np.array([[0, -1, v[1]], [1, 0, -v[0]], [-v[1], v[0], 0]])
        # calculate the constituents of the A matrix
        aaa1 = skew_u @ P1
        aaa2 = skew_v @ P2
        # construct the 4x4 A matrix
        bbb = np.concatenate((aaa1[0:2, :], aaa2[0:2, :]), axis=0)
        # find the nullspace of the A matrix
        _, _, v1 = np.linalg.svd(bbb)
        # normalize with respect to the last value in vector
        ccc = v1[:, 3]/v1[3, 3]
        # set the output point to be the 3D coordinates
        pts3D[ii, :] = ccc[0:3]

    return pts3D


def disambiguate_pose(Rs, Cs, pts3Ds):
    # initialize a variable to carry the maximum number of valid points found
    max_valid = 0
    # for each configuration
    for ii in range(len(Rs)):
        # set the number of valid points to zero
        n_valid = 0
        # for each point
        for jj in range(len(pts3Ds[ii])):
            # if the point is in front of the rotated camera and the camera facing z_vector = (0,0,1)
            if np.dot(Rs[ii][-1, :].T, (pts3Ds[ii][jj, :].reshape(3, 1) - Cs[ii])) > 0 and pts3Ds[ii][jj, 2] > 0:
                # increment the number of valid points
                n_valid = n_valid + 1

        # if the number of valid points is larger than the previous max number of valid points
        if n_valid >= max_valid:
            # set the output values
            R = Rs[ii]
            C = Cs[ii]
            pts3D = pts3Ds[ii]
            # update the max valid value
            max_valid = n_valid

    return R, C, pts3D


def compute_rectification(K, R, C):
    # calculate rx, ry, and rz
    rz_tilde = np.array([[0], [0], [1]])
    rx = (C/np.linalg.norm(C))
    rz_num = rz_tilde.T - (np.dot(rz_tilde.T , rx) @ rx.T)
    rz_den = np.linalg.norm(rz_num)
    rz = (rz_num/rz_den).T
    ry = np.cross(rz.T, rx.T)
    # calculate the rectified rotation matrix
    r_rect = np.concatenate((rx.T, ry, rz.T))
    # get the homography for each image
    H1 = K @ r_rect @ np.linalg.inv(K)
    H2 = K @ r_rect @ R.T @ np.linalg.inv(K)
    return H1, H2


def dense_match(img1, img2):
    # get dimensions of the input image
    m = (img1.shape[0])
    n = (img1.shape[1])
    # make sure the image is in the correct data type
    img1.astype(np.uint8)
    img2.astype(np.uint8)
    # initialize empty keypoint list
    keypoints = []
    # create a keypoint for each pixel
    for ii in range(m):
        for jj in range(n):
            keypoints.append(cv2.KeyPoint(jj, ii, 3))
    # extract sift feature from each keypoint location
    sift_img1 = cv2.xfeatures2d.SIFT_create()
    sift_img2 = cv2.xfeatures2d.SIFT_create()
    _, dense_feature_img1 = sift_img1.compute(img1, keypoints)
    _, dense_feature_img2 = sift_img2.compute(img2, keypoints)
    # reshape the dense features into a convenient shape
    dense_im1 = dense_feature_img1.reshape(m, n, 128)
    dense_im2 = dense_feature_img2.reshape(m, n, 128)
    # initialize output array
    disparity = np.zeros((m, n))
    # for each row
    for jj in range(m):
        # for each column
        for kk in range(n):
            # select the pixel of the left image
            im1_pixel = dense_im1[jj, kk, :]
            # select the pixels on right image along the epipolar line to the right of the left pixel index.
            im2_pixel = dense_im2[jj, kk::, :]
            # calculate the norm of the difference of sift descriptors
            ccc = np.linalg.norm(im1_pixel - im2_pixel, axis=1)
            # find the disparity between the pixel on the left image and the closest match on the right image.
            disparity[jj, kk] = np.argmin(ccc)

    return disparity


# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


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


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()


def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = np.dot(F, np.array([p[0], p[1], 1]).reshape(3, 1))
    p1, p2 = (0, -el[2] / el[1]), (img.shape[1], (-img_width * el[0] - el[2]) / el[1])
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('./left.bmp', 1)
    img_right = cv2.imread('./right.bmp', 1)
    # visualize_img_pair(img_left, img_right)

    # Step 1: find correspondences between image pair
    pts1, pts2 = find_match(img_left, img_right)
    # # visualize_find_match(img_left, img_right, pts1, pts2)

    # Step 2: compute fundamental matrix
    F = compute_F(pts1, pts2)
    # visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    # Step 3: computes four sets of camera poses
    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    # visualize_camera_poses(Rs, Cs)

    # # Step 4: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    # visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # # Step 5: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)

    # # Step 6: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    # visualize_img_pair(img_left_w, img_right_w)
    #
    # # Step 7: generate disparity map
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    disparity = dense_match(img_left_w, img_right_w)
    # visualize_disparity_map(disparity)

    # save to mat
    sio.savemat('stereo.mat', mdict={'pts1': pts1, 'pts2': pts2, 'F': F, 'pts3D': pts3D, 'H1': H1, 'H2': H2,
                                     'img_left_w': img_left_w, 'img_right_w': img_right_w, 'disparity': disparity})
