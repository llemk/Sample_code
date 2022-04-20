import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_differential_filter():
    # To do
    # create simple 3x3 sobel filter
    filter_dx = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    filter_dy = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))

    return filter_dx, filter_dy


def filter_image(im, filter):
    # To do
    # get image dimensions
    m = im.shape[0]
    n = im.shape[1]

    # get filter dimensions (assumed square)
    k = filter.shape[0]

    # number of zeros required to pad
    z = int(k / 2)

    # pad the image
    padded_im = np.ones([2 * z + m, 2 * z + n])
    padded_im[z:-z, z:-z] = im

    # initialize new array for the filtered image
    im_filtered = np.zeros([m, n])

    # fill in filtered image array
    for x in range(m):
        for y in range(n):
            im_filtered[x, y] = np.sum(np.multiply(padded_im[x:x + k, y:y + k], filter))

    return im_filtered


def get_gradient(im_dx, im_dy):
    # calculate gradient magnitude
    grad_mag = np.sqrt(np.square(im_dx) + np.square(im_dy))
    # constant to avoid dividing by zero
    e = 0.000001
    # calculate gradient angle
    grad_angle = np.arctan(np.divide(im_dy, im_dx + e))
    grad_angle += np.pi / 2
    # convert radians to degrees
    grad_angle = grad_angle * 180 / np.pi

    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    # To do
    # calculate the number of cells in the x and y direction
    big_m = grad_mag.shape[0] // cell_size
    big_n = grad_mag.shape[1] // cell_size

    # initialize the histogram
    ori_histo = np.zeros((big_m, big_n, 6))

    # for each cell
    for i in range(big_m - 1):
        for j in range(big_n - 1):

            # select the cell from the calculated gradient magnitude and gradient angle
            mag_cell = grad_mag[i * cell_size: i * cell_size + cell_size, j * cell_size: j * cell_size + cell_size - 1]
            ang_cell = grad_angle[i * cell_size: i * cell_size + cell_size,
                       j * cell_size: j * cell_size + cell_size - 1]

            # for each pixel in  the cell
            for x in range(cell_size - 1):
                for y in range(cell_size - 1):
                    angle = ang_cell[x, y]

                    # sum the magnitudes of each respective gradient angle
                    if 180 >= angle >= 165 or 0 <= angle < 15:
                        ori_histo[i, j, 0] += mag_cell[x, y]
                    elif 15 <= angle < 45:
                        ori_histo[i, j, 1] += mag_cell[x, y]
                    elif 45 <= angle < 75:
                        ori_histo[i, j, 2] += mag_cell[x, y]
                    elif 75 <= angle < 105:
                        ori_histo[i, j, 3] += mag_cell[x, y]
                    elif 105 <= angle < 135:
                        ori_histo[i, j, 4] += mag_cell[x, y]
                    elif 135 <= angle < 165:
                        ori_histo[i, j, 5] += mag_cell[x, y]
                    else:
                        print('angle out of bounds')

    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # To do
    # get histogram dimensions
    big_m = ori_histo.shape[0]
    big_n = ori_histo.shape[1]
    # calculate the depth of the block descriptor
    depth = 6 * block_size ** 2
    # initialize the normalized HOG array accounting for cells lost due to block normalization
    ori_histo_normalized = np.zeros(((big_m - (block_size - 1)), (big_n - (block_size - 1)), depth))
    e = 0.00001  # constant to avoid division by zero

    # for each cell
    for i in range(big_m - 1):
        for j in range(big_n - 1):
            # select the block
            block = ori_histo[i: i + block_size, j: j + block_size, :]
            # concatenate the block
            vect_block = np.reshape(block, depth)
            # calculate the denominator of the normalization function
            den = (np.sum(np.square(vect_block)) + e ** 2) ** 0.5
            # calculate the normalized HOG and place it into the normalized HOG array
            ori_histo_normalized[i, j, :] = vect_block / den

    return ori_histo_normalized


def extract_hog(im):
    # convert grey-scale image to double format
    im_copy = im.copy().astype('float') / 255.0
    # To do
    cell_size = 8
    # get x and y differential/sobel filters
    fil_dx, fil_dy = get_differential_filter()
    # generate differential filtered images
    im_dx = filter_image(im_copy, fil_dx)
    im_dy = filter_image(im_copy, fil_dy)
    # compute the gradients
    grad_mag, grad_angle = get_gradient(im_dx, im_dy)
    # build the HOG for all cells
    hist = build_histogram(grad_mag, grad_angle, cell_size)
    # build the normalized descriptor of all blocks
    desc = get_block_descriptor(hist, 2)
    # reshape the descriptor array into a vector
    hog = np.reshape(desc, -1)

    # visualize to verify. commented out and placed in main function to avoid execution during face_recognition func.
    # visualize_hog(im, hog, cell_size, 2)

    return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()


def face_recognition(I_target, I_template):
    # get the hog vector of the template
    hog_temp = extract_hog(I_template)
    # calculate the zero mean descriptor
    z_mean_temp = hog_temp - np.mean(hog_temp)
    # calculate the number of pixels before the template will overhang the edge of the target
    size_x = I_target.shape[1] - I_template.shape[1]
    size_y = I_target.shape[0] - I_template.shape[0]
    threshold = 0.5  # bounding box threshold
    # initialize bounding box array
    bounding_box = []
    s = 4   # stride length
    # for each patch of the target that matches the size of the template
    for i in range(0, size_x, s):
        for j in range(0, size_y, s):
            # select the patch of the target
            target_patch = I_target[j: j + I_template.shape[0], i: i + I_template.shape[1]]
            # get the HOG of the target
            hog_target = extract_hog(target_patch)
            # calculate the zero mean descriptor
            z_mean_target = hog_target - np.mean(hog_target)
            # calculate the normalized cross correlation
            ncc = np.dot(z_mean_target, z_mean_temp) / (
                        np.sqrt(np.sum(np.square(z_mean_target))) * np.sqrt(np.sum(np.square(z_mean_temp))))
            # if the normalized cross correlation is above the threshold
            if ncc > threshold:
                # add the bounding box coordinates of the NCC to the bounding box array
                bounding_box.append([i, j, ncc])

    # set the box size as the x dimension of the template
    box_size = I_template.shape[0]
    iou_limit = 0.5  # the limit for the intersection over union
    # calculate the area of the bounding box
    box_area = box_size ** 2
    # create a copy of the bounding box values
    bb_copy = np.asarray(bounding_box)
    # initialize array for the filtered bounding boxes
    bb_filtered = []

    # while there are unfiltered bounding boxes
    while len(bb_copy) != 0:
        # find the index of the bounding box with the largest NCC
        max_index = np.argmax(bb_copy[:, 2])
        # find the coordinates of the bounding box with the largest NCC value
        box_top_max = bb_copy[max_index, 1]
        box_left_max = bb_copy[max_index, 0]
        # add the bounding box with largest NCC value to the array of filtered bounding boxes
        bb_filtered.append(bb_copy[max_index, :])
        # delete the largest NCC bounding box from the unfiltered list
        bb_copy = np.delete(bb_copy, max_index, axis=0)
        # calculate the centerpoint of the bounding box with the largest NCC value
        centerpoint = [box_left_max + box_size // 2, box_top_max + box_size // 2]
        # array for the bounding boxes that are to be deleted
        del_rows = []
        # for each bounding box in the unfiltered bounding box array
        for k in range(bb_copy.shape[0]):
            # find the coordinates of the bounding box to be compared to the largest NCC value bounding box
            box_top = bb_copy[k, 1]
            box_left = bb_copy[k, 0]
            # calculate the centerpoint of the bounding box to be compared
            centerpoint1 = [box_left + box_size // 2, box_top + box_size // 2]
            # calculate the magnitude of the distance between centerpoints of the two bounding boxes
            difference = abs(np.subtract(centerpoint, centerpoint1))
            # if the distance between the centerpoints is small enough to cause overlap
            if difference[0] <= box_size and difference[1] <= box_size:
                # calculate the overlap of the bounding boxes
                overlap = (box_size - difference[0]) * (box_size - difference[1])
                # calculate the intersection over union of the two bounding boxes
                iou = overlap / (2 * box_area - overlap)
                # if the intersection over union is above the limit
                if iou > iou_limit:
                    # add the bounding box index to the array of rows to be deleted from the unfiltered list
                    del_rows.append(k)
        # delete the selected rows from the unfiltered list
        bb_copy = np.delete(bb_copy, del_rows, axis=0)
    # set the output to be an array of the filtered list of bounding boxes.
    bounding_boxes = np.asarray(bb_filtered)

    return bounding_boxes


def visualize_face_detection(I_target,bounding_boxes,box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.show()


if __name__=='__main__':

    im = cv2.imread('cameraman.tif', 0)
    hog = extract_hog(im)
    visualize_hog(im, hog, 8, 2)

    I_target= cv2.imread('target.png', 0)
    # MxN image

    I_template = cv2.imread('template.png', 0)
    # mxn  face template

    bounding_boxes=face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png')
    # MxN image (just for visualization)
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
    # this is visualization code.
