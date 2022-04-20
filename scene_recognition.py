import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath


def extract_dataset_info(data_path):
    # extract information from train.txt
    f = open(os.path.join(data_path, "train.txt"), "r")
    contents_train = f.readlines()
    label_classes, label_train_list, img_train_list = [], [], []
    for sample in contents_train:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        if label not in label_classes:
            label_classes.append(label)
        label_train_list.append(sample[0])
        img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
    print('Classes: {}'.format(label_classes))

    # extract information from test.txt
    f = open(os.path.join(data_path, "test.txt"), "r")
    contents_test = f.readlines()
    label_test_list, img_test_list = [], []
    for sample in contents_test:
        sample = sample.split()
        label, img_path = sample[0], sample[1]
        label_test_list.append(label)
        img_test_list.append(os.path.join(data_path, Path(
            PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

    return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def compute_dsift(img, stride, size):
    # calculate number of cells
    m = (img.shape[0]) // stride
    n = (img.shape[1]) // stride

    # initialize empty keypoint list
    keypoints = []
    # create a keypoint for each cell
    for ii in range(n):
        for jj in range(m):
            keypoints.append(cv2.KeyPoint(ii * stride + ((size - 1) / 2), jj * stride + ((size - 1) / 2), size))

    # extract sift feature from each keypoint location
    sift_query = cv2.SIFT_create()
    keypoints, dense_feature = sift_query.compute(img, keypoints)
    dense_feature.astype(np.int16)

    return dense_feature


def get_tiny_image(img, output_size):
    # calculate cell info
    width = img.shape[0] // output_size[0]
    height = img.shape[1] // output_size[1]
    total_pixel = width * height

    # initialize output
    feature = np.zeros(output_size)

    # compute the average value of each cell
    for ii in range(output_size[0]):
        for jj in range(output_size[1]):
            sum_pixels = np.sum(np.sum(img[ii * width:ii* width + width - 1, jj * height:jj*height + height - 1]))
            feature[ii,jj] = sum_pixels // total_pixel

    # normalize image to unit length
    feature = feature / np.linalg.norm(feature)
    # normalize image to zero mean
    tiny_total_pixel = output_size[0]*output_size[1]
    tiny_avg = np.sum(np.sum(feature)) / tiny_total_pixel
    feature = feature - tiny_avg

    return feature


def predict_knn(feature_train, label_train, feature_test, k):
    # find nearest neighbors
    n1 = NearestNeighbors(n_neighbors=k).fit(feature_test)
    dist, ind = n1.kneighbors(feature_train)

    # initialize labels for test images
    label_test_pred = np.zeros(feature_test.shape[0], label_train.shape[1])
    # for each test feature
    for kk in range(len(ind)):
        # get the indices of the nearest neighbors
        match = ind[kk]
        sum_labels = np.zeros((1, label_train.shape[1]))
        # sum the labels of all nearest neighbors
        for ll in range(len(match)):
            sum_labels = sum_labels + label_train[match[ll]]
        # select the most common label to be the predicted label
        label_test_pred[kk, np.argmax(sum_labels)] = 1

    return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # set tiny image size
    tiny_size = (16, 16)
    # initialize training features and training label arrays
    feature_train = np.zeros((len(img_train_list), tiny_size[0] * tiny_size[1]))
    label_train = np.zeros((len(img_train_list), len(label_classes)))
    # for each training image
    for ii in range(len(img_train_list)):
        img = cv2.imread(img_train_list[ii], 0)
        # extract feature
        tiny = get_tiny_image(img, tiny_size)
        # populate training feature and training label array
        feature_train[ii, :] = tiny.reshape(-1)
        label_train[ii, label_classes.index(label_train_list[ii])] = 1

    # initialize test features array
    feature_test = np.zeros((len(img_test_list), tiny_size[0] * tiny_size[1]))
    # for each test image
    for jj in range(len(img_test_list)):
        img = cv2.imread(img_test_list[jj], 0)
        # extract feature
        tiny = get_tiny_image(img, tiny_size)
        # pupulate test feature array
        feature_test[jj, :] = tiny.reshape(-1)

    # get prediction
    label_test_pred = predict_knn(feature_train, label_train, feature_test, 10)
    # get labels of test image
    label_test_array = np.asarray(label_test_list)
    # initialize confusion matrix
    confusion = np.zeros((len(label_classes), len(label_classes)))
    # for each classification
    for kk in range(len(label_classes)):
        # find each instance of the class in the test images
        class_instances = np.where(label_test_array == label_classes[kk])[0]
        # initialize the row of the confusioon matrix
        row = np.zeros((1, len(label_classes)))
        # sum all of the predicted labels given to test images of the class
        for ll in range(len(class_instances)):
            row = row + label_test_pred[class_instances[ll]]
        # normalize the values
        row = row / len(class_instances)
        # insert the row into the confusion matrix
        confusion[kk, :] = row
    # calculate the accuracy of the confusion matrix
    accuracy = np.trace(confusion) / len(label_classes)
    print('TINY+KNN Accuracy: ', accuracy)
    # visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def build_visual_dictionary(dense_feature_list, dic_size):
    # find clusters
    cluster = KMeans(n_clusters=dic_size).fit(dense_feature_list)
    # get cluster centers
    vocab = cluster.cluster_centers_
    return vocab


def compute_bow(feature, vocab):
    # match each feature to the nearest cluster center
    n1 = NearestNeighbors(n_neighbors=1).fit(vocab)
    dist, ind = n1.kneighbors(feature)

    # initialize the bow feature
    bow_feature = np.zeros((1, vocab.shape[0]))
    # for each input feature
    for ii in range(len(ind)):
        # increment the bow feature at the matching vocab
        bow_feature[0, ind[ii]] += 1

    # make bow feature unit length
    bow_feature = bow_feature / np.linalg.norm(bow_feature)
    return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
    # ** Code to construct dictionary **
    # # construct feature list for building dictionary
    # stride = 16
    # size = 16
    # dense_feature_list = np.empty((1, 128), dtype=np.int16)
    # for ii in range(len(img_train_list)):
    #     img = cv2.imread(img_train_list[ii])
    #     # get sift feature
    #     img_shift = compute_dsift(img, stride, size)
    #     # add dense sift feature to list
    #     dense_feature_list = np.concatenate((dense_feature_list, img_shift), axis=0)
    #
    # dic_size = 100
    # vocab = build_visual_dictionary(dense_feature_list, dic_size)
    # np.savetxt('vocab_dict', vocab)
    #
    # # ** end dictionary construction **

    # get dictionary
    vocab = np.loadtxt('vocab_dict')

    # set stride and size for dsift function
    stride = 32
    size = 16

    # initialize feature and label arrays
    bow_train = np.zeros((len(img_test_list), vocab.shape[0]))
    label_train = np.zeros((len(img_train_list), len(label_classes)))
    # for each training img
    for jj in range(len(img_train_list)):
        img = cv2.imread(img_train_list[jj], 0)
        # extract the feature
        img_shift = compute_dsift(img, stride, size)
        bow_img = compute_bow(img_shift, vocab)
        # populate the training feature and label array
        bow_train[jj, :] = bow_img
        label_train[jj, label_classes.index(label_train_list[jj])] = 1

    # construct bow for test images
    bow_test = np.zeros((len(img_test_list), vocab.shape[0]))
    for xx in range(len(img_test_list)):
        img = cv2.imread(img_test_list[xx], 0)
        # extract the feature
        img_shift = compute_dsift(img, stride, size)
        bow_img = compute_bow(img_shift, vocab)
        # populate the test feature array
        bow_test[xx, :] = bow_img

    # get prediction
    label_test_pred = predict_knn(bow_train, label_train, bow_test, 10)

    # create confusion matrix
    label_test_array = np.asarray(label_test_list)
    confusion = np.zeros((len(label_classes), len(label_classes)))
    # for each classification
    for kk in range(len(label_classes)):
        # find each instance of the class in the test images
        class_instances = np.where(label_test_array == label_classes[kk])[0]
        # initialize the row of the confusioon matrix
        row = np.zeros((1, len(label_classes)))
        # sum all of the predicted labels given to test images of the class
        for ll in range(len(class_instances)):
            row = row + label_test_pred[class_instances[ll]]
        # normalize the values
        row = row / len(class_instances)
        # insert the row into the confusion matrix
        confusion[kk, :] = row
    # calculate the accuracy of the confusion matrix
    accuracy = np.trace(confusion) / len(label_classes)
    print('BOW+KNN Accuracy: ', accuracy)
    # visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test, n_classes):
    # initialize an array of probabilities
    prediction_prob = np.zeros((feature_test.shape[0], n_classes))
    # for each class
    for ii in range(n_classes):
        # get the binary label
        binary_label = label_train[:, ii]
        # train svm for the class
        svm = LinearSVC()
        svm.fit(feature_train, binary_label)
        # get the confidence of the prediction for this classifier and populatethe probability array
        prediction_prob[:, ii] = svm.decision_function(feature_test)

    # initialize final label prediction array
    label_test_pred = np.zeros(prediction_prob.shape)
    # get the maximum probablity label
    max_index = np.argmax(prediction_prob, axis=1)
    # pupulate the final label prediction array
    for jj in range(len(max_index)):
        label_test_pred[jj, max_index[jj]]=1

    return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):

    # # ** Code to construct dictionary **
    # # construct feature list for building dictionary
    # stride = 16
    # size = 16
    # dense_feature_list = np.empty((1, 128), dtype=np.int16)
    # for ii in range(len(img_train_list)):
    #     img = cv2.imread(img_train_list[ii])
    #     # get sift feature
    #     img_shift = compute_dsift(img, stride, size)
    #     # add dense sift feature to list
    #     dense_feature_list = np.concatenate((dense_feature_list, img_shift), axis=0)
    #
    # dic_size = 100
    # vocab = build_visual_dictionary(dense_feature_list, dic_size)
    # np.savetxt('vocab_dict', vocab)
    #
    # # ** end dictionary construction **


    # get dictionary
    vocab = np.loadtxt('vocab_dict')

    stride = 32
    size = 16

    # initalize training feature and training label arrays
    bow_train = np.zeros((len(img_test_list), vocab.shape[0]))
    label_train = np.zeros((len(img_train_list), len(label_classes)))

    # for each training image
    for jj in range(len(img_train_list)):
        img = cv2.imread(img_train_list[jj], 0)
        # extract feature
        img_shift = compute_dsift(img, stride, size)
        bow_img = compute_bow(img_shift, vocab)
        # populate training feature and training label arrays
        bow_train[jj, :] = bow_img
        label_train[jj, label_classes.index(label_train_list[jj])] = 1

    # construct bow for test images
    bow_test = np.zeros((len(img_test_list), vocab.shape[0]))
    for xx in range(len(img_test_list)):
        img = cv2.imread(img_test_list[xx], 0)
        # extract feature
        img_shift = compute_dsift(img, stride, size)
        # populate test feature
        bow_img = compute_bow(img_shift, vocab)
        bow_test[xx, :] = bow_img

    # get prediction
    label_test_pred = predict_svm(bow_train, label_train, bow_test, len(label_classes))

    # create confusion matrix
    label_test_array = np.asarray(label_test_list)
    confusion = np.zeros((len(label_classes), len(label_classes)))
    # for each classification
    for kk in range(len(label_classes)):
        # find each instance of the class in the test images
        class_instances = np.where(label_test_array == label_classes[kk])[0]
        # initialize the row of the confusioon matrix
        row = np.zeros((1, len(label_classes)))
        # sum all of the predicted labels given to test images of the class
        for ll in range(len(class_instances)):
            row = row + label_test_pred[class_instances[ll]]
        # normalize the values
        row = row / len(class_instances)
        # insert the row into the confusion matrix
        confusion[kk, :] = row

    # calculate the accuracy of the confusion matrix
    accuracy = np.trace(confusion) / len(label_classes)
    print('BOW+SVM Accuracy: ', accuracy)
    # visualize_confusion_matrix(confusion, accuracy, label_classes)
    return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
    plt.title("accuracy = {:.3f}".format(accuracy))
    plt.imshow(confusion)
    ax, fig = plt.gca(), plt.gcf()
    plt.xticks(np.arange(len(label_classes)), label_classes)
    plt.yticks(np.arange(len(label_classes)), label_classes)
    # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
    # avoid top and bottom part of heatmap been cut
    ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # To do: replace with your dataset path
    label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info(
        "./scene_classification_data")

    classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)

    classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
    
    classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)