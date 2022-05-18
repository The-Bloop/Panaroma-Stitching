# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"

    overlap_arr, imgs = get_overlap(imgs)

    while(len(imgs) > 1):
        new_img = stitch_2_img(imgs[0], imgs[1])
        imgs.pop(1)
        imgs.pop(0)
        imgs.insert(1, new_img)

    print('Done')
    cv2.imwrite(savepath, new_img)
    return overlap_arr

def get_overlap(imgs):
    overlap_arr = []
    pop_arr = []
    for k in range(len(imgs)):
        img1 = imgs[k]
        overlap_vector = []
        for l in range(len(imgs)):
            if k == l:
                overlap_vector.append(1)
            else:
                img2 = imgs[l]

                sift2 = cv2.xfeatures2d.SIFT_create()
                g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                kp1, f1 = sift2.detectAndCompute(g1, None)
                kp2, f2 = sift2.detectAndCompute(g2, None)

                match_list = []

                for i in range(len(kp1)):
                    h1 = f1[i, 0:128]
                    match = compute_ssd(h1, f2)
                    if(match >= 0):
                        match_list.append([i, match])

                kp1_new = []
                kp2_new = []
                src_pts = []
                dst_pts = []
                for m in range(len(match_list)):
                    kp1_new.append(kp1[match_list[m][0]])
                    kp2_new.append(kp2[match_list[m][1]])
                    src_pts.append(np.float32(kp1[match_list[m][0]].pt))
                    dst_pts.append(np.float32(kp2[match_list[m][1]].pt))

                src_pts = np.array(src_pts)
                src_pts = src_pts.reshape(-1, 1, 2)
                dst_pts = np.array(dst_pts)
                dst_pts = dst_pts.reshape(-1, 1, 2)


                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                total_matches = np.sum(mask)
                x = img1.shape
                y = img2.shape

                x_perc = (total_matches / x[1])
                y_perc = (total_matches / y[1])

                if(x_perc >= 0.2 and y_perc >= 0.2):
                    overlap_vector.append(1)
                else:
                    overlap_vector.append(0)

        overlap_arr.append(overlap_vector)
        if(np.sum(overlap_vector) == 1):
            pop_arr.append(k)

    overlap_arr = (np.array(overlap_arr)).reshape((len(imgs), len(imgs)))

    if(len(pop_arr) >= 1):
        for p in range(1, len(pop_arr)):
            imgs.pop(pop_arr[-p])
        imgs.pop(pop_arr[0])

    return overlap_arr, imgs

def stitch_2_img(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, f1 = sift.detectAndCompute(img1, None)
    kp2, f2 = sift.detectAndCompute(img2, None)

    match_list = []
    for i in range(len(kp1)):
        h1 = f1[i, 0:128]
        match = compute_ssd(h1, f2)
        if (match >= 0):
            match_list.append([i, match])

    kp1_new = []
    kp2_new = []
    src_pts = []
    dst_pts = []
    for k in range(len(match_list)):
        kp1_new.append(kp1[match_list[k][0]])
        kp2_new.append(kp2[match_list[k][1]])
        src_pts.append(np.float32(kp1[match_list[k][0]].pt))
        dst_pts.append(np.float32(kp2[match_list[k][1]].pt))

    src_pts = np.array(src_pts)
    src_pts = src_pts.reshape(-1, 1, 2)
    dst_pts = np.array(dst_pts)
    dst_pts = dst_pts.reshape(-1, 1, 2)

    M1, mask1 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    x = []
    x = img1.shape
    y = img2.shape
    bb1 = np.float32([[0, 0], [0, x[0] - 1], [x[1] - 1, x[0] - 1], [x[1] - 1, 0]]).reshape(-1, 1, 2)

    dst = cv2.perspectiveTransform(bb1, M1)
    dst = dst.reshape(4, 2)
    bb_max = np.amax(dst, 0)
    bb_min = np.amin(dst, 0)

    # bb_l = [bb_max[0], bb_min[1]]
    # bb_w = [bb_min[0], bb_max[1]]
    #
    # l_sum = (bb_max[0] - bb_l[0]) ** 2 + (bb_max[1] - bb_l[1]) ** 2
    # w_sum = (bb_max[0] - bb_w[0]) ** 2 + (bb_max[1] - bb_w[1]) ** 2
    #
    # l = np.sqrt(l_sum)
    # w = np.sqrt(w_sum)
    # area = l * w
    # if (np.any(M1) != None):
    #     print(np.linalg.det(M1))
    # else:
    #     print("None")
    # print('Area : ', area)

    a = [1, 1]
    if(bb_min[0] > 0):
        bb_min[0] = 0
        a[0] = 0
    if (bb_min[1] > 0):
        bb_min[1] = 0
        a[1] = 0

    bb_max = bb_max - bb_min
    if (int(abs(bb_max[0])) < img2.shape[1] + int(abs(bb_min[0]))):
        bb_max[0] = img2.shape[1] + int(abs(bb_min[0]))
    if (int(abs(bb_max[1])) < img2.shape[0] + int(abs(bb_min[1]))):
        bb_max[1] = img2.shape[0] + int(abs(bb_min[1]))

    T = np.float32([[1, 0, -bb_min[0]], [0, 1, -bb_min[1]], [0, 0, 1]])
    M = np.dot(T, M1)

    img_stc = cv2.warpPerspective(img1, M, (int(abs(bb_max[0])), int(abs(bb_max[1]))))

    g_stc = cv2.cvtColor(img_stc, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    match_array = g_stc[int(abs(bb_min[1])):(img2.shape[0] + int(abs(bb_min[1]))), (int(abs(bb_min[0]))):(img2.shape[1] + int(abs(bb_min[0])))]

    log_array = np.less(match_array, g2)
    ind_array = np.where(log_array == 1)
    ind_array = np.transpose(np.array(ind_array))
    for i in range(len(ind_array)):
        img_stc[(ind_array[i][0] + int(abs(bb_min[1]))), (ind_array[i][1] + int(abs(bb_min[0])))] = img2[
            ind_array[i][0], ind_array[i][1]]

    # print('done')
    return img_stc

def compute_ssd(h1,h2):
    ssd_arr = np.sum(np.power((h2[:] - h1), 2), 1)
    ssd_unq = np.unique(ssd_arr)
    ssd_sort = sorted(ssd_arr)
    s1 = ssd_sort[0]
    s2 = ssd_sort[1]
    value = round(s1/s2)
    index = -1
    if(value < 0.5):
        ind = np.where(ssd_arr == s1)
        ind = np.array(ind)
        index = ind[0][0]
    return index


if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3', savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
