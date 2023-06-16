import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import lab.fast_LTS

def picchange():
    line_num = 0
    img1 = cv.imread('shot.jpg', 0)
    # 长宽
    height1, width1 = img1.shape
    cv.imshow('img0', img1)
    cv.waitKey(0)

    # img1 = cv.GaussianBlur(img1, (9, 9), 1)
    img_mean = cv.blur(img1, (3,3))
    out = img1 - (1-0.1) * img_mean
    out[out >= 0] = 255
    out[out < 0] = 0
    img1 = out.astype(np.uint8)

    img2 = cv.imread('shot.jpg', 1)

    # hist, bins = np.histogram(img1.ravel(), bins=100)
    # plt.hist(img1.ravel(), bins=100)
    # plt.show()
    
    # if bins[np.argmax(hist)] > 60:
    #     thr = bins[np.argmax(hist)] - 60
    # else:
    #     thr = 0

    cv.imshow('img1', img1)
    cv.waitKey(0)

    img_mask = cv.imread('mask.png', 0)
    mask = cv.resize(img_mask, (width1, height1))
    ret, mask = cv.threshold(mask, 128, 255, cv.THRESH_BINARY)
    img1 = cv.bitwise_not(img1)
    img1 = cv.bitwise_or(img1,img1,mask=mask)
    cv.imshow('img1', img1)
    cv.waitKey(0)
    
    # 腐蚀 因为黑白相反
    kernel1 = np.ones((3, 3), np.uint8)
    img1 = cv.erode(img1, kernel1)
    img1 = cv.dilate(img1, kernel1)
    img1 = cv.dilate(img1, kernel1)
    img1 = cv.erode(img1, kernel1)
    
    cv.imshow('img1', img1)
    cv.waitKey(0)

    '''
    contours, hierarchy = cv.findContours(img1.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # 统计选择的检测区域中的轮廓（有效轮廓）
    valid_cntrs = []  # 空列表 valid_cntrs 用于存放有效轮廓

    for cntr in contours:  # 遍历找到的所有轮廓
        x, y, w, h = cv.boundingRect(cntr)  # 轮廓的坐标
        if (cv.contourArea(cntr) >= 25) & (h < height1) & (h < width1):
            dmy = cv.rectangle(img1, (x, y), (x + w, y + h), (255, 0, 0), 2)
            line_num += 1
            cv.imshow('img', dmy)
    '''

    low_threshold = 10
    high_threshold = 20
    edges = cv.Canny(img1, low_threshold, high_threshold)

    cv.imshow('edges', edges)
    cv.waitKey(0)

    # 边缘检测
    contours, hierarchy = cv.findContours(img1.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # 统计选择的检测区域中的轮廓（有效轮廓）
    x_t = []  # 空列表 
    y_t = []  # 空列表 

    for cntr in contours:  # 遍历找到的所有轮廓
        x, y, w, h = cv.boundingRect(cntr)  # 轮廓的坐标

        if (cv.contourArea(cntr) <= 5) & (h < height1) & (h < width1):
            print("find one")
            line_image = cv.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 0), 2)
            x_t.append(x)
            y_t.append(y)

    
    cv.imshow('img3', line_image)
    cv.waitKey(0)

picchange()



