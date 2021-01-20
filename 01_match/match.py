
import cv2
import os
import sys
import time
import numpy as np
# from matplotlib import pyplot as plt


def match_orb(img1_path:str, img2_path:str):
    # 读取灰度图
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)

    # ORB特征计算
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 暴力匹配
    bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True) # 汉明距离
    matches = bf.match(des1, des2)

    # 可视化前10个效果
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1=img1, keypoints1=kp1,
                        img2=img2, keypoints2=kp2,
                        matches1to2=matches[:10],
                        outImg=None, flags=2)

    return img3


def match_sift(img1_path: str, img2_path: str):
    # 读取灰度图
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)
    # img1 = cv2.imread(img1_path)
    # img2 = cv2.imread(img2_path)

    # SIFT特征计算
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 暴力匹配
    bf = cv2.BFMatcher(normType=cv2.NORM_L1, crossCheck=True) # 欧式距离
    matches = bf.match(des1, des2)
    # 可视化前10个效果
    matches = sorted(matches, key=lambda x: x.distance)
    img_out = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

    # # Flann特征匹配 更高级的筛选方式
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2, k=2) 
    # goodMatch = []
    # for m, n in matches:
    #     # goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2
    #     if m.distance < 0.5*n.distance:
    #         goodMatch.append(m)
    
    # goodMatch = np.expand_dims(goodMatch, 1)
    # img_out = cv2.drawMatchesKnn(img1, kp1,
    #                              img2, kp2,
    #                              goodMatch, None, flags=2)
 
    return img_out


def match_suft(img1_path: str, img2_path: str):
    # 读取灰度图
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)

    # 使用SURF_create特征检测器 和 BFMatcher描述符
    surf = cv2.xfeatures2d.SURF_create()
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(normType=cv2.NORM_L1, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # 可视化前10个效果	
    img_out = cv2.drawMatches(img1, kp1, img2, kp2,
                              matches[:10], None, flags=2)
    
    return img_out


if __name__ == '__main__':
    # 输入数据
    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    start_time = time.time()
    match_img_orb = match_orb(img1_path, img2_path)
    match_img_sift = match_sift(img1_path, img2_path)
    match_img_suft = match_suft(img1_path, img2_path)
    cv2.imwrite("./orb_match.jpg", match_img_orb)
    cv2.imwrite("./sift_match.jpg", match_img_sift)
    cv2.imwrite("./suft_match.jpg", match_img_suft)
    # cv2.imshow("match", match_img_orb)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    end_time = time.time()
    print("Total Spend time：", str((end_time - start_time) / 60)[0:6] + "分钟")
