# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math

import warnings
warnings.filterwarnings('ignore')
import skimage.morphology

import matplotlib.pyplot as plt

# 透過部分の画像を消した上で画像を読み込む
def imread_cutting_bg(src):
    # 入力画像を取得
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)

    # 透過部分を意図的にカットする
    for i in range(len(img)):
        for j in range(len(img[0])):
            if (img.item(i,j,3) <= 140):
                img.itemset((i, j, 0), 0)
                img.itemset((i, j, 1), 0)
                img.itemset((i, j, 2), 0)
    
    return img

# 減色処理
def reduce_color(img):
    # img_src = cv2.imread('./image/karasu.jpg')
    Z = img.reshape((-1,4))

    # float32に変換
    Z = np.float32(Z)

    # K-Means法
    criteria = (cv2.TERM_CRITERIA_EPS, 10, 2.0)
    K = 64
    ret,label,center=cv2.kmeans(Z,
                                K,
                                None,
                                criteria,
                                10,
                                cv2.KMEANS_PP_CENTERS)

    # UINT8に変換
    center = np.uint8(center)
    res = center[label.flatten()]
    img_dst = res.reshape((img.shape))
    return img_dst

def extract_edge(img):
    # 白い部分を膨張させる.
    dilated = cv2.dilate(img, np.ones((3,3),np.uint8), iterations=1)

    # 差をとる.
    diff = cv2.absdiff(dilated, img)

    # 白黒反転
    contour = 255 - diff        

    return contour


# 漫画化フィルタ
def manga_filter(src):
    hoge = src
    color = src[:,:,0:3]
    alpha = src[:,:,3]
    inv_alpha = 255 - alpha

    # グレースケール変換
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    edge = cv2.adaptiveThreshold(cv2.max(gray, inv_alpha), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 10)
    erode_edge = cv2.erode(edge, np.ones((1,1), np.uint8))
    
    inv_alpha = 255 - cv2.erode(alpha, np.ones((1,1), np.uint8)) # アルファの境を広げる
    no_edge = cv2.inpaint(color, cv2.max(255 - erode_edge, inv_alpha), 11, cv2.INPAINT_NS)

    color_edge = color.copy()
    color_edge[edge > 128] = 0
    
    rgb = cv2.split(no_edge)
    result = cv2.merge([cv2.min(c, alpha) for c in rgb])

    result2 = blend(result, color_edge, 255-edge)
    
    new_edge = extract_edge(gray)

    return {
        "orig_color": color,
        "no_edge": no_edge,
        "alpha": alpha,
        "edge_color": color_edge,
        "edge_alpha": new_edge,
    }

def old_reduce(target_img):
    img = reduce_color(target_img)

    # 画像の漫画化
    img = manga_filter(img)
    size = (84,84)

    no_edge = cv2.resize(img["no_edge"], size)
    alpha = cv2.resize(img["alpha"], size, interpolation = cv2.INTER_NEAREST)
    no_edge[alpha<255] = (0,255,0)

    edge_color = cv2.resize(img["orig_color"], size, interpolation = cv2.INTER_LANCZOS4) #, interpolation = cv2.INTER_NEAREST)
    edge_alpha = cv2.resize(img["edge_alpha"], size, interpolation = cv2.INTER_NEAREST)

    result = blend(no_edge, edge_color, 255-edge_alpha)
    cv2.imwrite("result.png", result)
    return result

def skeletonize(edge):
    edge = edge[:,:,0]
    edge = skimage.morphology.skeletonize(edge.astype(np.bool))
    edge = edge.astype(np.uint8)
    edge[edge > 0] = 255
    w, h = np.shape(edge)
    return edge.reshape(w,h,1)

def shrink_edge(edge, size):
    edge = skeletonize(~edge)
    cv2.imwrite("edge_result_0.png", edge)
    result = np.zeros((size,size,1), np.uint8)
    w, h, channel = np.shape(edge)
    scale = 1.0 * w / size
    for y in range(size):
        for x in range(size):
            sx = int(x * scale)
            sy = int(y * scale)
            ex = int((x+1) * scale)
            ey = int((y+1) * scale)
            mx = 0
            for dx in range(sx,ex):
                for dy in range(sy,ey):
                    mx = max(mx, edge[dx,dy,0])
            result[x,y] = mx
    cv2.imwrite("edge_result.png", result)
    result = skeletonize(result)
    cv2.imwrite("edge_result_2.png", result)
    
def blend(img1, img2, mask):
    masked1 = img1.copy()
    masked2 = img2.copy()
    masked1[mask >= 255] = 0
    masked2[mask < 255] = 0
    return cv2.bitwise_or(masked1, masked2)

def main():
    # 入力画像を取得
    img = imread_cutting_bg("test.png")

    # 旧処理
    # hoge = old_reduce(img)

    # 画像減色
    img = reduce_color(img)
    cv2.imwrite("hoge.png", img)

if __name__ == '__main__':
    main()
    #edge = cv2.imread("edge.png")
    #shrink_edge(edge, 84)
