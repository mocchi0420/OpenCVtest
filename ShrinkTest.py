# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import copy

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
            if (img.item(i,j,3) <= 120):
                img.itemset((i, j, 0), 0)
                img.itemset((i, j, 1), 0)
                img.itemset((i, j, 2), 0)
                img.itemset((i, j, 3), 0)
            else:
                img.itemset((i, j, 3), 255)

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

def covert_to_dot_2by2(base):
    img = base
    for i in range(len(img)//2 - 1):
        for j in range(len(img[0])//2 - 1):
            r_val = (img.item(i*2, j*2, 0) + img.item(i*2 + 1, j*2, 0) + img.item(i*2, j*2 + 1, 0) + img.item(i*2 + 1, j*2 + 1, 0) )//4
            g_val = (img.item(i*2, j*2, 1) + img.item(i*2 + 1, j*2, 1) + img.item(i*2, j*2 + 1, 1) + img.item(i*2 + 1, j*2 + 1, 1) )//4
            b_val = (img.item(i*2, j*2, 2) + img.item(i*2 + 1, j*2, 2) + img.item(i*2, j*2 + 1, 2) + img.item(i*2 + 1, j*2 + 1, 2) )//4
            a_val = (img.item(i*2, j*2, 3) + img.item(i*2 + 1, j*2, 3) + img.item(i*2, j*2 + 1, 3) + img.item(i*2 + 1, j*2 + 1, 3) )//4

            for dx in range(2):
                for dy in range(2):
                    img.itemset((i*2 + dx, j*2 + dy, 0), r_val)
                    img.itemset((i*2 + dx, j*2 + dy, 1), g_val)
                    img.itemset((i*2 + dx, j*2 + dy, 2), b_val)
                    img.itemset((i*2 + dx, j*2 + dy, 3), a_val)
    return img

def covert_to_dot_3by3(base):
    img = base
    for i in range(len(img)//3 - 1):
        for j in range(len(img[0])//3 - 1):
            r_val = (img.item(i*3, j*3, 0) + img.item(i*3 + 1, j*3, 0) + img.item(i*3 + 2, j*3, 0)
                    +img.item(i*3, j*3 + 1, 0) + img.item(i*3 + 1, j*3 + 1, 0) + img.item(i*3 + 2, j*3 + 1, 0)
                    +img.item(i*3, j*3 + 2, 0) + img.item(i*3 + 1, j*3 + 2, 0) + img.item(i*3 + 2, j*3 + 2, 0)
                    )//9
            g_val = (img.item(i*3, j*3, 1) + img.item(i*3 + 1, j*3, 1) + img.item(i*3 + 2, j*3, 1)
                    +img.item(i*3, j*3 + 1, 1) + img.item(i*3 + 1, j*3 + 1, 1) + img.item(i*3 + 2, j*3 + 1, 1)
                    +img.item(i*3, j*3 + 2, 1) + img.item(i*3 + 1, j*3 + 2, 1) + img.item(i*3 + 2, j*3 + 2, 1)
                    )//9
            b_val = (img.item(i*3, j*3, 2) + img.item(i*3 + 1, j*3, 2) + img.item(i*3 + 2, j*3, 2)
                    +img.item(i*3, j*3 + 1, 2) + img.item(i*3 + 1, j*3 + 1, 2) + img.item(i*3 + 2, j*3 + 1, 2)
                    +img.item(i*3, j*3 + 2, 2) + img.item(i*3 + 1, j*3 + 2, 2) + img.item(i*3 + 2, j*3 + 2, 2)
                    )//9
            a_val = (img.item(i*3, j*3, 3) + img.item(i*3 + 1, j*3, 3) + img.item(i*3 + 2, j*3, 3)
                    +img.item(i*3, j*3 + 1, 3) + img.item(i*3 + 1, j*3 + 1, 3) + img.item(i*3 + 2, j*3 + 1, 3)
                    +img.item(i*3, j*3 + 2, 3) + img.item(i*3 + 1, j*3 + 2, 3) + img.item(i*3 + 2, j*3 + 2, 3)
                    )//9

            for dx in range(3):
                for dy in range(3):
                    img.itemset((i*3 + dx, j*3 + dy, 0), r_val)
                    img.itemset((i*3 + dx, j*3 + dy, 1), g_val)
                    img.itemset((i*3 + dx, j*3 + dy, 2), b_val)
                    img.itemset((i*3 + dx, j*3 + dy, 3), a_val)
    return img

def covert_to_dot_3by3_without_alpha(base):
    img = base

    for i in range(len(img)//3 - 1):
        for j in range(len(img[0])//3 - 1):
            #val = (img.item(i*3, j*3) + img.item(i*3 + 1, j*3) + img.item(i*3 + 2, j*3)
            #        +img.item(i*3, j*3 + 1) + img.item(i*3 + 1, j*3 + 1) + img.item(i*3 + 2, j*3 + 1)
            #        +img.item(i*3, j*3 + 2) + img.item(i*3 + 1, j*3 + 2) + img.item(i*3 + 2, j*3 + 2)
            #        )//9
            arr = []
            for dx in range(3):
                for dy in range(3):            
                    if(img.item(i*3 + dx, j*3 + dy) >= 240):
                        arr.append(255)
                    else:
                        arr.append(0)

            #arr = [img.item(i*3, j*3), img.item(i*3 + 1, j*3), img.item(i*3 + 2, j*3),
            #         img.item(i*3, j*3 + 1),img.item(i*3 + 1, j*3 + 1),img.item(i*3 + 2, j*3 + 1),
            #         img.item(i*3, j*3 + 2),img.item(i*3 + 1, j*3 + 2),img.item(i*3 + 2, j*3 + 2)]

            if(len([i for i in arr if (i == 255)]) >= 3):
                val = 255
            else:
                val = 0

            for dx in range(3):
                for dy in range(3):
                    img.itemset((i*3 + dx, j*3 + dy), val)
                    #if(val > 200):
                    #    img.itemset((i*3 + dx, j*3 + dy), 255)
                    #else:
                    #    img.itemset((i*3 + dx, j*3 + dy), 0)
    return img


def extract_edge(img):
    # 白い部分を膨張させる.
    dilated = cv2.dilate(img, np.ones((3,3),np.uint8), iterations=1)

    # 差をとる.
    diff = cv2.absdiff(dilated, img)

    # 白黒反転
    contour = 255 - diff        

    return contour

def extract_edge002(color):
    tmp = color
    for i in range(len(tmp)):
        for j in range(len(tmp[0])):
            if(tmp.item(i,j,3) < 255 and tmp.item(i,j,3) > 130):
                tmp.itemset((i,j,0), 255)
                tmp.itemset((i,j,1), 255)
                tmp.itemset((i,j,2), 255)
                #erosion.itemset((i,j), 255)
    cv2.imwrite("p0.png", tmp)
    gry = cv2.fastNlMeansDenoising(
        cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY), h=30)
    print(gry)

    _,bin_img = cv2.threshold(gry,150,255,cv2.THRESH_BINARY)
    res_img = np.zeros(bin_img.shape,np.uint8)
    cv2.imwrite("p1.png", bin_img)
    #cv2.imwrite("gray.png", gry)
    '''
    # カーネルサイズの設定
    kernel3 = np.ones((3, 3), np.uint8)
    # [1] 画像のグレースケール化とノイズ除去
    gray = cv2.fastNlMeansDenoising(
        cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), h=30)
    cv2.imwrite("gray.png", gray)
    # [2] dilate -> diff で線抽出
    dilation = cv2.dilate(gray, kernel3, iterations=1)
    diff = cv2.subtract(dilation, gray)
    diff_inv = 255 - diff
    # [3] 線を単色 -> 2値に
    _, edge = cv2.threshold(diff_inv, 230, 255, cv2.THRESH_BINARY)
    # [4] 線画とモノクロ2値を重ねる
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    #img = cv2.bitwise_and(color, edge)
    '''
    image, contours, hierarchy = cv2.findContours(gry, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(res_img,pts=[contours[-1]],color=(255,255,255))
    hoge = 255 - res_img
    cv2.imwrite("p2.png", hoge)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(hoge, kernel,iterations = 3)

    for i in range(len(hoge)):
        for j in range(len(hoge[0])):
            if(hoge.item(i,j) == dilation.item(i,j)):
                hoge.itemset((i,j), 255)

    return hoge

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
    # size = (84,84)
    size = (72,72)

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
    size = (72,72)
    # 入力画像を取得
    img = cv2.imread("test.png", cv2.IMREAD_UNCHANGED)
    pre = cv2.resize(img, size, interpolation = cv2.INTER_LANCZOS4) 
    cv2.imwrite("pre001.png", pre)

    img = imread_cutting_bg("test.png")
    pre = cv2.resize(img, size, interpolation = cv2.INTER_LANCZOS4) 
    cv2.imwrite("pre002.png", pre)
    # 旧処理
    # hoge = old_reduce(img)

    # 枠線出す
    edge = extract_edge002(img)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #edge = extract_edge(gray)
    #edge = covert_to_dot_3by3_without_alpha(edge)
    edge = cv2.resize(edge, size, interpolation = cv2.INTER_LANCZOS4) 
    cv2.imwrite("edge.png", edge)

    # 画像減色
    img = reduce_color(img)
    # cv2.imwrite("hoge.png", img)

    # 画像の色を3*3で同色変換
    img = covert_to_dot_3by3(img)

    size = (72,72)
    result = cv2.resize(img, size, interpolation = cv2.INTER_LANCZOS4) 
    cv2.imwrite("hoge.png", result)

    new_edge = copy.copy(result)
    for i in range(len(edge)):
        for j in range(len(edge[0])):
            if(edge[i][j] == 0):
                new_edge.itemset((i,j,0),0)
                new_edge.itemset((i,j,1),0)
                new_edge.itemset((i,j,2),0)
                new_edge.itemset((i,j,3),255)
            else:
                new_edge.itemset((i,j,0),255)
                new_edge.itemset((i,j,1),255)
                new_edge.itemset((i,j,2),255)
                new_edge.itemset((i,j,3),0)

    fuga = blend(result, new_edge, 255-edge)

    cv2.imwrite("fuga.png", fuga)

    old_reduce(result)

if __name__ == '__main__':
    main()
    #edge = cv2.imread("edge.png")
    #shrink_edge(edge, 84)
