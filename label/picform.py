# code:utf-8
import numpy as np
import cv2
import os

# 获取标注和图片
def pic_get(path, filein):
    file = filein.split('.')[-2]
    # 标注数据
    path_read = path + '\\data1\\annotations\\val\\' + file + '.png'
    # 标注缓存
    save = path + '\\data1\\annotations\\pic\\' + file + '.png'

    img = cv2.imread(path_read)
    cv2.imwrite(save,img)


def pic_read(path, filein):
    file = filein.split('.')[-2]
    # 标注原图
    path_read1 = path + '\\data1\\images\\val\\' + file + '.jpg'
    # 标注缓存
    path_read2 = path + '\\data1\\annotations\\pic\\' + file + '.png'

    img = pic_show(path_read1,path_read2)
    return img


def pic_show(read1,read2):
    img_read1=cv2.imread(read1)
    img_read2=cv2.imread(read2)
    w = img_read1.shape[0]
    h = img_read1.shape[1]

    ze0=np.zeros([w,h,2],dtype=np.uint8)

    imr10 = img_read1.copy()
    imr20 = img_read2.copy()
    imr21 = img_read2.copy()
    imr22 = img_read2.copy()
    imr23 = img_read2.copy()
    imr24 = img_read2.copy()
    # 刻度
    imr20[imr20==2]=255
    imr200=np.append(imr20[:,:,:1],ze0,axis=2)
    # 指针
    imr21[imr21==1]=255
    imr210=np.append(imr21[:,:,:2],ze0[:,:,:1],axis=2)
    # 数字
    imr22[imr22==3]=255
    imr220=np.append(ze0,imr22[:,:,:1],axis=2)
    # 单位
    imr23[imr23==4]=255
    imr230=np.append(ze0[:,:,:1],imr23[:,:,:2],axis=2)

    imr24[imr24==5]=255
    imr240=np.append(imr24[:,:,:1],ze0[:,:,:1],axis=2)
    imr241=np.append(ze0[:,:,:1],imr240,axis=2)

    imr2001=imr200|imr210
    imr2002=imr2001|imr220
    imr2003=imr2002|imr230
    imr2004=imr2003|imr241
    imr300 = imr2004|imr10
    # cv2.imshow('img_read2',imr300)
    # cv2.waitKey(0)
    
    return imr300
# 实时更新标注信息
def drawing_now(x,y,r,c,path, filein):
    file = filein.split('.')[-2]
    # 标注原图与缓存
    path_read2 = path + '\\data1\\annotations\\pic\\' + file + '.png'
    save = path + '\\data1\\annotations\\pic\\' + file + '.png'

    img_read=cv2.imread(path_read2)
    cc = [c,c,c]
    img = cv2.circle(img_read, (x,y), r, cc,-1)
    cv2.imwrite(save,img)
    print("draw now")

def save_out(path, filein):
    file = filein.split('.')[-2]
    # 标注缓存
    path_read2 = path + '\\data1\\annotations\\pic\\' + file + '.png'
    # 标注图片
    save = path + '\\data1\\annotations\\val\\' + file + '.png'
    img_read=cv2.imread(path_read2)
    cv2.imwrite(save,img_read)
    print("save now")
