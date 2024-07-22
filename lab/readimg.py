import cv2
import numpy as np
import fast_LTS as FL
import math
import rectify as rec
import time
import os
# PAI值
pi = math.pi
# 2 确定旋转角度
angle = pi* (-90) / 360 * 2 
can=math.cos(angle)
san=math.sin(angle)

def access_point(pic):
    contours, hierarchy = cv2.findContours(pic, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b: b[1][0], reverse=True))
    xi = []
    yi = []
    valid_cntrs = []
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        if (cv2.contourArea(cntr) >= 25)&(cv2.contourArea(cntr) <= 46384):
            valid_cntrs.append(cntr)
            ptrx=x+int(w/2)
            ptry=y+int(h/2)
            xi.append(ptrx)
            yi.append(ptry)
    return xi,yi

def cal_point(pic,actual_pic,point):
    contours, hierarchy = cv2.findContours(pic, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    x_point=point[0]
    y_point=point[1]
    
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        if (cv2.contourArea(cntr) >= 25) & (cv2.contourArea(cntr) <= 26384)&(np.absolute(x_point-x-w/2)<=70)&(np.absolute(y_point-y-h/2)<=70):
            out_pic=actual_pic[y:int(y+h+1),x:int(x+w+1)]
            break

    return out_pic


def trans_polar(point,r,xe,ye):

    # 极坐标转换后图像的高，可自己设置
    h = int(r / 1)
    # 极坐标转换后图像的宽，一般是原来圆形的周长
    w = int(2 * r * pi)

    # line_image = np.zeros((h, w, 3), dtype=np.uint8)
    col=point[0]
    row=point[1]

    # 角度，最后的-0.1是用于优化结果，可以自行调整
    theta = pi * 2 / w * (col + 1) - 0.1
    # 半径，减1防止超界
    rho = r - row 
    
    # 1 确定极坐标
    x0 = rho * math.cos(theta)
    y0 = rho * math.sin(theta)
    
    # 3 确定直角坐标
    x1 = x0 * can - y0 * san
    y1 = x0 * san + y0 * can
    
    # 4 切换为OpenCV图像坐标
    x0 = int(xe + x1)
    y0 = int(ye - y1)
    # 赋值
    
    return x0,y0




def readpic(obj,obj_org):
    # t1 = time.time() 
    pic_org=cv2.imread(obj_org,cv2.IMREAD_GRAYSCALE)
    pic=cv2.imread(obj,cv2.IMREAD_GRAYSCALE)
    # imw,imh=pic.shape[0],pic.shape[1]

    pic1=cv2.imread(obj)
    pic2=pic.copy()
    pic2[pic==2]=255
    # pic2[pic!=2]=0
    pic1[pic==2]=255
    # pic1[pic!=2]=0
    pic1[pic==1]=127
    pic1[pic==3]=200
    pic_trans=pic1.copy()
    # print(pic2)
    xi,yi=access_point(pic2.copy())
    _,vx,vy=FL.fast_lts(np.array(xi),np.array(yi))

    # print(vx.shape)
    # print(vx.shape)
    vv=np.vstack((vx,vy)).T
    vv=np.expand_dims(vv, axis=1)
    ellipse=cv2.fitEllipse(vv)
    (xe,ye),(ae,be),ang = ellipse
    reel=rec.rectifing(xe,ye,xi,yi)
    num=0
    while(reel.rectify()!=1):
        if num>20:
            break
        num+=1
        
    print(reel.ang)
    # cv2.ellipse(pic1, ellipse, (0,255,0),3)
    max_shaft = max(ae,be)
    # 短轴
    p_l_x = xe - (be / 2 * math.sin(ang * (math.pi) / 180))
    p_l_y = ye + (be / 2 * math.cos(ang * (math.pi) / 180))
    p_r_x = xe + (be / 2 * math.sin(ang * (math.pi) / 180))
    p_r_y = ye - (be / 2 * math.cos(ang * (math.pi) / 180))
    # 长轴
    p_t_x = xe + (ae / 2 * math.cos((180-ang) * (math.pi) / 180))
    p_t_y = ye - (ae / 2 * math.sin((180-ang) * (math.pi) / 180))
    p_b_x = xe - (ae / 2 * math.cos((180-ang) * (math.pi) / 180))
    p_b_y = ye + (ae / 2 * math.sin((180-ang) * (math.pi) / 180))

    # cv2.circle(pic1, (int(p_l_x),int(p_l_y)), 8, (127,0,0), -1)
    # cv2.circle(pic1, (int(p_r_x),int(p_r_y)), 8, (127,127,0), -1)

    # cv2.circle(pic1, (int(p_t_x),int(p_t_y)), 8, (255,0,255), -1)
    # cv2.circle(pic1, (int(p_b_x),int(p_b_y)), 8, (0,0,255), -1)

    if max_shaft == ae :
        p_o_x = xe - (ae / 2 * math.sin(ang * (math.pi) / 180))
        p_o_y = ye + (ae / 2 * math.cos(ang * (math.pi) / 180))
        p_e_x = xe + (ae / 2 * math.sin(ang * (math.pi) / 180))
        p_e_y = ye - (ae / 2 * math.cos(ang * (math.pi) / 180))
        points2 = np.float32([[p_o_x, p_o_y], [p_e_x, p_e_y],[p_t_x, p_t_y], [p_b_x, p_b_y]])

    elif max_shaft == be :
        p_o_x = xe + (be / 2 * math.cos((180-ang) * (math.pi) / 180))
        p_o_y = ye - (be / 2 * math.sin((180-ang) * (math.pi) / 180))
        p_e_x = xe - (be / 2 * math.cos((180-ang) * (math.pi) / 180))
        p_e_y = ye + (be / 2 * math.sin((180-ang) * (math.pi) / 180))
        points2 = np.float32([[p_l_x, p_l_y], [p_r_x, p_r_y],[p_o_x, p_o_y], [p_e_x, p_e_y]])
    

    # cv2.circle(pic_trans, (int(p_o_x),int(p_o_y)), 8, (0,127,127), -1)
    # cv2.circle(pic_trans, (int(p_e_x),int(p_e_y)), 8, (45,0,127), -1)
    
    points1 = np.float32([[p_l_x, p_l_y], [p_r_x, p_r_y], [p_t_x, p_t_y], [p_b_x, p_b_y]])

    mat_perspective = cv2.getPerspectiveTransform(points1, points2)

    image_perspective = cv2.warpPerspective(pic_trans, mat_perspective,
                                       (pic_trans.shape[1], pic_trans.shape[0]))
    org_perspective = cv2.warpPerspective(pic_org, mat_perspective,
                                       (pic_org.shape[1], pic_org.shape[0]))
    
    
    M = cv2.getRotationMatrix2D((int(xe),int(ye)), -reel.ang+90, 1)
    image_perspective = cv2.warpAffine(image_perspective, M, (pic_trans.shape[1], pic_trans.shape[0])) # 极坐标 仿射变换
    M = cv2.getRotationMatrix2D((int(xe),int(ye)), -90, 1)
    image_perspective3 = cv2.warpAffine(image_perspective, M, (pic_trans.shape[1], pic_trans.shape[0]))
    M = cv2.getRotationMatrix2D((int(xe),int(ye)), -reel.ang, 1)
    org_perspective = cv2.warpAffine(org_perspective, M, (pic_org.shape[1], pic_org.shape[0])) # 极坐标 仿射变换
    
    
    # cv2.imshow('out', org_perspective)
    # cv2.waitKey(0)

    # 遮罩和原图的矫正图
    # return image_perspective,org_perspective
    
    # 输入图像圆的半径，一般是宽高一半
    r = int(max_shaft/1.8)
    # line_image0 = trans_polar(image_perspective,r,xe,ye)
    line_image = cv2.warpPolar(image_perspective, (int(r),int(2*pi*r)), (int(xe),int(ye)), r, cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS)
    line_image = cv2.rotate(line_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    line_image = line_image[:int(0.5*r),:]
    # cv2.imshow('out1', line_image)
    # cv2.waitKey(0)


    gray_line_image_1=cv2.cvtColor(line_image,cv2.COLOR_BGR2GRAY)
    gray_line_image_1[gray_line_image_1==255]= 0
    gray_line_image_1[gray_line_image_1==127]= 255
    gray_line_image_1[gray_line_image_1!=255]= 0
    gray_line_image_2=cv2.cvtColor(line_image,cv2.COLOR_BGR2GRAY)
    gray_line_image_2[gray_line_image_2==255]= 255
    gray_line_image_2[gray_line_image_2!=255]= 0
    gray_line_image_3=cv2.cvtColor(line_image,cv2.COLOR_BGR2GRAY)
    gray_line_image_3[gray_line_image_3==255]= 0
    gray_line_image_3[gray_line_image_3==200]= 255
    gray_line_image_3[gray_line_image_3!=255]= 0

    image_perspective3=cv2.cvtColor(image_perspective3,cv2.COLOR_BGR2GRAY)
    image_perspective3[image_perspective3==255]=0
    image_perspective3[image_perspective3==200]=255
    image_perspective3[image_perspective3!=255]=0

    # 指针
    xline_1, yline_1 = access_point(gray_line_image_1)
    # 刻度点
    xline_2, yline_2 = access_point(gray_line_image_2)
    # 数字
    xline_3, yline_3 = access_point(gray_line_image_3)
    # 对应数字区间
    tmp_line = np.absolute(xline_1[0] - np.array(xline_3))
    ind_tmp0 = np.argmin(tmp_line)
    tmp_line[ind_tmp0] = 1024
    ind_tmp1 = np.argmin(tmp_line)
    # 对应刻度点区间
    tmp_line = np.absolute(xline_3[ind_tmp0] - np.array(xline_2))
    ind_kedu0 = np.argmin(tmp_line)

    tmp_line = np.absolute(xline_3[ind_tmp1] - np.array(xline_2))
    ind_kedu1 = np.argmin(tmp_line)
    # 刻度点区间百分比
    meter_place=np.absolute(xline_2[ind_kedu1]-xline_2[ind_kedu0])
    lit_index=min(ind_kedu1,ind_kedu0)

    meter_inplace=np.absolute(xline_2[lit_index]-xline_1[0])
    percent=meter_inplace/meter_place
    objname=obj.split('/')[-1].split('.')[0]
    # 保存信息
    dic={"object":objname,"length":str(len(xline_3)),"index1":str(ind_tmp0),"index0":str(ind_tmp1),"percent":str(percent)}

    out_cs=[]
    for ind in range(len(xline_3)):
        po_line=xline_3[ind],yline_3[ind]
        pot_line1 = trans_polar(po_line,r,xe,ye)
        outpic = cal_point(image_perspective3,org_perspective,pot_line1)
        out_cs.append(outpic)
        # cv2.imshow('out2', outpic)
        # cv2.waitKey(0)

    return out_cs,dic
    
    
    # t2 = time.time()
    # cv2.imshow('out', num2)
    # cv2.imshow('out0',line_image)
    # cv2.waitKey(0)
    
    # print(int(round((t2-t1) * 1000)))
    
    
    

# readpic()