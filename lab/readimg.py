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

# 下面两个值是要识别的颜色范围
lower_yellow = np.array([20, 20, 20])  # 黄色的下限
upper_yellow = np.array([30, 255, 255])  # 黄色上限
# color(lower_yellow, upper_yellow, 'yellow')
# 红色需要特殊处理
lower_red = np.array([0, 43, 46, 156, 43, 46])  # 红色阈值下界
higher_red = np.array([10, 255, 255, 180, 255, 255])  # 红色阈值上界
# color(lower_red, higher_red, 'red')
lower_green = np.array([35, 110, 106])  # 绿色阈值下界
higher_green = np.array([77, 255, 255])  # 绿色阈值上界
# color(lower_green, higher_green, 'green')
lower_blue = np.array([100, 43, 46])  # 蓝色阈值下界
upper_blue = np.array([160, 255, 255])  # 蓝色阈值下界



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

def access_scale(pic,length):
    contours, hierarchy = cv2.findContours(pic, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b: b[1][0], reverse=True))
    xi = np.array([])
    yi = np.array([])
    hi = np.array([])
    valid_cntrs = []
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        if (cv2.contourArea(cntr) >= 25)&(cv2.contourArea(cntr) <= 46384):
            valid_cntrs.append(cntr)
            ptrx=x+int(w/2)
            ptry=y+int(h/2)
            xi=np.append(xi,ptrx)
            yi=np.append(yi,ptry)
            hi=np.append(hi,h)
    indices = np.argpartition(-hi, length)[:length]
    return np.sort(xi[indices])[::-1],yi[indices]

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
    pic=cv2.imread(obj)
    # imw,imh=pic.shape[0],pic.shape[1]
    pic_hsv=cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
    # 刻度
    picg = cv2.inRange(pic_hsv, lower_green, higher_green)
    # 指针
    picr = cv2.inRange(pic_hsv, lower_red[:3], higher_red[:3])
    picr = picr + cv2.inRange(pic_hsv, lower_red[3:], higher_red[3:])
    # 单位
    picb = cv2.inRange(pic_hsv, lower_blue, upper_blue)
    # 数字
    picy = cv2.inRange(pic_hsv, lower_yellow, upper_yellow)
    # _,picg= cv2.threshold(picg,thresh=127,maxval=255,type=cv2.THRESH_TRUNC)
    _,picr= cv2.threshold(picr,thresh=127,maxval=255,type=cv2.THRESH_TRUNC)
    _,picb= cv2.threshold(picb,thresh=60,maxval=255,type=cv2.THRESH_TRUNC)
    _,picy= cv2.threshold(picy,thresh=200,maxval=255,type=cv2.THRESH_TRUNC)
    pic1=picg|picr|picb|picy
    # pic1=picg
    pic2=picg
    # pic1=pic.copy()
    # pic2=pic.copy()
    # pic2[pic==2]=255

    # pic1[pic==2]=255

    # pic1[pic==1]=127
    # pic1[pic==3]=200
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
    # cv2.ellipse(pic_org, ellipse, (0,127,0),2)
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

    # cv2.circle(pic_org, (int(p_l_x),int(p_l_y)), 5, (0,127,0), -1)
    # cv2.circle(pic_org, (int(p_r_x),int(p_r_y)), 5, (0,127,0), -1)
    # cv2.line(pic_org, (int(p_l_x),int(p_l_y)), (int(p_r_x),int(p_r_y)), (0,127,0), 2)
    # cv2.circle(pic_org, (int(p_t_x),int(p_t_y)), 5, (0,127,0), -1)
    # cv2.circle(pic_org, (int(p_b_x),int(p_b_y)), 5, (0,127,0), -1)
    # cv2.line(pic_org, (int(p_t_x),int(p_t_y)), (int(p_b_x),int(p_b_y)), (0,127,0), 2)

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
    

    # cv2.circle(pic_org, (int(p_o_x),int(p_o_y)), 5, (0,127,0), -1)
    # cv2.circle(pic_org, (int(p_e_x),int(p_e_y)), 5, (0,127,0), -1)
    # cv2.line(pic_org, (int(p_o_x),int(p_o_y)), (int(p_e_x),int(p_e_y)), (0,127,0), 2)
    # cv2.circle(pic_org, (int(xe),int(ye)), int(max_shaft/2), (0,127,0), 2)
    
    points1 = np.float32([[p_l_x, p_l_y], [p_r_x, p_r_y], [p_t_x, p_t_y], [p_b_x, p_b_y]])

    mat_perspective = cv2.getPerspectiveTransform(points1, points2)
    # 原图
    image_perspective = cv2.warpPerspective(pic_trans, mat_perspective,
                                       (pic_trans.shape[1], pic_trans.shape[0]))
    org_perspective1 = cv2.warpPerspective(pic_org, mat_perspective,
                                       (pic_org.shape[1], pic_org.shape[0]))
    
    
    M = cv2.getRotationMatrix2D((int(xe),int(ye)), -reel.ang+90, 1)
    image_perspective = cv2.warpAffine(image_perspective, M, (pic_trans.shape[1], pic_trans.shape[0])) # 极坐标 仿射变换
    M = cv2.getRotationMatrix2D((int(xe),int(ye)), -90, 1)
    image_perspective3 = cv2.warpAffine(image_perspective, M, (pic_trans.shape[1], pic_trans.shape[0]))
    M = cv2.getRotationMatrix2D((int(xe),int(ye)), -reel.ang, 1)
    org_perspective = cv2.warpAffine(org_perspective1, M, (pic_org.shape[1], pic_org.shape[0])) # 极坐标 仿射变换
    
    # cv2.circle(org_perspective, (int(p_l_x),int(p_l_y)), 5, (0,127,0), -1)
    # cv2.circle(org_perspective, (int(p_r_x),int(p_r_y)), 5, (0,127,0), -1)
    # cv2.line(org_perspective, (int(p_l_x),int(p_l_y)), (int(p_r_x),int(p_r_y)), (0,127,0), 2)
    # cv2.circle(org_perspective, (int(p_o_x),int(p_o_y)), 5, (0,127,0), -1)
    # cv2.circle(org_perspective, (int(p_e_x),int(p_e_y)), 5, (0,127,0), -1)
    # cv2.line(org_perspective1, (int(xe),0), (int(xe),600), (0,127,0), 2)
    # cv2.ellipse(org_perspective1, ellipse, (0,127,0),2)
    # cv2.circle(org_perspective1, (int(xe),int(ye)), int(max_shaft/2), (0,127,0), 2)
    # cv2.imshow('out', image_perspective3)
    # cv2.imwrite('./pic_org.png',org_perspective)
    # cv2.imshow('out1', org_perspective)
    
    
    
    # cv2.imwrite('./connect_r.png',picr)
    # cv2.imwrite('./connect_g.png',picg)
    # cv2.imwrite('./connect_b.png',picb)
    # cv2.imwrite('./connect_y.png',picy)
    # cv2.waitKey(0)

    # 遮罩和原图的矫正图
    # return image_perspective,org_perspective
    
    # 输入图像圆的半径，一般是宽高一半 
    r = int(max_shaft/1.8)
    # line_image0 = trans_polar(image_perspective,r,xe,ye)
    line_image = cv2.warpPolar(image_perspective, (int(r),int(2*pi*r)), (int(xe),int(ye)), r, cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS)
    line_image = cv2.rotate(line_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    line_image = line_image[:int(0.5*r),:]

    # 指针
    # gray_line_image_1=cv2.cvtColor(line_image,cv2.COLOR_BGR2GRAY)
    gray_line_image_1=line_image.copy()
    gray_line_image_1[gray_line_image_1==255]= 0
    gray_line_image_1[gray_line_image_1==127]= 255
    gray_line_image_1[gray_line_image_1!=255]= 0
    gray_line_image_1[80:,:]=0
    # cv2.imshow('out', gray_line_image_1)
    # cv2.waitKey(0)
    # gray_line_image_2=cv2.cvtColor(line_image,cv2.COLOR_BGR2GRAY)
    gray_line_image_2=line_image.copy()
    gray_line_image_2[gray_line_image_2==255]= 255
    gray_line_image_2[gray_line_image_2!=255]= 0
    
    # gray_line_image_3=cv2.cvtColor(line_image,cv2.COLOR_BGR2GRAY)
    gray_line_image_3=line_image.copy()
    gray_line_image_3[gray_line_image_3==255]= 0
    gray_line_image_3[gray_line_image_3==200]= 255
    gray_line_image_3[gray_line_image_3!=255]= 0
    
    # image_perspective3=cv2.cvtColor(image_perspective3,cv2.COLOR_BGR2GRAY)
    image_perspective3=image_perspective3.copy()
    image_perspective3[image_perspective3==255]=0
    image_perspective3[image_perspective3==200]=255
    image_perspective3[image_perspective3!=255]=0
    
    # 指针
    xline_1, yline_1 = access_point(gray_line_image_1)
    # 刻度点
    xline_2, yline_2 = access_point(gray_line_image_2)
    
    # 数字
    xline_3, yline_3 = access_point(gray_line_image_3)
    # 大刻度
    xline_4, yline_4 = access_scale(gray_line_image_2,len(xline_3))
    
    # line_show_img = np.zeros((line_image.shape[0], line_image.shape[1], 3), dtype = np.uint8)
    # for i in range(len(xline_2)):
    #     cv2.line(line_show_img, (xline_2[i],0), (xline_2[i],30), (0,255,0), 2)

    # cv2.line(line_show_img, (xline_1[0],0), (xline_1[0],line_image.shape[1]), (255,0,0), 2)
    
    # cv2.imwrite('./point.png', line_show_img)
    


    # 对应数字区间
    tmp_line = np.absolute(xline_1[0] - xline_4)
    ind_tmp0 = np.argmin(tmp_line)
    tmp_line[ind_tmp0] = 1024
    ind_tmp1 = np.argmin(tmp_line)
    # 对应刻度点区间
    tmp_line = np.absolute(xline_4[ind_tmp0] - np.array(xline_2))
    ind_kedu0 = np.argmin(tmp_line)

    tmp_line = np.absolute(xline_4[ind_tmp1] - np.array(xline_2))
    ind_kedu1 = np.argmin(tmp_line)

    tmp_line = np.absolute(xline_1[0] - np.array(xline_2))
    ind_kedu3 = np.argmin(tmp_line)

    # 刻度点大区间百分比
    meter_place=np.absolute(xline_2[ind_kedu1]-xline_2[ind_kedu0])
    lit_index=min(ind_kedu1,ind_kedu0)
    # 刻度点位置细分百分比
    
    if np.absolute(xline_2[ind_kedu3]- xline_1[0])<16 :
        ref_kedu = np.absolute(ind_kedu0-ind_kedu1)
        ind_kedu4 = np.absolute(ind_kedu3-lit_index)
        ref_percent = ind_kedu4/ref_kedu
        print(ref_percent)
    else:
        ref_kedu = np.absolute(ind_kedu0-ind_kedu1)
        ind_kedu4 = np.absolute(ind_kedu3-lit_index)
        tmp_line[ind_kedu3] = 1024
        ind_kedu5 = np.argmin(tmp_line)
        ind_kedu5 = np.absolute(ind_kedu5-lit_index)
        ref_percent = (ind_kedu5+ind_kedu4)/(ref_kedu*2)
        print(ref_percent)
    print(ind_kedu4)

    meter_inplace=np.absolute(xline_2[lit_index]-xline_1[0])
    # percent = (meter_inplace/meter_place)

    # print(percent)
    objname=obj.split('/')[-1].split('.')[0]
    print(objname)
    cv2.imshow('out', line_image)
    cv2.waitKey(0)
    # 保存信息
    dic={"object":objname,"length":str(len(xline_3)),"index1":str(ind_tmp0),"index0":str(ind_tmp1),"percent":str(ref_percent)}

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