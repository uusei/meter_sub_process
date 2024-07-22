import numpy as np
import cv2
import os
import readimg as ri
import time
import json
ann_file='./data00/annotations/test'
pic_file='./data00/images/test'
trans_file='./data00/annotations/trans_test'
annotate=['0','1','2','3','4','5','6','7','8','9','0','.']


def getMotionDsf(shape, angle, dist):
        xCenter = (shape[0] - 1) / 2
        yCenter = (shape[1] - 1) / 2
        sinVal = np.sin(angle * np.pi / 180)
        cosVal = np.cos(angle * np.pi / 180)
        PSF = np.zeros(shape)  # 点扩散函数
        for i in range(dist):  # 将对应角度上motion_dis个点置成1
            xOffset = round(sinVal * i)
            yOffset = round(cosVal * i)
            PSF[int(xCenter - xOffset), int(yCenter + yOffset)] = 1
        return PSF / PSF.sum()  # 归一化
 
def wienerFilter(input, PSF, eps, K=0.05):  # 维纳滤波，K=0.01
        fftImg = np.fft.fft2(input)
        fftPSF = np.fft.fft2(PSF) + eps
        fftWiener = np.conj(fftPSF) / (np.abs(fftPSF)**2 + K)
        imgWienerFilter = np.fft.ifft2(fftImg * fftWiener)
        imgWienerFilter = np.abs(np.fft.fftshift(imgWienerFilter))
        imgWienerFilter = imgWienerFilter.astype(np.uint8)
        return imgWienerFilter

def conect(dst_edge,val_edge):
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(dst_edge,connectivity=8)
    for (i, label) in enumerate(np.unique(labels)):
        # 如果是背景，忽略
        if label == 0:
            # print("[INFO] label: 0 (background)")
            continue
        numPixels = stats[i][-1]
        # 判断区域是否满足面积要求
        if numPixels > 10:
            val_edge[labels == label] = 255
    return val_edge


def cal_region(pics,objname):
    num=0

    for tmp in pics:
    
        x_len=tmp.shape[1]
        y_len=tmp.shape[0]
        dst = np.zeros_like(tmp)
        # hist_tmp = cv2.calcHist([tmp], [0], None, [256], [0,255])
        cv2.normalize(tmp,dst, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # PSF = getMotionDsf((h+1, w+1), 7, 6)  # 运动模糊函数    维纳 滤波
        # dst = wienerFilter(dst, PSF, 1e-6) 
        # dst=cv2.merge([dst])
        # kSize = (5,5)
        # kernalMean = np.ones(kSize, np.float32) / (kSize[0]*kSize[1])  # 生成归一化盒式核 
        # dst = cv2.filter2D(dst, -1, kernalMean)
        # m, n = 3, 3
        # order = 1/(m*n)
        # kernalMean = np.ones((m,n), np.float32)  # 生成盒式核  均方值 滤波
        # hPad = int((m-1) / 2)
        # wPad = int((n-1) / 2)
        # imgPad = np.pad(dst.copy(), ((hPad, m-hPad-1), (wPad, n-wPad-1)), mode="edge")
    
        # imgGeoMean = dst.copy()
        # for i in range(hPad, h + hPad):
        #     for j in range(wPad, 2 + wPad):
        #         prod = np.prod(imgPad[i-hPad:i+hPad+1, j-wPad:j+wPad+1]*1.0)
        #         imgGeoMean[i-hPad][j-wPad] = np.power(prod, order)
        
        hist = cv2.calcHist([dst], [0], None, [256], [0,255])
        # inter_hist=np.absolute(np.argmax(hist_tmp)-np.argmax(hist))
        
        # if inter_hist>30:
        ind= int(np.argmax(hist[:,0]))
        # else: 
        #     ind= int(np.argmax(hist[:,0])*0.66)
        
        # 引入边缘增强
        dst_edge = cv2.Canny(dst,0,ind*0.4,L2gradient=True)           
        # kernel = np.ones((3,3),np.uint8)
        # dst_edge = cv2.morphologyEx(dst_edge0, cv2.MORPH_CLOSE, kernel, iterations=1)
        # cv2.imshow('edge', dst_edge)
        val_edge=np.zeros_like(dst_edge)
        val_edge=conect(dst_edge,val_edge)
        
        kernel = np.ones((3,3),np.uint8)
        val_edge = cv2.dilate(val_edge, kernel, iterations=1)
        # val_edge = cv2.morphologyEx(val_edge, cv2.MORPH_CLOSE, kernel, iterations=1)
        # cv2.imshow('edge1', val_edge)
        
        # dst = cv2.dilate(dst, kernel, iterations=1)
        # dst = cv2.bitwise_xor(dst, dst0)
        # hist = cv2.calcHist([dst], [0], None, [256], [0,255])
        # dst1 = cv2.add(dst,val_edge)
        # dst1[dst1>255]=255
        dst0=dst.copy()
        hist = cv2.calcHist([dst], [0], None, [256], [0,255])
        ind= int(np.argmax(hist[:,0])*0.65)
        # cv2.imshow('out', dst)
        # cv2.waitKey(0)
        dst[dst<ind]=255
        dst[dst!=255]=0
        
        # cv2.imshow('out1', dst)
        # cv2.waitKey(0)
        
        kernel = np.ones((3,1),np.uint8)
        dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel, iterations=1)
        
        
        contours_sub, hierarchy_sub = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours_sub:
            boundingBoxes = [cv2.boundingRect(c) for c in contours_sub]
            (contours_sub, boundingBoxes) = zip(*sorted(zip(contours_sub, boundingBoxes), key=lambda b: b[1][0], reverse=False))
        else:
            ind=int(ind+50)
            dst=dst0
            
            dst[dst<ind]=255
            dst[dst!=255]=0
            # cv2.imshow('out1', dst)
            # cv2.waitKey(0)
            dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel, iterations=1)
            contours_sub, hierarchy_sub = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            boundingBoxes = [cv2.boundingRect(c) for c in contours_sub]
            (contours_sub, boundingBoxes) = zip(*sorted(zip(contours_sub, boundingBoxes), key=lambda b: b[1][0], reverse=False))


        for sub_cn in contours_sub:
            x, y, w, h = cv2.boundingRect(sub_cn)
            if (cv2.contourArea(sub_cn) >= 900) & ((h/w)<1.4):
                ind= int(np.argmax(hist[:,0])*0.45)
                
                dst0[dst0<ind]=255
                dst0[dst0!=255]=0                 
            
                kernel = np.ones((3,1),np.uint8)
                dst = cv2.morphologyEx(dst0, cv2.MORPH_OPEN, kernel, iterations=1)
                # cv2.imshow('out2', dst)
                
                contours_sub1, hierarchy_sub = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for sub_cn1 in contours_sub1:
                    x, y, w, h = cv2.boundingRect(sub_cn1)
                    if (cv2.contourArea(sub_cn) >= 1250)& ((h/w)<1.4):
                        dst1 = cv2.subtract(dst,val_edge)
                        kernel = np.ones((3,1),np.uint8)
                        # dst = cv2.erode(dst1, kernel, iterations=1)
                        dst = cv2.morphologyEx(dst1, cv2.MORPH_OPEN, kernel, iterations=1)
                        # cv2.imshow('out3', dst)
                        # cv2.waitKey(0)
        area = x_len*y_len*0.25*0.1
        x_len1=x_len*0.5
        x_len2=x_len*0.5
        contours_num, hierarchy_sub = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        boundingBoxes = [cv2.boundingRect(c) for c in contours_num]
        (contours_num, boundingBoxes) = zip(*sorted(zip(contours_num, boundingBoxes), key=lambda b: b[1][0], reverse=False))
        count=0
        flag=''
        for sub_num in contours_num:
            x, y, w, h = cv2.boundingRect(sub_num)
            
            if (cv2.contourArea(sub_num) >= area) :
                tmp_num=tmp[:,x:int(x+w+3)].copy()
                tmp_mask=dst[:,x:int(x+w+1)].copy()
                
                dst[:,x:int(x+w+1)]=0
                if x< x_len1:
                    dst[:,:int(x)]=0
                    flag='front'
                elif (x)>=x_len2:
                    dst[:,int(x):]=0
                    flag='fore'
                tmp_number=trans_file+'/sub_number/'+objname+'_'+str(num)+'_'+ flag +'.png'
                cv2.imwrite(tmp_number,tmp_num)
                # cv2.imshow('out_num_'+flag, tmp_num)
                # cv2.imshow('out_mask_'+flag, tmp_mask)
                # cv2.imshow('out_'+flag, dst)
                count=count+1
                break
        contours_num, hierarchy_sub = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours_num:
            boundingBoxes = [cv2.boundingRect(c) for c in contours_num]
            (contours_num, boundingBoxes) = zip(*sorted(zip(contours_num, boundingBoxes), key=lambda b: b[1][0], reverse=False))
        for sub_num in contours_num:
            x, y, w, h = cv2.boundingRect(sub_num)
            
            if (cv2.contourArea(sub_num) >= area) :
                tmp_num=tmp[:,x:int(x+w+3)].copy()
                tmp_mask=dst[:,x:int(x+w+1)].copy()
                
                dst[:,x:int(x+w+1)]=0
                if x< x_len1:
                    dst[:,:int(x)]=0
                    flag='front'
                elif (x)>=x_len2:
                    dst[:,int(x):]=0
                    flag='fore'
                tmp_number=trans_file+'/sub_number/'+objname+'_'+str(num)+'_'+ flag +'.png'
                cv2.imwrite(tmp_number,tmp_num)
                # cv2.imshow('out_num_'+flag, tmp_num)
                # cv2.imshow('out_mask_'+flag, tmp_mask)
                # cv2.imshow('out_'+flag, dst)
                count=count+1
                break
        contours_num, hierarchy_sub = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        for sub_num in contours_num:
            x, y, w, h = cv2.boundingRect(sub_num)
            
            if (cv2.contourArea(sub_num) >= 10):
                
                
                
                # cv2.imshow('out_mask'+str(count), tmp_mask)
                if count>0:
                   tmp_num=tmp[:,x:int(x+w+1)]
                   tmp_number=trans_file+'/sub_number/'+objname+'_'+str(num)+'_extra' +'.png'
                   cv2.imwrite(tmp_number,tmp_num)
                #    cv2.imshow('out_num_'+str(count), tmp_num)
                   dst[:,x:int(x+w+1)]=0
                else:
                   tmp_num=tmp[:,0:int(x+0.5*w+1)]
                   tmp_number=trans_file+'/sub_number/'+objname+'_'+str(num)+'_front' +'.png'
                   cv2.imwrite(tmp_number,tmp_num)
                #    cv2.imshow('out_num_'+'front', tmp_num)
                   tmp_num=tmp[:,int(x+0.5*w+1):]
                   tmp_number=trans_file+'/sub_number/'+objname+'_'+str(num)+'_fore' +'.png'
                   cv2.imwrite(tmp_number,tmp_num)
                #    cv2.imshow('out_num_'+'fore', tmp_num)
                # cv2.imshow('out_'+str(count), dst)
                count=0
                break
        
        # cv2.waitKey(5)
        # cv2.destroyAllWindows()
        tmp_file=trans_file+'/tmp_pic/'+objname+'_'+ str(num) +'.png'
        tmp_number=trans_file+'/numbers_pic/'+objname+'_'+ str(num) +'.png'
        
        # cv2.imwrite(tmp_file,dst)
        # cv2.imwrite(tmp_number,tmp)
        # cv2.waitKey(5)
        num= num+1
        
        del dst,tmp
        # tmp = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel, iterations=1)
        # tmp = cv2.resize(tmp, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_AREA)
    





def pic_trans(objects):
    t1 = time.time()
    ann_pth=ann_file+'/'+objects
    pic_pth=pic_file+'/'+objects.split('.')[0]+'.jpg'
    # ann=cv2.imread(ann_pth,cv2.IMREAD_GRAYSCALE)
    # pic=cv2.imread(pic_pth,cv2.IMREAD_GRAYSCALE)
    pics,dic=ri.readpic(ann_pth,pic_pth)
    
    filename=trans_file+'/sub_number/'+dic["object"]+'.json'
    with open(filename, 'w') as file_obj:
        json.dump(dic, file_obj)

    objname=objects.split('.')[0]

    cal_region(pics,objname)
    t2 = time.time()
    print(int(round((t2-t1) * 1000)))
    


# dirlist=os.listdir(ann_file)
# for file in dirlist:
#     if file.split('.')[-1] == 'png':
#         pic_trans(file)
pic_trans('c311.png')