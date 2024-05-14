import cv2
import numpy as np

# 计算象限位置
class rectifing:
    def __init__(self,cx,cy,vx,vy):
        self.cx=cx
        self.cy=cy
        self.vx=vx
        self.vy=vy
        # 四个象限 一一对应
        self.s1=np.empty((1,2))
        self.s2=np.empty((1,2))
        self.s3=np.empty((1,2))
        self.s4=np.empty((1,2))
        self.s_lst=np.array([0,0,0,0])
        self.ang=0
    
    def calc_angle(self,ax):
        # 判断斜率为正或负
        if ax >= 0:
            angle = np.rad2deg(np.arctan(ax))
        else:
            angle = 180 + np.rad2deg(np.arctan(ax))
        return angle

    
    def ang_adjust_org(self):
        if self.ang>=180 and self.ang<360:
            self.ang=-(180-(self.ang-180))
        elif self.ang>=360 and self.ang<450:
            self.ang=self.ang-360

    def find_quadrant_point(self, qua0,qua1):
        # 12象限
        if (qua0 in [0,1]) & (qua1 in [0,1]):
            ind1=np.argmin(self.s1[:,1])
            ind2=np.argmin(self.s2[:,1])
            
            px=int((self.s1[ind1,0]+self.s2[ind2,0])/2)
            py=int((self.s1[ind1,1]+self.s2[ind2,1])/2)
            if px <= self.cx:
                qua=1
            else:
                qua=0
            return self.s1[ind1,:],self.s2[ind2,:],qua
        # 23象限
        elif(qua0 in [1,2]) & (qua1 in [1,2]):
            ind1=np.argmin(self.s2[:,0])
            ind2=np.argmin(self.s3[:,0])
            
            px=int((self.s2[ind1,0]+self.s3[ind2,0])/2)
            py=int((self.s2[ind1,1]+self.s3[ind2,1])/2)
            if py <= self.cy:
                qua=1
            else:
                qua=2
            return self.s2[ind1,:],self.s3[ind2,:],qua
        # 34象限
        elif(qua0 in [2,3]) & (qua1 in [2,3]):
            ind1=np.argmax(self.s3[:,1])
            ind2=np.argmax(self.s4[:,1])
            
            px=int((self.s3[ind1,0]+self.s4[ind2,0])/2)
            py=int((self.s3[ind1,1]+self.s4[ind2,1])/2)
            if px <= self.cx:
                qua=2
            else:
                qua=3
            return self.s3[ind1,:],self.s4[ind2,:],qua
        # 41象限
        elif(qua0 in [0,3]) & (qua1 in [0,3]):
            ind1=np.argmax(self.s4[:,0])
            ind2=np.argmax(self.s1[:,0])
            
            px=int((self.s4[ind1,0]+self.s1[ind2,0])/2)
            py=int((self.s4[ind1,1]+self.s1[ind2,1])/2)
            if px <= self.cy:
                qua=0
            else:
                qua=3
            return self.s4[ind1,:],self.s1[ind2,:],qua
        
    def rectify(self):
        for i in range(len(self.vx)):
            self.vxy = np.array([self.vx[i],self.vy[i]])
            self.vxy = np.expand_dims(self.vxy, 0)
            if self.vx[i] <= self.cx:
                # 在二三象限
                if self.vy[i] <= self.cy:
                    # 在二象限
                    self.s2 = np.concatenate((self.s2,self.vxy),axis=0)
                elif self.vy[i] > self.cy:
                    # 在san象限
                    self.s3 = np.concatenate((self.s3,self.vxy),axis=0)
            if self.vx[i] > self.cx:
                # 在一四象限
                if self.vy[i] <= self.cy:
                    # 在一象限
                    self.s1 = np.concatenate((self.s1,self.vxy),axis=0)
                elif self.vy[i] > self.cy:
                    # 在四象限
                    self.s4 = np.concatenate((self.s4,self.vxy),axis=0)

        self.s1=np.delete(self.s1,0,axis=0)
        self.s2=np.delete(self.s2,0,axis=0)
        self.s3=np.delete(self.s3,0,axis=0)
        self.s4=np.delete(self.s4,0,axis=0)
        self.s_lst[0]=len(self.s1)
        self.s_lst[1]=len(self.s2)
        self.s_lst[2]=len(self.s3)
        self.s_lst[3]=len(self.s4)

        if len(self.s1)==0:
            self.s_lst[0]=999
        if len(self.s2)==0:
            self.s_lst[1]=999
        if len(self.s3)==0:
            self.s_lst[2]=999
        if len(self.s4)==0:
            self.s_lst[3]=999
        think=0
        for j in range(4):
            if self.s_lst[j]==999:
                think+=1
                self.ang=45+90*j+90
                self.ang_adjust_org()
            if think > 1 :
                self.ang = 0
                break        

        if think == 0:
            # 找到最小值的索引
            self.lst_min0=np.argmin(self.s_lst)
            self.s_lst[self.lst_min0]=999
            
            self.lst_min1=np.argmin(self.s_lst)
            if np.absolute(self.lst_min1-self.lst_min0)!=1:
                if ((self.lst_min1 in [0,3]) & (self.lst_min0 in [0,3])):
                    pass
                else:
                    return 0
                    # self.s_lst[self.lst_min1]=999
                    # self.lst_min1=np.argmin(self.s_lst)
            

            po1,po2,qua=self.find_quadrant_point(self.lst_min0,self.lst_min1)
            slope=(po1[1]-po2[1])/(po1[0]-po2[0])
            slope_out=self.calc_angle(slope)
            if qua in [0,3]:
                self.ang=180-slope_out
            elif qua in [1,2]:
                self.ang= (0-slope_out)
            if (qua in [2,3]) & (slope_out > 170):
                self.ang= 180-slope_out
            if (qua in [2,3]) & (slope_out < 10):
                self.ang= 0-slope_out
        return 1

            







    
    

    
        
   


