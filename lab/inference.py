from mct import run
import os
import cv2
import json
import glob
import numpy as np
import time
import pandas as pd
ann_file='./data00/annotations/trans_test/sub_number'

def reading(p1,p0,perc,num_list):   
    result=num_list[p1]-num_list[p0]
    result=result*perc
    return result

def get_num(ann_file):
    tp_read=list()
    dis0,dis1,fc0,fc1=run.create_model('./mct/Grid_MCT_IN1K/checkpoints/DINOv2_EsViT_0.1_False_0.pt')
    
    dirlist=os.listdir(ann_file)
    for file in dirlist:
        if file.split('.')[-1] == 'json':
            read=readjs(file,dis0,dis1,fc0,fc1)
            tp_read.append(read)
    tp_read=pd.DataFrame(tp_read)
    tp_read.to_excel(ann_file+'/inference.xlsx',index=False)

def readjs(js_file,dis0,dis1,fc0,fc1):
    js=os.path.join(ann_file,js_file)
    with open(js) as file_obj:
        info = json.load(file_obj)
    
    read_num=[]

    for i in range(int(info["length"])):
        picname = info["object"]+'_'+str(i)
        candiatelist=glob.glob(ann_file+'/'+picname+'*.png')
        extra=''
        front=''
        fore=''
        if len(candiatelist) == 1:
            img=cv2.imread(candiatelist[0])
            out_tmp=run.run_inference(img,dis0,dis1,fc0,fc1)
            read_num.append(out_tmp)

        elif len(candiatelist) == 2:
            cand=[x for x in candiatelist if 'front' in x]
            img=cv2.imread(cand[0])
            out_tmp0=run.run_inference(img,dis0,dis1,fc0,fc1)

            
            cand=[x for x in candiatelist if 'front' not in x]
            img=cv2.imread(cand[0])
            out_tmp1=run.run_inference(img,dis0,dis1,fc0,fc1)

            if (out_tmp0=='0') & (out_tmp1 !='.'):
                out_tmp=out_tmp0+'.'+out_tmp1
            elif out_tmp1 !='.':
                out_tmp=out_tmp0+out_tmp1
            else:
                out_tmp='0'
            
            read_num.append(out_tmp)

        elif len(candiatelist) > 2:
            cand=[x for x in candiatelist if 'front' in x]
            img=cv2.imread(cand[0])
            
            out_tmp0=run.run_inference(img,dis0,dis1,fc0,fc1)
            # read_num.append(out_tmp0)

            cand=[x for x in candiatelist if 'fore' in x]
            img=cv2.imread(cand[0])
            out_tmp1=run.run_inference(img,dis0,dis1,fc0,fc1)
            cand=[x for x in candiatelist if 'extra' in x]
            img=cv2.imread(cand[0])
            out_tmp2=run.run_inference(img,dis0,dis1,fc0,fc1)
            
            if (out_tmp1=='.') & (out_tmp2!='.') :
                out_tmp=out_tmp0+'.'+out_tmp2
            elif out_tmp2=='.':
                out_tmp=out_tmp0+'.'+out_tmp1
            else:
                out_tmp='0'
            read_num.append(out_tmp)

        else:
            read_num.append('0')
        
    
    read_num=[float(x) for x in read_num]
    read_num=np.array(read_num)
    print(read_num)
    t1=time.time()
    for n in range(3):
        
        sub_read=read_num[1:]-read_num[:-1]
        elemt,cot=np.unique(sub_read, return_counts=True)
        elind=elemt>0
        elemfix=elemt[elind]
        if elemfix:
            trcot=cot[elind]
            cut=elemfix[np.argmax(trcot)]
            fix=(sub_read!=cut).tolist()
            
            for k in range(len(fix)):
                if fix[k]:
                    if (k+1)<len(fix):
                        if fix[k+1]==False:
                            read_num[k+1]=read_num[k+2]-cut
                    if (k-1)>=0:
                        if fix[k-1]==False:
                            read_num[k+1]=read_num[k]+cut
        else:
            break

    # "index1": "4", "index0": "3", "percent": "0.5964125560538116"
    first=int(info["index1"])
    second=int(info["index0"])
    perc=float(info["percent"])
    fin=round(reading(first,second,perc,read_num),3)
    np.append(read_num, fin)
    t2=time.time()
    print(t2-t1)
    print(read_num)
    return read_num

get_num(ann_file)




# dis0,dis1,fc0,fc1=run.create_model('./mct/Grid_MCT_IN1K/checkpoints/DINOv2_EsViT_0.1_False_0.pt')
# img_path =os.path.join(ann_file, 'c311_5_front.png')
# img=cv2.imread(img_path)
# run.run_inference(img,dis0,dis1,fc0,fc1)

