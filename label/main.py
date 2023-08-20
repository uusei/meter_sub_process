# code:utf-8
from PyQt5.QtWidgets import *
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import *
import sys
import os
from main_label import UI_label
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import QVideoWidget
import time
import numpy as np
from PyQt5.QtGui import QCursor
import cv2
import file_operator as fop
import picform as pfm
import json

class data_func(QMainWindow, UI_label):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.initparam()
        
        self.pushButton0.clicked.connect(self.openfile)
        self.pushButton1.clicked.connect(self.last_pic)
        self.pushButton2.clicked.connect(self.next_pic)
        self.pushButton4.clicked.connect(self.savepic)
        self.pushButton5.clicked.connect(self.draw_pic)
        self.pushButton6.clicked.connect(self.setcolor)
        self.pushButton7.clicked.connect(self.resetcolor)
        self.pushButton8.clicked.connect(self.setsize)
        self.pushButton9.clicked.connect(self.resetsize)
        self.pushButton10.clicked.connect(self.jump_pic)
        self.pushButton0.setShortcut(Qt.Key_Return)
        self.pushButton1.setShortcut('n')
        self.pushButton2.setShortcut('m')
        self.pushButton5.setShortcut('d')


    def initparam(self):
        
        self.Update_s = Update1()
        self.thout = Update2()
        self.isdrawing = False
        self.color = 1
        self.sizeorg = 1
        self.picdir=0
        self.startpic = False
        self.path=os.getcwd()
        if os.path.isfile('set.json'):
            filename = "set.json"
            with open(filename) as file_obj:
                numbers = json.load(file_obj)
            self.picdir = numbers["num"]
        elif not os.path.isfile('set.json'):
            print('create a file')
            numbers = {"num":self.picdir, "path":"G:\\pyproject\\meter_label\\data1\\images\\train"}
            filename = "set.json"
            with open(filename, 'w') as file_obj:
                json.dump(numbers, file_obj)
        self.lineEdit.setText(numbers['path'])

    def picplay(self):
        self.frame = pfm.pic_read(self.path,self.listpic[self.picdir])
        self.currentpic = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        if self.currentpic.any():
            self.h1, self.w1, ch = self.currentpic.shape
            bytesPerLine = ch * self.w1
            convertToQtFormat = QtGui.QImage(self.currentpic.data, self.w1, self.h1, bytesPerLine, QtGui.QImage.Format_RGB888)
            convertToQtFormat = convertToQtFormat.scaled(800, 800)
            self.label.setPixmap(QtGui.QPixmap(convertToQtFormat))

    def openfile(self):
        if self.startpic:
            self.thout.start()
            self.thout.stoop.connect(self.Update_s.stop)
        self.isdrawing = False
        self.txt0=self.lineEdit.text()    
        if self.txt0:
            self.countpic,self.listpic=fop.list_pic(self.txt0)
            self.lineEdit_2.setText(str(self.picdir))
            self.label_show.setText(self.listpic[self.picdir])
            pfm.pic_get(self.path,self.listpic[self.picdir])
            self.Update_s.start()
            self.Update_s.date2.connect(self.picplay)
            self.startpic = True
        else:
            print('invalid')

    def last_pic(self):
        pfm.save_out(self.path,self.listpic[self.picdir])
        if self.startpic:
            self.startpic = False
            self.thout.run()
            self.thout.stoop.connect(self.Update_s.stop)
            
        self.isdrawing = False
        if self.txt0:
            if self.picdir > 0:
                self.picdir-=1
            else:
                self.picdir=0
            self.lineEdit_2.setText(str(self.picdir))
            self.label_show.setText(self.listpic[self.picdir])
            pfm.pic_get(self.path,self.listpic[self.picdir])
            self.startpic = True
            self.thout.run()
            self.thout.stoop.connect(self.Update_s.goon)
            
    def next_pic(self):
        pfm.save_out(self.path,self.listpic[self.picdir])
        if self.startpic:
            self.startpic = False
            self.thout.run()
            self.thout.stoop.connect(self.Update_s.stop)
        self.isdrawing = False
        if self.txt0:
            if self.picdir < (self.countpic-1):
                self.picdir+=1
            else:
                self.picdir=self.countpic-1
            self.lineEdit_2.setText(str(self.picdir))
            self.label_show.setText(self.listpic[self.picdir])
            pfm.pic_get(self.path,self.listpic[self.picdir])
            self.startpic = True
            self.thout.run()
            self.thout.stoop.connect(self.Update_s.goon)

    def jump_pic(self):
        pfm.save_out(self.path,self.listpic[self.picdir])
        if self.startpic:
            self.startpic = False
            self.thout.run()
            self.thout.stoop.connect(self.Update_s.stop)
        self.isdrawing = False  
        if self.txt0 :
            if int(self.lineEdit_2.text())<= self.countpic:
                self.picdir=int(self.lineEdit_2.text()) 
                self.label_show.setText(self.listpic[self.picdir])
                pfm.pic_get(self.path,self.listpic[self.picdir])
                self.startpic = True
                self.thout.run()
                self.thout.stoop.connect(self.Update_s.goon)


    def draw_pic(self):
        if self.startpic:
            self.isdrawing = not self.isdrawing
            print('drawing turn')

    def savepic(self):
        self.isdrawing = False
        pfm.save_out(self.path,self.listpic[self.picdir])
        numbers = {"num":self.picdir, "path":self.lineEdit.text()}
        filename = "set.json"
        with open(filename, 'w') as file_obj:
            json.dump(numbers, file_obj)

    def setcolor(self):
        self.isdrawing = False
        self.color=self.textEdit0.toPlainText()
    
    def resetcolor(self):
        self.isdrawing = False
        self.textEdit0.setText(self.color)

    def setsize(self):
        self.isdrawing = False
        self.sizeorg = self.textEdit1.toPlainText()

    def resetsize(self):
        self.isdrawing = False
        self.textEdit1.setText(self.sizeorg)

    def mousePressEvent(self, e):
        pos = e.pos()
        sizeg =self.label.geometry()
        xp = pos.x()-sizeg.x()
        yp = pos.y()-sizeg.y()
        if self.isdrawing:
            pfm.drawing_now(round(xp/800*self.w1),round(yp/800*self.h1),int(self.sizeorg),int(self.color),self.path,self.listpic[self.picdir])
            self.setCursor(QCursor(Qt.OpenHandCursor))  # 更改鼠标图标


    def mouseMoveEvent(self, e):
        pos = e.pos()
        sizeg =self.label.geometry()
        xp = pos.x()-sizeg.x()
        yp = pos.y()-sizeg.y()
        if self.isdrawing:
            pfm.drawing_now(round(xp/800*self.w1),round(yp/800*self.h1),int(self.sizeorg),int(self.color),self.path,self.listpic[self.picdir])

    def mouseReleaseEvent(self, e):
        pos = e.pos()
        sizeg =self.label.geometry()
        xp = pos.x()-sizeg.x()
        yp = pos.y()-sizeg.y()
        if self.isdrawing:
            pfm.drawing_now(round(xp/800*self.w1),round(yp/800*self.h1),int(self.sizeorg),int(self.color),self.path,self.listpic[self.picdir])
            self.setCursor(QCursor(Qt.ArrowCursor))

# 线程1 实现更新图片
class Update1(QThread):
    date2 = pyqtSignal()

    def __init__(self):
        super(Update1, self).__init__()
        self.flag = True

    def run(self):
        while True:
            time.sleep(0.1)
            if self.flag == True:
                self.date2.emit()  # 发射信号
                print('en')
            else:
                print('off')
                continue

    def stop(self):
        self.flag = not self.flag

    def goon(self):
        self.flag = not self.flag

# 线程1 实现更新图片
class Update2(QThread):
    stoop = pyqtSignal()

    def __init__(self):
        super(Update2, self).__init__()

    def run(self):
        for i in range(2):
            time.sleep(0.1)       
            self.stoop.emit()  # 发射信号

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mk1 = data_func()
    mk1.show()
    sys.exit(app.exec_())