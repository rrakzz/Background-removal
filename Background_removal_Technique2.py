from statistics import mean 
import cv2
import glob, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

Input_path = 'D:\\Face\\images\\'
Output_path = 'D:\\Face\\Output\\'
Image_format = 'jpeg'
Threshold_per = 0.1 #0 to 1 Threshold_percntage

files = list(filter(os.path.isfile, glob.glob(Input_path+f'*.{Image_format}')))
# files.sort(key=lambda x: os.path.getctime(x))

for i,filename in enumerate(files):
    Image_name = filename.split('\\')[-1].split('.')[0]
    print("Processing:", Image_name)
    
    im = Image.open(filename)
    px = im.load() 
    width, height = im.size 
    left = width * 0.05
    top = height * 0.05
    right = width * 0.1
    bottom = height * 0.2

    im1 = im.crop((left, top, right, bottom)) 
    px1 = im1.load() 

    B,G,R = [], [], []
    for j in range(int(right)):
        for i in range(int(left)):
            Bl, Gl, Rl = px1[i,j]
            B.append(Bl)
            G.append(Gl)
            R.append(Rl)
    B,G,R = mean(B), mean(G), mean(R)
    print("Average BGR values of top left corner", B,G,R)

    for j in range(im.size[0]):
        for i in range(im.size[1]):
            fB, fG, fR = px[j,i]
            if abs(fR-R) < Threshold_per*R and abs(fG-G) < Threshold_per*G and abs(fB-B) < Threshold_per*B:
                px[j,i] = (255, 255, 255)  

    im.save(Output_path+f"{Image_name}_WB.png")