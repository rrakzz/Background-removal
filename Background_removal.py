from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFilter
from statistics import mean 
from skimage.io import imread, imshow
import numpy as np
import cv2
import glob, os

Input_path = 'D:\\Face\\stores\\'
Output_path = 'D:\\Face\\stores_out\\'
Image_format = 'jpg'

files = list(filter(os.path.isfile, glob.glob(Input_path+"*."+Image_format)))
# files.sort(key=lambda x: os.path.getctime(x))

for i,filename in enumerate(files):
    Image_name = filename.split('\\')[-1].split('.')[0]
    print("Started Processing ("+str(i+1)+"/"+str(len(files))+"): "+ Image_name)
    
    # K-means clustering technique to group the colours
    In = cv2.imread(filename) # dividing by 255 to bring the pixel values between 0 and 1
    pic = cv2.imread(filename)/255 
    pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])
    kmeans = KMeans(n_clusters=4, random_state=0).fit(pic_n)
    pic2show = kmeans.cluster_centers_[kmeans.labels_]
    cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
    cv2.imwrite(Output_path+Image_name+"_kmeans.png", cluster_pic*255)

    # Cropping the colour from the top left corner 
    im = Image.open(Output_path+Image_name+"_kmeans.png")
    px1 = im.load() 
    width, height = im.size 
    left = width * 0.05
    top = height * 0.05
    right = width * 0.1
    bottom = height * 0.2

    im1 = im.crop((left, top, right, bottom)) 
    im1.save(Output_path+'tempL.png') #Saving the cropped image for reference
    
    # Removing the background colour and filling white colour from left
    px = im1.load() 
    aR, aG, aB = [], [], []
    for j in range(im1.size[0]):
        for i in range(im1.size[1]):
            fR, fG, fB = px[j,i]
            aR.append(fR)
            aG.append(fG)
            aB.append(fB)
    #         print(fR, fG, fB)
    for j in range(im.size[0]):
        for i in range(im.size[1]):
            fR, fG, fB = px1[j,i]
            if abs(fR-mean(aR)) < 0.1*fR and abs(fG-mean(aG)) < 0.1*fG and abs(fB-mean(aB)) < 0.1*fB :
                px1[j,i] = (255, 255, 255) 
                
    # Cropping the colour from the top right corner   
#     width, height = im.size 
#     left = width * 0.9
#     top = height* 0.05
#     right = width * 0.95
#     bottom =  height* 0.2

#     im1 = im.crop((left, top, right, bottom)) 
#     im1.save(Output_path+'tempR.png') #Saving the cropped image for reference
    
#     # Removing the background colour and filling white colour from right
#     px = im1.load() 
#     aR, aG, aB = [], [], []
#     for j in range(im1.size[0]):
#         for i in range(im1.size[1]):
#             fR, fG, fB = px[j,i]
#             aR.append(fR)
#             aG.append(fG)
#             aB.append(fB)
#     #         print(fR, fG, fB)
#     for j in range(im.size[0]):
#         for i in range(im.size[1]):
#             fR, fG, fB = px1[j,i]
#             if abs(fR-mean(aR)) < 0.1*fR and abs(fG-mean(aG)) < 0.1*fG and abs(fB-mean(aB)) < 0.1*fB :
#                 px1[j,i] = (255, 255, 255) 
    im.save(Output_path+Image_name+"_whiteBG.png")
    
    # Creating the mask image by replacing other than white pixels into black
    im1 = Image.open(Output_path+Image_name+"_whiteBG.png")
    px1 = im1.load() 
    for j in range(im1.size[0]):
        for i in range(im1.size[1]):
            if px1[j,i][0] == 255 and px1[j,i][1] == 255 and px1[j,i][2] == 255:
                px1[j,i] = (255, 255, 255)
            else:
                px1[j,i] = (0, 0, 0)    
    im1.save(Output_path+Image_name+"_Mask.png")
    
    img = cv2.imread(Output_path+Image_name+"_Mask.png")
    kernel = np.ones((5,5),np.uint8)
#     erosion = cv2.erode(img,kernel,iterations = 2)
#     dilation = cv2.dilate(erosion,kernel,iterations = 3)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(Output_path+Image_name+"_Mask1.png", opening)
#     plt.imshow(erosion,cmap = 'gray')
#     plt.savefig(Output_path+f"{Image_name}_Mask1.png")

    im = Image.open(Output_path+Image_name+"_Mask1.png")
    px = im.load() 
    px[0,0]

    Black_start = 0
    Black_end = 0
    for i in range(im.size[1]):
        for j in range(im.size[0]):
            if i > 0.1*im.size[1] and j > 0.05*im.size[0] and j < 0.95*im.size[0]:
                if px[j,i][0] == 0:
                    Black_start = j
                    break
        for j in reversed(range(im.size[0])):
            if i > 0.1*im.size[1] and j > 0.05*im.size[0] and j < 0.95*im.size[0]:
                if px[j,i][0] == 0:
                    Black_end = j
                    break
        for k in range(Black_start, Black_end):
            px[k,i] = (0, 0, 0)

    im.save(Output_path+Image_name+"_Mask2.png")

    # Merging White Background image and Original image along with Mask
    # Gaussian Blur used to smooth edges
    im1 = Image.open(Output_path+Image_name+"_whiteBG.png")
    im2 = Image.open(filename)
    mask = Image.open(Output_path+Image_name+"_Mask2.png").convert('L').resize(im1.size)
    mask_blur = mask.filter(ImageFilter.GaussianBlur(2))
    im_final = Image.composite(im1, im2, mask_blur)
    im_final.save(Output_path+Image_name+"_Final.png")
    
    Out = cv2.imread(Output_path+Image_name+"_Final.png")
    numpy_horizontal_concat = np.concatenate((In,Out), axis=1)
    cv2.imshow('Input vs Output', numpy_horizontal_concat)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()