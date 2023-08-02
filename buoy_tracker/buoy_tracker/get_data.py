import cv2 
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def bouy_images(filename,colors):
    video = cv2.VideoCapture(filename)
    tot_frames = int(video.get(7))
    count = 0
    bgr = {colors[0]:(52,232,235),colors[1]:(52,137,235),colors[2]:(52,235,86)}
    for color in colors:
        path = 'Raw Data/'+color
        if os.path.exists(path):
            for img in os.listdir(path):
                if img.endswith('.png'):
                    os.remove(path+'/'+img)
        else:
            os.mkdir(path)


    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    while video.isOpened():
        ret, frame = video.read()
        
        if ret:
            h,w,ch = frame.shape
            frames = {}
            frame_w_text = frame.copy() 
            text = 'Frame ' + str(count+1) + ' of ' + str(tot_frames)
            cv2.putText(frame_w_text,text,(w-200,h-10),font,1,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(frame_w_text,text,(w-200,h-10),font,1,(255,255,255),1,cv2.LINE_AA)
            bouy_count = 0

            for color in colors:
                color_frame = frame_w_text.copy()
                cv2.rectangle(color_frame,(0,h-40),(40,h),bgr[color],-1)
                frames[color]=color_frame

            for color in colors:
                clicked = False
                roi = []
                text1 = 'Draw a rectangle arround the ' + color + ' bouy'
                text2 = 'If the bouy is not present press ESC'
                
                cv2.putText(frames[color],text1,(10,20),font,1,(0,0,0),2,cv2.LINE_AA)
                cv2.putText(frames[color],text1,(10,20),font,1,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(frames[color],text2,(10,40),font,1,(0,0,0),2,cv2.LINE_AA)
                cv2.putText(frames[color],text2,(10,40),font,1,(255,255,255),1,cv2.LINE_AA)
                

                cv2.imshow('Frame',frames[color])
                (x,y,w,h) = cv2.selectROI("Frame", frames[color], fromCenter=False,showCrosshair=False)

                if w == 0 or h == 0:
                    continue

                else:
                    bouy_count += 1
                    bouy = frame[y:y+h,x:x+w]
                    mask = np.zeros(bouy.shape[:2],np.uint8)
                    cv2.ellipse(mask,(w//2,h//2),(w//2,h//2),0,0,360,255,-1)
                    bouy = cv2.bitwise_and(bouy,bouy,mask = mask)
                    # cv2.imshow('Bouy',bouy) 
                    # cv2.waitkey(0)
                    # cv2.destroyWindow('Bouy')
                    path = 'Raw Data/'+color+'/'+str(count)+'.png'
                    cv2.imwrite(path,bouy)
                    print('The image was saved')
                    print()
                    
            if bouy_count == 0:
                video.release()
                cv2.destroyAllWindows()

            count +=1
        else:
            video.release()


def sort_data(path,colors):
    image_paths = {}

    # Store the paths for all images in image_paths
    for color in colors:
        paths = []
        color_path = path + '/' + color
        if os.path.exists(color_path):
            for img in os.listdir(color_path):
                if img.endswith('.png'):
                    paths.append(color_path+'/'+img)
        image_paths[color] = paths

    # Get indicies for training and testing and put images into relevant folder
    for color in colors:
        path_list = image_paths[color]
    
        tot = len(path_list)
        train = int(.7*tot)

        train_inds = []
        while len(train_inds) < train:
            rand_ind = random.randint(0,tot-1)
            if rand_ind not in train_inds:
                train_inds.append(rand_ind)

        train_path = 'Training Data/'+color
        if os.path.exists(train_path):
            for img in os.listdir(train_path):
                if img.endswith('.png'):
                    os.remove(train_path+'/'+img)
        else:
            os.mkdir(train_path)
        
        test_path = 'Testing Data/'+color
        if os.path.exists(test_path):
            for img in os.listdir(test_path):
                if img.endswith('.png'):
                    os.remove(test_path+'/'+img)
        else:
            os.mkdir(test_path)

        for ind,path in enumerate(path_list):
            img = cv2.imread(path)
            img_name = path.split('/')[-1]
            
            if ind in train_inds:
                path = train_path + '/' + img_name
                cv2.imwrite(path,img)
            else:
                path = test_path + '/' + img_name
                cv2.imwrite(path,img)

def generate_dataset(path,colorspace):
    dataset = []
    for img_name in os.listdir(path):
        if img_name.endswith('.png'):
            # print('\n'+path+'/'+img_name)
            img = cv2.imread(path+'/'+img_name)
            if colorspace == 'BGR':
                b = img[:,:,0].flatten()
                g = img[:,:,1].flatten()
                r = img[:,:,2].flatten()
                data = np.stack((b, g, r))

            elif colorspace == 'HSV':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h = img[:,:,0].flatten()
                s = img[:,:,1].flatten()
                v = img[:,:,2].flatten()
                data = np.stack((h, s, v))

            idx = np.argwhere(np.all(data[..., :] == 0, axis=0))

            nonzero = np.delete(data, idx, axis=1)

            if len(dataset) == 0:
                dataset = nonzero
            else:
                dataset = np.concatenate((dataset,nonzero),axis = 1)
                
    return dataset




if __name__ == '__main__':

    get_new_raw = False
    sort = True
    bouy_colors = ['yellow','orange','green']
    colorspace = 'BGR' #HSV or BGR

    if get_new_raw:
        print('Running this program will ovewrite all previous data')
        ans_str = input('Press y to continue: ')

        if ans_str.lower() == 'y':
            filename = '../media/detectbuoy.avi'
            bouy_images(filename,bouy_colors)
        else:
            exit()
    if sort:
        sort_data('Raw Data',bouy_colors)

    training_data = {}
    testing_data = {}
    for color in bouy_colors:
        print('\n'+color)
        train_path = 'Training Data/'+color
        test_path = 'Testing Data/'+color

        train_data= generate_dataset(train_path,colorspace)
        print(train_data.shape)
        test_data= generate_dataset(test_path,colorspace)
        print(test_data.shape)

        training_data[color] = train_data
        testing_data[color] = test_data

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if colorspace == 'BGR':
        for color in bouy_colors:
            data = training_data[color]
            b = data[0,:]
            g = data[1,:]
            r = data[2,:]
            ax.scatter(b,g,r,color = color)

        ax.set_xlabel('Blue')
        ax.set_ylabel('Green')
        ax.set_zlabel('Red')
    elif colorspace == 'HSV':
        for color in bouy_colors:
            data = training_data[color]
            h = data[0,:]
            s = data[1,:]
            v = data[2,:]
            ax.scatter(h,s,v,color = color)

        ax.set_xlabel('Hue')
        ax.set_ylabel('Saturation')
        ax.set_zlabel('Value')


    plt.show()

    







    
