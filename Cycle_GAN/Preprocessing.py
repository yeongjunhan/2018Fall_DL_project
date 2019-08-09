import os
import platform
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import cv2
from glob import glob
# %matplotlib inline


# path = '/home/yeongjun/Deep Learning Lab/Project/DAT' # Test용으로, 임시
path = os.path.dirname(os.path.realpath(__file__)) #True path
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
font = cv2.FONT_HERSHEY_SIMPLEX
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', 
            '(21, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

'''
Convert Image to RGB
Option : COLOR_BGR2RGB, COLOR_BGR2GRAY , ...
'''
def convert_to_RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

'''
e.g.
Input : '(0,2)' -> Output : 0
Input : '(15,20)' -> Output : 15
'''

def convert_list(string_list):
    return string_list.split(',')[0].replace('(', '')


'''
directory -> 상위 디렉터리. 디렉터리 하나에 다 넣어두자.
file name -> use glob
lbp.xml 도 dir 에 있어야 함.
image -> jpg로 저장
option : haar, lbp or lbp_imp


--저장(혹은 저장 되어있는) 경로

1.크롤링한 데이터 경로
CycleGAN-tensorflow/datasets/crawling

2. face detection용 데이터 경로
CycleGAN-tensorflow/face_detection/

3. 데이터 전처리 후 저장 경로 (test도 동일)
CycleGAN-tensorflow/dataset/man2baby/trainA : train 에 쓰일 남자 사진
CycleGAN-tensorflow/dataset/man2baby/trainB : train 에 쓰일 여자 사진
CycleGAN-tensorflow/dataset/man2baby/trainC/ : train 에 쓰일 아기 사진

4.Input 경로
dir_path = os.getcwd() 또는 os.path.dirname(os.path.realpath(__file__)), 즉 CYcleGAN-tensorlfow/

--필요 files

CycleGAN-tensorflow/face_detection 에 있어야 하고,

haarcascade_frontalface_alt.xml
lbpcascade_frontalface.xml
lbpcascade_frontalface_improved.xml

deploy_age.prototxt
deploy_gender.prototxt
age_net.caffemodel
gender_net.caffemodel

'''
def face_save(dir_path, option = 'haar'):
    if not os.path.exists(os.path.join(dir_path, 'faces')):
        os.mkdir(os.path.join(dir_path, 'faces') )
    if option == 'haar':
        face_cascade = cv2.CascadeClassifier(os.path.join(dir_path, '/face_detection/haarcascade_frontalface_alt.xml'))
    elif option == 'lbp':
        face_cascade = cv2.CascadeClassifier(os.path.join(dir_path, '/face_detection/lbpcascade_frontalface.xml'))
    else:
        face_cascade = cv2.CascadeClassifier(os.path.join(dir_path, '/face_detection/lbpcascade_frontalface_improved.xml'))
    
    image_list = glob(dir_path + '/datasets/crawling/*.jpg')
    age_net = cv2.dnn.readNetFromCaffe(dir_path + "/face_detection/deploy_age.prototxt", 
                               dir_path + "/face_detection/age_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe(dir_path + "/face_detection/deploy_gender.prototxt", 
                                  dir_path + "/face_detection/gender_net.caffemodel")
    print("Total of images  : ",len(image_list))
    
    detected_faces, predicted_ages, predicted_genders = [], [], []
    num_trainA = 0
    num_trainB = 0
    num_trainC = 0
    num_testA = 0
    num_testB = 0
    num_testC = 0
    for j,image_dir in enumerate(image_list):
        print("="*30)
        origin_image = cv2.imread(image_dir).copy()
        gray_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5)
        
        print(" [%d] # of faces  : %d" % ((j+1),len(faces)))
        print("# of items for train : [{A},{B},{C}]".format(A = num_trainA, B = num_trainB, C = num_trainC))
        print("# of items for test : [{A},{B},{C}]".format(A = num_testA, B = num_testB, C = num_testC))
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(origin_image, (x,y), (x+w, y+h), (0,225,0), 2)
            face_image = origin_image[y:y+h, x:x+w].copy()
            detected_faces.append(face_image)
            
            blob = cv2.dnn.blobFromImage(face_image, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            predicted_genders.append(gender)
            
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            predicted_ages.append(age)
            
            #horse2zebra trainA : 1067, trainB : 1334, testA : 120, testB : 140
            
            if len(faces) != 0:
                #train set
                resized_face_RGB = cv2.resize(convert_to_RGB(face_image), dsize = (256,256), fx = 1, fy = 1, interpolation = cv2.INTER_LINEAR)
                least_age = convert_list(age)

                if not os.path.exists(os.path.join(dir_path, 'datasets/man2baby/trainA')):
                    os.mkdir(os.path.join(dir_path, 'datasets/man2baby/trainA'))
                if not os.path.exists(os.path.join(dir_path, 'datasets/man2baby/trainB')):
                    os.mkdir(os.path.join(dir_path, 'datasets/man2baby/trainB'))
                if not os.path.exists(os.path.join(dir_path, 'datasets/man2baby/trainC')):
                    os.mkdir(os.path.join(dir_path, 'datasets/man2baby/trainC'))
                if not os.path.exists(os.path.join(dir_path, 'datasets/man2baby/testA')):
                    os.mkdir(os.path.join(dir_path, 'datasets/man2baby/testA'))
                if not os.path.exists(os.path.join(dir_path, 'datasets/man2baby/testB')):
                    os.mkdir(os.path.join(dir_path, 'datasets/man2baby/testB'))
                if not os.path.exists(os.path.join(dir_path, 'datasets/man2baby/testC')):
                    os.mkdir(os.path.join(dir_path, 'datasets/man2baby/testC'))

                if least_age >= 20 and gender == 'Male':
                    if num_trainA <= 999:
                        cv2.imwrite(os.path.join(dir_path, 'datasets/man2baby/trainA'),resized_face_RGB)
                        num_trainA +=1
                    else:
                        pass
                    
                if least_age >= 20 and gender =='Female':
                    if num_trainB <= 999:
                        cv2.imwrite(os.path.join(dir_path, 'datasets/man2baby/trainB'),resized_face_RGB)
                        num_trainB +=1
                    else:
                        pass
                    
                if least_age < 20:
                    if num_trainC <= 999:
                        cv2.imwrite(os.path.join(dir_path, 'datasets/man2baby/trainC'),resized_face_RGB)
                        num_trainC +=1
                    else:
                        pass

                if num_trainA+num_trainB+num_trainC == 3000:
                    #test set
                    if least_age >= 20 and gender == 'Male':
                        if num_testA <= 149:
                            cv2.imwrite(os.path.join(dir_path, 'datasets/man2baby/testA'),resized_face_RGB)
                            num_testA +=1
                        else:
                            pass

                    if least_age >= 20 and gender =='Female':
                        if num_testB <= 149:
                            cv2.imwrite(os.path.join(dir_path, 'datasets/man2baby/testB'),resized_face_RGB)
                            num_testB +=1
                        else:
                            pass

                    if least_age < 20:
                        if num_testC <= 149:
                            cv2.imwrite(os.path.join(dir_path, 'datasets/man2baby/testC'),resized_face_RGB)
                            num_testC +=1
                        else:
                            pass

                if num_trainA + num_trainB + num_trainC == 3000 and num_testA + num_testB + num_testC ==450:
                    print("Finished")
                    break
    #                 plt.imshow(resized_face_RGB)
    return detected_faces, predicted_ages, predicted_genders

detected_faces, predicted_ages, predicted_genders = face_save(path, option = 'haar')
