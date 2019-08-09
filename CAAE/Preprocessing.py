#Preprocessing
import os
import platform
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import cv2
import shutil
from glob import glob
from datetime import datetime
#web crawl
import argparse
import json
import itertools
import logging
import re
import os
import uuid
import sys
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

'''
Use command : 

python3 Preprocessing.py --search_list father mother family teenage man men girl woman women businessman businesswoman whitefamily asianfamily adult adults prefessional --num_images 100

python3 Preprocessing.py --phase face_detection  --use_shutil True

father, dad, parents, family, aisan family
white family, family picture, mom, mother, baby
infant, baby boy, baby girl
man and girls ...

'''

class Preprocessing(object):
    def __init__(self,
                 crawl_save_dir,
                 num_images,
                 search_list,
                 age_list,
                 REQUEST_HEADER,
                 MODEL_MEAN_VALUES,
                 resize_scale,
                 gender_dict,
                 faceimage_save_dir,
                 use_shutil
                 ):
        
        self.crawl_save_dir=  crawl_save_dir
        self.num_images = num_images
        self.search_list = search_list
        self.age_list = age_list
        self.REQUEST_HEADER = REQUEST_HEADER
        self.gender_dict = gender_dict
        self.faceimage_save_dir = faceimage_save_dir
        self.resize_scale = resize_scale
        self.MODEL_MEAN_VALUES = MODEL_MEAN_VALUES
        self.use_shutil = use_shutil
    '''
    논문에(Face-Aging-CAAE) 나온 Image 는 제대로 working 하는 것을 확인함.
    궁금하면 ./save/test/test_as_{Male or Female}.png 를 확인해볼것.

    UTKFace Image 이름 형식 :  [age]_[gender]_[race]_[date&time].jpg
    FaceAging.py 가 저것에 맞추어서 coding 되어 있으니 형식을 맞춰주자...

    Age_list : FaceAging.py에 있는 label 에 맞게 class 를 나눔(UTKFace Image 는 0~116살 까지)
    gender_list : 0 이 Male, 1이 Female 로 저장 하기 위해 dictionary
    race 는 쓰지 않으므로 0을 default 로 해서 저장.
    date&time은 지금 돌리는 시점으로 한다.

    자꾸 해서 귀찮으니까 web_crawling.py랑 파일 합치자..(web_crawling을 argparse list 로 받아오게 해서...)
    
    Directory 는 기존의 CycleGAN-Tensorflow 와 datasets -> data 빼고는 같게..
    
    '''
    
    def configure_logging(self):
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter('[%(asctime)s %(levelname)s %(module)s]: %(message)s'))
        logger.addHandler(handler)
        self._logger = logger
    
    
    def get_soup(self, url):
        response = urlopen(Request(url, headers=self.REQUEST_HEADER))
        return BeautifulSoup(response, 'html.parser')

    def get_query_url(self, search):
        return "https://www.google.co.in/search?q=%s&source=lnms&tbm=isch" % search

    def extract_images_from_soup(self,soup):
        image_elements = soup.find_all("div", {"class": "rg_meta"})
        metadata_dicts = (json.loads(e.text) for e in image_elements)
        link_type_records = ((d["ou"], d["ity"]) for d in metadata_dicts)
        return link_type_records

    def extract_images(self, search):
        url = self.get_query_url(search)
        self._logger.info("Souping")
        soup = self.get_soup(url)
        self._logger.info("Extracting image urls")
        link_type_records = self.extract_images_from_soup(soup)
        return itertools.islice(link_type_records, self.num_images)

    def get_raw_image(self, url):
        req = Request(url, headers=self.REQUEST_HEADER)
        resp = urlopen(req)
        return resp.read()

    def save_image(self, raw_image, image_type):
        extension = image_type if image_type else 'png'
        file_name = str(uuid.uuid4().hex) + "." + extension
        save_path = os.path.join(self.crawl_save_dir, file_name)
        with open(save_path, 'wb+') as image_file:
            image_file.write(raw_image)

    def download_images_to_dir(self, images, search):
        for i, (url, image_type) in enumerate(images):
            try:
                self._logger.info("Making request (%d/%d) for '%s'", i, self.num_images, search)
                raw_image = self.get_raw_image(url)
                self.save_image(raw_image, image_type)
            except Exception as e:
                self._logger.exception(e)

    def run(self):
        for search in self.search_list:
#             try:
            query = '+'.join(search.split())
            self._logger.info("Extracting image links")
            images = self.extract_images(search)
            self._logger.info("Downloading images")
            self.download_images_to_dir(images, search)
            self._logger.info("Finished")
#             except:
#             self._logger.info("Why exception occurs?? AttributeError??")

    def convert_list(self, string_list):
        return int((int(string_list.split(',')[0].replace('(', '')) + int(string_list.split(',')[1].replace(')', '')))/2)


    def face_save(self, dir_path, option = 'haar'):
        
        print("Current # of face images in directory : %s"%(len(os.listdir(self.faceimage_save_dir))))
        if option == 'haar':
            face_cascade = cv2.CascadeClassifier(os.path.join(dir_path, 'face_detection/haarcascade_frontalface_alt.xml'))
        elif option == 'lbp':
            face_cascade = cv2.CascadeClassifier(os.path.join(dir_path, 'face_detection/lbpcascade_frontalface.xml'))
        else:
            face_cascade = cv2.CascadeClassifier(os.path.join(dir_path, 'face_detection/lbpcascade_frontalface_improved.xml'))

        image_list = glob(self.crawl_save_dir + '/*.jpg') + glob(self.crawl_save_dir + '/*.png') + glob(self.crawl_save_dir + '/*.jpeg')
        age_net = cv2.dnn.readNetFromCaffe(dir_path + "/face_detection/deploy_age.prototxt", 
                                   dir_path + "/face_detection/age_net.caffemodel")
        gender_net = cv2.dnn.readNetFromCaffe(dir_path + "/face_detection/deploy_gender.prototxt", 
                                      dir_path + "/face_detection/gender_net.caffemodel")
        print("# of images to detect  : ",len(image_list))
    
        detected_faces, predicted_ages, predicted_genders = [], [], []
        num = 0
        for j,image_dir in enumerate(image_list):

            if type(cv2.imread(image_dir)) == type(None):
                continue
            origin_image = cv2.imread(image_dir)
            gray_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5)

            if (j+1)%10 == 0:
                print("="*45)
                print(" [Step : %d] # of stacked files  : %d" % ((j+1),num), flush = True)
            for i, (x, y, w, h) in enumerate(faces):
                cv2.rectangle(origin_image, (x,y), (x+w, y+h), (0,225,0), 2)
                face_image = origin_image[y:y+h, x:x+w].copy()
                detected_faces.append(face_image)

                blob = cv2.dnn.blobFromImage(face_image, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = list(gender_dict.keys())[gender_preds[0].argmax()]
                predicted_genders.append(gender)

                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]
                predicted_ages.append(age)

                #UTKFace data : 23708, size : 200x200x3
                if len(faces) != 0:
                    resized_face_RGB = cv2.resize(face_image, dsize = self.resize_scale, fx = 1, fy = 1, interpolation = cv2.INTER_CUBIC)
                    mean_age = self.convert_list(age)
                    if not os.path.exists(self.faceimage_save_dir):
                        os.mkdir(self.faceimage_save_dir)
                    #[age]_[gender]_[race]_[date&time].jpg
                    cv2.imwrite(self.faceimage_save_dir + str(mean_age) + '_' + 
                                str(self.gender_dict[gender])+ '_' + str(0) + '_' + datetime.now().strftime('%Y%m%d%H%M%S') + str(i)+ 
                                '.jpg.chip.jpg',resized_face_RGB)
                    num += 1

        print("Total # of Images in directory  :  %d"%(len(os.listdir(self.faceimage_save_dir))))
        print("Done")
        return
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape Google images')
    parser.add_argument('-c', '--crawl_save_dir', default='./data/web_crawling', type=str, help='save directory')
    parser.add_argument('-n', '--num_images', default=1, type=int, help='num images to save')
    parser.add_argument('-l', '--search_list', default = [], nargs = '+', help = 'list the items to search')
    parser.add_argument('-s', '--use_shutil', default = False, type = bool, help = 'True then delete the face image directory')
    parser.add_argument('-f', '--faceimage_save_dir', default = './data/man2baby/', type = str, help = 'where to save face image')
    parser.add_argument('-r', '--resize_scale', default = (200,200), type = tuple, help = 'write resize scale')
    parser.add_argument('-p', '--phase', default = 'web_crawl', type = str, help = 'write the phase')
    args = parser.parse_args()
    
    age_list = ['(0, 5)', '(6, 10)', '(11, 15)', '(16, 20)', 
                '(21, 30)', '(31, 40)', '(41, 50)', '(51, 60)', '(61, 70)', '(71, 100)']
    gender_dict = {'Male' : 0, 'Female' : 1}
    
    REQUEST_HEADER = {'User-Agent': "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


    if args.phase == 'web_crawl':
        if not os.path.exists(args.crawl_save_dir):
            os.mkdir(args.crawl_save_dir)
        print("Search list : " + str(args.search_list))
        Preprocessing = Preprocessing(crawl_save_dir = args.crawl_save_dir,
                                      num_images = args.num_images,
                                      search_list = args.search_list,
                                      age_list = age_list,
                                      REQUEST_HEADER = REQUEST_HEADER,
                                      MODEL_MEAN_VALUES = MODEL_MEAN_VALUES,
                                      resize_scale = args.resize_scale,
                                      gender_dict = gender_dict,
                                      faceimage_save_dir = args.faceimage_save_dir,
                                      use_shutil = args.use_shutil)
        logging = Preprocessing.configure_logging()
        Preprocessing.run()
    
        current_dir = os.path.dirname(os.path.realpath(__file__))
        
        if not os.path.exists(args.faceimage_save_dir): 
            os.mkdir(args.faceimage_save_dir)
            
        if args.use_shutil == 'True':
            try:
                shutil.rmtree(args.faceimage_save_dir)
            except OSError as e:
                if e.errno == 2:
                    print('No such file or directory to remove')
                    pass
                else:
                    raise
        
        Preprocessing.face_save(current_dir, option = 'haar')
    
    
    if args.phase == 'face_detection':
        
        if not os.path.exists(args.faceimage_save_dir): 
            os.mkdir(args.faceimage_save_dir)
        if args.use_shutil == 'True':
            try:
                shutil.rmtree(args.faceimage_save_dir)
            except OSError as e:
                if e.errno == 2:
                    print('No such file or directory to remove')
                    pass
                else:
                    raise

        
        Preprocessing = Preprocessing(crawl_save_dir = args.crawl_save_dir,
                                      num_images = args.num_images,
                                      search_list = args.search_list,
                                      age_list = age_list,
                                      REQUEST_HEADER = REQUEST_HEADER,
                                      MODEL_MEAN_VALUES = MODEL_MEAN_VALUES,
                                      resize_scale = args.resize_scale,
                                      gender_dict = gender_dict,
                                      faceimage_save_dir = args.faceimage_save_dir,
                                      use_shutil = args.use_shutil)
        
        current_dir = os.path.dirname(os.path.realpath(__file__))
        Preprocessing.face_save(current_dir, option = 'haar')

    

