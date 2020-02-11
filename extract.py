import scipy.io as sio
import numpy as np
import os
import gc
import six.moves.urllib as urllib
import cv2
import time
import xml.etree.cElementTree as ET
import random
import shutil as sh
from shutil import copyfile
import zipfile
from PIL import Image
import sys
import csv


def save_csv(csv_path, csv_content):
    with open(csv_path, 'w') as csvfile:
        wr = csv.writer(csvfile)
        for i in range(len(csv_content)):
            wr.writerow(csv_content[i])

def get_bbox_visualize(base_path, dir):
    create_directory("masks")
    create_directory("masks/1")
    create_directory("masks/2")
    create_directory("masks/3")
    create_directory("masks/4")
    create_directory("images")
    image_path_array = []
    for root, dirs, filenames in os.walk(base_path + dir):
        for f in filenames:
            if(f.split(".")[1] == "jpg"):
                img_path = base_path + dir + "/" + f
                image_path_array.append(img_path)

    #sort image_path_array to ensure its in the low to high order expected in polygon.mat
    image_path_array.sort()
    boxes = sio.loadmat(
        base_path + dir + "/polygons.mat")
    # there are 100 of these per folder in the egohands dataset
    polygons = boxes["polygons"][0]
    pointindex = 0
    gotcha = False
    for first in polygons:
        mask_number = 0

        font = cv2.FONT_HERSHEY_SIMPLEX

        img_id = image_path_array[pointindex]
        img = cv2.imread(img_id)
        delta = int((img.shape[1] - img.shape[0])/2)
        img_params = {}
        img_params["width"] = np.size(img, 1)
        img_params["height"] = np.size(img, 0)
        head, tail = os.path.split(img_id)
        img_params["filename"] = tail
        img_params["path"] = os.path.abspath(img_id)
        img_params["type"] = "train"
        pointindex += 1

        boxarray = []
        csvholder = []
        mask = np.full((img.shape[0], img.shape[1]) , 0, np.uint8)
        for pointlist in first:
            pst = np.empty((0, 2), int)
            max_x = max_y = min_x = min_y = height = width = 0

            findex = 0
            lastPoint = []
            for point in pointlist:
                if(len(point) == 2):
                    x = int(point[0])
                    y = int(point[1])
                    lastPoint = [x,y]

                    if(findex == 0):
                        min_x = x
                        min_y = y
                    findex += 1
                    max_x = x if (x > max_x) else max_x
                    min_x = x if (x < min_x) else min_x
                    max_y = y if (y > max_y) else max_y
                    min_y = y if (y < min_y) else min_y
                    appeno = np.array([[x, y]])
                    pst = np.append(pst, appeno, axis=0)
                    cv2.putText(img, ".", (x, y), font, 0.7,
                                (255, 255, 255), 2, cv2.LINE_AA)
            if(len(lastPoint) == 2):
                tmp_mask = np.zeros_like(mask)
                mask_number+=1
                cv2.polylines(tmp_mask, [pst], True, 255, 5)
                cv2.fillPoly(tmp_mask, [pst], 255, 4, 0)
                threshold = tmp_mask > 0
                new_mask = threshold.astype(np.uint8) * 255
                output_mask_path = "masks/" +str(mask_number) +'/'+ (img_id.split("/")[-1]).split(".")[0] + '.png'
                cv2.imwrite(output_mask_path, new_mask)
                mask += new_mask
            hold = {}
            hold['minx'] = min_x
            hold['miny'] = min_y
            hold['maxx'] = max_x
            hold['maxy'] = max_y
            if (min_x > 0 and min_y > 0 and max_x > 0 and max_y > 0):
                boxarray.append(hold)
                labelrow = [tail,
                            np.size(img, 1), np.size(img, 0), "hand", min_x, min_y, max_x, max_y]
                csvholder.append(labelrow)

            cv2.polylines(img, [pst], True, (0, 255, 255), 1)
            cv2.rectangle(img, (min_x, max_y),
                          (max_x, min_y), (0, 255, 0), 1)

        if(mask_number == 0):
            continue # ignore image if there is no mask, after all what can you learn from it?
        output_img_path = "images/" + (img_id.split("/")[-1]).split(".")[0] + '.png'
        cv2.imwrite(output_img_path, cv2.imread(img_id))


def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def generate_label_files(image_dir):
    header = ['filename', 'width', 'height',
              'class', 'xmin', 'ymin', 'xmax', 'ymax']
    for root, dirs, filenames in os.walk(image_dir):
        for dir in dirs:
            csvholder = []
            csvholder.append(header)
            loop_index = 0
            for f in os.listdir(image_dir + dir):
                if(f.split(".")[1] == "csv"):
                    loop_index += 1
                    csv_file = open(image_dir + dir + "/" + f, 'r')
                    reader = csv.reader(csv_file)
                    for row in reader:
                        csvholder.append(row)
                    csv_file.close()
                    os.remove(image_dir + dir + "/" + f)
            save_csv(image_dir + dir + "/" + dir + "_labels.csv", csvholder)
            print("Saved label csv for ", dir, image_dir +
                  dir + "/" + dir + "_labels.csv")


def generate_masks(image_dir):
    image_counter = 0
    for root, dirs, filenames in os.walk(image_dir):
        for dir in dirs:
            image_counter +=100
            print(image_counter)
            get_bbox_visualize(image_dir, dir)

    print("mask generation complete!")
    
def rename_files(image_dir):
    print("Renaming files")
    loop_index = 0
    for root, dirs, filenames in os.walk(image_dir):
        for dir in dirs:
            for f in os.listdir(image_dir + dir):
                if (dir not in f):
                    if(f.split(".")[1] == "jpg"):
                        loop_index += 1
                        os.rename(image_dir + dir +
                                  "/" + f, image_dir + dir +
                                  "/" + dir + "_" + f)
                else:
                    break

def extract_folder(dataset_path):
    if not os.path.exists("egohands"):
        zip_ref = zipfile.ZipFile(dataset_path, 'r')
        print("> Extracting Dataset files")
        zip_ref.extractall("egohands")
        print("> Extraction complete")
        zip_ref.close()
        rename_files("egohands/_LABELLED_SAMPLES/")
    generate_masks("egohands/_LABELLED_SAMPLES/")

    
def download_egohands_dataset(dataset_url, dataset_path):
    if not os.path.exists(dataset_path):
        print("downloading egohands dataset")
        opener = urllib.request.URLopener()
        opener.retrieve(dataset_url, dataset_path)
        print("> download complete")
    extract_folder(dataset_path)

    
EGOHANDS_DATASET_URL = "http://vision.soic.indiana.edu/egohands_files/egohands_data.zip"
EGO_HANDS_FILE = "egohands_data.zip"

download_egohands_dataset(EGOHANDS_DATASET_URL, EGO_HANDS_FILE)
