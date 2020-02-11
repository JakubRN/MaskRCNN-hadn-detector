from threading import Thread
import cv2, time
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
      
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(.01)

    def show_frame(self):
        # Display frames in main program
        cv2.imshow('frame', self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

device = torch.device('cuda')

class MaskRCNN_maniuplator(object):
    def __init__(self, video_stream):
        
        self.video_stream_widget=video_stream
        self.masks = None
        self.new_mask_available = False
        self.mask_accuracy = 0.5

        # our dataset has two classes only - background and person
        num_classes = 2
        # get the model using our helper function
        self.model = get_instance_segmentation_model(num_classes)
        # move model to the right device
        self.model.to(device)

        resume = 'best_checkpoint_new.tar'
        checkpoint = torch.load(resume)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        # print("=> loaded checkpoint '{}' (epoch {})" .format(resume, checkpoint['epoch']))


    def start(self):
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        counter = 0
        start = time.time()
        while True:
            if self.video_stream_widget.capture.isOpened():
                if self.video_stream_widget.status:
                    counter +=1
                    curr = time.time()
                    if(curr - start > 1):
                        print("Current FPS: ", counter /(curr - start))
                        counter = 0
                        start = time.time()
                    photo = cv2.cvtColor(self.video_stream_widget.frame, cv2.COLOR_BGR2RGB)
                    result_feed = photo *(1.0/255.0)
                    result_feed = result_feed.transpose(2,1,0)
                    with torch.no_grad():
                        prediction = self.model([torch.as_tensor(result_feed, dtype=torch.float32).to(device)])
                    # pix = img.mul(255).permute(1, 2, 0).byte().numpy()
                    mask = prediction[0]['masks'].permute(1, 0, 3, 2)[0].cpu().detach().numpy()
                    boxes = prediction[0]['boxes'].cpu().detach().numpy()
                    scores = prediction[0]['scores'].cpu().detach().numpy()
                    # print(boxes[0])
                    # print(mask.shape[0])
                    if(mask.shape[0] > 0):
                        self.masks = mask[0][int(boxes[0][0]):int(boxes[0][2]),int(boxes[0][1]):int(boxes[0][3])]
                        self.new_mask_available = True
                        # print(self.masks.shape)
                        # cv2.rectangle(self.masks, (boxes[0][1], boxes[0][0]), (boxes[0][3], boxes[0][2]), 255, 5)
                        self.out_mask = mask[0]
                        cv2.rectangle(self.out_mask, (boxes[0][1], boxes[0][0]), (boxes[0][3], boxes[0][2]), 1, 5)
                        # for j in range(boxes.shape[0]):
                        #     print(boxes[j])
                            # if(scores[j] > 0.9):
                            #     cv2.rectangle(self.out_mask, (boxes[j][1], boxes[j][0]), (boxes[j][3], boxes[j][2]), 1, 5)
                        # for j in range(1, mask.shape[0]):
                        #     # if(scores[j] > self.mask_accuracy):
                        #     self.masks = self.masks + mask[j]



from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from itertools import chain
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from pynput.keyboard import Key, Controller
import math

keyboard = Controller()
currentValue = 7
valuesMap = {
    1: (Key.right,),
    2: (Key.right, Key.up),
    3: (Key.right, Key.down),
    4: (Key.left,),
    5: (Key.left, Key.up),
    6: (Key.left, Key.down),
    7: (Key.up, '/'),
    8: (Key.down, '/'),
    9: (Key.right, '/'),
    10: (Key.left, '/'),
}

class Main_Classificator(object):
    def __init__(self, maskrcnn_manipulator_object):
        self.maskrcnn_manipulator_object = maskrcnn_manipulator_object

        data, labels = self.readImages()
        features = self.getFeatures(data)

        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)
        features = self.normalize(features)

        self.pca = PCA(n_components=0.95)
        features = self.pca.fit_transform(features)
        print(self.pca.explained_variance_ratio_)

        x_tr, x_tst, y_tr, y_tst = train_test_split(features, labels, test_size = 0.3)
        self.model_SVC = SVC(kernel='linear')
        self.model_SVC.fit(x_tr, y_tr)
        Z = self.model_SVC.predict(x_tst)
        print(confusion_matrix(y_tst, Z))
        print(accuracy_score(y_tst, Z, normalize=True))
    
    def start(self):
        self.thread = Thread(target=self.classify, args=())
        self.thread.daemon = True
        self.thread.start()

    def processImage(self, imgdata, purple = False, HSVdata = None):
        hsv_light_purple = np.array([100, 100, 100])
        hsv_dark_purple = np.array([120, 200, 200])
        if purple:
            hsv_light_purple = np.array([155, 110, 0])
            hsv_dark_purple = np.array([180, 255, 255])
        elif HSVdata is not None:
            hsv_light_purple = np.array(HSVdata[0:3])
            hsv_dark_purple = np.array(HSVdata[3:6])
        hsv = cv2.cvtColor(cv2.resize(imgdata, (640, 360)), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_light_purple, hsv_dark_purple)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if(len(contours) == 0):
            return np.zeros((30, 30))
        longest_contour = contours[0]
        for c in contours:
            if(len(c) > len(longest_contour)):
                longest_contour = c
        H1_contours = np.zeros_like(mask)
        cv2.drawContours(H1_contours, [longest_contour], 0, 255, -1)
        # cv2.drawContours(H low_contours, contours, -1, (0,255,0), 2, hierarchy=hierarchy, maxLevel = 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        dilation = cv2.dilate(H1_contours ,kernel, iterations = 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
        
        contours, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        centers = [None]*len(contours)
        radius = [None]*len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
            centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
        
        crop_img = closing[int(boundRect[-1][1]):int(boundRect[-1][1])+boundRect[-1][3], \
                int(boundRect[-1][0]):int(boundRect[-1][0])+boundRect[-1][2]]
        return cv2.resize(crop_img, (60, 60))

    def readImages(self):
        images = []
        labels = []
        for index, data in enumerate(os.walk("data/")):
            if(len(data[2]) is not 0):
                print(index, data[0], data[1])
            for img in data[2]:
                imgdata = cv2.imread(data[0] + "/" + img)
                # photo = cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB)
                # result_feed = photo *(1.0/255.0)
                # result_feed = result_feed.transpose(2,1,0)
                # with torch.no_grad():
                #     prediction = self.maskrcnn_manipulator_object.model([torch.as_tensor(result_feed, dtype=torch.float32).to(device)])
                # # pix = img.mul(255).permute(1, 2, 0).byte().numpy()
                # mask = prediction[0]['masks'].permute(1, 0, 3, 2)[0].cpu().detach().numpy()
                # boxes = prediction[0]['boxes'].cpu().detach().numpy()
                # scores = prediction[0]['scores'].cpu().detach().numpy()
                # print(scores)
                # out_mask = mask[0]
                # if(mask.shape[0] > 0):
                #     for j in range(boxes.shape[0]):
                #         if(scores[j] > 0.9):
                #             cv2.rectangle(out_mask, (boxes[j][1], boxes[j][0]), (boxes[j][3], boxes[j][2]), 1, 5)
                #     for j in range(1, mask.shape[0]):
                #         if(scores[j] > 0.9):
                #             out_mask = out_mask + mask[j]
                # imgdata[out_mask > 0.5] = 255
                # cv2.imshow("tmpwindow", imgdata)
                # key = cv2.waitKey(-1)
                # if key == ord('q'):
                #     cv2.destroyAllWindows()
                #     exit(1)
                outImg = self.processImage(imgdata, purple=True)


                images.append(outImg)
                labels.append(index)
        return images, labels
        
    def getFeatures(self,data):
        contours = [ cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0] for img in data]
        c_area = np.array([cv2.contourArea(c[-1]) for c in contours])
        c_len = np.array([cv2.arcLength(c[-1],True) for c in contours])
        Moments = [cv2.moments(img) for img in data]
        HuMoments = [cv2.HuMoments(mom) for mom in Moments]
        centres =[(M["m10"]/M["m00"], M["m01"]/M["m00"]) for M in Moments]
        Moments=np.array([list(moment.values()) for moment in Moments])
        HuMoments=np.array([[-math.log(abs(hu)) for hu in chain.from_iterable(moment)] for moment in HuMoments])
        centres=np.array(centres)
        metadata = np.vstack((c_area, c_len)).T
        metadata = np.concatenate((centres, Moments, HuMoments), axis = 1)
        return metadata

    def normalize(self, data):
        return ( data - self.mean) / self.std


    def classify(self):
        # while True:
        if self.maskrcnn_manipulator_object.new_mask_available:
            self.maskrcnn_manipulator_object.new_mask_available = False
            resized_mask = cv2.resize(self.maskrcnn_manipulator_object.masks, (100, 100))
            ret,thresh1 = cv2.threshold(resized_mask,self.maskrcnn_manipulator_object.mask_accuracy,1,cv2.THRESH_BINARY)
            chunkFeatures = self.normalize(self.getFeatures([thresh1.astype(np.uint8)]))
            chunkFeatures = self.pca.transform(chunkFeatures)
            results = self.model_SVC.predict(chunkFeatures)
            print(results)
            # unique, counts =np.unique(results, return_counts=True)
            # for value, numberOfOccurences in zip(unique, counts):
            #     if numberOfOccurences >= maxNumberOfFrames:
            #         if(value != mycv):
            #             print(value)
            #             mycv = value
            #         if(gameStarted):
            #             applyMove(value)
                


if __name__ == '__main__':

    imageNumber = {}
    for i in range(1,11):
        imageNumber[i] = 100
    try:
        os.mkdir("data")
        for i in range(1,11):
            os.mkdir("data/" + str(i))
    except FileExistsError as identifier:
        print("directories already exist")
    
    video_stream_widget = VideoStreamWidget()
    maskrcnn_calculation = MaskRCNN_maniuplator(video_stream_widget)
    # main_classificator = Main_Classificator(maskrcnn_calculation)
    def on_trackbar(val):
        maskrcnn_calculation.mask_accuracy = val / 100

    window_name = "test"
    cv2.namedWindow(window_name)
    cv2.createTrackbar("pick accuracy", window_name , int(maskrcnn_calculation.mask_accuracy * 100), 100, on_trackbar)
    while True:
        try:
            frame = np.array(video_stream_widget.frame)
            if(maskrcnn_calculation.masks is not None):
                cv2.imshow("mask", maskrcnn_calculation.masks)
                # main_classificator.classify()
                frame[maskrcnn_calculation.out_mask > maskrcnn_calculation.mask_accuracy] = 255
                cv2.imshow(window_name, frame)
                c = cv2.waitKey(0)
                print(c)
                cv2.imshow(window_name, frame)
                if c == ord('q'):
                    video_stream_widget.capture.release()
                    cv2.destroyAllWindows()
                    exit(1)
                elif c >=49 and c <= 58:
                    mystring = "data/" + str(c - 48) + "/" + str(imageNumber[c - 48]) + ".png"

                    imageNumber[c - 48] +=1

                    cv2.imwrite(mystring, video_stream_widget.frame)
                elif c == 48:
                    mystring = "data/" + str(c - 38) + "/" + str(imageNumber[c - 38]) + ".png"

                    imageNumber[c - 38] +=1

                    cv2.imwrite(mystring, video_stream_widget.frame)
            
            else:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    video_stream_widget.capture.release()
                    cv2.destroyAllWindows()
                    exit(1)
                elif key == ord(' '):
                    maskrcnn_calculation.start()
                    # main_classificator.start()

        except AttributeError:
            pass