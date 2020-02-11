from threading import Thread
import cv2, time
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
import os
import numpy as np
import torch
import torch.utils.data

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
        self.last_position = None
        # our dataset has two classes only - background and person
        num_classes = 2
        # get the model using our helper function
        self.model = get_instance_segmentation_model(num_classes)
        # move model to the right device
        self.model.to(device)
        resume = 'best_checkpoint.tar'
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
                    with torch.no_grad():
                        prediction = self.model([F.to_tensor(photo).to(device)])
                    # pix = img.mul(255).permute(1, 2, 0).byte().numpy()
                    mask = prediction[0]['masks'].permute(1, 0, 2, 3)[0].cpu().detach().numpy()
                    boxes = prediction[0]['boxes'].cpu().detach().numpy()
                    scores = prediction[0]['scores'].cpu().detach().numpy()
                    # print(boxes[0])
                    # print(mask.shape[0])

                    if(mask.shape[0] > 0):
                        if(self.last_position is None):
                            best_index = 0
                            self.last_position = np.array([(boxes[0][2] + boxes[0][0])/2, (boxes[0][3] + boxes[0][1])/2])
                            # print(self.last_position)
                        else:
                            best_index = -1
                            curr_best_dist = 128000
                            for j in range(boxes.shape[0]):
                                position = np.array([(boxes[j][2] + boxes[j][0])/2, (boxes[j][3] + boxes[j][1])/2])
                                # print(position)
                                dist = np.linalg.norm(position - self.last_position)
                                if(dist < curr_best_dist):
                                    best_index = j
                                    curr_best_dist = dist
                            # print("best: ", self.last_position)
                            if(curr_best_dist > 120):
                                continue
                            self.last_position = np.array([(boxes[best_index][2] + boxes[best_index][0])/2, (boxes[best_index][3] + boxes[best_index][1])/2])
                        
                        self.masks = mask[best_index][int(boxes[best_index][1]):int(boxes[best_index][3]),int(boxes[best_index][0]):int(boxes[best_index][2])]
                        self.new_mask_available = True
                        # print(self.masks.shape)
                        # cv2.rectangle(self.masks, (boxes[0][1], boxes[0][0]), (boxes[0][3], boxes[0][2]), 255, 5)
                        self.out_mask = mask[best_index]
                        # cv2.rectangle(self.out_mask, (boxes[best_index][0], boxes[best_index][1]), (boxes[best_index][2], boxes[best_index][3]), 1, 5)
                        for j in range(1, mask.shape[0]):
                            # if(scores[j] > self.mask_accuracy):
                            self.out_mask = self.out_mask + mask[j]



from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from itertools import chain
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from pynput.keyboard import Key, Controller
import math
import pickle as pk
from collections import Counter, deque
from torchvision.transforms import functional as F
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
    11: (),
}
def applyMove(value):
    global currentValue
    if(value == currentValue): 
        return
    for keycode in valuesMap[currentValue]:
        keyboard.release(keycode)
    for keycode in valuesMap[value]:
        keyboard.press(keycode)
    currentValue = value

class Main_Classificator(object):
    def __init__(self, maskrcnn_manipulator_object):
        self.maskrcnn_manipulator_object = maskrcnn_manipulator_object
        self.queue_size = 7
        self.recentResults = deque()
        if os.path.isfile('classification_model.pkl'):
            with open('classification_model.pkl', 'rb') as pickle_file:
                self.pca = pk.load(pickle_file)
                self.model_SVC = pk.load(pickle_file)
                self.mean = pk.load(pickle_file)
                self.std = pk.load(pickle_file)
        else:
            data, labels = self.readImages()
            features = self.getFeatures(data)
            self.mean = np.mean(features, axis=0)
            self.std = np.std(features, axis=0)
            features = self.normalize(features)

            self.pca = PCA(n_components=0.95)
            features = self.pca.fit_transform(features)
            print(self.pca.explained_variance_ratio_)
            x_tr, x_tst, y_tr, y_tst = train_test_split(features, labels, test_size = 0.3)
            self.model_SVC = SVC(kernel='linear', class_weight='balanced')
            self.model_SVC.fit(x_tr, y_tr)
            Z = self.model_SVC.predict(x_tst)
            print(confusion_matrix(y_tst, Z))
            print(accuracy_score(y_tst, Z, normalize=True))
            with open('classification_model.pkl', 'wb') as pickle_file:
                pk.dump(self.pca, pickle_file)
                pk.dump(self.model_SVC, pickle_file)
                pk.dump(self.mean, pickle_file)
                pk.dump(self.std, pickle_file)

    def start(self):
        self.thread = Thread(target=self.classify, args=())
        self.thread.daemon = True
        self.thread.start()

    def readImages(self):
        images = []
        labels = []
        for index, data in enumerate(os.walk("data/")):
            if(len(data[2]) is not 0):
                print(index, data[0], data[1])
            for img in data[2]:
                img_path = data[0] + "/" + img
                imgdata = cv2.imread(img_path)
                rgb_photo = cv2.cvtColor(imgdata, cv2.COLOR_BGR2RGB)
                # result_feed = rgb_photo.astype(np.float32) * (1.0/255.0)
                # result_feed = result_feed.transpose(2,1,0)
                
                with torch.no_grad():
                    # prediction = self.maskrcnn_manipulator_object.model([torch.as_tensor(result_feed, dtype=torch.float32).to(device)])
                    prediction = self.maskrcnn_manipulator_object.model([F.to_tensor(rgb_photo).to(device)])
                # pix = img.mul(255).permute(1, 2, 0).byte().numpy()
                mask = prediction[0]['masks'].permute(1, 0, 2, 3)[0].cpu().detach().numpy()
                # mask = prediction[0]['masks'].cpu().detach().numpy()
                boxes = prediction[0]['boxes'].cpu().detach().numpy()
                scores = prediction[0]['scores'].cpu().detach().numpy()
                # out_mask = mask[0]
                # print(prediction)
                # cv2.rectangle(out_mask, (boxes[0][1], boxes[0][0]), (boxes[0][3], boxes[0][2]), 1, 5)
                # if(mask.shape[0] > 0):
                #     for j in range(boxes.shape[0]):
                #         print(j)
                #         cv2.rectangle(out_mask, (boxes[j][0], boxes[j][1]), (boxes[j][2], boxes[j][3]), 1, 5)
                #     for j in range(1, mask.shape[0]):
                #         out_mask = out_mask + mask[j]
                # imgdata[out_mask > 0.5] = 255
                # cv2.imshow("tmpwindow", imgdata)
                outImg = cv2.resize(mask[0][int(boxes[0][1]):int(boxes[0][3]),int(boxes[0][0]):int(boxes[0][2])], (64, 64))
                # cv2.imshow("tmpwindow2", outImg)
                # key = cv2.waitKey(0)
                # if key == ord('q'):
                #      exit(1)
                # outImg = self.processImage(imgdata, purple=True)

                label_ind = int(data[0].split('/')[-1])
                # print(label_ind)
                images.append(outImg)
                labels.append(label_ind)
        cv2.destroyAllWindows()
        return images, labels
        
    def getFeatures(self,data):
        # contours = [ cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0] for img in data]
        # c_area = np.array([cv2.contourArea(c[-1]) for c in contours])
        # c_len = np.array([cv2.arcLength(c[-1],True) for c in contours])
        Moments = [cv2.moments(img) for img in data]
        HuMoments = [cv2.HuMoments(mom) for mom in Moments]
        centres =[(M["m10"]/M["m00"], M["m01"]/M["m00"]) for M in Moments]
        Moments=np.array([list(moment.values()) for moment in Moments])
        HuMoments=np.array([[-math.log(abs(hu)) for hu in chain.from_iterable(moment)] for moment in HuMoments])
        centres=np.array(centres)
        # metadata = np.vstack((c_area, c_len)).T
        metadata = np.concatenate((centres, Moments, HuMoments), axis = 1)
        return metadata

    def normalize(self, data):
        return ( data - self.mean) / self.std


    def classify(self):
        # while True:
        if self.maskrcnn_manipulator_object.new_mask_available:
            self.maskrcnn_manipulator_object.new_mask_available = False
            resized_mask = cv2.resize(self.maskrcnn_manipulator_object.masks, (64, 64))
            chunkFeatures = self.normalize(self.getFeatures([resized_mask]))
            chunkFeatures = self.pca.transform(chunkFeatures)
            results = self.model_SVC.predict(chunkFeatures)
            self.recentResults.append(results[0])
            if(len(self.recentResults) == self.queue_size):
                best_class, score = Counter(self.recentResults).most_common(1)[0]
                if(best_class != currentValue):
                    if(score >= 2):
                        applyMove(best_class)
                        print("switching " , best_class)
                    else:
                        print("not sure")
                        applyMove(11)
                self.recentResults.popleft()
                


if __name__ == '__main__':
    print("hello")
    video_stream_widget = VideoStreamWidget()
    maskrcnn_calculation = MaskRCNN_maniuplator(video_stream_widget)
    main_classificator = Main_Classificator(maskrcnn_calculation)
    def on_trackbar(val):
        maskrcnn_calculation.mask_accuracy = val / 100

    window_name = "test"
    cv2.namedWindow(window_name)
    cv2.createTrackbar("pick accuracy", window_name , int(maskrcnn_calculation.mask_accuracy * 100), 100, on_trackbar)
    paused = True
    while True:
        try:
            frame = np.array(video_stream_widget.frame)
            if(maskrcnn_calculation.masks is not None):
                cv2.imshow("mask", maskrcnn_calculation.masks)
                frame[maskrcnn_calculation.out_mask > maskrcnn_calculation.mask_accuracy] = 255
                if not paused:
                    main_classificator.classify()
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                video_stream_widget.capture.release()
                cv2.destroyAllWindows()
                exit(1)
            elif key == ord('s'):
                maskrcnn_calculation.start()
            elif key == ord(' '):
                if(paused):
                    applyMove(11)
                    paused = False
                else:
                    paused = True

        except AttributeError:
            pass