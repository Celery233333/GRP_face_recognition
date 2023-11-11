import cv2 as cv
import os
import utils
import numpy as np
from net.mtcnn import mtcnn
from net.inception import InceptionResNetV1

class face():

    # initialize the models
    def __init__(self):
        # initialize the dataset 
        self.dataset = []

        # set up the mtcnn
        self.mtcnn_model = mtcnn()
        self.threshold = [0.5,0.6,0.8]

        # set up the facenet
        self.facenet_model = InceptionResNetV1()
        self.facenet_model.load_weights('model_data/facenet_keras.h5')

        # set up the resolution-fix model
        self.superres = cv.dnn_superres.DnnSuperResImpl_create()
        self.superres.readModel("model_data/FSRCNN_x3.pb")
        self.superres.setModel("fsrcnn", 3)

    # detect and process human face from the input image
    def face_detect(self,image):
        # convert to RGB type
        img = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # detect all the faces in the image
        faces = self.mtcnn_model.detectFace(img, self.threshold)

        # process the face
        if len(faces) == 1:
            img = utils.face_process(faces[0],img)
        else:
            exit("Please make sure there is (only) one person on the screen!\n")
    
        return img

    # add one picture to dataset (call it after each test)
    # should use exactly file name (not path)
    def add_dataset(self,path):
        # detect glass first
        if (self.glass_detect(path) == 1):
            return 1

        name = path.split(".")[1]
        if name == "jpg" or name == "png":
            try:
                # upload the image
                img = cv.imread(path)
            except:
                exit("\""+path+"\""+" does not exist\n")

            image = cv.imread(path)
            img = self.face_detect(image)
            img = np.expand_dims(img,0)

            encoding = utils.calc_128_vec(self.facenet_model,img)
            self.dataset.append(encoding)

    def clear_dataset(self):
        self.dataset = []

    # compare the current face with faces in dataset
    def compare(self,path):
        try:
            # upload the image
            img = cv.imread(path)
        except:
            exit("\""+path+"\""+" does not exist\n")

        if len(self.dataset) == 0:
            print("dataset is empty!")
            return 2

        # update the quality of OCR picture
        upscaled = self.superres.upsample(img)
        bicubic = cv.resize(img,(upscaled.shape[1], upscaled.shape[0]),interpolation=cv.INTER_CUBIC)

        # detect face in OCR picture
        img = self.face_detect(bicubic)
        
        img = utils.letterbox(img,(160,160))
        # expand dimension to fit keras
        img = np.expand_dims(img,0)
            
        # encoding the face
        encoding = utils.calc_128_vec(self.facenet_model, img)

        distances = []
        for data in self.dataset :
            # calculate and comapre the distance
            similarity = utils.face_distance(encoding,data)
            distances.append(similarity)
        
        # check the best match
        similarity = max(distances)
        print("confidence: "+ str(similarity*100))

        # weight is 0.8
        if similarity >= 0.8:
            print("sucess\n")
            return 1
        else: 
            print("fail\n")
            return 0

    # detect smile from input image
    def smile_detect(self,path):
        try:
            # upload the image
            image = cv.imread(path)
        except:
            exit("\""+path+"\""+" does not exist\n")

        # detect glass first
        if (self.glass_detect(path) == 1):
            return 2

        img = self.face_detect(image)
        gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)

        detector = cv.CascadeClassifier("model_data/haarcascade_smile.xml")
        smile = detector.detectMultiScale(gray,1.1,50)
        
        if len(smile) == 1:
            print("smile\n")
            return 1
        else:
            ### ask to take photo again ###
            print("no smile\n")
            return 0

    # detect glass from input image 
    # should add this function to all operations
    def glass_detect(self,path):
        try:
            # upload the image
            image = cv.imread(path)
        except:
            exit("\""+path+"\""+" does not exist\n")

        img = self.face_detect(image)
        gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)

        detector = cv.CascadeClassifier("model_data/haarcascade_mcs_eyepair_big.xml")
        glass = detector.detectMultiScale(gray,1.05,30)
        
        if len(glass) == 0:
            print("glass\n")
            return 1
        else:
            print("no glass\n")
            return 0

if __name__ == "__main__":
    celery = face()

    celery.add_dataset("1.jpg")
    celery.add_dataset("2.jpg")
    celery.add_dataset("3.jpg")

    celery.compare("1.jpg")
    celery.compare("test2.jpg")

    celery.smile_detect("test1.jpg")
    celery.glass_detect("1.jpg")