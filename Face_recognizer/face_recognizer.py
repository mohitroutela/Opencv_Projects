# Import openCV module
import cv2
# import the face detection using LBP from the code written
from face_detector import *
# os module for reading traing files from directories and paths
import os
# numpy to convert python lists to numpy arrays as it is needed by opencv
# face recognizers
import numpy as np

# writing the names of the subjects
subjects = ["sallu bhai", "messi the great"]


def prepare_training_data(data_folder_path):
    '''this function will read all person's training images,detect face from each image and will 
    return two lists of exactly same size,one list of faces and another lists of labels for each face '''
    # get the directories from the folder
    directories = os.listdir(data_folder_path)
    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # go through each directory and read images within it
    for dir_name in directories:
        # my dir starts with letter 's' so
        # ignore any non-relevent directories if any
        if not dir_name.startswith("s"):
            continue
        # extract label number of person from the directories
        label = int(dir_name.replace("s", ""))
        # this will give you output like 1 if you have directories like "s1"
        # this path is where your's images will reside of your folder like
        # traing-data/s1
        subject_dir_path = data_folder_path + "/" + dir_name
        # get the images of the respective subjects from the folder i.e s0 or
        # s1
        subject_images_names = os.listdir(subject_dir_path)
        # go through each image,read image,detect face and add face to the list of faces
        # counter=0
        for image_name in subject_images_names:
            # counter+=1
            # ignore system files like starting with .
            if image_name.startswith("."):
                continue
            # building image path like trainig-data/s1/1.jpg
            image_path = subject_dir_path + "/" + image_name
            # read the image
            image = cv2.imread(image_path)
            # dispaly the image on an window
            cv2.imshow("training an image...", image)
            # wait for some seconds before closing the window
            cv2.waitKey(100)
            # detect face using the function return earlier
            face, rectangle = detect_faces(image)
            #print("face "+str(counter))
            # print(face)
            # checking which faces are detected and which are not with the help of couter
            # initialized at line no 33
            # print(rectangle)
            # print("\n")
            # ignoring the faces that are not detected
            if face is not None:
                # add face to the list of faces
                faces.append(face)
                # add labels for this face
                labels.append(label)
    # print(faces)
    # print(labels)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels


print("Preparing training data ...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")
print("Total faces: ", len(faces))
print("Total labels:", len(labels))
# Creating our face recognizer,here I am using the
# LBP face recogizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))


def draw_rectangle(img, rect):
    x, y, w, h = rect
    # this function is explained in face detection algorith
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# function to draw text on the given images
# staring from the x,y,co-ordinates


def draw_text(img, text, x, y):
    cv2.putText(img,text,(x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    #cv2.putText(img, text, startPoint, font, fontSize, rgbColor, lineWidth)
# this function will predict the person on the image passed and draws a rectangle around it
# with the name of the subject


def predict(test_img):
    # make a copy of it
    # so that original won't change
    img = test_img.copy()
    # detect_face around the image
    face, rect = detect_faces(img)
    # predict the image using face recogizer defined above
    #label = face_recognizer.predict(face)
    #print("label is ")
    #print(label)
    label,x=face_recognizer.predict(face)
    # get the name of the label
    label_text = subjects[label]
    # draw rectangle
    draw_rectangle(img, rect)
    # draw name
    draw_text(img, label_text, rect[0], rect[1] - 5)
    return img


print("predicting images ...")
# load test-images
test_img1 = cv2.imread("test-data/2.jpeg")
test_img2 = cv2.imread("test-data/3.jpeg")
# perform prediction
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
print("prediction done")
# display images
cv2.imshow(subjects[0], predicted_img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow(subjects[1], predicted_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
