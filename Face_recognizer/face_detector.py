import cv2
def detect_faces(img):
    '''This is the function to detect faces using LBP face detector '''
    # convert the test image to gray scale image as opencv face detector
    # expects gray image
    gray_scale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # loading opencv face detector
    face_cascade_classifier = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml")
    # detecting faces from images
    faces = face_cascade_classifier.detectMultiScale(
        gray_scale_img, scaleFactor=1.06, minNeighbors=5)
    '''This is a general function to detect objects, in this case, 
    it'll detect faces since we called in the face cascade. 
    If it finds a face, it returns a list of positions of said face in the form “Rect(x,y,w,h).”,
    if not, then returns “None”.'''
    if(len(faces)==0):
        return None,None
    else:
        x,y,w,h=faces[0]
        return gray_scale_img[y:y+w,x:x+h],faces[0]
