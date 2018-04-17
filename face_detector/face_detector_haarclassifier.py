import cv2

#load cascade classifier training file for haarcascade 
haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img=cv2.imread("photo.jpg")
gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#converts the coloured image to gray
#detect multiscale (some images may be closer to camera than others) images
faces=haar_face_cascade.detectMultiScale(gray_image,
	scaleFactor=1.05,minNeighbors=5)
print(type(faces))
print(faces)

for x,y,w,h in faces:
	img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	#the first parameter takes image,2nd & 3rd parameter builds an rectangle.4th parameter shows color of the rectangle(BGR foramt)
	#5th parameter width of the window
resized_image=cv2.resize(img,(int(img.shape[0]/2),int(img.shape[1]/2)))
cv2.imshow("pic",resized_image) #pic is the name of the window
cv2.imwrite("output_of_haar.jpg",resized_image)
#write the resized image into new file
cv2.waitKey(0) #so that user can close the window
'''the user presses any button then the window will be closed '''
cv2.destroyAllWindows()# by pressing any button the window will be closed