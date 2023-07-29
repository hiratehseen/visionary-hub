import cv2
import face_recognition
 
imgTrain = face_recognition.load_image_file('xx (2).jpg')
imgTrain = cv2.cvtColor(imgTrain,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('xx (3).jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
 
faceLoc = face_recognition.face_locations(imgTrain)[0]
encodeElon = face_recognition.face_encodings(imgTrain)[0]
cv2.rectangle(imgTrain,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
 
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
 
results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
 
cv2.imshow('Zhang Jingyi',imgTrain)
cv2.imshow('Zhang Jingyi Test',imgTest)
cv2.waitKey(0)