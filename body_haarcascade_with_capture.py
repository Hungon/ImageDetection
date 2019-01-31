import numpy as np
import cv2



if __name__ == '__main__':
    # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    body_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_upperbody.xml')
    #https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bodies = body_cascade.detectMultiScale(gray, 1.1, 3)

        for (x,y,w,h) in bodies:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            print("bx:"+str(x)+" by:"+str(y))

            faces = face_cascade.detectMultiScale(roi_gray)
            for (fx,fy,fw,fh) in faces:
                print("fx:"+str(fx)+" fy:"+str(fy))
                cv2.rectangle(roi_color,(fx,fy),(fx+fw,fy+fh),(0,255,0),2)

        cv2.imshow('img',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
