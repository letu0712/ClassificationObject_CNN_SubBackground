import cv2
import numpy as np
from tensorflow.python.keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
from numpy import expand_dims

model = load_model("modelHumanCatCar.h5")
cap = cv2.VideoCapture('s.mp4')
_,frame = cap.read()
width,height,_ = frame.shape
background = cv2.imread("Background.jpg",0)
fps = cap.get(cv2.CAP_PROP_FPS)
wait_time = 1000/fps


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

ones_erode = np.ones((3,3))
ones_dilate = np.ones((19,19))
while True:
    _,frame = cap.read()
    cv2.imshow("Video",frame)
    frame_clone = frame.copy()

    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(frame,background)
    cv2.imshow("Diff",diff)

    _,threshold = cv2.threshold(diff,80,255,type = cv2.THRESH_BINARY)
    cv2.imshow("Threshold",threshold)

    erode = cv2.erode(threshold,ones_erode)
    cv2.imshow("Erode",erode)

    dilate = cv2.dilate(erode,ones_dilate)
    cv2.imshow("Dilate",dilate)

    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        for cont in contours:
            x, y, w, h = cv2.boundingRect(cont)

            cv2.rectangle(frame_clone, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame_clone,'FPS: '+str(int(fps)),(20,20),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0),2)
            cv2.imshow("Result",frame_clone)


            ROI = frame_clone[y:y+h, x:x+w]

            img = cv2.resize(ROI, (224, 224))
            img = img_to_array(img)
            input = expand_dims(img, 0)

            inputGen = ImageDataGenerator(rescale=1. / 255)
            input_generator = inputGen.flow(input, batch_size=32)
            predict = model.predict_generator(input_generator)
            # print(predict)
            label = np.argmax(predict, axis=1)
            # print(label)

            if label == 2:
                cv2.putText(frame_clone,"Human" + str(np.round(predict,3)),(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
            elif label == 0:
                cv2.putText(frame_clone, "Car" + str(np.round(predict, 3)), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                            (255, 255, 0), 2)
            else:
                cv2.putText(frame_clone, "Cat"+str(np.round(predict,3)), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

    out.write(frame_clone)
    if cv2.waitKey(int(wait_time)) == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()




