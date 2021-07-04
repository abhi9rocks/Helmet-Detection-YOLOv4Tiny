import cv2
import time
import numpy as np
import winsound
from PIL import Image

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.5

cap = cv2.VideoCapture("test.mp4")
prevTime = 0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

size = (width, height)

result = cv2.VideoWriter('output.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

while cap.isOpened(): 
    ret, frame = cap.read()
    frame = np.array(frame)
    
    class_names = []
    with open("coco.names", "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]

    
    colors = np.random.uniform(0,255,size=(len(class_names),3))
    net = cv2.dnn.readNet("custom-yolov4-tiny-detector_best.weights", "custom-yolov4-tiny-detector.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(640, 640), scale=1/255, swapRB=True)
    
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    
    
    
    for (classid, score, box) in zip(classes, scores, boxes):
        for (classid, score, box) in zip(classes, scores, boxes):
            img2=frame[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
            x=int((box[2])/2)
            y=int((box[3])/3.5)
            ifa = Image.fromarray(img2)
            color = ifa.getpixel((x,y))
            label = "%s : %f" % (class_names[classid[0]], score)
            cv2.rectangle(frame, box,color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2)
            duration = 1000  # milliseconds
            freq = 440  # Hz
            winsound.Beep(freq, duration)
            result.write(frame)

        

        
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        result.release()
        cv2.destroyAllWindows()
        break 