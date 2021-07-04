import cv2 
import numpy as np
from matplotlib import pyplot as plt


CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.5

class_names = []
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

img = cv2.imread("Images/hard_hat_workers1578.png")

colors = np.random.uniform(0,255,size=(len(class_names),3))

net = cv2.dnn.readNet("custom-yolov4-tiny-detector_best.weights", "custom-yolov4-tiny-detector.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(640, 640), scale=1/255, swapRB=True)

classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

for (classid, score, box) in zip(classes, scores, boxes):
  color=colors[classid[0]]
  label = "%s : %f" % (class_names[classid[0]], score)
  cv2.rectangle(img, box,color, 2)
  cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,255), 2)   
  

cv2.imshow('frame',img)
cv2.waitKey(0) 