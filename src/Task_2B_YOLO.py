import cv2
import numpy as np
from yolov5 import models
#from yolov5 import non_max
import os
import torch

# Load the pre-trained weights
YOLO_V5_MODEL = models
#YOLO_V5_MODEL.evaluate()

# Load the images from the folder
images = []
for image_files in os.listdir("image"):
    img = cv2.imread(os.path.join("image", image_files))
    images.append(img)

# Perform object detection on each image
for image in images:
    # Convert the image to a tensor
    tensor = torch.from_numpy(image).unsqueeze(0).to(models.device).float32()

    # Perform object detection
    outputs = YOLO_V5_MODEL(tensor)

    # Use non-max suppression to remove overlapping detections
    boxes, labels, scores = [o.numpy() for o in outputs]
    boxes = np.moveaxis(boxes, 1, -1)
    boxes, labels, scores = non_max_suppression(boxes, labels, scores)

    # Draw the bounding boxes and class labels on the image
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{YOLO_V5_MODEL.classes[label]} ({score * 100:.2f}%)"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
