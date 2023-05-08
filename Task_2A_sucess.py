## Initialising libraries.
import cv2
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#Reading coordinates from file made.
coordinates = []
with open('Coordinatess.csv') as coord_file:
    reader = csv.reader(coord_file)
    for row in reader:
        coordinates.append(row)

#Importing the images and reading them.
images = []
for image_files in os.listdir("image"):
    img = cv2.imread(os.path.join("image", image_files))
    images.append(img)

# Drawing the bounding boxes on the images with corresponding coordinates.
for i, img in enumerate(images):
    if i>=len(coordinates):
        break
    coordinatess = coordinates[i]
    x, y, w, h = [int(c) if c != "" else 0 for c in coordinatess]
    bounded_images = cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,225), 2)
    plt.imshow(bounded_images)
    plt.axis('on')
    plt.show()


