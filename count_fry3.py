import cv2
import numpy as np
import os


def count_fry(img, bg):
    # make gray images
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(bg,cv2.COLOR_BGR2GRAY)

    # get the difference between the two images
    diff = cv2.absdiff(gray, bg_gray)

    # threshold using otsu or binary
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #cv2.imshow("Threshold", thresh)

    kernel = np.ones((3,3), np.uint8)
    erode = cv2.erode(thresh, kernel, iterations=1) 
    dilate = cv2.dilate(erode, kernel, iterations=1)

    contours, _ = cv2.findContours(dilate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)
        cv2.drawContours(img,[hull],-1,(0,255,255),3)
        cv2.drawContours(img,[contours[i]],-1,(255,0,0),2)

    number_of_objects_in_image = len(contours)

    for i in range(len(contours)):
        contour_area = cv2.contourArea(contours[i])
        hull_area = cv2.contourArea(hull_list[i])
        if hull_area > contour_area + contour_area/3:
            number_of_objects_in_image = number_of_objects_in_image + 1
    return number_of_objects_in_image

# root
path = "lab03/IMAGES/"

# save counts to calcualte accuracy
counts_100 = []
counts_200 = []
counts_300 = []
counts_400 = []

# take note which folder
folder = 0

# walk through all images
for (root, dirs, file) in os.walk(path):
    # keeping the counts
    counts = []

    for f in file:
        # there are 4 bg images
        if 'background' in f:
            folder = folder + 1
            bg = cv2.imread(os.path.join(root, f).replace("\\","/"))
            continue
        # count frys in image
        img = cv2.imread(os.path.join(root, f).replace("\\","/"))
        counts.append(count_fry(img, bg))
    
    if folder == 1:
        counts_100 = counts
    elif folder == 2:
        counts_200 = counts
    elif folder == 3:
        counts_300 = counts
    else:
        counts_400 = counts

# calculate percentage per image
counts_100 = [x/100 for x in counts_100]
counts_200 = [x/200 for x in counts_200]
counts_300 = [x/300 for x in counts_300]
counts_400 = [x/400 for x in counts_400]

# get average accuracy for each folder
accuracy_100 = sum(counts_100)/len(counts_100)
accuracy_200 = sum(counts_200)/len(counts_200)
accuracy_300 = sum(counts_300)/len(counts_300)
accuracy_400 = sum(counts_400)/len(counts_400)

print("Accuracy for images with 100 frys: ", accuracy_100)
print("Accuracy for images with 200 frys: ", accuracy_200)
print("Accuracy for images with 300 frys: ", accuracy_300)
print("Accuracy for images with 400 frys: ", accuracy_400)



