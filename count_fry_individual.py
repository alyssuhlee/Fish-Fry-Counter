import cv2
import numpy as np

# Read image
img = cv2.imread('lab03/IMAGES/400/my_photo-11.jpg')
bg = cv2.imread('lab03/IMAGES/400/background.jpg')

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
print(number_of_objects_in_image)

for i in range(len(contours)):
    contour_area = cv2.contourArea(contours[i])
    hull_area = cv2.contourArea(hull_list[i])
    if hull_area > contour_area + 60:
        number_of_objects_in_image = number_of_objects_in_image + 1

print ("The number of frys in this image: ", str(number_of_objects_in_image))

cv2.putText(img, f"Fry Count: {number_of_objects_in_image}", (10, 30), 
            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 225, 0), 2)

img = cv2.resize(img, (800, 500))
cv2.imshow("diff(img1, img2)", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()

