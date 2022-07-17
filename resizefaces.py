import cv2
import os
path = r"F:\Nu\semester 6\R & D\dataresized\0"
myList = os.listdir(path)
print(myList)
# Read the input image
for i in myList:
        print(i)
        img1 = cv2.imread(path+ "/" +str(i), cv2.IMREAD_UNCHANGED)
        width = 32
        height = 32
        dim = (width, height)

# resize image
        resized = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
        print(type(resized))
        cv2.imwrite((r"F:\\Nu\\semester 6\\R & D\\pics\\" + str(i)),resized)
