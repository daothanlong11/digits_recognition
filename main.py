import cv2
import numpy as np
import os
from preprocess import *
from imutils import paths

full_path = os.path.realpath(__file__)
path = os.path.dirname(full_path)


drawing = False  # true if mouse is pressed
mode = True
font = cv2.FONT_HERSHEY_SIMPLEX
# mouse callback function


def roi(location):
    xbar = []
    ybar = []
    for value in location:
        xbar.append(value[0])
        ybar.append(value[1])
    min_x = min(xbar)
    max_x = max(xbar)
    min_y = min(ybar)
    max_y = max(ybar)
    return min_x, max_y, max_x, min_y


location = []
number = 0
line_color = (255, 255, 255)


def paint_draw(event, former_x, former_y, flags, param):
    global current_former_x, current_former_y, drawing, mode, number,result
    if event == cv2.EVENT_LBUTTONDOWN:
        location.clear()
        drawing = True
        current_former_x, current_former_y = former_x, former_y
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.line(image, (current_former_x, current_former_y),
                         (former_x, former_y), line_color, 20)
                current_former_x = former_x
                current_former_y = former_y
        location.append([current_former_x, current_former_y])

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.line(image, (current_former_x, current_former_y),
                     (former_x, former_y), line_color, 5)
            current_former_x = former_x
            current_former_y = former_y
        x1, y1, x2, y2 = roi(location)
        cropping = False
        cv2.rectangle(image, (x1-20, y1+20), (x2+20, y2-20), (0, 0, 255), 2)
        #cv2.imshow("image", image)
        crop_img = image[y2-18:y1+18, x1-18:x2+18]
        cv2.imwrite(path+"/digits/digit.png", crop_img)
        argv = path+"/digits/digit.png"

        img_pre = preprocess(argv)
        result = clf.predict([img_pre])[0]
        cv2.putText(image,'number: %d'%result,(x1-40, y1+40),font,0.8,(0,255,0))

             

    return former_x, former_y

image = cv2.imread(path+"/background/black_1.jpg")
cv2.namedWindow("OpenCV Paint Brush")
#cv2.namedWindow("Result")
result = 0
cv2.setMouseCallback('OpenCV Paint Brush', paint_draw)

while(1):
    cv2.imshow('OpenCV Paint Brush', image)
    
    if cv2.waitKey(1) == ord('r'):  # reset new paint
        image = cv2.imread(path+"/background/black_1.jpg")
        cv2.imshow('OpenCV Paint Brush', image)

    if cv2.waitKey(1) == ord('y'):  
        imagePaths = list(paths.list_images(path + "/digits_preprocess/{}/".format(result)))
        if len(imagePaths) < 1:
            img = cv2.imread(path+"/digits_preprocess/preprocess_number/digit_pre.png")
            cv2.imwrite(path+"/digits_preprocess/{}/0.png".format(result),img)
            print("save new datasets success")
        else:
            img = cv2.imread(path+"/digits_preprocess/preprocess_number/digit_pre.png")
            imagePath = imagePaths[len(imagePaths)-1]
            name = imagePath.split(os.path.sep)[-1]
            name = int(name.split('.')[-2])
            cv2.imwrite(path+"/digits_preprocess/{}/{}.png".format(result,name+1),img)
            print("save new datasets success")
    
cv2.destroyAllWindows()
