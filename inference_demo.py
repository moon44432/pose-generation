import cv2
import os

image = None
copied_image = None
target_point = (0, 0)

def mouse_click(event, x, y, flags, param):
    global copied_image
    global target_point
    if event == cv2.EVENT_FLAG_LBUTTON:
        target_point = (x, y)
        cv2.circle(copied_image, (x, y), 10, (0, 0, 255), -1)
        cv2.imshow('image', copied_image)
        copied_image = image.copy()

def inference(image_path):
    global image
    global copied_image
    image = cv2.imread(image_path)
    copied_image = image.copy()
    print(image.shape)
    cv2.imshow('image', copied_image)
    cv2.setMouseCallback('image', mouse_click)

    while True:
        k = cv2.waitKey(0)
        if k == 13: # press ENTER
            cv2.destroyAllWindows()
            break
    print(target_point)

    ## model inference ##

    ## pose visualize ##

if __name__ == '__main__':
    image_path = 'test.jpg'
    inference(image_path)