"""
main.py
"""
import os
import time
from pathlib import Path
import glob
import cv2

# face detect
def face_detect(file_name):
    #open image
    image = cv2.imread(file_name)

    #gray scale convert
    #Don't use "_" as the first character of filename
    #Don't use multi byte character as the filename 
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #features extraction
    cascade_path = "./model/haarcascade_frontalface_default.xml"
    assert os.path.isfile(cascade_path), 'haarcascade_frontalface_default.xml not exists'
    cascade = cv2.CascadeClassifier(cascade_path)

    #face detect
    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

    if len(facerect) > 0:
        #rect = (x, y, w, h)
        max_wxh = 0 
        for rect in facerect:
            if(max_wxh < rect[2] * rect[3]):
                max_rect = rect
            
        return (max_rect[1], max_rect[1] + max_rect[3], max_rect[0], max_rect[0] + max_rect[2])
    return (0, 0, 0, 0)

# 顔切り出し
def cut_face(file_name, cut_rect):
    """cut_face
        cut_rect = [top, bottom, left, right]
    """
    output_file = "." + os.sep + "work" + os.sep + "face" + os.sep
    temp_path = Path(file_name)
    output_file += (temp_path.parts)[1] + ".jpg"

    print("output : " + output_file)
    img = cv2.imread(file_name)
    img1 = img[cut_rect[0] : cut_rect[1], cut_rect[2] : cut_rect[3]]
    cv2.imwrite(output_file, img1)

if __name__ == "__main__":
    print("begin")

    INPUT_PATH = "." + os.sep + "data" + os.sep
    input_list = glob.glob(INPUT_PATH + "*" + os.sep + "*.jpg")

    for input_file in input_list:
        rect = face_detect(input_file)
        # print("rect : " + str(rect))

        cut_face(input_file, rect)

    print("end")

