"""
main.py
"""
import os
import time
import glob
import cv2

# face detect
def face_detect(file_name):
    #open image
    image = cv2.imread(file_name)

    #gray scale convert
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("test", image_gray)
    # cv2.waitKey(0)

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
            # color = (255, 255, 255)
            # cv2.rectangle(image, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
            # cv2.imshow("test", image)

            if(max_wxh < rect[2] * rect[3]):
                max_rect = rect
            
        return (max_rect[1], max_rect[1] + max_rect[3], max_rect[0], max_rect[0] + max_rect[2])
    return (0, 0, 0, 0)

# 顔切り出し
def cut_face(file_name, cut_rect):
    """cut_face
        cut_rect = [top, bottom, left, right]
    """
    output_file = str(os.path.splitext(file_name)[0]) + "_face.jpg"
    # print("output : " + output_file)
    img = cv2.imread(file_name)
    img1 = img[cut_rect[0] : cut_rect[1], cut_rect[2] : cut_rect[3]]
    cv2.imwrite(output_file, img1)

# 顔特徴量抽出
def feature_extraction():
    pass

# 顔検出
# 顔認識


if __name__ == "__main__":
    print("begin")

    INPUT_PATH = "." + os.sep + "data" + os.sep
    input_list = glob.glob(INPUT_PATH + "*" + os.sep + "*.jpg")
    print(input_list)

    for input_file in input_list:
        rect = face_detect(input_file)
        # print("rect : " + str(rect))

        cut_face(input_file, rect)

    print("end")