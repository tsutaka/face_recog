"""
main.py
"""
# pip install cmake
# pip install dlib
# install VS2019
# import dlib
# from skimage import io
import cv2

# 顔検出
# def face_detect(file_name):
#     face_detector = dlib.get_frontal_face_detector()
#     image = io.imread(file_name)
#     detected_faces = face_detector(image, 1)

#     save_image = cv2.imread(file_name, cv2.IMREAD_COLOR)

#     for i, face_rect in enumerate(detected_faces):
#         cv2.rectangle(save_image, tuple([face_rect.left(),face_rect.top()]), tuple([face_rect.right(),face_rect.bottom()]), (0, 0,255), thickness=2)
#         cv2.imwrite('complete_'+file_name, save_image)

# 顔切り出し
def cut_face(file_name, cut_rect):
    """cut_face

        cut_rect = [top, bottom, left, right]
    """
    output_file = "out.jpg"
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

    INPUT_FILE = "image/001_Charles/440px-Prince_Charles_Ireland-4.jpg"
    print(INPUT_FILE)

    # face_detect()

    rect = [0, 50, 0, 50]
    cut_face(INPUT_FILE, rect)

    print("end")