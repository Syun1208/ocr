from api import recognizer
import cv2


if __name__ == "__main__":
    image = cv2.imread('/home/hoangtv/Desktop/Long/customer/SimpleHTR/data/line.png')
    results = recognizer(image)
    print(results)