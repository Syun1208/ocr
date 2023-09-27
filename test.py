from api import recognizer
import cv2
import os
from paddleocr import PPStructure,draw_structure_result,save_structure_res


def test_ppstructure():
    table_engine = PPStructure(show_log=True, image_orientation=True)

    save_folder = './output_ppstructure'
    os.makedirs(save_folder, exist_ok=True)
    img_path = './PaddleOCR/ppstructure/docs/table/1.png'
    img = cv2.imread(img_path)
    result = table_engine(img)
    save_structure_res(result, save_folder,os.path.basename(img_path).split('.')[0])

    for line in result:
        line.pop('img')
        print(line)

    from PIL import Image

    font_path = './PaddleOCR/doc/fonts/simfang.ttf' # PaddleOCR下提供字体包
    image = Image.open(img_path).convert('RGB')
    im_show = draw_structure_result(image, result,font_path=font_path)
    im_show = Image.fromarray(im_show)
    im_show.save(os.path.join(save_folder, 'result.jpg'))

if __name__ == "__main__":
    # image = cv2.imread('/home/hoangtv/Desktop/Long/customer/SimpleHTR/data/line.png')
    # results = recognizer(image)
    # print(results)
    test_ppstructure()