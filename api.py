from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.requests import Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from multiprocessing import Pool
import uvicorn
import time
import requests
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)
import os
import cv2
import base64
import numpy as np
import sys
from pathlib import Path
import logging
import urllib.request
import tensorflow as tf
from tensorflow.python.framework import ops
from paddleocr import PPStructure,draw_structure_result,save_structure_res


FILE = Path(__file__).resolve()
# Read folder containing file path
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
# main work directory
WORK_DIR = os.path.dirname(ROOT)


sys.path.append('./SimpleHTR')
sys.path.append('./SimpleHTR/src')
sys.path.append('./PaddleOCR')

from SimpleHTR.src.model import Model, DecoderType
from SimpleHTR.src.main import parse_args, char_list_from_file, infer
from PaddleOCR.ppstructure.table.predict_table import table_recognizer_ppocr
app = FastAPI()
output_dir = './ouput'
os.makedirs(output_dir, exist_ok=True)


# load models
refine_net = load_refinenet_model(cuda=True)
craft_net = load_craftnet_model(cuda=True)
args = parse_args()
decoder_mapping = {'bestpath': DecoderType.BestPath,
                    'beamsearch': DecoderType.BeamSearch,
                    'wordbeamsearch': DecoderType.WordBeamSearch}
decoder_type = decoder_mapping['wordbeamsearch']
ops.reset_default_graph()
model = Model(char_list_from_file(), decoder_type, must_restore=True, dump=args.dump)
table_engine = PPStructure(show_log=True, image_orientation=True)

class UserRequest(BaseModel):
    image_url: str
    image_base64: str

def base64_to_image(im_b64):
    im_bytes = base64.b64decode(im_b64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img

def recognizer(image):

    results = {}
    list_text = []
    list_score = []

    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        cuda=True,
        long_size=1280
    )

    start = time.time()
    for i, box in enumerate(prediction_result["boxes"]):
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        point1, point2, point3, point4 = box
        x, y, w, h = point1[0], point1[1], point2[0] - point1[0], point4[1]-point1[1]
        crop_img = image[y:y+h, x:x+w]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        recognized, probability = infer(model, crop_img)

        list_text.append(recognized)
        list_score.append(probability)

    results['text'] = list_text
    results['score'] = list(map(lambda x: float(x), list_score))

    results['time_inference'] = time.time() - start

    
    return results

@app.get('/')
async def health_check():
    return {'message': 'Hello Anh Long'}


@app.post('/ocr')
async def ocr(request: UserRequest):
    if request.image_base64 != "string" or request.image_base64 != "":
        logging.info("USER CHOOSES IMAGE BASE64 MODE")
        image = base64_to_image(request.image_base64)
    elif request.image_url != "string" or request.image_url != "":
        logging.info("USER CHOOSES IMAGE URL MODE")
        response = urllib.request.urlopen(request.image_url)
        image_data = np.asarray(bytearray(response.read()), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    results = recognizer(image)

    return JSONResponse(content=jsonable_encoder(results))

@app.post("/ocr_test")
async def ocr_test(file :UploadFile = File(...)):
    if file.filename.split('.')[-1] in ("jpg", "jpeg", "png"):
        pass
    else:
        raise HTTPException(status_code=415, detail="Item not found")
    logging.info("USER CHOOSES UPLOADING FILE MODE")
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)    
    results = recognizer(image)
    
    return JSONResponse(content=jsonable_encoder(results))
@app.post('/kie')
async def kie(request: UserRequest):
    pass

@app.post('/table_recognizer')
async def table_recognizer(request: UserRequest):
    if request.image_base64 != "string" or request.image_base64 != "":
        logging.info("USER CHOOSES IMAGE BASE64 MODE")
        image = base64_to_image(request.image_base64)
    elif request.image_url != "string" or request.image_url != "":
        logging.info("USER CHOOSES IMAGE URL MODE")
        response = urllib.request.urlopen(request.image_url)
        image_data = np.asarray(bytearray(response.read()), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    result = table_recognizer(image)

    return HTMLResponse(content=result['html'], status_code=200)

    
    
    

@app.post('/table_recognizer_test')
async def table_recognizer_test(file :UploadFile = File(...)):
    if file.filename.split('.')[-1] in ("jpg", "jpeg", "png"):
        pass
    else:
        raise HTTPException(status_code=415, detail="Item not found")
    logging.info("USER CHOOSES UPLOADING FILE MODE")
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 
    result = table_recognizer_ppocr(image)

    return HTMLResponse(content=result['html'], status_code=200)   

def test():
    image = cv2.imread('/home/hoangtv/Desktop/Long/customer/PaddleOCR/doc/datasets/table_tal_demo/2.jpg')
    results = table_recognizer_ppocr(image)
    print(results['html'])
if __name__ == "__main__":
    # base64_code = "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"
    # image = base64_to_image(base64_code)
    # results = recognizer(image)
    # test()
    uvicorn.run("api:app", host='0.0.0.0', port=8080, reload=True, workers=Pool()._processes)