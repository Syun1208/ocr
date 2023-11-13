from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.requests import Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from multiprocessing import Pool
import uvicorn
import time
from bs4 import BeautifulSoup
import json
import requests
# from craft_text_detector import (
#     read_image,
#     load_craftnet_model,
#     load_refinenet_model,
#     get_prediction,
#     export_detected_regions,
#     export_extra_results,
#     empty_cuda_cache
# )
import os
import cv2
import base64
import numpy as np
import sys
from pathlib import Path
import logging
import torch
import urllib.request
import tensorflow as tf
from tensorflow.python.framework import ops
# from paddleocr import PPStructure,draw_structure_result,save_structure_res


FILE = Path(__file__).resolve()
# Read folder containing file path
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
# main work directory
WORK_DIR = os.path.dirname(ROOT)


# sys.path.append('./SimpleHTR')
# sys.path.append('./SimpleHTR/src')
sys.path.append('./PaddleOCR')

# from SimpleHTR.src.model import Model, DecoderType
# from SimpleHTR.src.main import parse_args, char_list_from_file, infer
from PaddleOCR.ppstructure.table.predict_table import table_recognizer_ppocr
app = FastAPI()
output_dir = './ouput'
os.makedirs(output_dir, exist_ok=True)

device = True if torch.cuda.is_available() else False
# load models
# refine_net = load_refinenet_model(cuda=device)
# craft_net = load_craftnet_model(cuda=device)
# args = parse_args()
# decoder_mapping = {'bestpath': DecoderType.BestPath,
#                     'beamsearch': DecoderType.BeamSearch,
#                     'wordbeamsearch': DecoderType.WordBeamSearch}
# decoder_type = decoder_mapping['wordbeamsearch']
# ops.reset_default_graph()
# model = Model(char_list_from_file(), decoder_type, must_restore=True, dump=args.dump)
# table_engine = PPStructure(show_log=True, image_orientation=True)

class UserRequest(BaseModel):
    image_url: str
    image_base64: str

def base64_to_image(im_b64):
    im_bytes = base64.b64decode(im_b64)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img

# def recognizer(image):

#     results = {}
#     list_text = []
#     list_score = []

#     prediction_result = get_prediction(
#         image=image,
#         craft_net=craft_net,
#         refine_net=refine_net,
#         text_threshold=0.7,
#         link_threshold=0.4,
#         low_text=0.4,
#         cuda=True,
#         long_size=1280
#     )

#     start = time.time()
#     for i, box in enumerate(prediction_result["boxes"]):
#         box = np.array(box).astype(np.int32).reshape(-1, 2)
#         point1, point2, point3, point4 = box
#         x, y, w, h = point1[0], point1[1], point2[0] - point1[0], point4[1]-point1[1]
#         crop_img = image[y:y+h, x:x+w]
#         crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
#         recognized, probability = infer(model, crop_img)

#         list_text.append(recognized)
#         list_score.append(probability)

#     results['text'] = list_text
#     results['score'] = list(map(lambda x: float(x), list_score))

#     results['time_inference'] = time.time() - start

    
#     return results

@app.get('/')
async def health_check():
    return {'message': 'Hello Anh Long'}


# @app.post('/ocr')
# async def ocr(request: UserRequest):
#     if request.image_base64 != "string" or request.image_base64 != "":
#         logging.info("USER CHOOSES IMAGE BASE64 MODE")
#         image = base64_to_image(request.image_base64)
#     elif request.image_url != "string" or request.image_url != "":
#         logging.info("USER CHOOSES IMAGE URL MODE")
#         response = urllib.request.urlopen(request.image_url)
#         image_data = np.asarray(bytearray(response.read()), dtype=np.uint8)
#         image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

#     results = recognizer(image)

#     return JSONResponse(content=jsonable_encoder(results))

# @app.post("/ocr_test")
# async def ocr_test(file :UploadFile = File(...)):
#     if file.filename.split('.')[-1] in ("jpg", "jpeg", "png"):
#         pass
#     else:
#         raise HTTPException(status_code=415, detail="Item not found")
#     logging.info("USER CHOOSES UPLOADING FILE MODE")
#     contents = await file.read()
#     nparr = np.fromstring(contents, np.uint8)
#     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)    
#     results = recognizer(image)
    
#     return JSONResponse(content=jsonable_encoder(results))
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

    result = table_recognizer_ppocr(image)

    return HTMLResponse(content=result['html'], status_code=200)

    
@app.post('/table_recognizer_json')
async def table_recognizer_json(request: UserRequest):
    if request.image_base64 != "string" or request.image_base64 != "":
        logging.info("USER CHOOSES IMAGE BASE64 MODE")
        image = base64_to_image(request.image_base64)
    elif request.image_url != "string" or request.image_url != "":
        logging.info("USER CHOOSES IMAGE URL MODE")
        response = urllib.request.urlopen(request.image_url)
        image_data = np.asarray(bytearray(response.read()), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    result_tr_ppocr = table_recognizer_ppocr(image)
    soup = BeautifulSoup(result_tr_ppocr['html'], 'html.parser')

    # Extract data from HTML and format it into JSON
    data = {"drug": [], "Full_Directions": []}

    for row in soup.find_all('tbody')[0].find_all('tr'):
        columns = row.find_all('td')
        data["drug"].append(columns[0].get_text(strip=True))
        data["Full_Directions"].append(columns[1].get_text(strip=True))

    # Organize the data into a list of dictionaries
    result = [{"drug": drug, "Full_Directions": direction} for drug, direction in zip(data["drug"], data["Full_Directions"])]

    format_response = {'html': result_tr_ppocr['html'], 'json': result}

    return JSONResponse(content=jsonable_encoder(format_response), status_code=200)

    

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


if __name__ == "__main__":
    uvicorn.run("api:app", host='0.0.0.0', port=8080, reload=True, workers=Pool()._processes)