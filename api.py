from fastapi import FastAPI
from fastapi.requests import Request
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

FILE = Path(__file__).resolve()
# Read folder containing file path
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
# main work directory
WORK_DIR = os.path.dirname(ROOT)


sys.path.append(os.path.join(ROOT, 'SimpleHTR'))
sys.path.append(os.path.join(ROOT, 'SimpleHTR/src'))


from SimpleHTR.src.model import Model, DecoderType
from SimpleHTR.src.main import parse_args, char_list_from_file, infer

app = FastAPI()
output_dir = './ouput'
os.makedirs(output_dir, exist_ok=True)


# load models
def load_models():
    refine_net = load_refinenet_model(cuda=True)
    craft_net = load_craftnet_model(cuda=True)
    args = parse_args()
    decoder_mapping = {'bestpath': DecoderType.BestPath,
                        'beamsearch': DecoderType.BeamSearch,
                        'wordbeamsearch': DecoderType.WordBeamSearch}
    decoder_type = decoder_mapping['wordbeamsearch']
    model = Model(char_list_from_file(), decoder_type, must_restore=True, dump=args.dump)

    return craft_net, refine_net, model

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

    craft_net, refine_net, model = load_models()

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
        results['score'] = list_score

    results['time_inference'] = time.time() - start
    
    return results

@app.get('/')
async def health_check():
    return {'message': 'Hello Anh Long'}


@app.post('/ocr')
async def ocr(request: UserRequest):
    if request.image_base64:
        image = base64_to_image(request.image_base64)
    elif request.image_url:
        image = requests.get(request.image_url)

    results = recognizer(image)

    return results
    


if __name__ == "__main__":
    uvicorn.run("api:app", host='0.0.0.0', port=8080, reload=True, workers=Pool()._processes)