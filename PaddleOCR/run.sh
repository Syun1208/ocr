python ppstructure/table/predict_table.py \
    --det_model_dir=/home/hoangtv/Desktop/Long/customer/PaddleOCR/ppstructure/inference/ch_PP-OCRv3_det_infer \
    --rec_model_dir=/home/hoangtv/Desktop/Long/customer/PaddleOCR/ppstructure/inference/ch_PP-OCRv3_rec_infer  \
    --table_model_dir=/home/hoangtv/Desktop/Long/customer/PaddleOCR/ppstructure/inference/ch_ppstructure_mobile_v2.0_SLANet_infer \
    --rec_char_dict_path=./ppocr/utils/ppocr_keys_v1.txt \
    --table_char_dict_path=./ppocr/utils/dict/table_structure_dict_ch.txt \
    --image_dir=/home/hoangtv/Desktop/Long/customer/PaddleOCR/doc/datasets/table_tal_demo/2.jpg \
    --output=./output/table