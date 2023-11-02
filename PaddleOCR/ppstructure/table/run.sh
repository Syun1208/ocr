python predict_table.py \
    --det_model_dir=/home/hoangtv/Desktop/Long/customer/PaddleOCR/ppstructure/table/pretrained_table_en/en_PP-OCRv3_det_infer \
    --rec_model_dir=/home/hoangtv/Desktop/Long/customer/PaddleOCR/ppstructure/table/pretrained_table_en/en_PP-OCRv3_rec_infer  \
    --table_model_dir=/home/hoangtv/Desktop/Long/customer/PaddleOCR/ppstructure/table/pretrained_table_en/en_ppstructure_mobile_v2.0_SLANet_infer \
    --rec_char_dict_path=/home/hoangtv/Desktop/Long/customer/PaddleOCR/ppstructure/table/pretrained_table_en/en_dict.txt \
    --table_char_dict_path=/home/hoangtv/Desktop/Long/customer/PaddleOCR/ppstructure/table/pretrained_table_en/table_structure_dict.txt \
    --image_dir=/home/hoangtv/Desktop/Long/customer/PaddleOCR/ppstructure/table/data/test.png \
    --output=../output/table