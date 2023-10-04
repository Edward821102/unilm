import numpy as np
from transformers import AutoModelForTokenClassification
from transformers import AutoProcessor
from PIL import Image
import json
import pandas as pd
from enum import Enum
from pathlib import Path
from torch.nn.functional import softmax
import torch

class LabelId(Enum):
    Item_value              = 0
    price_value             = 1    
    other                   = 2
    total_value             = 3
    quantity_value_of_unit  = 4
    unit_price_key          = 5
    quantity_key            = 6
    sales_amount_key        = 7
    seller_tax_id_key       = 8
    seller_tax_id_value     = 9
    total_char_key          = 10
    sales_tax_key           = 11
    quantity_key_of_unit    = 12
    sales_tax_value         = 13
    quantity_value          = 14
    id_number_key           = 15
    id_number_value         = 16
    buyer_tax_id_key        = 17
    buyer_tax_id_value      = 18
    Item_key                = 19
    sales_amount_value      = 20
    date                    = 21
    total_key               = 22
    price_key               = 23
    total_char_value        = 24
    unit_price_value        = 25


processor = AutoProcessor.from_pretrained("/home/edward/桌面/權重/LayoutLmv3/electronic_200_v3(混合框)", apply_ocr=False)
model = AutoModelForTokenClassification.from_pretrained("/home/edward/桌面/權重/LayoutLmv3/electronic_200_v3(混合框)")

labels = []
with open('/home/edward/桌面/標注數據/electronic_lmv3_200_v3/class_list.txt', "r", encoding="utf-8") as f:
    for i in f.readlines():
        i = i.replace('\n', '')
        index, label = i.split(' ')
        labels.append(label)

id2label = {v: k for v, k in enumerate(labels)}
label2color = {
    "Item_value": "black",
    "price_value": "black",
    "other": "black",
    "total_value": "black",
    "quantity_value_of_unit": "black",
    "unit_price_key": "black",
    "quantity_key": "black",
    "sales_amount_key": "black",
    "seller_tax_id_key": "black",
    "seller_tax_id_value": "black",
    "total_char_key": "black",
    "sales_tax_key": "black",
    "quantity_key_of_unit": "black",
    "sales_tax_value": "black",
    "quantity_value": "black",
    "id_number_key": "black",
    "id_number_value": "black",
    "buyer_tax_id_key": "black",
    "buyer_tax_id_value": "black",
    "Item_key": "black",
    "sales_amount_value": "black",
    "date": "black",
    "total_key": "black",
    "price_key": "black",
    "total_char_value": "black",
    "unit_price_value": "black"
  }


def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

def iob_to_label(label):
    return label

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]
def process_dict(data):
    if data != []:
        max_dict = max(data, key=lambda x: max(x.keys()))
        max_key = max(max_dict.keys())
        return max_dict[max_key]
    else:
        return []
    

def process_image(target_path:str):
    correct_counts = 0
    big_info = {}
    with open(target_path, "r", encoding="utf-8") as f:
        for i in f.readlines():
            # if "AA9662.jpg" not in i:
            #     continue
            important_infos = {"file_name":"", "id_number":"", "buyer_tax_id":"", "date":"", "sales_amount_value":"", "sales_tax_value":"", "total_value":"", "seller_tax_id":"", "other":""}
            dict_i = json.loads(i)
            image_name = Path(dict_i['file_name']).name
            print(image_name)
            important_infos["file_name"] = image_name
            image = Image.open(f"{dict_i['file_name']}")
            width, height = image.size
            words = []
            boxes = []  
            for index in range(len(dict_i["annotations"])):
                words.append(dict_i["annotations"][index]["text"])
                box = dict_i["annotations"][index]["box"]
                resort_box = [box[0][0], box[0][1], box[2][0], box[2][1]]
                new_box = normalize_bbox(resort_box, width, height)
                boxes.append(new_box)
            print(len(words))
            encoding = processor(images=image, text=words, boxes=boxes, truncation=True, return_offsets_mapping=True, return_tensors="pt")
            offset_mapping = encoding.pop('offset_mapping')
            outputs = model(**encoding)
            predictions = outputs.logits.argmax(-1).squeeze().tolist() # logits of shape (batch_size, num_classes)
            token_boxes = encoding.bbox.squeeze().tolist()
            scores = softmax(outputs.logits, dim=-1).squeeze().tolist()
            # # only keep non-subword predictions
            is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
            # print(len([i for i in is_subword if i==True]))
            true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]][1:-1]
            true_scores = [max(pred) for idx, pred in enumerate(scores) if not is_subword[idx]][1:-1]
            true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]][1:-1]
            outputs = model(**encoding)
            predictions = outputs.logits.argmax(-1).squeeze().tolist() # logits of shape (batch_size, num_classes)
            token_boxes = encoding.bbox.squeeze().tolist()
            is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0


            repeat_box = []
            idx = 0
            id_numbers, buyer_tax_ids, dates, seller_tax_ids, sales_amount_values, sales_tax_values, total_values, others = [], [], [], [], [], [], [], []
            num = 0
            for prediction, score ,box in zip(true_predictions, true_scores, true_boxes):
                num+=1
                if repeat_box != box:
                    # print(words[idx], prediction, score)
                    if prediction == "id_number_value":
                        id_numbers.append({score:words[idx]})
                    elif prediction == "buyer_tax_id_value":
                        buyer_tax_ids.append({score:words[idx]})
                    elif prediction == "date":
                        dates.append({score:words[idx]})
                    elif prediction == "seller_tax_id_value":
                        seller_tax_ids.append({score:words[idx]})
                    elif prediction == "sales_amount_value":
                        sales_amount_values.append({score:words[idx]})
                    elif prediction == "sales_tax_value":
                        sales_tax_values.append({score:words[idx]})
                    elif prediction == "total_value":
                        total_values.append({score:words[idx]})
                    else:
                        others.append({score:words[idx]})
                    idx += 1
                repeat_box = box
            
            important_infos["id_number"] = process_dict(id_numbers)
            important_infos["buyer_tax_id"] = process_dict(buyer_tax_ids)
            important_infos["date"] = process_dict(dates)
            important_infos["seller_tax_id"] = process_dict(seller_tax_ids)
            important_infos["sales_amount_value"] = process_dict(sales_amount_values)
            important_infos["sales_tax_value"] = process_dict(sales_tax_values)
            important_infos["total_value"] = process_dict(total_values)
            important_infos["other"] = others
            big_info.setdefault(correct_counts, important_infos)
            correct_counts += 1
        print(num)
        # print(big_info)
        # df = pd.DataFrame.from_dict(big_info, orient='index')
        # df.to_excel(f"/home/edward/桌面/QQQQQ.xlsx", index=False)
target_path = '/home/edward/桌面/標注數據/問題/test.txt'
process_image(target_path)