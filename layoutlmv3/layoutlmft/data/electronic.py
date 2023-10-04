import json
import os
from pathlib import Path
import datasets
from PIL import Image
import pandas as pd

def load_image(image_path):
    image = Image.open(image_path)
    w, h = image.size
    return image, (w, h)

def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]

class DatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for OurReceipt Dataset"""
    def __init__(self, **kwargs):
        """BuilderConfig for OurReceipt Dataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DatasetConfig, self).__init__(**kwargs)


class Electronic(datasets.GeneratorBasedBuilder):

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names = ['Item_value', 'price_value', 'other', 'total_value', 'quantity_value_of_unit', 'unit_price_key', 'quantity_key', 'sales_amount_key', 'seller_tax_id_key', 'seller_tax_id_value', 'total_char_key', 'sales_tax_key', 'quantity_key_of_unit', 'sales_tax_value', 'quantity_value', 'id_number_key', 'id_number_value', 'buyer_tax_id_key', 'buyer_tax_id_value', 'Item_key', 'sales_amount_value', 'date', 'total_key', 'price_key', 'total_char_value', 'unit_price_value']
                            )
                    ),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="",
        )


    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        """Uses local files located with data_dir"""
        dest = Path('/home/edward/桌面/標注數據/electronic_lmv3_200_v3')

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": dest/"train.txt", "dest": dest}
            ),            
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": dest/"test.txt", "dest": dest}
            ),
        ]

    def _generate_examples(self, filepath, dest):

        df = pd.read_csv(dest/'class_list.txt', delimiter='\s', header=None)
        id2labels = dict(zip(df[0].tolist(), df[1].tolist()))

        item_list = []
        with open(filepath, 'r') as f:
            for line in f:
                item_list.append(line.rstrip('\n\r'))
        
        for guid, fname in enumerate(item_list):
            print('*' *50 ,fname)
            data = json.loads(fname)
            image_path = dest/data['file_name']
            image, size = load_image(image_path)
            boxes = [[i['box'][0][0], i['box'][0][1], i['box'][2][0], i['box'][2][1]] for i in data['annotations']]

            text = [i['text'] for i in data['annotations']]
            label = [id2labels[i['label']] for i in data['annotations']]
            
            boxes = [normalize_bbox(box, size) for box in boxes]
            
            flag = 0

            for i in boxes:
              for j in i:
                if j > 1000:
                  flag += 1
                  pass
            if flag > 0: 
                print(image_path)
 
            yield guid, {"id": str(guid), "words": text, "bboxes": boxes, "ner_tags": label, "image_path": image_path}