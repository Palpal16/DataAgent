import os
import json
from Agent.utils import compare_csv

if __name__ == "__main__":
    PREFIX = 'claude' # options: 'my', 'claude', 'gpt'
    index = 1
    DATASET_FILE_PATH = f"evaluation/{PREFIX}_dataset.json"
    if os.path.exists(DATASET_FILE_PATH):
        with open(DATASET_FILE_PATH, 'r') as f:
            dataset = json.load(f)
    
    for i in range(len(dataset)):
        data = dataset[i]
        path_gt = data['gt_csv_path']
        path_gen = data['gen_csv_path']
        
        columns_names_iou, rows_iou, data_iou = compare_csv(path_gt, path_gen)
        print(f"Index: {i}\nColumns Names IoU: {columns_names_iou:.2f} -- Rows IoU: {rows_iou:.2f} -- Data IoU: {data_iou:.2f}")
        if 'annotated_accuracy' in data:
            print(f"Expected IoU: {data['annotated_accuracy']}")
        print('\n')