import pandas as pd
import sys
import os
import json
import numpy as np
from collections import Counter

def compare_csv(csv1_path, csv2_path):
    """
    Calculate IoU using multisets for proper duplicate handling.
    Column-order independent row comparison.
    """
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    
    # 1. Column names IoU (unchanged)
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    columns_names_iou = len(cols1 & cols2) / len(cols1 | cols2) if cols1 | cols2 else 0.0
    
    # 2. Column values IoU - FIXED with multisets
    cols_intersection = list(cols1 & cols2)
    if cols_intersection:
        cols_counter1 = Counter()
        cols_counter2 = Counter()
        
        for col in cols_intersection:
            cols_counter1.update((col, val) for val in df1[col])
            cols_counter2.update((col, val) for val in df2[col])
        
        intersection = cols_counter1 & cols_counter2
        union = cols_counter1 | cols_counter2
        columns_iou = sum(intersection.values()) / sum(union.values()) if union else 0.0
    else:
        columns_iou = 0.0
    
    # 3. Row IoU - FIXED: column-order independent + multisets
    if cols_intersection:
        sorted_cols = sorted(cols_intersection)  # Sort for consistency
        
        rows1 = [tuple(row) for row in df1[sorted_cols].values]
        rows2 = [tuple(row) for row in df2[sorted_cols].values]
        
        rows_counter1 = Counter(rows1)
        rows_counter2 = Counter(rows2)
        
        intersection = rows_counter1 & rows_counter2
        union = rows_counter1 | rows_counter2
        rows_iou = sum(intersection.values()) / sum(union.values()) if union else 0.0
    else:
        rows_iou = 0.0
    
    # 4. Overall data IoU - FIXED with multisets
    data_counter1 = Counter(df1.values.flatten())
    data_counter2 = Counter(df2.values.flatten())
    
    intersection = data_counter1 & data_counter2
    union = data_counter1 | data_counter2
    data_iou = sum(intersection.values()) / sum(union.values()) if union else 0.0
    
    final_rows_iou = columns_names_iou * rows_iou
    final_columns_iou = columns_names_iou * columns_iou
    
    return columns_names_iou, final_rows_iou, final_columns_iou, data_iou


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
        
        columns_names_iou, rows_iou, columns_iou, data_iou = compare_csv(path_gt, path_gen)
        print(f"Index: {i}\nColumns Names IoU: {columns_names_iou:.2f} -- Rows IoU: {rows_iou:.2f} -- Columns IoU: {columns_iou:.2f}")
        print(f"Data IoU: {data_iou:.2f} -- Expected IoU: {data['annotated_accuracy']}")
        print('\n')