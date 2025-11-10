import pandas as pd
import sys
import os
import json
import numpy as np

def compare_csv(csv1_path, csv2_path):
    """Calculate IoU for columns, rows, and overall between two CSV files."""
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    
    # Column IoU
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    cols_intersection = len(cols1.intersection(cols2))
    cols_union = len(cols1.union(cols2))
    columns_iou = cols_intersection / cols_union if cols_union > 0 else 0.0
    
    # Row IoU
    cols_intersection = list(cols1.intersection(cols2))
    if cols_intersection:
        df1_subset = df1[cols_intersection].apply(lambda x: tuple(x), axis=1)
        df2_subset = df2[cols_intersection].apply(lambda x: tuple(x), axis=1)
        rows1 = set(df1_subset)
        rows2 = set(df2_subset)
        rows_intersection = len(rows1.intersection(rows2))
        rows_union = len(rows1.union(rows2))
        rows_iou = rows_intersection / rows_union if rows_union > 0 else 0.0
    else:
        rows_iou = 0.0
    
    # Overall IoU
    iou = columns_iou*rows_iou
    
    return rows_iou, columns_iou, iou


if __name__ == "__main__":
    PREFIX = 'gpt' # options: 'my', 'claude', 'gpt'
    index = 1
    DATASET_FILE_PATH = f"evaluation/{PREFIX}_dataset.json"
    if os.path.exists(DATASET_FILE_PATH):
        with open(DATASET_FILE_PATH, 'r') as f:
            dataset = json.load(f)
    
    for i in range(len(dataset)):
        data = dataset[i]
        path_gt = data['gt_csv_path']
        path_gen = data['gen_csv_path']
        
        rows_iou, columns_iou, iou = compare_csv(path_gt, path_gen)
        print(f"Index: {i} -- Rows IoU: {rows_iou:.2f} -- Columns IoU: {columns_iou:.2f}")
        print(f"Computed IoU: {iou:.2f} -- Expected IoU: {data['annotated_accuracy']}")
        print('\n')
