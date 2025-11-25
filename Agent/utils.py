import pandas as pd
import sys
import os
import json
import numpy as np
import csv
from typing import Dict, List, Tuple
from collections import Counter

def text_to_csv(text: str) -> List[List[str]]:
    """Convert text table to CSV rows.
    
    Handles both space-separated and pipe-separated formats.
    """
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    if not lines:
        return []
    
    rows = []
    for line in lines:
        # Try splitting by multiple spaces first
        if '  ' in line:
            parts = [p.strip() for p in line.split() if p.strip()]
        # Try pipe separator
        elif '|' in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
        # Fallback to comma
        else:
            parts = [p.strip() for p in line.split(',') if p.strip()]
        
        if parts:
            rows.append(parts)
    
    return rows

def save_csv(rows: List[List[str]], filepath: str):
    """Save rows to CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def compare_csv(csv1_path, csv2_path):
    """
    Calculate IoU using multisets for proper duplicate handling.
    Column-order independent row comparison.
    """
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    
    # 1. Column names IoU
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    columns_names_iou = len(cols1 & cols2) / len(cols1 | cols2) if cols1 | cols2 else 0.0
    
    # 2. Overall data IoU
    data_counter1 = Counter(df1.values.flatten())
    data_counter2 = Counter(df2.values.flatten())
    
    intersection = data_counter1 & data_counter2
    union = data_counter1 | data_counter2
    data_iou = sum(intersection.values()) / sum(union.values()) if union else 0.0

    # 3. Row IoU
    cols_intersection = list(cols1 & cols2)
    if cols_intersection:
        sorted_cols = sorted(cols_intersection)  # Sort for consistency
        
        rows1 = [tuple(row) for row in df1[sorted_cols].values]
        rows2 = [tuple(row) for row in df2[sorted_cols].values]
        
        rows_counter1 = Counter(rows1)
        rows_counter2 = Counter(rows2)
        
        intersection = rows_counter1 & rows_counter2
        union = rows_counter1 | rows_counter2
        rows_iou = sum(intersection.values()) / sum(union.values()) if union else 0.0
        final_rows_iou = columns_names_iou * rows_iou
    else:
        rows_iou = 0.0
    
    return columns_names_iou, final_rows_iou, data_iou

def best_of_n(agent, prompt: str, expected_csv: str = None, n: int=3, temperature=0.1) -> Dict:
    if isinstance(temperature, (list, tuple)) and len(temperature) == 2:
        temperatures = np.linspace(temperature[0], temperature[1], n)
    else:
        temperatures = [temperature] * n
    max_score=-1
    original_temp = agent.llm.temperature
    for i,temp in enumerate(temperatures):
        print(f"\n--- Attempt {i+1}/{n} (temperature={temp:.2f}) ---")
        agent.llm.temperature = temp
        try:
            ret = agent.run(prompt, only_lookup=True)
            csv_path = expected_csv.replace('_gt.csv', f'_gen_temp_{temp:.2f}.csv')
            result_rows = text_to_csv(ret['data'])
            save_csv(result_rows, csv_path)
            columns_names_iou, rows_iou, data_iou = compare_csv(expected_csv, csv_path)
            print(f"Columns Names IoU: {columns_names_iou:.2f} -- Rows IoU: {rows_iou:.2f} -- Data IoU: {data_iou:.2f}")
            score = (rows_iou + data_iou)/3
            print(f"Score: {score:.2f}")
            if score>max_score:
                max_score=score
                best_result = {
                    'result': ret,
                    'score': score,
                    'temperature': temp,
                    'attempt': i + 1
                }
        except Exception as e:
            print(f"Error in attempt {i+1}: {e}") 

    if max_score == -1:
        return []
    agent.llm.temperature = original_temp
    return best_result