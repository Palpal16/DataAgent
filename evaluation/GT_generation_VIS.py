import duckdb
import pandas as pd
import json
import os
import sys
from typing import Dict, List, Tuple

import readline

def _parse_chart_config(raw_text: str) -> Dict[str, str]:
    text = raw_text.strip().strip("`")
    # Attempt to extract JSON from possible code fences or prose
    try:
        # If there's a fenced block like ```json ... ``` remove it
        if text.lower().startswith("json"):  # e.g., "json\n{...}"
            text = text[4:].strip()
        if text.startswith("{") and text.endswith("}"):
            return json.loads(text)
        # Try to find first JSON object in text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    except Exception:
        pass
    # Fallback minimal schema
    return {
        "chart_type": "line",
        "x_axis": "date",
        "y_axis": "value",
        "title": "Chart",
    }

# Add workspace root to sys.path
workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

PREFIX = 'vis' #options: 'our', 'claude', 'gpt', 'gpt_columns', 'final'

TRANSACTION_DATA_FILE_PATH = 'data/Store_Sales_Price_Elasticity_Promotions_Data.parquet'
DATASET_FILE_PATH = f"evaluation/{PREFIX}_dataset.json"
table_name = "sales"

CHART_CONFIGURATION_PROMPT = (
    "Return a compact JSON object describing a chart configuration to visualize the data.\n"
    "Keys: chart_type (bar|line|area|scatter), x_axis (string), y_axis (string), title (string).\n"
    "Only return minified JSON, no markdown, no backticks.\n"
    "Data to consider (plain text table excerpt): {data}\n"
    "Visualization goal: {visualization_goal}"
)

if os.path.exists(DATASET_FILE_PATH):
    with open(DATASET_FILE_PATH, 'r') as f:
        dataset = json.load(f)
    print(f'\nLoaded existing dataset with {len(dataset)} entries.')
    i=0
    for entry in dataset:
        formatted_prompt = CHART_CONFIGURATION_PROMPT.format(visualization_goal=entry["prompt"], data=entry["gt_data"])
        print(f'Question {i+1}: {entry["prompt"]}')
        print('\nPrompt for LLM:')
        print(f'{"="*80}')
        print(formatted_prompt)
        print('-'*80)
        
        # Ask user to input the SQL query
        user_vis = input('\nPlease enter the VIS gt result for this question: ')

        # Data extraction part
        gt_vis = user_vis.strip()
        raw = gt_vis.content if hasattr(gt_vis, "content") else str(gt_vis)
        chart_config = _parse_chart_config(raw)
        chart_config["data"] = entry["gt_data"]
        dataset[i]['gt_vis'] = chart_config
        i=i+1

    # Save dataset as json file
    with open(f"evaluation/{PREFIX}_dataset.json", 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f'\n{"="*80}')
    print(f'Dataset completed and saved!')
    print(f'Total entries: {len(dataset)}')
    print(f'{"="*80}')