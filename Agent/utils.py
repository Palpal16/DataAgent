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
    try:
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)
    except Exception as e:
        print(f"Error while loading csvs for evaluation: {e}") 
        return 0. , 0. , 0.
    
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
        final_rows_iou = 0.0
    
    return columns_names_iou, final_rows_iou, data_iou

def judge_analysis(
    prompt: str,
    sql_query: str,
    data: str,
    analysis: str,
    judge_model: str = "gpt-oss:20b",
    ollama_url: str = "http://localhost:11434"
) -> Tuple[float, Dict]:
    """Evaluate data analysis quality using LLM-as-a-Judge.
    
    Args:
        prompt: Original user question
        sql_query: SQL query that was executed
        data: SQL results (ground truth)
        analysis: LLM's analysis text to evaluate
        judge_model: Ollama model name for judging (default: llama3.1:70b)
        ollama_url: Ollama server URL
    
    Returns:
        float: Overall score (0,1) = average of correctness, completeness, faithfulness and the detailed_evaluation of the judge
    """
    from langchain_ollama import ChatOllama
    
    JUDGE_PROMPT = """You are an expert evaluator assessing a data analysis response.
For the evaluation is important you consider the information that was available for the analysis, if the SQL result is wrong or has missing data, this problem shouldn't affect the analysis score.

### CONTEXT
USER QUESTION: {prompt}
SQL QUERY: {sql_query}
SQL RESULTS: 
{data}

### ANALYSIS TO EVALUATE
{analysis}

### EVALUATION RUBRIC (Rate 1-5 for each)

**CORRECTNESS (1-5)**
Does the analysis accurately interpret the SQL results? Are numerical values correct?
[1=Wrong, 3=Mostly correct, 5=Perfect]

**COMPLETENESS (1-5)**
Does it fully address all parts of the user's question using available data?
[1=Incomplete, 3=Main points covered, 5=Comprehensive]

**FAITHFULNESS (1-5)**
Does it only use information from SQL results? No hallucinated facts?
[1=Major hallucinations, 3=Minor issues, 5=Fully grounded]

### OUTPUT
Return ONLY valid JSON:
{{
  "correctness": {{"score": <1-5>, "reasoning": "<brief>", "issues": []}},
  "completeness": {{"score": <1-5>, "reasoning": "<brief>", "missing": []}},
  "faithfulness": {{"score": <1-5>, "reasoning": "<brief>", "hallucinations": []}}
}}"""

    try:
        # Create judge LLM
        judge_llm = ChatOllama(
            model=judge_model,
            temperature=0.2,
            base_url=ollama_url,
            max_tokens=1000
        )
        
        # Truncate data if too long
        truncated_data = data[:2000] if len(data) > 2000 else data
        
        # Get judgment
        formatted_prompt = JUDGE_PROMPT.format(
            prompt=prompt,
            sql_query=sql_query,
            data=truncated_data,
            analysis=analysis
        )
        
        response = judge_llm.invoke(formatted_prompt)
        raw_content = response.content if hasattr(response, "content") else str(response)
        
        # Parse JSON
        evaluation = _parse_judge_json(raw_content)
        
        # Compute overall score (average of 3 criteria)
        scores = [
            evaluation.get("correctness", {}).get("score", 0),
            evaluation.get("completeness", {}).get("score", 0),
            evaluation.get("faithfulness", {}).get("score", 0)
        ]
        score = sum(scores) / 3.0
        overall_score = (score - 1) / 4.0
        
        evaluation["overall_score"] = overall_score
        return overall_score, evaluation
            
    except Exception as e:
        print(f"Judge evaluation error: {e}")
        return (0.0, {"error": str(e)})


def _parse_judge_json(raw_text: str) -> Dict:
    """Parse judge JSON response with robust error handling."""
    try:
        # Clean markdown and find JSON
        content = raw_text.strip().replace("``````", "").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
        
        start = content.find("{")
        end = content.rfind("}")
        
        if start != -1 and end != -1:
            parsed = json.loads(content[start:end+1])
            
            # Ensure all criteria exist
            for criterion in ["correctness", "completeness", "faithfulness"]:
                if criterion not in parsed:
                    parsed[criterion] = {"score": 0, "reasoning": "Missing", "issues": []}
            
            return parsed
    except Exception as e:
        print(f"JSON parse error: {e}")
    
    # Fallback
    return {
        "correctness": {"score": 0, "reasoning": "Parse failed", "issues": []},
        "completeness": {"score": 0, "reasoning": "Parse failed", "missing": []},
        "faithfulness": {"score": 0, "reasoning": "Parse failed", "hallucinations": []}
    }


def best_of_n(
    agent, 
    prompt: str, 
    expected_csv: str = None, 
    csv_path: str = None, 
    n: int = 3, 
    temperature: float = 0.1,
    lookup_only: bool = False,
    judge_model: str = "gpt-oss:20b"
) -> Tuple[Dict, float]:
    """Run agent n times and return best result based on evaluation score(s).
    
    Args:
        agent: SalesDataAgent instance
        prompt: User question
        expected_csv: Path to ground truth CSV (for lookup evaluation)
        csv_path: Path to save generated CSV
        n: Number of attempts
        temperature: Single float or (min, max) tuple for temperature range
        lookup_only: If True, only run lookup+CSV eval; if False, run full agent with both evals
        judge_model: Model name for LLM-as-a-judge evaluation
    
    Returns:
        Tuple of (best_result_dict, score_variance)
    """
    # Temperature scheduling
    if isinstance(temperature, (list, tuple)) and len(temperature) == 2:
        temperatures = np.linspace(temperature[0], temperature[1], n)
    else:
        temperatures = [temperature] * n
    
    original_temp = agent.llm.temperature
    best_result = None
    max_score = -1
    all_scores = []
    
    for i, temp in enumerate(temperatures):
        print(f"\n--- Attempt {i+1}/{n} (temperature={temp:.2f}) ---")
        agent.llm.temperature = temp
        
        try:
            # Run agent
            if lookup_only:
                ret = agent.run(prompt, only_lookup=True)
            else:
                ret = agent.run(prompt, no_vis=True)
            
            # Evaluation 1: CSV/Data IoU (if lookup data available)
            csv_score = None
            if expected_csv and csv_path and ret.get('data'):
                try:
                    result_rows = text_to_csv(ret['data'])
                    save_csv(result_rows, csv_path)
                    columns_names_iou, rows_iou, data_iou = compare_csv(expected_csv, csv_path)
                    print(f"CSV Eval - Columns: {columns_names_iou:.2f} | Rows: {rows_iou:.2f} | Data: {data_iou:.2f}")
                    csv_score = data_iou
                except Exception as e:
                    print(f"CSV evaluation failed: {e}")
                    csv_score = 0.0
            
            # Evaluation 2: LLM-as-a-Judge (if analysis available)
            judge_score = None
            if not lookup_only and ret.get('answer'):
                try:
                    judge_score, details = judge_analysis(
                        prompt=ret.get("prompt", prompt),
                        sql_query=ret.get("sql_query", ""),
                        data=ret.get("data", ""),
                        analysis=ret.get("answer", [""])[0],
                        judge_model=judge_model,
                        ollama_url=agent.ollama_url
                    )
                    print(f"Judge Eval - Score: {judge_score:.2f}")
                except Exception as e:
                    print(f"Judge evaluation failed: {e}")
                    judge_score = 0.0
            
            # Compute final score
            if lookup_only:
                # Only CSV score matters
                final_score = csv_score if csv_score is not None else 0.0
            else:
                # Multiply both scores (both in 0-1 range)
                csv_score = csv_score if csv_score is not None else 0.0
                judge_score = judge_score if judge_score is not None else 0.0
                final_score = csv_score * judge_score
                print(f"Combined Score: {csv_score:.2f} × {judge_score:.2f} = {final_score:.2f}")
            
            all_scores.append(final_score)
            
            # Track best
            if final_score > max_score:
                max_score = final_score
                best_result = {
                    'result': ret,
                    'score': final_score,
                    'csv_score': csv_score,
                    'judge_score': judge_score,
                    'temperature': temp
                }
                
        except Exception as e:
            print(f"Error in attempt {i+1}: {e}")
            all_scores.append(0.0)
    
    # Restore original temperature
    agent.llm.temperature = original_temp
    
    # Calculate variance
    if not all_scores or max_score == -1:
        return {}, 0.0
    
    score_variance = max(all_scores) - min(all_scores)
    
    return best_result, score_variance