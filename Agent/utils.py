import pandas as pd
import sys
import os
import json
import numpy as np
import csv
from typing import Dict, List, Tuple, Optional
from collections import Counter
import math
import re
import subprocess
import tempfile
from functools import partial

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

def get_evaluation_functions(
    *,
    lookup_only: bool = False,
    # CSV evaluation options
    gt_csv_path: Optional[str] = None,
    py_csv_eval: bool = False,
    cpp_csv_eval: bool = False,
    evaluator_exe: Optional[str] = None,
    eval_keys: Optional[str] = None,
    # Text evaluation options
    gt_text_path: Optional[str] = None,
    bleu_text_eval: bool = False,
    bleu_impl: str = "simple",
    spice_text_eval: bool = False,
    spice_jar: Optional[str] = None,
    spice_java_bin: str = "java",
    llm_text_eval: bool = False,
) -> tuple[Optional[callable], Optional[callable]]:
    """Get evaluation functions based on command-line arguments.
    
    Args:
        lookup_only: If True, only CSV evaluation is relevant (no text analysis)
        py_csv_eval: Use Python CSV evaluator
        cpp_csv_eval: Use C++ CSV evaluator
        evaluator_exe: Path to C++ evaluator executable
        eval_keys: Comma-separated key columns for comparison
        spice_text_eval: Use SPICE for text evaluation
        bleu_text_eval: Use BLEU for text evaluation
        llm_text_eval: Use LLM for text evaluation
        bleu_impl: BLEU implementation ("simple" or "nltk")
        spice_jar: Path to SPICE jar file
        spice_java_bin: Java executable for SPICE
    
    Returns:
        Tuple of (csv_eval_fn, text_eval_fn), either can be None
    """
    csv_eval_fn = None
    text_eval_fn = None
    
    # CSV Evaluation
    if gt_csv_path:
        if py_csv_eval:
            csv_eval_fn = partial(compare_csv,csv2_path=gt_csv_path)
        elif cpp_csv_eval:
            if evaluator_exe is None:
                print("Cannot use --cpp_csv_eval because --evaluator-exe is not available") #TODO: make into warning
            
            keys = [k.strip() for k in (eval_keys or "").split(",") if k.strip()] or None
            csv_eval_fn = partial(run_cpp_comparator, evaluator_exe=evaluator_exe, expected_csv=gt_csv_path, keys=keys)

    # Load ground truth if provided
    if gt_text_path:
        try:
            with open(gt_text_path, 'r', encoding='utf-8') as f:
                gt_text = f.read()
        except Exception as e:
            print(f"Failed to read expected analysis file: {str(e)}")
            gt_text = None

    if gt_text and not lookup_only:
        if spice_text_eval:
            try:
                check_spice_jar_runnable(spice_jar=spice_jar, java_bin=spice_java_bin)
            except Exception as e:
                print(json.dumps({"error": f"SPICE precheck failed: {str(e)}"}, indent=2)) #TODO make into warning
            
            text_eval_fn = partial(spice_score_java, reference=gt_text, spice_jar=spice_jar, java_bin=spice_java_bin)
            
        elif bleu_text_eval:
            if bleu_impl == "nltk":
                text_eval_fn = partial(bleu_score_nltk,reference=gt_text, max_n=4, smooth=True)
            else:  # simple
                text_eval_fn = partial(bleu_score,reference=gt_text, max_n=4, smooth=True)
                
        elif llm_text_eval:
            def text_eval_llm(generated_text: str, expected_text: str) -> float:
                """LLM-as-judge text evaluator."""
                # TODO: Implement LLM-as-judge evaluation
                print(f"[Text Eval] LLM-as-judge: comparing texts")
                return 0.0  # Placeholder
            text_eval_fn = text_eval_llm
    
    return csv_eval_fn, text_eval_fn

def run_cpp_comparator(
    *,
    evaluator_exe: str,
    actual_csv: str,
    expected_csv: str,
    keys: Optional[List[str]] = None,
    case_insensitive: bool = False,
    stream_debug: bool = False,
) -> Dict:
    args = [evaluator_exe, "--actual", actual_csv, "--expected", expected_csv]
    if keys:
        args += ["--key", ",".join(keys)]
    if case_insensitive:
        args += ["--case-insensitive"]

    # If stream_debug is True, inherit stderr so C++ debug (sent to stderr) prints to terminal.
    # Keep stdout captured to parse JSON report.
    if stream_debug:
        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=None, text=True)
    else:
        proc = subprocess.run(args, capture_output=True, text=True)
    stdout = proc.stdout.strip()
    try:
        report = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError:
        report = {"equal": False, "error": "Invalid JSON from comparator", "raw": stdout}
    report["exit_code"] = proc.returncode
    if proc.returncode not in (0, 1):
        # Non-comparison error, include stderr
        report.setdefault("error", proc.stderr.strip())
    return report

def _tokenize_for_bleu(text: str) -> List[str]:
    """Simple, dependency-free tokenization (words + numbers) for BLEU."""
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", (text or "").lower())

def bleu_score(reference: str, hypothesis: str, *, max_n: int = 4, smooth: bool = True) -> float:
    """Compute a simple BLEU score (0..1) with optional add-one smoothing.

    Intended for quick evaluation of generated analysis text; not a full SacreBLEU replacement.
    """
    ref_tokens = _tokenize_for_bleu(reference)
    hyp_tokens = _tokenize_for_bleu(hypothesis)
    if not hyp_tokens or not ref_tokens:
        return 0.0

    def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        return [tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)]

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        hyp_ngrams = ngrams(hyp_tokens, n)
        ref_ngrams = ngrams(ref_tokens, n)
        if not hyp_ngrams:
            precisions.append(0.0)
            continue
        hyp_counts: Dict[Tuple[str, ...], int] = {}
        ref_counts: Dict[Tuple[str, ...], int] = {}
        for g in hyp_ngrams:
            hyp_counts[g] = hyp_counts.get(g, 0) + 1
        for g in ref_ngrams:
            ref_counts[g] = ref_counts.get(g, 0) + 1

        match = 0
        total = 0
        for g, c in hyp_counts.items():
            total += c
            match += min(c, ref_counts.get(g, 0))
        precisions.append((match + 1.0) / (total + 1.0) if smooth else (match / total if total else 0.0))

    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - (ref_len / max(hyp_len, 1)))

    if any(p <= 0.0 for p in precisions):
        return 0.0
    log_mean = sum(math.log(p) for p in precisions) / float(max_n)
    return float(bp * math.exp(log_mean))

def bleu_score_nltk(reference: str, hypothesis: str, *, max_n: int = 4, smooth: bool = True) -> float:
    """Compute BLEU (0..1) using NLTK's `sentence_bleu`.

    Requires:
        `pip install nltk`
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu  # type: ignore
        from nltk.translate.bleu_score import SmoothingFunction  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("NLTK is not installed. Install it with `pip install nltk`.") from e

    ref_tokens = _tokenize_for_bleu(reference)
    hyp_tokens = _tokenize_for_bleu(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0

    n = int(max(1, min(4, max_n)))
    if n == 1:
        weights = (1.0, 0.0, 0.0, 0.0)
    elif n == 2:
        weights = (0.5, 0.5, 0.0, 0.0)
    elif n == 3:
        weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0)
    else:
        weights = (0.25, 0.25, 0.25, 0.25)

    smoothing = SmoothingFunction().method1 if smooth else None
    score = sentence_bleu([ref_tokens], hyp_tokens, weights=weights, smoothing_function=smoothing)
    # NLTK returns a float in [0,1]
    return float(score)

def check_spice_jar_runnable(
    *,
    spice_jar: str,
    java_bin: str = "java",
    timeout_seconds: int = 10,
) -> None:
    """Fail-fast validation that the SPICE jar path exists and Java can execute it.

    This prevents spending time running the agent only to later fail with
    "Unable to access jarfile ...".
    """
    if not spice_jar:
        raise ValueError("spice_jar is required")

    jar_abs = os.path.abspath(spice_jar)
    if not os.path.exists(jar_abs):
        raise FileNotFoundError(f"SPICE jar not found: {jar_abs}")

    # If this is the common SPICE-1.0 bundle, ensure Stanford CoreNLP jars are present in lib/.
    jar_dir = os.path.dirname(jar_abs)
    lib_dir = os.path.join(jar_dir, "lib")
    if os.path.isdir(lib_dir):
        has_corenlp_code = any(
            fn.startswith("stanford-corenlp-") and fn.endswith(".jar") and "models" not in fn
            for fn in os.listdir(lib_dir)
        )
        has_corenlp_models = any(
            fn.startswith("stanford-corenlp-") and fn.endswith(".jar") and "models" in fn
            for fn in os.listdir(lib_dir)
        )
        if not (has_corenlp_code and has_corenlp_models):
            raise RuntimeError(
                "SPICE requires Stanford CoreNLP jars in the SPICE lib/ folder. "
                f"Missing in: {lib_dir}. "
                "The SPICE-1.0 bundle includes a script `get_stanford_models.sh` (Linux/macOS); "
                "on Windows, download CoreNLP 3.6.0 jars and place them into lib/ "
                "(both the code jar and the models jar)."
            )

        # On Windows, SPICE uses LMDB JNI. The bundle provides a win64 JNI jar; if Java is 32-bit,
        # it will fail at runtime with UnsatisfiedLinkError (lmdbjni32...).
        has_lmdb_win64 = any(fn.startswith("lmdbjni-win64-") and fn.endswith(".jar") for fn in os.listdir(lib_dir))
        if os.name == "nt" and has_lmdb_win64:
            try:
                ver = subprocess.run([java_bin, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_seconds)
                ver_text = (ver.stderr or "") + "\n" + (ver.stdout or "")
                if "64-Bit" not in ver_text and "64-bit" not in ver_text:
                    raise RuntimeError(
                        "Your Java appears to be 32-bit, but SPICE-1.0 on Windows requires 64-bit Java "
                        "(lmdbjni-win64). Install a 64-bit JDK/JRE and ensure it is on PATH."
                    )
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Java not found ('{java_bin}'). Install Java and ensure it's on PATH, or pass --spice-java-bin."
                ) from e

    cmd = [java_bin, "-jar", jar_abs]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
            cwd=os.path.dirname(jar_abs) or None,
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Java not found ('{java_bin}'). Install Java and ensure it's on PATH, or pass --spice-java-bin."
        ) from e
    except subprocess.TimeoutExpired:
        # If it runs longer than timeout, we assume the jar starts (good enough for this check).
        return

    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    combined = (stderr + "\n" + stdout).strip().lower()

    if "unable to access jarfile" in combined:
        raise RuntimeError(f"Java cannot access the jar: {jar_abs}")
    if "no main manifest attribute" in combined:
        raise RuntimeError(f"Jar is not runnable (no main manifest attribute): {jar_abs}")

    # Otherwise: even if return code is non-zero, many jars print usage/help and exit -> OK.

def spice_score_java(
    reference: str,
    hypothesis: str,
    *,
    spice_jar: str,
    java_bin: str = "java",
    timeout_seconds: int = 120,
) -> float:
    """Compute SPICE score (0..1) by calling the official Java SPICE jar.

    This uses the common COCO-caption SPICE JSON format:
      [{"image_id": 0, "test": "<candidate>", "refs": ["<ref1>", "<ref2>", ...]}]

    Args:
        reference: Ground-truth/reference text.
        hypothesis: Generated text to evaluate.
        spice_jar: Path to SPICE jar (e.g., spice-1.0.jar).
        java_bin: Java executable to use.
        timeout_seconds: Kill the Java process if it exceeds this time.

    Returns:
        SPICE F-score in [0,1].
    """
    if not spice_jar:
        raise ValueError("spice_jar is required")
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise TypeError("reference and hypothesis must be strings")
    if not reference.strip() or not hypothesis.strip():
        return 0.0

    # Use absolute paths to avoid cwd-related issues inside the Java tool
    spice_jar_abs = os.path.abspath(spice_jar)
    jar_dir = os.path.dirname(spice_jar_abs)

    payload = [
        {
            "image_id": 0,
            "test": hypothesis,
            "refs": [reference],
        }
    ]

    with tempfile.TemporaryDirectory() as td:
        in_json = os.path.abspath(os.path.join(td, "spice_in.json"))
        out_json = os.path.abspath(os.path.join(td, "spice_out.json"))
        with open(in_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        # Simple command: java -Xmx8G -jar spice-*.jar input.json
        cmd = [
            java_bin,
            "-Xmx8G",  # Add memory limit like your working command
            "-jar",
            spice_jar_abs,
            in_json,
            "-out",
            out_json, 
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
                cwd=jar_dir,  # Run from jar directory
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"SPICE timed out after {timeout_seconds}s") from e
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr or "").strip()
            raise RuntimeError(f"SPICE failed: {stderr}") from e

        if not os.path.exists(out_json):
            raise RuntimeError("SPICE did not produce an output file")

        with open(out_json, "r", encoding="utf-8") as f:
            out = json.load(f)

        # Expected (COCO-caption): list with one element; element has `scores` -> `All` -> `f`
        try:
            item = out[0] if isinstance(out, list) else out
            scores = item.get("scores") or {}
            all_scores = scores.get("All") or scores.get("all") or {}
            f1 = all_scores.get("f") or all_scores.get("f1")
            return float(f1) if f1 is not None else 0.0
        except Exception:
            return 0.0