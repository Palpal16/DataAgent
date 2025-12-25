from flask import Flask, request, jsonify
from Agent.data_agent import SalesDataAgent
from Agent.utils import bleu_score, bleu_score_nltk, spice_score_java, check_spice_jar_runnable

app = Flask(__name__)
agent = SalesDataAgent()

@app.route('/call-agent', methods=['POST'])
def call_agent():
    payload = request.get_json(silent=True) or {}
    prompt = payload.get('prompt')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    # Optional arguments (mirror CLI)
    visualization_goal = payload.get("goal") or payload.get("visualization_goal")
    data_path = payload.get("data_path") or payload.get("data")
    model = payload.get("model")
    lookup_only = bool(payload.get("lookup_only") or payload.get("lookup-only") or payload.get("lookupOnly") or False)
    output_csv = payload.get("output_csv") or payload.get("output-csv")

    best_of_n = payload.get("best_of_n") or payload.get("best-of-n") or 1
    best_of_n_temp_min = payload.get("best_of_n_temp_min") or payload.get("best-of-n-temp-min")
    best_of_n_temp_max = payload.get("best_of_n_temp_max") or payload.get("best-of-n-temp-max")

    expected_csv = payload.get("expected_csv") or payload.get("expected-csv")
    evaluator_exe = payload.get("evaluator_exe") or payload.get("evaluator-exe")
    eval_keys = payload.get("eval_keys") or payload.get("eval-keys")
    analyze_only = bool(payload.get("analyze_only") or payload.get("analyze-only") or payload.get("analyzeOnly") or False)
    expected_analysis = payload.get("expected_analysis") or payload.get("expected-analysis")
    bleu_impl = (payload.get("bleu_impl") or payload.get("bleu-impl") or "simple")
    analysis_metric = (payload.get("analysis_metric") or payload.get("analysis-metric") or "bleu")
    spice_jar = payload.get("spice_jar") or payload.get("spice-jar")
    spice_java_bin = payload.get("spice_java_bin") or payload.get("spice-java-bin") or "java"
    # Fail fast for SPICE (avoid running LLM/DuckDB if jar is missing/unrunnable)
    if str(analysis_metric).lower() == "spice":
        try:
            check_spice_jar_runnable(spice_jar=str(spice_jar or ""), java_bin=str(spice_java_bin))
        except Exception as e:
            return jsonify({"error": f"SPICE precheck failed: {str(e)}"}), 400

    # Allow per-request agent overrides (optional)
    req_agent = agent
    if model or data_path:
        req_agent = SalesDataAgent(
            model=model or "llama3.2:3b",
            data_path=data_path,
        )

    # Run agent
    result = req_agent.run(
        prompt=prompt,
        visualization_goal=visualization_goal,
        only_lookup=lookup_only,
        analyze_only=analyze_only,
        output_csv=output_csv,
        best_of_n=int(best_of_n) if best_of_n is not None else 1,
        best_of_n_temp_min=float(best_of_n_temp_min) if best_of_n_temp_min is not None else None,
        best_of_n_temp_max=float(best_of_n_temp_max) if best_of_n_temp_max is not None else None,
    )

    # Optional analysis evaluation (generated analysis vs expected analysis)
    if expected_analysis is not None:
        try:
            hyp = result.get("analyze_data") or ""
            if str(analysis_metric).lower() == "spice":
                if not spice_jar:
                    raise ValueError("SPICE selected but spice_jar was not provided")
                spice_val = spice_score_java(
                    str(expected_analysis),
                    str(hyp),
                    spice_jar=str(spice_jar),
                    java_bin=str(spice_java_bin),
                )
                result["analysis_evaluation"] = {"metric": "spice", "impl": "java", "spice": spice_val}
            else:
                if str(bleu_impl).lower() == "nltk":
                    bleu_val = bleu_score_nltk(str(expected_analysis), str(hyp), max_n=4, smooth=True)
                    impl = "nltk"
                else:
                    bleu_val = bleu_score(str(expected_analysis), str(hyp), max_n=4, smooth=True)
                    impl = "simple"
                result["analysis_evaluation"] = {"metric": "bleu", "impl": impl, "bleu": bleu_val}
        except Exception as e:
            result["analysis_evaluation"] = {"error": f"Analysis evaluation failed: {str(e)}"}

    # Optional comparison (if requested)
    if expected_csv:
        # Reuse the same comparison logic as CLI: IoU by default, C++ comparator if exe provided.
        try:
            if evaluator_exe:
                from evaluator import run_cpp_comparator
                keys = [k.strip() for k in str(eval_keys or "").split(",") if k.strip()] or None
                result["evaluation"] = run_cpp_comparator(
                    evaluator_exe=str(evaluator_exe),
                    actual_csv=str(output_csv or result.get("output_csv")),
                    expected_csv=str(expected_csv),
                    keys=keys,
                )
            else:
                from Agent.utils import compare_csv
                c_iou, r_iou, d_iou = compare_csv(str(expected_csv), str(output_csv or result.get("output_csv")))
                result["evaluation"] = {
                    "mode": "iou",
                    "columns_iou": c_iou,
                    "rows_iou": r_iou,
                    "data_iou": d_iou,
                }
        except Exception as e:
            result["evaluation"] = {"equal": False, "error": f"Evaluation failed: {str(e)}"}
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

