
from flask import Flask, request, jsonify
import os
import tempfile
from Agent.data_agent import SalesDataAgent
from Agent.utils import get_evaluation_functions, prepare_gt_from_dataset

app = Flask(__name__)

# Initialize a default agent (can be overridden per request)
agent = SalesDataAgent()

@app.route('/call-agent', methods=['POST'])
def call_agent():
    """Run `SalesDataAgent` via HTTP.

    This endpoint executes a single prompt and optionally evaluates results (CSV IoU and/or
    text metrics). It returns the full agent result plus a few convenience fields.

    Request JSON (minimal):
        {
          "prompt": "Show me sales in Nov 2021"
        }

    Common options:
        - "model": str (e.g. "llama3.2:3b", "openai:gpt-4o-mini", "anthropic:claude-3-5-sonnet-latest")
        - "visualization_goal": str
        - "lookup_only": bool  (only lookup)
        - "no_vis": bool       (lookup + analysis, no visualization)
        - "best_of_n": int
        - "temp": float
        - "temp_max": float
        - "save_dir": str
        - "enable_codecarbon": bool

    Evaluation options (optional):
        - "dataset": str (JSON dataset; auto-generates gt files into save_dir)
        - "gt_csv": str
        - "gt_text": str
        - "py_csv_eval": bool
        - "cpp_csv_eval": bool
        - "evaluator_exe": str
        - "eval_keys": str
        - "iou_type": "rows" | "columns" | "table"
        - "bleu_text_eval": bool
        - "spice_text_eval": bool
        - "llm_text_eval": bool
        - "bleu_nltk": bool
        - "spice_jar": str
        - "spice_java_bin": str

    Tracing options (optional):
        - "enable_tracing": bool
        - "phoenix_endpoint": str
        - "project_name": str

    Response (success, HTTP 200):
        {
          "status": "success",
          "result": { ...full agent output... },
          "save_dir": "...",
          "csv_score": 0.0-1.0,         # if enabled
          "text_score": 0.0-1.0,        # if enabled
          "score_variance": 0.0-1.0,    # if best_of_n>1
          "answer": [...],             # convenience alias for result.answer
          "data_preview": "...",
          "sql_query": "..."
        }

    Response (error):
        - HTTP 400 for invalid payload / evaluation configuration
        - HTTP 500 for runtime errors (LLM/tool failures)
    """
    payload = request.get_json(silent=True) or {}

    # Required parameter
    prompt = payload.get('prompt')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400

    # Extract parameters with support for various naming conventions
    visualization_goal = payload.get("visualization_goal") or payload.get("goal")
    data_path = payload.get("data_path") or payload.get("data")
    model = payload.get("model")

    # Agent behavior flags
    lookup_only = bool(payload.get("lookup_only") or payload.get("lookup-only") or payload.get("lookupOnly", False))
    no_vis = bool(payload.get("no_vis") or payload.get("no-vis") or payload.get("noVis", False))

    # Best-of-n parameters
    best_of_n = int(payload.get("best_of_n") or payload.get("best-of-n", 1))
    temp = payload.get("temp")
    if temp is not None:
        temp = float(temp)
    temp_max = payload.get("temp_max") or payload.get("temp-max")
    if temp_max is not None:
        temp_max = float(temp_max)

    # Save directory
    save_dir = payload.get("save_dir") or payload.get("save-dir")
    if not save_dir:
        save_dir = tempfile.mkdtemp(prefix="api_agent_runs_")

    # Ground truth paths
    dataset_path = payload.get("dataset") or payload.get("dataset_path")
    gt_csv = payload.get("gt_csv") or payload.get("gt-csv")
    gt_text_path = payload.get("gt_text") or payload.get("gt-text")

    # CSV evaluation options
    py_csv_eval = bool(payload.get("py_csv_eval") or payload.get("py-csv-eval", False))
    cpp_csv_eval = bool(payload.get("cpp_csv_eval") or payload.get("cpp-csv-eval", False))
    evaluator_exe = payload.get("evaluator_exe") or payload.get("evaluator-exe")
    eval_keys = payload.get("eval_keys") or payload.get("eval-keys")
    iou_type = payload.get("iou_type", "rows")

    # Text evaluation options
    bleu_text_eval = bool(payload.get("bleu_text_eval") or payload.get("bleu-text-eval", False))
    spice_text_eval = bool(payload.get("spice_text_eval") or payload.get("spice-text-eval", False))
    llm_text_eval = bool(payload.get("llm_text_eval") or payload.get("llm-text-eval", False))
    bleu_nltk = bool(payload.get("bleu_nltk") or payload.get("bleu-nltk", False))
    spice_jar = payload.get("spice_jar") or payload.get("spice-jar")
    spice_java_bin = payload.get("spice_java_bin") or payload.get("spice-java-bin", "java")

    # Phoenix tracing options
    enable_tracing = bool(payload.get("enable_tracing") or payload.get("enable-tracing", False))
    phoenix_endpoint = payload.get("phoenix_endpoint") or payload.get("phoenix-endpoint", "http://localhost:6006/v1/traces")
    project_name = payload.get("project_name") or payload.get("project-name", "evaluating-agent")

    # CodeCarbon option
    enable_codecarbon = bool(payload.get("enable_codecarbon") or payload.get("enable-codecarbon", False))

    # Auto-generate gt files from dataset if provided
    if dataset_path:
        try:
            prep = prepare_gt_from_dataset(
                prompt=prompt,
                dataset_path=dataset_path,
                output_dir=save_dir,
                gt_csv_filename="gt_data.csv",
                gt_results_filename="gt_results.json",
                gt_text_filename="gt_analysis.txt",
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if not gt_csv:
            gt_csv = prep["gt_csv_path"]
        if not gt_text_path and prep.get("gt_text_path"):
            gt_text_path = prep["gt_text_path"]

    # Create agent (use global or create new one with custom params)
    req_agent = agent
    if model or data_path or enable_tracing:
        try:
            req_agent = SalesDataAgent(
                model=model or "llama3.2:3b",
                temperature=temp or 0.1,
                data_path=data_path,
                enable_tracing=enable_tracing,
                phoenix_endpoint=phoenix_endpoint,
                project_name=project_name,
            )
        except Exception as e:
            return jsonify({"error": f"Failed to create agent: {str(e)}"}), 500

    # Get evaluation functions from utils
    try:
        csv_eval_fn, text_eval_fn = get_evaluation_functions(
            lookup_only=lookup_only,
            gt_csv_path=gt_csv,
            py_csv_eval=py_csv_eval,
            cpp_csv_eval=cpp_csv_eval,
            evaluator_exe=evaluator_exe,
            eval_keys=eval_keys,
            gt_text_path=gt_text_path,
            iou_type=iou_type,
            spice_text_eval=spice_text_eval,
            bleu_text_eval=bleu_text_eval,
            llm_text_eval=llm_text_eval,
            bleu_nltk=bleu_nltk,
            spice_jar=spice_jar,
            spice_java_bin=spice_java_bin,
        )
    except Exception as e:
        return jsonify({"error": f"Failed to create evaluation functions: {str(e)}"}), 400

    # Run agent
    try:
        result, score_variance = req_agent.run(
            prompt,
            visualization_goal=visualization_goal,
            lookup_only=lookup_only,
            no_vis=no_vis,
            best_of_n=best_of_n,
            temp=temp,
            temp_max=temp_max,
            csv_eval_fn=csv_eval_fn,
            text_eval_fn=text_eval_fn,
            save_dir=save_dir,
            enable_codecarbon=enable_codecarbon,
        )

        # Prepare response
        response = {
            "status": "success",
            "result": result,
            "save_dir": save_dir,
        }

        # Add score variance if best-of-n was used
        if best_of_n > 1:
            response["score_variance"] = score_variance

        # Add evaluation scores if available
        if "csv_score" in result:
            response["csv_score"] = result["csv_score"]
        if "text_score" in result:
            response["text_score"] = result["text_score"]

        # Extract and include key fields for convenience
        if "answer" in result:
            response["answer"] = result["answer"]
        if "data" in result:
            response["data_preview"] = result["data"][:500] + "..." if len(result["data"]) > 500 else result["data"]
        if "sql_query" in result:
            response["sql_query"] = result["sql_query"]
        if "error" in result:
            response["error"] = result["error"]

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "type": type(e).__name__
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint.

    Returns:
        200 JSON with a simple status and the default model name.
        503 JSON if the agent cannot be initialized.
    """
    try:
        # Check if agent can be initialized
        test_agent = SalesDataAgent()
        return jsonify({"status": "healthy", "model": test_agent.llm.model})
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 503

@app.route('/models', methods=['GET'])
def list_models():
    """List available Ollama models from the configured Ollama host.

    Reads `OLLAMA_HOST` (default: http://localhost:11434) and calls `/api/tags`.

    Returns:
        200 JSON: {"models": ["llama3.2:3b", ...]}
        500 JSON on errors.
    """
    try:
        import requests
        ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        response = requests.get(f"{ollama_url}/api/tags")
        models = response.json().get("models", [])
        return jsonify({"models": [m["name"] for m in models]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
