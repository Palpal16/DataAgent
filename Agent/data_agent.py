"""Sales Data Agent using LangGraph, DuckDB, and Ollama (LLaMA).

This module exposes a class `SalesDataAgent` that orchestrates:
- DuckDB SQL over a local parquet file
- LLM-driven tool routing (lookup → analyze → visualize)
- Chart configuration extraction and chart code generation

Usage example:
    from Agent.data_agent import SalesDataAgent

    agent = SalesDataAgent()
    result = agent.run("Show me the sales in Nov 2021")
    print(result["answer"])  # Ordered list of steps/outputs (analysis text, then code)
"""

from __future__ import annotations

import requests
import json
import os
import difflib
from functools import partial
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd
from typing_extensions import NotRequired, TypedDict
# Support both `python -m Agent.data_agent ...` and running this file directly.
try:
    from .utils import text_to_csv, save_csv, compare_csv  # type: ignore
except Exception:  # pragma: no cover
    from Agent.utils import text_to_csv, save_csv, compare_csv  # type: ignore

# Optional energy/emissions tracking via CodeCarbon
try:
    from codecarbon import EmissionsTracker  # type: ignore
    print("CodeCarbon is available")
    _CODECARBON_AVAILABLE = True
except Exception:
    print("CodeCarbon is not available not using it")
    EmissionsTracker = None  # type: ignore
    _CODECARBON_AVAILABLE = False

from langgraph.graph import END, StateGraph
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

import math
import re

# Optional tracing/instrumentation (Phoenix / OpenInference)
try:
    from phoenix.otel import register as phoenix_register
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry.trace import StatusCode
    _PHOENIX_AVAILABLE = True
except Exception:  # pragma: no cover - tracing is optional
    StatusCode = None  # type: ignore
    _PHOENIX_AVAILABLE = False
    #print exception
    print(Exception)


# Mirror utils_0.py printing of langgraph version
import langgraph
import langgraph.version
print(langgraph.version)


# -----------------------------
# Constants / Defaults
# -----------------------------

DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "Store_Sales_Price_Elasticity_Promotions_Data.parquet"
)

def _build_llm(
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    streaming: bool,
    ollama_url: str,
) :
    """Create an LLM client from a model string.

    Supported:
    - Ollama (default): model like "llama3.2:3b"
    - OpenAI: model like "openai:gpt-4o-mini" or "gpt-4o-mini"
    - Anthropic: model like "anthropic:claude-3-5-sonnet-latest" or "claude-3-5-sonnet-latest"
    """
    m = (model or "").strip()
    m_lower = m.lower()

    # Prefix-based selection
    if m_lower.startswith("openai:"):
        openai_model = m.split(":", 1)[1].strip()
        return ChatOpenAI(model=openai_model, temperature=temperature, max_tokens=max_tokens, streaming=streaming)

    if m_lower.startswith("anthropic:") or m_lower.startswith("claude:"):
        anthropic_model = m.split(":", 1)[1].strip()
        return ChatAnthropic(model=anthropic_model, temperature=temperature, max_tokens=max_tokens, streaming=streaming)

    # Heuristic selection if no prefix provided
    if m_lower.startswith("gpt-") or m_lower.startswith("o1-") or m_lower.startswith("o3-") or m_lower.startswith("chatgpt"):
        return ChatOpenAI(model=m, temperature=temperature, max_tokens=max_tokens, streaming=streaming)

    if m_lower.startswith("claude-"):
        return ChatAnthropic(model=m, temperature=temperature, max_tokens=max_tokens, streaming=streaming)

    # Default to Ollama
    return ChatOllama(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        base_url=ollama_url,
    )

# -----------------------------
# State Definition
# -----------------------------

class State(TypedDict):
    prompt: str
    data: Optional[str]
    analyze_data: Optional[str]
    answer: List[str]
    visualization_goal: Optional[str]
    chart_config: Optional[dict]
    tool_choice: NotRequired[str]
    error: NotRequired[str]
    sql_query: Optional[str]


# -----------------------------
# LLM Helpers
# -----------------------------

SQL_GENERATION_PROMPT = """Generate an SQL query based on the prompt.
Please just reply with the SQL query and NO MORE, just the query.
The prompt is : {prompt}. The available columns are: {columns}. The table name is: {table_name}.
If you need to use a DATE column with LIKE or pattern matching, first CAST it to VARCHAR like this: CAST(date_column AS VARCHAR) LIKE '%2021-11%'.
Return only the SQL query, with no explanations or markdown formatting.
"""



def generate_sql_query(state: State, columns: List[str], table_name: str, llm) -> str:
    """Generate a parameterized SQL query with the LLM based on the user prompt.

    Args:
        state: Conversation state containing the user prompt.
        columns: Available column names in the table.
        table_name: Name of the temporary DuckDB table to query.
        llm: ChatOllama instance used to generate the SQL.

    Returns:
        A plain SQL string suitable for DuckDB. Any markdown fences are stripped.
    """
    formatted_prompt = SQL_GENERATION_PROMPT.format(
        prompt=state["prompt"], columns=columns, table_name=table_name
    )
    response = llm.invoke(formatted_prompt)
    sql_query = response.content if hasattr(response, "content") else str(response)
    cleaned_sql = (
        sql_query.strip()
        .replace("```sql", "")
        .replace("```", "")
    )
    print("Generated SQL Query:\n", cleaned_sql)
    return cleaned_sql

def _lookup_sales_data_once(state: State, llm, tracer=None, csv_output_path: Optional[str] = None) -> Dict:
    """Look up sales data from a parquet file using LLM-generated SQL over DuckDB.

    This function registers the parquet data as a temporary DuckDB table, asks the
    LLM to generate an SQL query from the user's prompt and available columns, then
    executes the query and stores a text-formatted table in state['data'].

    Args:
        state: Conversation state; must include 'prompt'.
        data_path: Filesystem path to the parquet dataset. // ADD LATER
        llm: ChatOllama instance used for prompt-to-SQL generation.

    Returns:
        Updated state containing 'data' (string table) or 'error'.
    """
    table_name = "sales"
    df = pd.read_parquet(DEFAULT_DATA_PATH)
    duckdb.sql("DROP TABLE IF EXISTS sales")
    duckdb.register("df", df)
    duckdb.sql(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    sql_query = generate_sql_query(state, df.columns.tolist(), table_name, llm)
    try:
        result_df = duckdb.sql(sql_query).df()
        result_str = result_df.to_string()

        # Save to CSV if output path is provided (prefer DataFrame->CSV for fidelity)
        if csv_output_path:
            os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
            result_df.to_csv(csv_output_path, index=False)
        if tracer is not None:
            try:
                with tracer.start_as_current_span("sql_query_exec", openinference_span_kind="tool") as span:  # type: ignore[attr-defined]
                    span.set_input(state.get("prompt", ""))  # type: ignore[attr-defined]
                    span.set_output(result_str)  # type: ignore[attr-defined]
                    if StatusCode is not None:
                        span.set_status(StatusCode.OK)  # type: ignore[attr-defined]
            except Exception:
                pass
        out = {**state, "data": result_str, "sql_query": sql_query}
        if csv_output_path:
            out["output_csv"] = csv_output_path
        return out
    except Exception as e: # If the SQL fails, return empty results
        print(f"Error accessing data: {str(e)}")
        return {**state, "data": "", "sql_query": sql_query, "error": f"Error accessing data: {str(e)}"}

def _score_lookup_result(result_df: Optional[pd.DataFrame], error: Optional[str]) -> Tuple[int, int, int]:
    """Heuristic score for best-of-n selection.

    Returns a tuple so Python's tuple comparison can pick the "best":
    - ok_flag: 1 if no error else 0
    - row_count: number of rows in result (higher is better)
    - col_count: number of columns in result (higher is better)
    """
    ok_flag = 0 if error else 1
    if result_df is None:
        return (ok_flag, 0, 0)
    try:
        rows, cols = int(result_df.shape[0]), int(result_df.shape[1])
    except Exception:
        rows, cols = 0, 0
    return (ok_flag, rows, cols)

def lookup_sales_data(state: State, llm, tracer=None, csv_output_path: Optional[str] = None) -> Dict:
    """Look up sales data (optionally using best-of-n for SQL generation)."""
    # Allow passing the CSV output path via either:
    # - function arg (csv_output_path=...)
    # - state["output_csv"] (set by CLI/run)
    if not csv_output_path:
        try:
            csv_output_path = state.get("output_csv")  # type: ignore[assignment]
        except Exception:
            csv_output_path = None

    # Optional best-of-n self-consistency
    best_of_n = 1
    try:
        best_of_n = int(state.get("best_of_n", 1))  # type: ignore[arg-type]
    except Exception:
        best_of_n = 1
    best_of_n = max(1, best_of_n)

    # Temperature schedule for best-of-n
    try:
        t_min = float(state.get("best_of_n_temp_min", getattr(llm, "temperature", 0.1) or 0.1))  # type: ignore[arg-type]
    except Exception:
        t_min = getattr(llm, "temperature", 0.1) or 0.1
    try:
        t_max = float(state.get("best_of_n_temp_max", t_min))  # type: ignore[arg-type]
    except Exception:
        t_max = t_min

    if best_of_n <= 1:
        return _lookup_sales_data_once(state, llm, tracer, csv_output_path=csv_output_path)

    original_temp = getattr(llm, "temperature", None)
    temps: List[float]
    if best_of_n == 1:
        temps = [t_min]
    elif abs(t_max - t_min) < 1e-9:
        temps = [t_min] * best_of_n
    else:
        # linear schedule
        step = (t_max - t_min) / float(best_of_n - 1)
        temps = [t_min + i * step for i in range(best_of_n)]

    best_score: Tuple[int, int, int] = (-1, -1, -1)
    best_state: Optional[Dict] = None

    for i, temp in enumerate(temps):
        try:
            llm.temperature = float(temp)  # type: ignore[attr-defined]
        except Exception:
            pass

        attempt_state = _lookup_sales_data_once(state, llm, tracer, csv_output_path=None)
        err = attempt_state.get("error")
        sql = attempt_state.get("sql_query")

        # Re-run the SQL to get a DF for scoring (cheap, local DuckDB)
        result_df: Optional[pd.DataFrame] = None
        try:
            if sql:
                result_df = duckdb.sql(sql).df()
        except Exception:
            result_df = None

        score = _score_lookup_result(result_df, err)
        if score > best_score:
            best_score = score
            best_state = attempt_state

        print(f"[best-of-n] attempt {i+1}/{best_of_n} temp={temp:.3f} score={score} error={bool(err)}")

    # Restore temperature
    try:
        if original_temp is not None:
            llm.temperature = original_temp  # type: ignore[attr-defined]
    except Exception:
        pass

    final_state = best_state or _lookup_sales_data_once(state, llm, tracer, csv_output_path=None)

    # Save only the best attempt (if requested). Prefer best SQL -> DataFrame -> CSV.
    if csv_output_path:
        saved = False
        try:
            sql_best = final_state.get("sql_query")
            if sql_best:
                df_best = duckdb.sql(sql_best).df()
                os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
                df_best.to_csv(csv_output_path, index=False)
                saved = True
        except Exception:
            saved = False

        if not saved:
            # Fallback to the older string->rows approach
            try:
                result_rows = text_to_csv(final_state.get("data", "") or "")
                save_csv(result_rows, csv_output_path)
                saved = True
            except Exception as e:
                final_state = {**final_state, "error": f"CSV save failed: {str(e)}"}

        if saved:
            final_state = {**final_state, "output_csv": csv_output_path}

    final_state = {**final_state, "best_of_n": best_of_n, "best_of_n_temp_min": t_min, "best_of_n_temp_max": t_max}
    return final_state

def decide_tool(state: State, llm, tracer=None) -> State:
    """Select the next tool to run given the current conversation state.

    The LLM is prompted with the available tools and minimal state. The raw
    response is normalized against a fixed list of valid tool names.

    Tool selection constraints:
    - If no data is present, force 'lookup_sales_data' before analysis/visualization.
    - If more than one answer message is present, end the flow ('end').

    Args:
        state: Conversation state.
        llm: ChatOllama instance used to decide the tool.

    Returns:
        Updated state including 'tool_choice'.
    """
    tools_description = """You have access to the following tools to help you with your task:

    - lookup_sales_data: Look up sales data from a parquet file using SQL.
    - analyzing_data: Analyze the sales data for trends and insights.
    - create_visualization: Create visualizations based on the sales data.
    - end: End the conversation if the task is complete.

    Based on the actual state and the user prompt, decide which tool to use next.
    """

    decision_prompt = f"""
    {tools_description}
    Current state:
    - Prompt: {state.get('prompt')}
    - Answer so far: {state.get('answer', [])}
    - Visualization goal: {state.get('visualization_goal')}
    - Tool used last: {state.get('tool_choice')}
    Decide the next tool among: lookup_sales_data, analyzing_data, create_visualization.
    Respond with only the tool name, or 'end'.
    Keep in mind:
    - Do NOT reuse a tool that was already used earlier in the conversation.
    - If analysis and visualization are both completed, respond with "end".
    - If all relevant tools for the prompt have been used, respond with "end".
    """

    try:
        current_prompt = state.get("prompt", "")
        current_answer = state.get("answer", [])
        visualization_goal = state.get("visualization_goal")
        chart_config = state.get("chart_config")
        analyzed_data = state.get("analyze_data")

        response = llm.invoke(decision_prompt)
        tool_choice = response.content.strip().lower()
        valid_tools = ["lookup_sales_data", "analyzing_data", "create_visualization", "end"]
        closest_match = difflib.get_close_matches(tool_choice, valid_tools, n=1, cutoff=0.6)
        matched_tool = closest_match[0] if closest_match else "lookup_sales_data"

        if matched_tool in ["analyzing_data", "create_visualization"] and not state.get("data"):
            matched_tool = "lookup_sales_data"
        elif len(state.get("answer", [])) > 1:
            matched_tool = "end"

        # Tracing span for tool choice (optional)
        if tracer is not None:
            try:
                with tracer.start_as_current_span("tool_choice", openinference_span_kind="tool") as span:  # type: ignore[attr-defined]
                    # Minimal, robust attributes to avoid dtype issues
                    span.set_attributes({  # type: ignore[attr-defined]
                        "prompt": str(current_prompt),
                        "tool_choice": str(matched_tool),
                    })
                    span.set_input(str(current_prompt))  # type: ignore[attr-defined]
                    span.set_output(str(matched_tool))  # type: ignore[attr-defined]
                    if StatusCode is not None:
                        span.set_status(StatusCode.OK)  # type: ignore[attr-defined]
            except Exception:
                pass

        print(f"Tool selected: {matched_tool}")

        return {
            **state,
            "prompt": current_prompt,
            "answer": current_answer,
            "analyze_data": analyzed_data,
            "visualization_goal": visualization_goal,
            "chart_config": chart_config,
            "tool_choice": matched_tool,
        }
    except Exception as e:
        print(f"Error deciding tool: {str(e)}")
        return {**state, "error": f"Error accessing data: {str(e)}"}


DATA_ANALYSIS_PROMPT = (
    "Analyze the following data: {data}\n"
    "Your job is to answer the following question: {prompt}"
)


def analyze_sales_data(state: State, llm, tracer=None) -> Dict:
    """Run only the analysis step (expects state['data'] to be present)."""
    try:
        formatted_prompt = DATA_ANALYSIS_PROMPT.format(
            data=state.get("data", ""), prompt=state.get("prompt", "")
        )
        analysis_result = llm.invoke(formatted_prompt)
        analysis_text = analysis_result.content if hasattr(analysis_result, "content") else str(analysis_result)
        # Optional tracing
        if tracer is not None:
            try:
                with tracer.start_as_current_span("data_analysis", openinference_span_kind="tool") as span:  # type: ignore[attr-defined]
                    span.set_input(state.get("prompt", ""))  # type: ignore[attr-defined]
                    span.set_output(str(analysis_text))  # type: ignore[attr-defined]
                    if StatusCode is not None:
                        span.set_status(StatusCode.OK)  # type: ignore[attr-defined]
            except Exception:
                pass
        return {
            **state,
            "analyze_data": analysis_text,
            "answer": state.get("answer", []) + [analysis_text],
        }
    except Exception as e:
        return {**state, "error": f"Error analyzing data: {str(e)}"}


def _tokenize_for_bleu(text: str) -> List[str]:
    # Simple, dependency-free tokenization (words + numbers).
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", (text or "").lower())


def bleu_score(reference: str, hypothesis: str, *, max_n: int = 4, smooth: bool = True) -> float:
    """Compute a simple BLEU score (0..1) with optional add-one smoothing.

    This is intended for quick evaluation of analysis text; it's not a full SacreBLEU replacement.
    """
    ref_tokens = _tokenize_for_bleu(reference)
    hyp_tokens = _tokenize_for_bleu(hypothesis)
    if not hyp_tokens:
        return 0.0
    if not ref_tokens:
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
        # Count ngrams
        hyp_counts: Dict[Tuple[str, ...], int] = {}
        ref_counts: Dict[Tuple[str, ...], int] = {}
        for g in hyp_ngrams:
            hyp_counts[g] = hyp_counts.get(g, 0) + 1
        for g in ref_ngrams:
            ref_counts[g] = ref_counts.get(g, 0) + 1
        # Clipped matches
        match = 0
        total = 0
        for g, c in hyp_counts.items():
            total += c
            match += min(c, ref_counts.get(g, 0))
        if smooth:
            precisions.append((match + 1.0) / (total + 1.0))
        else:
            precisions.append(match / total if total else 0.0)

    # Brevity penalty
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)
    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - (ref_len / hyp_len))

    # Geometric mean of precisions
    if any(p <= 0.0 for p in precisions):
        return 0.0
    log_mean = sum(math.log(p) for p in precisions) / float(max_n)
    return float(bp * math.exp(log_mean))

CHART_CONFIGURATION_PROMPT = (
    "Return a compact JSON object describing a chart configuration to visualize the data.\n"
    "Keys: chart_type (bar|line|area|scatter), x_axis (string), y_axis (string), title (string).\n"
    "Only return minified JSON, no markdown, no backticks.\n"
    "Data to consider (plain text table excerpt): {data}\n"
    "Visualization goal: {visualization_goal}"
)


def _parse_chart_config(raw_text: str) -> Dict[str, str]:
    """Parse a chart configuration JSON from a raw LLM response.

    The function attempts to tolerate code fences and extra prose, extracting the
    first JSON object it can find. On failure, a minimal default schema is
    returned.

    Args:
        raw_text: Raw text from the LLM expected to contain a JSON object.

    Returns:
        A dictionary with keys: 'chart_type', 'x_axis', 'y_axis', 'title'.
    """
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


def extract_chart_config(state: State, llm) -> State:
    """Infer a compact chart configuration from the looked-up data.

    Prompts the LLM to return a minified JSON config and parses it into a
    Python dict. The original text-formatted data is attached as config['data'].

    Args:
        state: Conversation state; should include 'data' and optionally 'visualization_goal'.
        llm: ChatOllama instance used to infer the chart configuration.

    Returns:
        Updated state including 'chart_config' or None if no data.
    """
    data_text = state.get("data") or ""
    if not data_text:
        return {**state, "chart_config": None}

    visualization_goal = state.get("visualization_goal") or state.get("prompt", "Chart")
    formatted_prompt = CHART_CONFIGURATION_PROMPT.format(
        data=data_text, visualization_goal=visualization_goal
    )
    response = llm.invoke(formatted_prompt)
    raw = response.content if hasattr(response, "content") else str(response)
    chart_config = _parse_chart_config(raw)
    chart_config["data"] = data_text
    print("Este es el chart config: "+str(chart_config))
    return {**state, "chart_config": chart_config}


CREATE_CHART_PROMPT = (
    "Write Python code to create a chart using matplotlib given the config.\n"
    "Only return code, no markdown fences or commentary. The code must:\n"
    "- import matplotlib.pyplot as plt\n"
    "- build a simple chart of type config['chart_type'] using axes config['x_axis'], config['y_axis'] if possible\n"
    "- set the title to config['title']\n"
    "- call plt.tight_layout() and plt.show() at the end\n"
    "config: {config}"
)


def create_chart(state: State, llm) -> str:
    """Ask the LLM to emit matplotlib code for the given chart configuration.

    Args:
        state: Conversation state; must include 'chart_config'.
        llm: ChatOllama instance used to generate the plotting code.

    Returns:
        A Python code string (without markdown fences) that, when executed,
        renders the chart using matplotlib.
    """
    formatted_prompt = CREATE_CHART_PROMPT.format(config=state.get("chart_config", {}))
    response = llm.invoke(formatted_prompt)
    code = response.content if hasattr(response, "content") else str(response)
    # clean any accidental fences
    return code.replace("```python", "").replace("```", "").strip()


def create_visualization(state: State, llm) -> State:
    """Create a visualization by first extracting config and then generating code.

    Args:
        state: Conversation state; should include 'data'.
        llm: ChatOllama instance used for config extraction and code generation.

    Returns:
        Updated state with 'chart_config' and the generated code appended to 'answer'.
    """
    try:
        with_config = extract_chart_config(state, llm)
        code = create_chart(with_config, llm)
        return {
            **with_config,
            "answer": with_config.get("answer", []) + [code],
        }
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return {**state, "error": f"Error accessing data: {str(e)}"}


def route_to_tool(state: State) -> str:
    """Return the next node key for the graph based on 'tool_choice' in state.

    Args:
        state: Conversation state that may include 'tool_choice'.

    Returns:
        One of: 'lookup_sales_data' | 'analyzing_data' | 'create_visualization' | 'end'.
    """
    tool_choice = state.get("tool_choice", "lookup_sales_data")
    valid_tools = ["lookup_sales_data", "analyzing_data", "create_visualization", "end"]
    return tool_choice if tool_choice in valid_tools else "end"


# -----------------------------
# Public Agent Class
# -----------------------------

class SalesDataAgent:
    """End-to-end agent to query, analyze, and visualize sales data.

    The agent builds a LangGraph with tool-selection, data lookup (DuckDB over
    parquet), LLM-based analysis, and visualization code generation. Use `run()`
    to execute a single prompt through the flow.
    """
    def __init__(
        self,
        *,
        model: str = "llama3.2:3b",
        temperature: float = 0.1,
        max_tokens: int = 2000,
        streaming: bool = True,
        data_path: Optional[str] = None,
        ollama_url: Optional[str] = None,
        enable_tracing: bool = False,
        phoenix_api_key: Optional[str] = None,
        phoenix_endpoint: Optional[str] = None,
        project_name: str = "evaluating-agent",
    ) -> None:
        """Initialize the agent and compile the graph.

        Args:
            model: Ollama model name.
            temperature: Sampling temperature for the LLM.
            max_tokens: Generation token limit.
            streaming: Whether to stream tokens from the LLM.
            data_path: Optional override for the parquet dataset path.
            ollama_url: Optional override for Ollama base URL; defaults to OLLAMA_HOST or http://localhost:11434.
        """
        self.ollama_url = ollama_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        # LLM provider selection:
        # - default: Ollama (local)
        # - OpenAI: pass model like "openai:gpt-4o-mini" (requires OPENAI_API_KEY)
        # - Anthropic: pass model like "anthropic:claude-3-5-sonnet-latest" (requires ANTHROPIC_API_KEY)
        self.llm = _build_llm(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            ollama_url=self.ollama_url,
        )
        self.data_path = data_path or DEFAULT_DATA_PATH

        # Optional Phoenix/OpenInference tracing integration
        self.tracer = None
        self.tracing_enabled = False
        if enable_tracing and _PHOENIX_AVAILABLE:
            try:
                # Environment variables similar to utils_0.py
                if phoenix_api_key:
                    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={phoenix_api_key}"
                    os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={phoenix_api_key}"
                if phoenix_endpoint:
                    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_endpoint

                tracer_provider = phoenix_register(
                    project_name=project_name,
                    endpoint=(phoenix_endpoint or "https://app.phoenix.arize.com/v1/traces"),
                )
                LangChainInstrumentor(tracer_provider=tracer_provider).instrument(skip_dep_check=True)
                self.tracer = tracer_provider.get_tracer(__name__)
                self.tracing_enabled = True
            except Exception as _:
                self.tracer = None
                self.tracing_enabled = False

        self.graph = self._build_graph()
        self.run_checked = False

    def check_ollama(self):
        try:
            self.llm.invoke("Hello, how are you?")
            print("Ollama is running locally")
            return True
        except Exception as e:
            print(e)
            return False

    def check_model(self):
        """Check if the model is running locally"""
        try:
            base = self.ollama_url.rstrip("/")
            requests.get(f"{base}/api/version", timeout=3).json()
            print("Server is running locally")
            return self.check_ollama()
        except Exception as e:
            print(e)
            return False


    def _build_graph(self):
        """Construct and compile the LangGraph for the agent run loop."""
        graph = StateGraph(State)

        # Capture the LLM in closures so nodes accept only (state)
        llm = self.llm
        tracer = self.tracer

        def analyzing_data_node(state: State) -> Dict:
            """Ask the LLM to analyze the looked-up data in the context of the prompt.

            Args:
                state: Conversation state; should include 'data' and 'prompt'.
                llm: ChatOllama instance used for the analysis.

            Returns:
                Updated state including 'analyze_data' and the analysis appended to 'answer'.
            """
            try:
                print("Data to analyze:\n", state.get("data", ""))
                formatted_prompt = DATA_ANALYSIS_PROMPT.format(
                    data=state.get("data", ""), prompt=state.get("prompt", "")
                )
                analysis_result = llm.invoke(formatted_prompt)
                analysis_text = analysis_result.content if hasattr(analysis_result, "content") else str(analysis_result)
                if tracer is not None:
                    try:
                        with tracer.start_as_current_span("data_analysis", openinference_span_kind="tool") as span:  # type: ignore[attr-defined]
                            span.set_input(state.get("prompt", ""))  # type: ignore[attr-defined]
                            span.set_output(str(analysis_text))  # type: ignore[attr-defined]
                            if StatusCode is not None:
                                span.set_status(StatusCode.OK)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                return {
                    **state,
                    "analyze_data": analysis_text,
                    "answer": state.get("answer", []) + [analysis_text],
                }
            except Exception as e:
                print(f"Error analyzing data: {str(e)}")
                return {**state, "error": f"Error accessing data: {str(e)}"}

        def create_visualization_node(state: State) -> Dict:
            """Create a visualization by first extracting config and then generating code.

            Args:
                state: Conversation state; should include 'data'.
                llm: ChatOllama instance used for config extraction and code generation.

            Returns:
                Updated state with 'chart_config' and the generated code appended to 'answer'.
            """
            try:
                with_config = extract_chart_config(state, llm)
                code = create_chart(with_config, llm)
                if tracer is not None:
                    try:
                        with tracer.start_as_current_span("gen_visualization", openinference_span_kind="tool") as span:  # type: ignore[attr-defined]
                            span.set_input(str(state.get("prompt", "")))  # type: ignore[attr-defined]
                            span.set_output(str(code))  # type: ignore[attr-defined]
                            if StatusCode is not None:
                                span.set_status(StatusCode.OK)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                return {
                    **with_config,
                    "answer": with_config.get("answer", []) + [code],
                }
            except Exception as e:
                print(f"Error creating visualization: {str(e)}")
                return {**state, "error": f"Error accessing data: {str(e)}"}

        graph.add_node("decide_tool", partial(decide_tool, llm=llm, tracer=tracer))
        graph.add_node("lookup_sales_data", partial(lookup_sales_data, llm=llm, tracer=tracer))
        graph.add_node("analyzing_data", analyzing_data_node)
        graph.add_node("create_visualization", create_visualization_node)
        graph.set_entry_point("decide_tool")

        graph.add_conditional_edges(
            "decide_tool",
            route_to_tool,
            {
                "lookup_sales_data": "lookup_sales_data",
                "analyzing_data": "analyzing_data",
                "create_visualization": "create_visualization",
                "end": END,
            },
        )

        graph.add_edge("lookup_sales_data", "decide_tool")
        graph.add_edge("analyzing_data", "decide_tool")
        graph.add_edge("create_visualization", "decide_tool")
        
        return graph.compile()

    def _run_core(
        self,
        state: Dict,
        visualization_goal: Optional[str],
        initial_state: Optional[Dict],
        only_lookup: bool,
        output_csv: Optional[str] = None,
        best_of_n: int = 1,
        best_of_n_temp_min: Optional[float] = None,
        best_of_n_temp_max: Optional[float] = None,
        analyze_only: bool = False,
    ) -> Dict:
        """Core execution logic for a single agent run.

        Kept separate from `run` so that wrappers like CodeCarbon or tracing
        can be applied cleanly around this method.
        """
        if visualization_goal:
            state["visualization_goal"] = visualization_goal
        if initial_state:
            state.update(initial_state)
        if output_csv:
            state["output_csv"] = output_csv
        # Best-of-n config (consumed by lookup_sales_data)
        try:
            state["best_of_n"] = int(best_of_n)
        except Exception:
            state["best_of_n"] = 1
        if best_of_n_temp_min is not None:
            state["best_of_n_temp_min"] = float(best_of_n_temp_min)
        if best_of_n_temp_max is not None:
            state["best_of_n_temp_max"] = float(best_of_n_temp_max)

        # Shortcut: run lookup then analysis only (no tool-routing / no visualization)
        if analyze_only:
            looked_up = lookup_sales_data(state, self.llm, self.tracer, csv_output_path=output_csv)
            return analyze_sales_data(looked_up, self.llm, self.tracer)

        # Shortcut: run only the lookup tool
        if only_lookup:
            print("[Agent] Running only lookup_sales_data")
            try:
                if self.tracing_enabled and self.tracer is not None:
                    with self.tracer.start_as_current_span("AgentRun_LookupOnly", openinference_span_kind="agent") as span:  # type: ignore[attr-defined]
                        span.set_input(state)  # type: ignore[attr-defined]
                        result = lookup_sales_data(state, self.llm, self.tracer, csv_output_path=output_csv)
                        span.set_output(result)  # type: ignore[attr-defined]
                        if StatusCode is not None:
                            span.set_status(StatusCode.OK)  # type: ignore[attr-defined]
                        return result
                else:
                    result = lookup_sales_data(state, self.llm, csv_output_path=output_csv)
                    return result
            except Exception as _e:
                return {**state, "error": f"Lookup failed: {str(_e)}"}

        print("Running the graph...")
        if self.tracing_enabled and self.tracer is not None:
            try:
                with self.tracer.start_as_current_span("AgentRun", openinference_span_kind="agent") as span:  # type: ignore[attr-defined]
                    print("[LangGraph] Starting LangGraph execution with tracing")
                    span.set_input(state)  # type: ignore[attr-defined]
                    result = self.graph.invoke(state)
                    span.set_output(result)  # type: ignore[attr-defined]
                    if StatusCode is not None:
                        span.set_status(StatusCode.OK)  # type: ignore[attr-defined]
                    print("[LangGraph] LangGraph execution completed")
                    return result
            except Exception:
                # Fallback to non-traced execution on any tracing error
                result = self.graph.invoke(state)
                return result
        else:
            print("[LangGraph] Starting LangGraph execution")
            result = self.graph.invoke(state)
            print("[LangGraph] LangGraph execution completed")
            return result

    def run(
        self,
        prompt: str,
        *,
        visualization_goal: Optional[str] = None,
        initial_state: Optional[Dict] = None,
        only_lookup: bool = False,
        output_csv: Optional[str] = None,
        best_of_n: int = 1,
        best_of_n_temp_min: Optional[float] = None,
        best_of_n_temp_max: Optional[float] = None,
        analyze_only: bool = False,
    ) -> Dict:
        """Execute the agent for a single prompt.

        Args:
            prompt: Natural-language request or question.
            visualization_goal: Optional explicit goal for charts; defaults to the prompt.
            initial_state: Optional seed state to merge in before execution.

        Returns:
            The final state dictionary produced by the compiled graph execution.
        """
        state = {
            "prompt": prompt,
        }
        if not self.run_checked:
            print("Checking the model can run locally")
            self.run_checked = self.check_model()
        
        if not self.run_checked:
            print("Model is not running locally, remember to run ollama serve")
            return {**state, "error": "Model is not running locally, remember to run ollama serve"}
        else:
            # Wrap the execution with CodeCarbon tracker if available
            if _CODECARBON_AVAILABLE:
                try:
                    with EmissionsTracker(  # type: ignore[call-arg]
                        project_name="SalesDataAgent",
                        output_dir="codecarbon",
                        save_to_file=True,
                        measure_power_secs=1,
                        log_level="error",
                    ):
                        return self._run_core(
                            state,
                            visualization_goal,
                            initial_state,
                            only_lookup,
                            output_csv=output_csv,
                            best_of_n=best_of_n,
                            best_of_n_temp_min=best_of_n_temp_min,
                            best_of_n_temp_max=best_of_n_temp_max,
                            analyze_only=analyze_only,
                        )
                except Exception:
                    # If tracker initialization fails, run without it
                    return self._run_core(
                        state,
                        visualization_goal,
                        initial_state,
                        only_lookup,
                        output_csv=output_csv,
                        best_of_n=best_of_n,
                        best_of_n_temp_min=best_of_n_temp_min,
                        best_of_n_temp_max=best_of_n_temp_max,
                        analyze_only=analyze_only,
                    )
            else:
                return self._run_core(
                    state,
                    visualization_goal,
                    initial_state,
                    only_lookup,
                    output_csv=output_csv,
                    best_of_n=best_of_n,
                    best_of_n_temp_min=best_of_n_temp_min,
                    best_of_n_temp_max=best_of_n_temp_max,
                    analyze_only=analyze_only,
                )

    def draw_graph(self) -> str:
        """Return an ASCII rendering of the compiled graph if available."""
        try:
            from IPython.display import Image, display
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except Exception:
            # Fallback if mermaid is not available
            print(self.graph.get_graph().print_ascii())


__all__ = ["SalesDataAgent", "State"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Sales Data Agent")
    parser.add_argument("prompt", type=str, help="User prompt/question")
    parser.add_argument("--data", dest="data_path", type=str, default=DEFAULT_DATA_PATH, help="Path to parquet file")
    parser.add_argument("--goal", dest="visualization_goal", type=str, default=None, help="Optional viz goal")
    parser.add_argument("--model", dest="model", type=str, default="llama3.2:3b", help="Ollama model name (e.g., llama3.2:3b)")
    # Only-lookup flag
    parser.add_argument("--lookup-only", dest="lookup_only", action="store_true", help="Run only the lookup_sales_data tool")
    parser.add_argument('--output-csv', dest='output_csv', type=str, help='Path to save the output CSV file')
    parser.add_argument("--best-of-n", dest="best_of_n", type=int, default=1, help="Run lookup N times (self-consistency) and pick the best SQL result")
    parser.add_argument("--best-of-n-temp-min", dest="best_of_n_temp_min", type=float, default=None, help="Min temperature for best-of-n schedule")
    parser.add_argument("--best-of-n-temp-max", dest="best_of_n_temp_max", type=float, default=None, help="Max temperature for best-of-n schedule")
    parser.add_argument("--analyze-only", dest="analyze_only", action="store_true", help="Run lookup then analysis (no visualization)")
    parser.add_argument("--expected-analysis", dest="expected_analysis", type=str, default=None, help="Ground-truth analysis text to compute BLEU against")
    parser.add_argument("--expected-analysis-file", dest="expected_analysis_file", type=str, default=None, help="Path to a text file containing ground-truth analysis (BLEU)")
    parser.add_argument("--expected-csv", dest="expected_csv", type=str, default=None, help="Path to ground-truth/expected CSV to compare against")
    parser.add_argument("--evaluator-exe", dest="evaluator_exe", type=str, default=None, help="Path to C++ comparator executable (optional)")
    parser.add_argument("--eval-keys", dest="eval_keys", type=str, default=None, help="Comma-separated key columns for C++ comparator (optional)")
    args = parser.parse_args()

    agent = SalesDataAgent(model=args.model, data_path=args.data_path)
    output = agent.run(
        args.prompt,
        visualization_goal=args.visualization_goal,
        only_lookup=args.lookup_only,
        output_csv=args.output_csv,
        best_of_n=args.best_of_n,
        best_of_n_temp_min=args.best_of_n_temp_min,
        best_of_n_temp_max=args.best_of_n_temp_max,
        analyze_only=args.analyze_only,
    )

    # Optional: analysis BLEU evaluation (generated analysis vs expected analysis)
    expected_analysis_text = args.expected_analysis
    if not expected_analysis_text and args.expected_analysis_file:
        try:
            with open(args.expected_analysis_file, "r", encoding="utf-8") as f:
                expected_analysis_text = f.read()
        except Exception as e:
            output["analysis_evaluation"] = {"error": f"Failed to read expected analysis file: {str(e)}"}
            expected_analysis_text = None

    if expected_analysis_text is not None:
        try:
            hyp = output.get("analyze_data") or ""
            output["analysis_evaluation"] = {
                "metric": "bleu",
                "bleu": bleu_score(expected_analysis_text, hyp, max_n=4, smooth=True),
            }
        except Exception as e:
            output["analysis_evaluation"] = {"error": f"BLEU evaluation failed: {str(e)}"}

    # Optional: compare actual vs expected CSV (IoU by default, C++ comparator if --evaluator-exe is provided)
    if args.expected_csv:
        actual_csv = args.output_csv or output.get("output_csv")
        try:
            if args.evaluator_exe:
                from evaluator import run_cpp_comparator  # type: ignore

                keys = [k.strip() for k in (args.eval_keys or "").split(",") if k.strip()] or None
                output["evaluation"] = run_cpp_comparator(
                    evaluator_exe=args.evaluator_exe,
                    actual_csv=str(actual_csv),
                    expected_csv=str(args.expected_csv),
                    keys=keys,
                )
            else:
                c_iou, r_iou, d_iou = compare_csv(str(args.expected_csv), str(actual_csv))
                output["evaluation"] = {
                    "mode": "iou",
                    "columns_iou": c_iou,
                    "rows_iou": r_iou,
                    "data_iou": d_iou,
                }
        except Exception as e:
            output["evaluation"] = {"equal": False, "error": f"Evaluation failed: {str(e)}"}

    # Minimal printout
    print(json.dumps({k: v for k, v in output.items() if k != "data"}, indent=2))


