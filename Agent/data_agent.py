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
from typing import Dict, List, Optional
import tempfile
import numpy as np
import argparse

import duckdb
import pandas as pd
from typing_extensions import NotRequired, TypedDict

from langgraph.graph import END, StateGraph
from langchain_ollama import ChatOllama

try:
    from Agent.utils import *
except ImportError:
    from utils import *

# Optional energy/emissions tracking via CodeCarbon
try:
    from codecarbon import EmissionsTracker  # type: ignore
    print("CodeCarbon is available")
    _CODECARBON_AVAILABLE = True
except Exception:
    print("CodeCarbon is not available, not using it")
    EmissionsTracker = None  # type: ignore
    _CODECARBON_AVAILABLE = False

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

# -----------------------------
# State Definition
# -----------------------------

class State(TypedDict):
    prompt: str
    data: Optional[str]
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



def generate_sql_query(state: State, columns: List[str], table_name: str, llm: ChatOllama) -> str:
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

def lookup_sales_data(state: State, llm: ChatOllama, tracer=None) -> Dict:
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
        if tracer is not None:
            try:
                with tracer.start_as_current_span("sql_query_exec", openinference_span_kind="tool") as span:  # type: ignore[attr-defined]
                    span.set_input(state.get("prompt", ""))  # type: ignore[attr-defined]
                    span.set_output(result_str)  # type: ignore[attr-defined]
                    if StatusCode is not None:
                        span.set_status(StatusCode.OK)  # type: ignore[attr-defined]
            except Exception:
                pass
        return {**state, "data": result_str, "sql_query": sql_query}
    except Exception as e: # If the SQL fails, return empty results
        print(f"Error accessing data: {str(e)}")
        return {**state, "data": "", "sql_query": sql_query, "error": f"Error accessing data: {str(e)}"}

DATA_ANALYSIS_PROMPT = """ Your goal is to give a clear answer to this question: {prompt}.
Use the information available and return only a direct answer to the question.
The only data available is the data extracted from another agent using this SQL code: {sql_query}
The output of the SQL you can use to answer is this data: {data}
"""

def analyzing_data(state: State, llm: ChatOllama, tracer=None) -> Dict:
    """Ask the LLM to analyze the looked-up data in the context of the prompt.

    Args:
        state: Conversation state; should include 'data' and 'prompt'.
        llm: ChatOllama instance used for the analysis.

    Returns:
        Updated state including the analysis appended to 'answer'.
    """
    try:
        #print("Data to analyze:\n", state.get("data", ""))
        formatted_prompt = DATA_ANALYSIS_PROMPT.format(
            data=state.get("data", ""), prompt=state.get("prompt", ""), sql_query=state.get("sql_query","")
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
            "answer": state.get("answer", []) + [analysis_text],
        }
    except Exception as e:
        print(f"Error analyzing data: {str(e)}")
        return {**state, "error": f"Error accessing data: {str(e)}"}

def decide_tool(state: State, llm: ChatOllama, tracer=None) -> State:
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
            "visualization_goal": visualization_goal,
            "chart_config": chart_config,
            "tool_choice": matched_tool,
        }
    except Exception as e:
        print(f"Error deciding tool: {str(e)}")
        return {**state, "error": f"Error accessing data: {str(e)}"}
    

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


def extract_chart_config(state: State, llm: ChatOllama) -> State:
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
    print("This is the cart_config: "+str(chart_config))
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


def create_chart(state: State, llm: ChatOllama) -> str:
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

    
def create_visualization(state: State, llm: ChatOllama, tracer=None) -> State:
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
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            base_url=self.ollama_url,
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

        graph.add_node("decide_tool", partial(decide_tool, llm=llm, tracer=tracer))
        graph.add_node("lookup_sales_data", partial(lookup_sales_data, llm=llm, tracer=tracer))
        graph.add_node("analyzing_data", partial(analyzing_data, llm=llm, tracer=tracer))
        graph.add_node("create_visualization", partial(create_visualization, llm=llm, tracer=tracer))
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
    
    def draw_graph(self) -> str:
        """Return an ASCII rendering of the compiled graph if available."""
        try:
            from IPython.display import Image, display
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except Exception:
            # Fallback if mermaid is not available
            print(self.graph.get_graph().print_ascii())

    def run_core(
        self,
        prompt: str,
        *,
        visualization_goal: Optional[str] = None,
        lookup_only: bool = False,
        no_vis: bool = False
    ) -> Dict:
        """Execute the agent for a single prompt.

        Args:
            prompt: Natural-language request or question.
            visualization_goal: Optional explicit goal for charts; defaults to the prompt.

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
    
        if lookup_only:
            print("[Agent] Running only lookup_sales_data")
            try:
                if self.tracing_enabled and self.tracer is not None:
                    with self.tracer.start_as_current_span("AgentRun_LookupOnly", openinference_span_kind="agent") as span:  # type: ignore[attr-defined]
                        span.set_input(state)  # type: ignore[attr-defined]
                        result = lookup_sales_data(state, self.llm, self.tracer)
                        span.set_output(result)  # type: ignore[attr-defined]
                        if StatusCode is not None:
                            span.set_status(StatusCode.OK)  # type: ignore[attr-defined]
                        return result
                else:
                    result = lookup_sales_data(state, self.llm)
                    return result
            except Exception as _e:
                return {**state, "error": f"Lookup failed: {str(_e)}"}
        if no_vis:
            print("[Agent] Running agent without visualization")
            try:
                if self.tracing_enabled and self.tracer is not None:
                    with self.tracer.start_as_current_span("AgentRun_NoVis", openinference_span_kind="agent") as span:  # type: ignore[attr-defined]
                        span.set_input(state)  # type: ignore[attr-defined]
                        state = lookup_sales_data(state, self.llm, self.tracer)
                        result = analyzing_data(state, self.llm, self.tracer)
                        print(f"\nAgent response: {result.get('answer', [None])[0]}")
                        span.set_output(result)  # type: ignore[attr-defined]
                        if StatusCode is not None:
                            span.set_status(StatusCode.OK)  # type: ignore[attr-defined]
                        return result
                else:
                    state = lookup_sales_data(state, self.llm)
                    result = analyzing_data(state, self.llm, self.tracer)
                    print(f"\nAgent response: {result.get('answer', [None])[0]}")
                    return result
            except Exception as _e:
                print(f"Lookup failed: {str(_e)}")
                return {**state, "error": f"Lookup failed: {str(_e)}"}
        
        if visualization_goal:
            state["visualization_goal"] = visualization_goal
        print("Running the graph...")
        if self.tracing_enabled and self.tracer is not None:
            try:
                with self.tracer.start_as_current_span("AgentRun", openinference_span_kind="agent") as span:  # type: ignore[attr-defined]
                    print("[LangGraph] Starting LangGraph execution with tracing")
                    span.set_input(state)  # type: ignore[attr-defined]
                    result = self.graph.invoke(state)
                    print(f"\nAgent response: {result.get('answer', [])}")
                    span.set_output(result)  # type: ignore[attr-defined]
                    if StatusCode is not None:
                        span.set_status(StatusCode.OK)  # type: ignore[attr-defined]
                    print("[LangGraph] LangGraph execution completed")
                    return result
            except Exception:
                # Fallback to non-traced execution on any tracing error
                result = self.graph.invoke(state)
                print(f"\nAgent response: {result.get('answer', [])}")
                return result
        else:
            print("[LangGraph] Starting LangGraph execution")
            result = self.graph.invoke(state)
            print("[LangGraph] LangGraph execution completed")
            return result
    
    def _run_with_evaluation(
        self,
        *,
        prompt: str,
        visualization_goal: Optional[str] = None,
        lookup_only: bool = False,
        no_vis: bool = False,
        best_of_n: int = 1,
        temp: Optional[float] = None,
        temp_max: Optional[float] = None,
        csv_eval_fn: Optional[callable] = None,
        text_eval_fn: Optional[callable] = None,
        save_dir: Optional[str] = None,
    ) -> Dict:
        """Core evaluation logic extracted from run() for CodeCarbon wrapping."""
        
        if best_of_n > 1 and temp is not None and temp_max is not None:
            temps = np.linspace(temp, temp_max, best_of_n).tolist()
        else:
            temps = [temp if temp is not None else self.llm.temperature] * best_of_n
        
        print(f"[Agent] Running best-of-{best_of_n} with temperatures: {temps}")
        
        all_results = []
        all_scores = []
        
        for i in range(best_of_n):
            original_temp = self.llm.temperature
            self.llm.temperature = temps[i]
            
            try:
                result = self.run_core(
                    prompt,
                    visualization_goal=visualization_goal,
                    lookup_only=lookup_only,
                    no_vis=no_vis
                )

                # Save CSV
                csv_path = None
                if result.get("data"):
                    csv_path = os.path.join(save_dir, f"run_data.csv")
                    result_rows = text_to_csv(result['data'])
                    save_csv(result_rows, csv_path)
                
                # Extract analysis text
                analysis_text = result.get("answer", [None])[0] if result.get("answer") else None
                
                # Evaluate
                score = 0.0
                csv_score = None
                text_score = None
                
                if csv_eval_fn:
                    csv_score = csv_eval_fn(csv_path)
                    score += csv_score
                    result["csv_score"] = csv_score
                
                if text_eval_fn:
                    text_score = text_eval_fn(analysis_text)
                    score += text_score
                    result["text_score"] = text_score
                
                result["temperature"]= temps[i]

                all_results.append(result)
                all_scores.append(score)
                
            except Exception as e:
                print(f"Error: {str(e)}")
                
        self.llm.temperature = original_temp
        if not all_scores:
            return {}, 0.0
        
        best_idx = int(np.argmax(all_scores))
        best_result = all_results[best_idx]
        
        results_path = os.path.join(save_dir, "all_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        score_variance = (max(all_scores) - min(all_scores))/max(all_scores) if max(all_scores) != 0 else 0.0
        return best_result, score_variance
            
    def run(
        self,
        prompt: str,
        *,
        visualization_goal: Optional[str] = None,
        lookup_only: bool = False,
        no_vis: bool = False,
        best_of_n: int = 1,
        temp: Optional[float] = None,
        temp_max: Optional[float] = None,
        csv_eval_fn: Optional[callable] = None,
        text_eval_fn: Optional[callable] = None,
        save_dir: Optional[str] = None,
        enable_codecarbon: bool = False,
    ) -> Dict:
        
        if save_dir is None:
            save_dir = tempfile.mkdtemp(prefix="agent_runs_")
        os.makedirs(save_dir, exist_ok=True)
        
        # Wrap execution with CodeCarbon if requested and available
        if enable_codecarbon and _CODECARBON_AVAILABLE:
            codecarbon_dir = os.path.join(save_dir, "codecarbon")
            os.makedirs(codecarbon_dir, exist_ok=True)
            try:
                with EmissionsTracker(  # type: ignore[call-arg]
                    project_name="SalesDataAgent",
                    output_dir=codecarbon_dir,
                    save_to_file=True,
                    measure_power_secs=1,
                    log_level="error",
                ):
                    return self._run_with_evaluation(
                        prompt=prompt,
                        visualization_goal=visualization_goal,
                        lookup_only=lookup_only,
                        no_vis=no_vis,
                        best_of_n=best_of_n,
                        temp=temp,
                        temp_max=temp_max,
                        csv_eval_fn=csv_eval_fn,
                        text_eval_fn=text_eval_fn,
                        save_dir=save_dir,
                    )
            except Exception as e:
                print(f"CodeCarbon tracking failed: {e}, continuing without it")
                # Fall through to run without CodeCarbon
        
        return self._run_with_evaluation(
            prompt=prompt,
            visualization_goal=visualization_goal,
            lookup_only=lookup_only,
            no_vis=no_vis,
            best_of_n=best_of_n,
            temp=temp,
            temp_max=temp_max,
            csv_eval_fn=csv_eval_fn,
            text_eval_fn=text_eval_fn,
            save_dir=save_dir,
        )

__all__ = ["SalesDataAgent", "State"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the Sales Data Agent")
    parser.add_argument("prompt", type=str, help="User prompt/question")
    parser.add_argument("--gt_csv", type=str, default=None, help="Path to ground-truth CSV file")
    parser.add_argument("--gt_text", type=str, default=None, help="Path to a text file containing the ground-truth")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save run results")

    parser.add_argument("--data", dest="data_path", type=str, default=DEFAULT_DATA_PATH, help="Path to parquet file")
    parser.add_argument("--goal", dest="visualization_goal", type=str, default=None, help="Optional visualization goal")
    parser.add_argument("--model", type=str, default="llama3.2:3b", help="Ollama model name")
       
    # Agent type options
    agent_group = parser.add_mutually_exclusive_group()
    agent_group.add_argument("--lookup_only", action="store_true", help="Only run data lookup")
    agent_group.add_argument("--no_vis", action="store_true", help="Run lookup then analysis (no visualization)")

    # Best-of-n options
    parser.add_argument("--best_of_n", type=int, default=1, help="Run agent N times and pick the best result")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature used to build the agent and as minimum for best-of-n")
    parser.add_argument("--temp-max", type=float, default=None, help="Max temperature for best-of-n, if not provided best-of-n runs without modifying the temperature")

    # CSV evaluation options
    csv_eval_group = parser.add_mutually_exclusive_group()
    csv_eval_group.add_argument("--py_csv_eval", action="store_true", help="Use Python evaluator for CSV IoU")
    csv_eval_group.add_argument("--cpp_csv_eval", action="store_true", help="Use C++ evaluator for CSV IoU")
    parser.add_argument("--evaluator_exe", type=str, default=None, help="Path to C++ comparator executable")
    parser.add_argument("--eval_keys", type=str, default=None, help="Comma-separated key columns for C++ comparator")
    parser.add_argument("--iou_type", type=str, default="rows", choices=["columns", "rows", "table"], help="Type of IoU to use for CSV evaluation, choose between 'columns', 'rows', 'table'")

    # Text evaluation options
    text_eval_group = parser.add_mutually_exclusive_group()
    text_eval_group.add_argument("--spice_text_eval", action="store_true")
    text_eval_group.add_argument("--bleu_text_eval", action="store_true")
    text_eval_group.add_argument("--llm_text_eval", action="store_true") 
    parser.add_argument("--bleu_nltk", action="store_true", help="Use nltk for BLEU implementation instead of simple BLEU")
    parser.add_argument("--spice_jar", type=str, default=None, help="Path to SPICE jar (e.g., spice-1.0.jar)")
    parser.add_argument("--spice_java_bin", type=str, default="java", help="Java executable for SPICE")

    # Phoenix tracking options
    parser.add_argument("--enable_tracing", action="store_true", help="Enable Phoenix tracing/tracking")
    parser.add_argument("--phoenix_endpoint", type=str, default="http://localhost:6006/v1/traces", help="Phoenix endpoint URL (default: https://app.phoenix.arize.com/v1/traces)")
    parser.add_argument("--project_name", type=str, default="evaluating-agent", help="Phoenix project name")

    # CodeCarbon options
    parser.add_argument("--enable_codecarbon", action="store_true", help="Enable CodeCarbon energy/emissions tracking")
    
    args = parser.parse_args()

    # Create agent
    agent = SalesDataAgent(
        model=args.model, 
        temperature=args.temp, 
        data_path=args.data_path,
        enable_tracing=args.enable_tracing,
        phoenix_endpoint=args.phoenix_endpoint,
        project_name=args.project_name,
    )

    # Get evaluation functions based on arguments
    csv_eval_fn, text_eval_fn = get_evaluation_functions(
        lookup_only=args.lookup_only,
        gt_csv_path = args.gt_csv,
        py_csv_eval=args.py_csv_eval,
        cpp_csv_eval=args.cpp_csv_eval,
        evaluator_exe=args.evaluator_exe,
        eval_keys=args.eval_keys,
        gt_text_path=args.gt_text,
        iou_type=args.iou_type,
        spice_text_eval=args.spice_text_eval,
        bleu_text_eval=args.bleu_text_eval,
        llm_text_eval=args.llm_text_eval,
        bleu_nltk=args.bleu_nltk,
        spice_jar=args.spice_jar,
        spice_java_bin=args.spice_java_bin,
    )

    # Run agent
    output, score_variance = agent.run(
        args.prompt,
        visualization_goal=args.visualization_goal,
        lookup_only=args.lookup_only,
        no_vis=args.no_vis,
        best_of_n=args.best_of_n,
        temp=args.temp,
        temp_max=args.temp_max,
        csv_eval_fn=csv_eval_fn,
        text_eval_fn=text_eval_fn,
        save_dir=args.save_dir,
        enable_codecarbon=args.enable_codecarbon,
    )
    
    # Print results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    if args.best_of_n > 1:
        print(f"Score variance: {score_variance:.4f}")
    print(f"Answer: {output.get('answer', [])}")
    if args.save_dir or args.best_of_n > 1:
        print(f"Results saved to: {args.save_dir or 'temp directory'}")