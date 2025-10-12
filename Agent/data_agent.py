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

import duckdb
import pandas as pd
from typing_extensions import NotRequired, TypedDict

from langgraph.graph import END, StateGraph
from langchain_ollama import ChatOllama


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
    analyze_data: Optional[str]
    answer: List[str]
    visualization_goal: Optional[str]
    chart_config: Optional[dict]
    tool_choice: NotRequired[str]
    error: NotRequired[str]


# -----------------------------
# LLM Helpers
# -----------------------------

SQL_GENERATION_PROMPT = (
    "Generate an SQL query based on the prompt. "
    "Please just reply with the SQL query and NO MORE, just the query. "
    "The prompt is : {prompt} "
    "The available columns are: {columns}. "
    "The table name is: {table_name}. "
    "If you need to use a DATE column with LIKE or pattern matching, first CAST it to VARCHAR like this: "
    "CAST(date_column AS VARCHAR) LIKE '%2021-11%' "
    "Return only the SQL query, with no explanations or markdown formatting."
)


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
    return (
        sql_query.strip()
        .replace("```sql", "")
        .replace("```", "")
    )


def lookup_sales_data(state: State, data_path: str, llm: ChatOllama) -> State:
    """Look up sales data from a parquet file using LLM-generated SQL over DuckDB.

    This function registers the parquet data as a temporary DuckDB table, asks the
    LLM to generate an SQL query from the user's prompt and available columns, then
    executes the query and stores a text-formatted table in state['data'].

    Args:
        state: Conversation state; must include 'prompt'.
        data_path: Filesystem path to the parquet dataset.
        llm: ChatOllama instance used for prompt-to-SQL generation.

    Returns:
        Updated state containing 'data' (string table) or 'error'.
    """
    try:
        table_name = "sales"
        df = pd.read_parquet(data_path)
        duckdb.sql("DROP TABLE IF EXISTS sales")
        duckdb.register("df", df)
        duckdb.sql(f"CREATE TABLE {table_name} AS SELECT * FROM df")
        sql_query = generate_sql_query(state, df.columns.tolist(), table_name, llm)
        result_df = duckdb.sql(sql_query).df()
        result_str = result_df.to_string(index=False)
        return {**state, "data": result_str}
    except Exception as e:
        return {**state, "error": f"Error accessing data: {str(e)}"}


def decide_tool(state: State, llm: ChatOllama) -> State:
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
    tools_description = (
        "You have access to the following tools to help you with your task:\n\n"
        "- lookup_sales_data: Look up sales data from a parquet file using SQL.\n"
        "- analyzing_data: Analyze the sales data for trends and insights.\n"
        "- create_visualization: Create visualizations based on the sales data.\n"
        "- end: End the conversation if the task is complete.\n\n"
        "Based on the actual state and the user prompt, decide which tool to use next."
    )

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
        return {**state, "error": f"Error accessing data: {str(e)}"}


DATA_ANALYSIS_PROMPT = (
    "Analyze the following data: {data}\n"
    "Your job is to answer the following question: {prompt}"
)


def analyzing_data(state: State, llm: ChatOllama) -> State:
    """Ask the LLM to analyze the looked-up data in the context of the prompt.

    Args:
        state: Conversation state; should include 'data' and 'prompt'.
        llm: ChatOllama instance used for the analysis.

    Returns:
        Updated state including 'analyze_data' and the analysis appended to 'answer'.
    """
    try:
        formatted_prompt = DATA_ANALYSIS_PROMPT.format(
            data=state.get("data", ""), prompt=state.get("prompt", "")
        )
        analysis_result = llm.invoke(formatted_prompt)
        analysis_text = analysis_result.content if hasattr(analysis_result, "content") else str(analysis_result)
        return {
            **state,
            "analyze_data": analysis_text,
            "answer": state.get("answer", []) + [analysis_text],
        }
    except Exception as e:
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


def create_visualization(state: State, llm: ChatOllama) -> State:
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
        model: str = "llama3.2",
        temperature: float = 0.1,
        max_tokens: int = 2000,
        streaming: bool = True,
        data_path: Optional[str] = None,
    ) -> None:
        """Initialize the agent and compile the graph.

        Args:
            model: Ollama model name.
            temperature: Sampling temperature for the LLM.
            max_tokens: Generation token limit.
            streaming: Whether to stream tokens from the LLM.
            data_path: Optional override for the parquet dataset path.
        """
        self.llm = ChatOllama(
            model=model, temperature=temperature, max_tokens=max_tokens, streaming=streaming
        )
        self.data_path = data_path or DEFAULT_DATA_PATH
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
            requests.get("http://localhost:11434/api/version", timeout=3).json()
            print("Server is running locally")
            return self.check_ollama()
        except Exception as e:
            print(e)
            return False

    def _build_graph(self):
        """Construct and compile the LangGraph for the agent run loop."""
        graph = StateGraph(State)

        decide_tool_with_llm = partial(decide_tool, llm=self.llm)
        lookup_with_ctx = partial(lookup_sales_data, data_path=self.data_path, llm=self.llm)
        analyze_with_llm = partial(analyzing_data, llm=self.llm)
        create_viz_with_llm = partial(create_visualization, llm=self.llm)

        graph.add_node("decide_tool", decide_tool_with_llm)
        graph.add_node("lookup_sales_data", lookup_with_ctx)
        graph.add_node("analyzing_data", analyze_with_llm)
        graph.add_node("create_visualization", create_viz_with_llm)
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

    def run(self, prompt: str, *, visualization_goal: Optional[str] = None, initial_state: Optional[Dict] = None) -> Dict:
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
            return {**state, "error": "Model is not running locally, remember to run ollama serve"}
        else:
            if visualization_goal:
                state["visualization_goal"] = visualization_goal
            if initial_state:
                state.update(initial_state)
            print("Running the graph...")
            return self.graph.invoke(state)

    def draw_graph_ascii(self) -> str:
        """Return an ASCII rendering of the compiled graph if available."""
        try:
            return self.graph.get_graph().print_ascii()
        except Exception:
            return "[graph visualization unavailable]"


__all__ = ["SalesDataAgent", "State"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Sales Data Agent")
    parser.add_argument("prompt", type=str, help="User prompt/question")
    parser.add_argument("--data", dest="data_path", type=str, default=None, help="Path to parquet file")
    parser.add_argument("--goal", dest="visualization_goal", type=str, default=None, help="Optional viz goal")
    args = parser.parse_args()

    agent = SalesDataAgent(data_path=args.data_path)
    output = agent.run(args.prompt, visualization_goal=args.visualization_goal)
    # Minimal printout
    print(json.dumps({k: v for k, v in output.items() if k != "data"}, indent=2))


