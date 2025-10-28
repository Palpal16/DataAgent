## Sales Data Agent

An LLM-powered agent that queries a local parquet dataset with DuckDB, analyzes the results, and generates visualization code. It uses a LangGraph workflow and an Ollama-hosted model (default: `llama3.2`).

### What it does
- Lookup: converts natural language into SQL via the LLM and runs it on DuckDB over the parquet file at `data/Store_Sales_Price_Elasticity_Promotions_Data.parquet`.
- Analyze: asks the LLM to summarize/interpret the results.
- Visualize: requests a minimal chart configuration and emits matplotlib code to plot it.

---

## Requirements
- Python 3.10+
- Ollama running locally (`https://ollama.com`) with a model pulled (default: `llama3.2`)
- Parquet file present at `data/Store_Sales_Price_Elasticity_Promotions_Data.parquet`

Install Python deps (from the project root):
```powershell
pip install langgraph langchain-ollama duckdb pandas pyarrow matplotlib
```

---

## Start Ollama locally

1) Install Ollama for your OS from `https://ollama.com/download`.

2) Pull a model (default used in code is `llama3.2`):
```powershell
ollama pull llama3.2
```

3) Start the Ollama server (keeps API on `http://localhost:11434`):
```powershell
ollama serve
```

4) Verify it is running:
```powershell
curl http://localhost:11434/api/version
ollama list
ollama ps
```

If you need to run against a remote server, ensure the API is reachable and set any required environment variables or update the code to point to the remote host.

---

## Project layout
```
DataAgent/
  Agent/
    data_agent.py        # SalesDataAgent class and LangGraph wiring
  data/
    Store_Sales_Price_Elasticity_Promotions_Data.parquet
  LangChainAgent.ipynb   # Original notebook prototype
  readme.MD              # This file
```

---

## Using the agent

### From a Jupyter notebook
```python
from Agent.data_agent import SalesDataAgent

agent = SalesDataAgent(
    model="llama3.2",
    temperature=0.1,
    max_tokens=2000,
    streaming=True,
)

result = agent.run(
    "Show me the sales in Nov 2021",
    visualization_goal="Sales trend for Nov 2021",
)

print("Final tool:", result.get("tool_choice"))
print("Chart config:", result.get("chart_config"))
print("Answer steps:", len(result.get("answer", [])))

# If the last answer is chart code, execute it to render the chart
if result.get("chart_config") and result.get("answer"):
    exec(result["answer"][-1], globals(), locals())
```

---

## Tracing with Phoenix (optional)

You can enable OpenInference/Phoenix tracing to visualize your agent runs (spans like AgentRun, tool_choice, sql_query_exec, data_analysis, gen_visualization).

### 1) Install tracing dependencies
```powershell
pip install phoenix openinference-instrumentation-langchain opentelemetry-api
```

### 2) Choose where to send traces
- Self-hosted Phoenix (local): set endpoint to `http://localhost:6006/v1/traces`
- Phoenix Cloud: use endpoint `https://app.phoenix.arize.com/v1/traces` and your API key

### 3) Start Phoenix locally (self-hosted) (Only if running locally)
```powershell
phoenix serve
# or explicitly
# phoenix serve --host 0.0.0.0 --port 6006
```

Open the UI at `http://localhost:6006`.

### 4) Enable tracing in SalesDataAgent
```python
from Agent.data_agent import SalesDataAgent

# Self-hosted example
agent = SalesDataAgent(
    enable_tracing=True,
    phoenix_endpoint="http://localhost:6006/v1/traces",
    project_name="evaluating-agent",
)

# Cloud example
# agent = SalesDataAgent(
#     enable_tracing=True,
#     phoenix_endpoint="https://app.phoenix.arize.com/v1/traces",
#     phoenix_api_key="<YOUR_API_KEY>",
#     project_name="evaluating-agent",
# )

ret = agent.run("What was the most popular product SKU?")
```

Alternatively, set the endpoint via environment variable before creating the agent:
```python
import os
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006/v1/traces"
from Agent.data_agent import SalesDataAgent
agent = SalesDataAgent(enable_tracing=True)
```

### 5) View your traces
- Self-hosted UI: open `http://localhost:6006`, go to Traces, select the project (default: `evaluating-agent`).
- Cloud UI: open `https://app.phoenix.arize.com`, Traces → select your project.

You should see spans named: `AgentRun`, `tool_choice`, `sql_query_exec`, `data_analysis`, `gen_visualization`.

### Troubleshooting tracing
- Verify the console shows: `[LangGraph] Starting LangGraph execution with tracing`.
- Confirm the endpoint includes `/v1/traces` and is reachable.
- Make sure dependencies are installed: `phoenix`, `openinference-instrumentation-langchain`, `opentelemetry-api`.
- For Cloud, ensure `phoenix_api_key` is set and valid.

### From the command line
```powershell
python -m Agent.data_agent "Show me the sales in Nov 2021" --goal "Sales trend for Nov 2021"
```

You can override the parquet path:
```powershell
python -m Agent.data_agent "Top products by revenue" --data "C:\\path\\to\\your.parquet"
```

---

## Run with Docker

Build the image (from project root):
```powershell
docker build -t data-agent .
```

Build explicitly with this Dockerfile (if building from elsewhere):
```powershell
docker build -f Dockerfile -t data-agent .
```

Run (Docker Desktop on Windows/macOS):
```powershell
docker run --rm -e OLLAMA_HOST=http://host.docker.internal:11434 \
  data-agent "Show me the sales in Nov 2021" --goal "Sales trend for Nov 2021"
```

Run (Linux host, replace with your host IP or network alias):
```bash
docker run --rm -e OLLAMA_HOST=http://192.168.1.10:11434 \
  data-agent "Top products by revenue"
```

Use a custom parquet by mounting it over the default path inside the container:
```powershell
docker run --rm -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -v C:\\path\\to\\your.parquet:/app/data/Store_Sales_Price_Elasticity_Promotions_Data.parquet \
  data-agent "Show weekly sales trend in 2021"
```

Override the entrypoint (run any Python you want inside the image):
```powershell
docker run --rm -it --entrypoint bash data-agent
# inside the container:
python -m Agent.data_agent "Top products by revenue" --goal "Top-5"
```

Notes:
- The container expects an Ollama server to be reachable at `OLLAMA_HOST`.
- Default `ENTRYPOINT` is `python -m Agent.data_agent`; any arguments after the image name are passed to the agent.

---

## Configuration
- Change model: pass `model="<name>"` to `SalesDataAgent(...)`.
- Custom parquet: pass `data_path="..."` in the constructor, or use `--data` in CLI.
- Visualization goal: pass `visualization_goal="..."` to `run()` or `--goal` in CLI.

---

## Troubleshooting
- Check Ollama is up:
  - `curl http://localhost:11434/api/version`
  - `ollama list` and `ollama ps`
- Ensure the model is pulled (e.g., `ollama pull llama3.2`).
- If you see connection errors, confirm no firewall is blocking `localhost:11434`.
- If SQL fails, skim the printed `data`/columns and adjust the prompt (e.g., include date formatting hints like `CAST(date_col AS VARCHAR)`).

---

## High-level flow
1. Decide tool (LLM): choose lookup → analyze → visualize → end.
2. Lookup (DuckDB): parquet → temp table → LLM SQL → query → text table in state.
3. Analyze (LLM): summarize/answer with reference to the result data.
4. Visualize (LLM): emit compact config → generate matplotlib code to plot.

The agent exposes a single `run(prompt, visualization_goal=None, initial_state=None)` entry point and returns the final state with an ordered `answer` list (analysis and then chart code when applicable).
