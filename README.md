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

### From the command line
```powershell
python -m Agent.data_agent "Show me the sales in Nov 2021" --goal "Sales trend for Nov 2021"
```

You can override the parquet path:
```powershell
python -m Agent.data_agent "Top products by revenue" --data "C:\\path\\to\\your.parquet"
```

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
