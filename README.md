## Sales Data Agent

An LLM-powered agent that queries a local parquet dataset with DuckDB, analyzes the results, and can generate matplotlib visualization code. Supports Ollama (default), OpenAI, and Anthropic models.

### What it does
- Lookup: LLM generates SQL and runs it with DuckDB over `data/Store_Sales_Price_Elasticity_Promotions_Data.parquet`
- Analyze: LLM summarizes and interprets the result
- Visualize: LLM emits a compact chart config + matplotlib code

---

## Requirements
- Python 3.10+
- Ollama running locally with a pulled model **or** an API key for OpenAI/Anthropic
- Parquet file at `data/Store_Sales_Price_Elasticity_Promotions_Data.parquet`

Install deps:
```powershell
pip install -r requirements.txt
```

---

## Start Ollama locally
```powershell
ollama pull llama3.2:3b
ollama serve
curl http://localhost:11434/api/version
```

If using a remote Ollama, set `OLLAMA_HOST` to the reachable URL.

---

## Project layout
```
DataAgent/
  Agent/                 # Core agent
  data/                  # Parquet dataset
  evaluation/            # Evaluation tools
  my_cpp/                # C++ comparator + runners
  jmx_files/             # JMeter artifacts
```

---

## Using the agent

### Python
```python
from Agent.data_agent import SalesDataAgent

agent = SalesDataAgent(model="llama3.2:3b", temperature=0.1)
result = agent.run("Show me the sales in Nov 2021", visualization_goal="Sales trend for Nov 2021")
print(result.get("tool_choice"))
```

### CLI
```powershell
python -m Agent.data_agent "Show me the sales in Nov 2021" --goal "Sales trend for Nov 2021"
```

Lookup only (save CSV):
```powershell
python -m Agent.data_agent "What were the sales in November 2021?" --lookup-only --output-csv "results/sales_november_2021.csv"
```

Best-of-N (lookup only):
```powershell
python -m Agent.data_agent "What were the sales in November 2021?" --lookup-only `
  --best-of-n 5 --best-of-n-temp-min 0.0 --best-of-n-temp-max 0.6 `
  --output-csv "results/sales_november_2021.csv"
```

Override parquet:
```powershell
python -m Agent.data_agent "Top products by revenue" --data "C:\path\to\your.parquet"
```

---

## LLM providers
- Ollama (default): `--model "llama3.2:3b"`, optional `OLLAMA_HOST`
- OpenAI: `--model "openai:gpt-4o-mini"` with `OPENAI_API_KEY`
- Anthropic: `--model "anthropic:claude-3-5-sonnet-latest"` with `ANTHROPIC_API_KEY`

PowerShell env vars:
```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
$env:ANTHROPIC_API_KEY="YOUR_KEY"
$env:OLLAMA_HOST="http://localhost:11434"
```

---

## HTTP API (Flask)
Start:
```powershell
python agent_api.py
```

Call:
```powershell
$body = @{
  prompt = "What were the sales in November 2021?"
  lookup_only = $true
  best_of_n = 5
  best_of_n_temp_min = 0.0
  best_of_n_temp_max = 0.6
  output_csv = "results/sales_november_2021.csv"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:5000/call-agent" -ContentType "application/json" -Body $body
```

Use the C++ comparator via the API by adding:
- `evaluator_exe`: e.g. `".\my_cpp\build\resultcmp.exe"`
- `eval_keys`: e.g. `"week,store_id"`

---

## Evaluation

Python comparator (IoU metrics):
```powershell
python -m Agent.data_agent "What were the sales in November 2021?" --lookup-only `
  --output-csv "results/sales_november_2021.csv" `
  --expected-csv "results/real_sales_november_2021.csv"
```

C++ comparator (optional):
```powershell
cd my_cpp
cmake -S . -B build -G "Ninja"
cmake --build build --config Release
cd ..

python -m Agent.data_agent "Weekly sales in 2021" `
  --lookup-only `
  --output-csv "results/weekly_sales_2021.csv" `
  --expected-csv "C:\path\to\expected.csv" `
  --evaluator-exe ".\my_cpp\build\resultcmp.exe" `
  --eval-keys "week,store_id"
```

---

## agent_config_runner (C++)
Build:
```powershell
cd my_cpp
cmake -S . -B build -G "Ninja"
cmake --build build --config Release
```

Run:
```powershell
.\build\agent_config_runner.exe
.\build\agent_config_runner.exe path\to\agent_config.yaml
```

Batch mode (agent_config.yaml):
```yaml
test_cases_json: ./test_cases.json
run_batch: true
```

---

## JMeter (optional)
1) Install Java 8+
2) Download JMeter: https://jmeter.apache.org/download_jmeter.cgi
3) Run `jmeter.bat` (Windows) or `./jmeter` (macOS/Linux)

---

## Tracing with Phoenix (optional)
Install:
```powershell
pip install arize-phoenix openinference-instrumentation-langchain opentelemetry-api
```

Enable:
```python
from Agent.data_agent import SalesDataAgent
agent = SalesDataAgent(enable_tracing=True, phoenix_endpoint="http://localhost:6006/v1/traces")
agent.run("What was the most popular product SKU?")
```

---

## Docker (optional)
```powershell
docker build -t data-agent .
docker run --rm -e OLLAMA_HOST=http://host.docker.internal:11434 `
  data-agent "Show me the sales in Nov 2021" --goal "Sales trend for Nov 2021"
```

---

## Optional: SPICE (Java) for analysis-text evaluation
Prereqs: Java + a SPICE jar (e.g. `spice-1.0.jar`)

CLI example:
```powershell
python -m Agent.data_agent "What were the sum of sales in November 2021?" `
  --analyze-only `
  --expected-analysis-file "results/expected_analysis.txt" `
  --analysis-metric spice `
  --spice-jar "spice/spice-1.0.jar"
```

---

## Energy and emissions (CodeCarbon)
Enable via `SalesDataAgent.run(...)` and read `codecarbon/emissions.csv`.

```powershell
pip install codecarbon
carbonboard --filepath "codecarbon/emissions.csv" --port 8050
```

---

## High-level flow
1. Decide tool (LLM): lookup → analyze → visualize → end
2. Lookup (DuckDB): parquet → temp table → SQL → query → text table
3. Analyze (LLM): summarize/answer
4. Visualize (LLM): chart config + matplotlib code

---

## Troubleshooting
- Ollama running: `curl http://localhost:11434/api/version`
- Pull model: `ollama pull llama3.2:3b`
- Parquet path exists and readable
- If SQL fails, add hints (e.g., `CAST(date_col AS VARCHAR)`)
