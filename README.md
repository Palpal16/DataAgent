## Sales Data Agent

LLM agent that queries a local parquet dataset with DuckDB, summarizes results, and optionally generates matplotlib code. Supports Ollama (default), OpenAI, and Anthropic models.

### Quick start
1) Install deps
```powershell
pip install -r requirements.txt
```

2) Ensure the parquet exists
`data/Store_Sales_Price_Elasticity_Promotions_Data.parquet`

3) Run with Ollama (default)
```powershell
ollama pull llama3.2:3b
ollama serve
python -m Agent.data_agent "Show me the sales in Nov 2021" --goal "Sales trend for Nov 2021"
```

### LLM providers
- Ollama (default): `--model "llama3.2:3b"` and `OLLAMA_HOST` if remote
- OpenAI: `--model "openai:gpt-4o-mini"` and `OPENAI_API_KEY`
- Anthropic: `--model "anthropic:claude-3-5-sonnet-latest"` and `ANTHROPIC_API_KEY`

PowerShell env vars:
```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
$env:ANTHROPIC_API_KEY="YOUR_KEY"
$env:OLLAMA_HOST="http://localhost:11434"
```

### Python usage
```python
from Agent.data_agent import SalesDataAgent

agent = SalesDataAgent(model="llama3.2:3b", temperature=0.1)
result = agent.run("Show me the sales in Nov 2021", visualization_goal="Sales trend for Nov 2021")
print(result.get("tool_choice"))
```

### CLI patterns
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

### HTTP API (Flask)
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

### Evaluation (Python)
Compare run CSV vs ground truth:
```powershell
python -m Agent.data_agent "What were the sales in November 2021?" --lookup-only `
  --output-csv "results/sales_november_2021.csv" `
  --expected-csv "results/real_sales_november_2021.csv"
```

### C++ comparator (optional)
Build:
```powershell
cd my_cpp
cmake -S . -B build -G "Ninja"
cmake --build build --config Release
cd ..
```

Use:
```powershell
python -m Agent.data_agent "Weekly sales in 2021" `
  --lookup-only `
  --output-csv "results/weekly_sales_2021.csv" `
  --expected-csv "C:\path\to\expected.csv" `
  --evaluator-exe ".\my_cpp\build\resultcmp.exe" `
  --eval-keys "week,store_id"
```

### agent_config_runner (C++)
Build:
```powershell
cd my_cpp
cmake -S . -B build -G "Ninja"
cmake --build build --config Release
```

Run:
```powershell
.\build\agent_config_runner.exe
# or
.\build\agent_config_runner.exe path\to\agent_config.yaml
```

Batch mode (agent_config.yaml):
```yaml
test_cases_json: ./test_cases.json
run_batch: true
```

### JMeter (optional)
1) Install Java (8+)
2) Download JMeter: https://jmeter.apache.org/download_jmeter.cgi
3) Run `jmeter.bat` (Windows) or `./jmeter` (macOS/Linux)

### Tracing with Phoenix (optional)
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

### Docker (optional)
```powershell
docker build -t data-agent .
docker run --rm -e OLLAMA_HOST=http://host.docker.internal:11434 `
  data-agent "Show me the sales in Nov 2021" --goal "Sales trend for Nov 2021"
```

### Troubleshooting
- Ollama running: `curl http://localhost:11434/api/version`
- Pull model: `ollama pull llama3.2:3b`
- Parquet path exists and readable
- If SQL fails, add hints (e.g., `CAST(date_col AS VARCHAR)`)
