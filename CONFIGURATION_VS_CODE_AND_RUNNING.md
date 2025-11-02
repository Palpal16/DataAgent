## VS Code configuration and running guide

This document shows how to configure, build, and run the project fully within VS Code on Windows, including the C++ evaluator and the Python agent.

---

## Prerequisites

- Visual Studio 2022 with C++ workload
- VS Code + CMake Tools extension
- Python 3.10+ and required packages:
```powershell
pip install langgraph langchain-ollama duckdb pandas pyarrow matplotlib
```
- Ollama installed and a model pulled (default used here: `llama3.2:3b`):
```powershell
ollama pull llama3.2:3b
```

---

## Configure and build the C++ evaluator in VS Code

1) Open the project root in VS Code: `DataAgent/`

2) Configure with the preset:
   - Press `Ctrl+Shift+P` → "CMake: Configure"
   - Choose: `Visual Studio Community 2022 Release - x86_amd64`

3) Build:
   - Press `Ctrl+Shift+P` → "CMake: Build"
   - If asked, pick `Debug` (or `Release` if preferred)

4) Resulting executable path (depending on config):
   - Debug:
     - `cpp_evaluator\out\build\Visual Studio Community 2022 Release - x86_amd64\Debug\resultcmp.exe`
   - Release:
     - `cpp_evaluator\out\build\Visual Studio Community 2022 Release - x86_amd64\Release\resultcmp.exe`

5) Verify the executable exists (PowerShell):
```powershell
Test-Path 'cpp_evaluator\out\build\Visual Studio Community 2022 Release - x86_amd64\Debug\resultcmp.exe'
```

6) Optional: run help
```powershell
& 'cpp_evaluator\out\build\Visual Studio Community 2022 Release - x86_amd64\Debug\resultcmp.exe' --help
```

Tip: If `cmake` isn’t on PATH in your terminal, build via VS Code commands (Command Palette) or use Developer PowerShell for VS.

---

## Running the Python agent from VS Code terminal

Always run from the project root so the `Agent` package resolves:
```powershell
cd C:\Users\Recup\OneDrive\Documentos\Books\PACS\DataAgent
```

### Run lookup only (no analysis/visualization) with model `llama3.2:3b`
```powershell
python -m Agent.data_agent "Weekly sales in 2021" `
  --lookup-only `
  --model "llama3.2:3b"
```

### Evaluate lookup results against an expected CSV using the C++ comparator
Debug build example (quote paths that have spaces):
```powershell
python -m Agent.data_agent "Weekly sales in 2021" `
  --lookup-only `
  --model "llama3.2:3b" `
  --expected-csv C:\Users\Recup\OneDrive\Documentos\Books\PACS\DataAgent\evaluation.csv `
  --evaluator-exe "C:\Users\Recup\OneDrive\Documentos\Books\PACS\DataAgent\cpp_evaluator\out\build\Visual Studio Community 2022 Release - x86_amd64\Debug\resultcmp.exe" `
  --eval-keys week,store_id --eval-float-rel 1e-6 --eval-float-abs 1e-8
```

Release build example:
```powershell
python -m Agent.data_agent "Weekly sales in 2021" `
  --lookup-only `
  --model "llama3.2:3b" `
  --expected-csv C:\Users\Recup\OneDrive\Documentos\Books\PACS\DataAgent\evaluation.csv `
  --evaluator-exe "C:\Users\Recup\OneDrive\Documentos\Books\PACS\DataAgent\cpp_evaluator\out\build\Visual Studio Community 2022 Release - x86_amd64\Release\resultcmp.exe" `
  --eval-keys week,store_id --eval-float-rel 1e-6 --eval-float-abs 1e-8
```

If you see `ModuleNotFoundError: No module named 'Agent'`, ensure you are in the project root or set `PYTHONPATH` to the project root before running.

---

## Creating an expected CSV baseline

If you don’t have an expected file yet, generate one from the current lookup result:
```powershell
python - << 'PY'
from Agent.data_agent import SalesDataAgent
from io import StringIO
import pandas as pd

agent = SalesDataAgent(model="llama3.2:3b")
res = agent.run("Weekly sales in 2021", only_lookup=True)
df = pd.read_fwf(StringIO(res.get("data","")))
df.to_csv(r"C:\Users\Recup\OneDrive\Documentos\Books\PACS\DataAgent\evaluation.csv", index=False)
print("Saved expected CSV.")
PY
```

Then re-run the agent with `--expected-csv` and you should see `evaluation.equal = true` (assuming the data hasn’t changed and keys/tolerances are correct).

---

## Troubleshooting

- EXE not found: search for it
```powershell
Get-ChildItem -Recurse -File -Filter resultcmp.exe | Select-Object -Expand FullName
```

- Path quoting: if a path contains spaces, wrap it in double quotes.

- Debug vs Release: ensure your `--evaluator-exe` points to the correct configuration folder.

- Keys and tolerances: if `equal=false`, try setting proper key columns and adjust tolerances:
  - `--eval-keys key1,key2`
  - `--eval-float-abs 1e-8 --eval-float-rel 1e-6`

- Model not running: make sure Ollama is serving (`ollama serve`) and the model exists (`ollama pull llama3.2:3b`).

---

## Summary

1. Configure and build the C++ evaluator via VS Code CMake commands.
2. Run the Python agent from the project root, adding `--model` and optionally `--lookup-only`.
3. Compare results with the C++ evaluator using `--expected-csv` and `--evaluator-exe`.
4. Create/update your expected CSV with the provided Python snippet when needed.


