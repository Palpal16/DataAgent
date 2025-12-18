// agent_orchestrator: run Python agent to generate CSV, then compare with resultcmp.
//
// Typical usage (from repo root):
//   cpp_evaluator/build/agent_orchestrator.exe ^
//     --prompt "What were the sales in November 2021?" ^
//     --output-csv "results/sales_november_2021.csv" ^
//     --expected-csv "results/real_sales_november_2021.csv" ^
//     --keys "week,store_id"
//
// Notes:
// - This tool shells out to Python (`python -m Agent.data_agent ...`) and then shells out to `resultcmp`.
// - It does NOT embed Python; it expects your environment can run the Python module.

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::string quote_arg(const std::string& s) {
    // Simple quoting for Windows/cmd and POSIX shells: wrap in double quotes and escape internal quotes.
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else out.push_back(c);
    }
    out.push_back('"');
    return out;
}

static bool starts_with(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

static std::string exe_suffix() {
#ifdef _WIN32
    return ".exe";
#else
    return "";
#endif
}

struct Args {
    std::string prompt;
    std::string python = "python";
    std::string agent_module = "Agent.data_agent";
    std::string model;
    std::string data_path;
    std::string goal;

    bool lookup_only = true;
    std::string output_csv;

    int best_of_n = 1;
    std::string best_of_n_temp_min;
    std::string best_of_n_temp_max;

    bool compare = true;
    std::string expected_csv;
    std::string keys; // comma-separated

    std::string resultcmp_path; // optional override
};

static void print_usage() {
    std::cerr
        << "agent_orchestrator\n"
        << "Runs Python agent -> writes CSV -> (optionally) compares with resultcmp.\n\n"
        << "Required:\n"
        << "  --prompt <text>\n"
        << "  --output-csv <path>\n"
        << "Optional (agent):\n"
        << "  --model <name>                 (e.g., llama3.2:3b | openai:gpt-4o-mini | anthropic:claude-3-5-sonnet-latest)\n"
        << "  --data <parquet_path>\n"
        << "  --goal <text>\n"
        << "  --lookup-only / --no-lookup-only (default: lookup-only)\n"
        << "  --best-of-n <N>\n"
        << "  --best-of-n-temp-min <float>\n"
        << "  --best-of-n-temp-max <float>\n"
        << "Optional (compare):\n"
        << "  --expected-csv <path>          (enables comparison)\n"
        << "  --keys <col1,col2>             (passed to resultcmp as --key)\n"
        << "  --resultcmp <path_to_resultcmp.exe>\n"
        << "  --no-compare                   (skip compare step)\n"
        << "Optional (python):\n"
        << "  --python <python_exe>          (default: python)\n"
        << "  --agent-module <module>        (default: Agent.data_agent)\n";
}

static bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                return {};
            }
            return std::string(argv[++i]);
        };

        if (arg == "--prompt") a.prompt = need(arg);
        else if (arg == "--output-csv") a.output_csv = need(arg);
        else if (arg == "--expected-csv") a.expected_csv = need(arg);
        else if (arg == "--keys") a.keys = need(arg);
        else if (arg == "--model") a.model = need(arg);
        else if (arg == "--data") a.data_path = need(arg);
        else if (arg == "--goal") a.goal = need(arg);
        else if (arg == "--python") a.python = need(arg);
        else if (arg == "--agent-module") a.agent_module = need(arg);
        else if (arg == "--best-of-n") a.best_of_n = std::max(1, std::atoi(need(arg).c_str()));
        else if (arg == "--best-of-n-temp-min") a.best_of_n_temp_min = need(arg);
        else if (arg == "--best-of-n-temp-max") a.best_of_n_temp_max = need(arg);
        else if (arg == "--lookup-only") a.lookup_only = true;
        else if (arg == "--no-lookup-only") a.lookup_only = false;
        else if (arg == "--no-compare") a.compare = false;
        else if (arg == "--resultcmp") a.resultcmp_path = need(arg);
        else if (arg == "-h" || arg == "--help") return false;
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }

    if (a.prompt.empty() || a.output_csv.empty()) {
        std::cerr << "Missing required args: --prompt and/or --output-csv\n";
        return false;
    }
    if (a.compare && a.expected_csv.empty()) {
        // If compare is enabled, require expected-csv.
        std::cerr << "Comparison enabled but missing --expected-csv\n";
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) {
        print_usage();
        return 2;
    }

    // Build Python command (mirrors Agent/data_agent.py CLI)
    std::ostringstream py;
    py << a.python << " -m " << a.agent_module << " " << quote_arg(a.prompt);
    if (!a.data_path.empty()) py << " --data " << quote_arg(a.data_path);
    if (!a.goal.empty()) py << " --goal " << quote_arg(a.goal);
    if (!a.model.empty()) py << " --model " << quote_arg(a.model);
    if (a.lookup_only) py << " --lookup-only";
    py << " --output-csv " << quote_arg(a.output_csv);
    if (a.best_of_n > 1) {
        py << " --best-of-n " << a.best_of_n;
        if (!a.best_of_n_temp_min.empty()) py << " --best-of-n-temp-min " << quote_arg(a.best_of_n_temp_min);
        if (!a.best_of_n_temp_max.empty()) py << " --best-of-n-temp-max " << quote_arg(a.best_of_n_temp_max);
    }

    std::cerr << "[orchestrator] Running Python agent:\n" << py.str() << "\n";
    int py_rc = std::system(py.str().c_str());
    if (py_rc != 0) {
        std::cerr << "[orchestrator] Python agent failed (exit code " << py_rc << ")\n";
        return 10;
    }

    if (!a.compare) {
        std::cerr << "[orchestrator] --no-compare set; done.\n";
        return 0;
    }

    // Determine resultcmp path (default: sibling executable in same directory).
    std::filesystem::path resultcmp = a.resultcmp_path.empty()
        ? (std::filesystem::path(argv[0]).parent_path() / ("resultcmp" + exe_suffix()))
        : std::filesystem::path(a.resultcmp_path);

    std::ostringstream cmp;
    cmp << quote_arg(resultcmp.string())
        << " --actual " << quote_arg(a.output_csv)
        << " --expected " << quote_arg(a.expected_csv);
    if (!a.keys.empty()) cmp << " --key " << quote_arg(a.keys);

    std::cerr << "[orchestrator] Running comparator:\n" << cmp.str() << "\n";
    int cmp_rc = std::system(cmp.str().c_str());

    // resultcmp returns 0 if equal, 1 if not equal, else >1 error.
    if (cmp_rc == 0) return 0;
    if (cmp_rc == 1) return 1;
    return 11;
}


