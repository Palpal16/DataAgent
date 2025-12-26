
// agent_orchestrator: run Python agent to generate CSV, then compare with resultcmp.
//
// Typical usage (from repo root):
//   cpp_evaluator/build/agent_orchestrator.exe ^
//     --prompt "What were the sales in November 2021?" ^
//     --save-dir "results/" ^
//     --expected-csv "results/real_sales_november_2021.csv" ^
//     --keys "week,store_id"
//
// Notes:
// - This tool shells out to Python (`python -m Agent.data_agent ...`) and then shells out to `resultcmp`.
// - It does NOT embed Python; it expects your environment can run the Python module.
// - The new version uses --save-dir instead of --output-csv (CSV is auto-saved as run_data.csv)

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
    bool no_vis = false;
    std::string save_dir;
    std::string output_csv;  // For backward compatibility, will be derived from save_dir
    int best_of_n = 1;
    std::string temp;
    std::string temp_max;

    // Evaluation options
    bool compare = true;
    std::string expected_csv;
    std::string expected_text;
    std::string keys;  // comma-separated
    std::string resultcmp_path;  // optional override

    // CSV evaluation
    bool py_csv_eval = false;
    bool cpp_csv_eval = false;
    std::string evaluator_exe;
    std::string eval_keys;
    std::string iou_type = "rows";

    // Text evaluation
    bool bleu_text_eval = false;
    bool spice_text_eval = false;
    bool llm_text_eval = false;
    bool bleu_nltk = false;
    std::string spice_jar;
    std::string spice_java_bin = "java";

    // Tracing and tracking
    bool enable_tracing = false;
    std::string phoenix_endpoint;
    std::string project_name = "evaluating-agent";
    bool enable_codecarbon = false;
};

static void print_usage() {
    std::cerr
        << "agent_orchestrator\n"
        << "Runs Python agent -> writes CSV -> (optionally) compares with resultcmp.\n\n"
        << "Required:\n"
        << "  --prompt <text>\n"
        << "  --save-dir <path>         (CSV will be saved as <path>/run_data.csv)\n"
        << "\n"
        << "Optional (agent behavior):\n"
        << "  --model <name>            (e.g., llama3.2:3b)\n"
        << "  --data <path>             Path to parquet file\n"
        << "  --goal <text>             Visualization goal\n"
        << "  --lookup-only             Only run data lookup (default)\n"
        << "  --no-lookup-only          Run full agent pipeline\n"
        << "  --no-vis                  Run without visualization\n"
        << "\n"
        << "Optional (best-of-n):\n"
        << "  --best-of-n <N>           Run N times and pick best\n"
        << "  --temp <float>            Base/minimum temperature\n"
        << "  --temp-max <float>        Maximum temperature for best-of-n\n"
        << "\n"
        << "Optional (evaluation - CSV):\n"
        << "  --expected-csv <path>     Ground truth CSV for comparison\n"
        << "  --py-csv-eval             Use Python IoU evaluator\n"
        << "  --cpp-csv-eval            Use C++ evaluator\n"
        << "  --evaluator-exe <path>    Path to C++ evaluator executable\n"
        << "  --eval-keys <keys>        Comma-separated keys for evaluator\n"
        << "  --iou-type <type>         IoU type: rows, columns, or table (default: rows)\n"
        << "\n"
        << "Optional (evaluation - text):\n"
        << "  --expected-text <path>    Ground truth text file for analysis evaluation\n"
        << "  --bleu-text-eval          Use BLEU for text evaluation\n"
        << "  --bleu-nltk               Use NLTK BLEU implementation\n"
        << "  --spice-text-eval         Use SPICE for text evaluation\n"
        << "  --spice-jar <path>        Path to SPICE jar\n"
        << "  --spice-java-bin <path>   Java executable (default: java)\n"
        << "  --llm-text-eval           Use LLM for text evaluation\n"
        << "\n"
        << "Optional (comparison with resultcmp):\n"
        << "  --keys <keys>             Comma-separated keys (passed to resultcmp as --key)\n"
        << "  --resultcmp <path>        Path to resultcmp executable\n"
        << "  --no-compare              Skip resultcmp comparison\n"
        << "\n"
        << "Optional (tracing/tracking):\n"
        << "  --enable-tracing          Enable Phoenix tracing\n"
        << "  --phoenix-endpoint <url>  Phoenix endpoint URL\n"
        << "  --project-name <name>     Phoenix project name\n"
        << "  --enable-codecarbon       Enable CodeCarbon energy tracking\n"
        << "\n"
        << "Optional (python):\n"
        << "  --python <cmd>            Python command (default: python)\n"
        << "  --agent-module <name>     Agent module name (default: Agent.data_agent)\n";
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

        // Core arguments
        if (arg == "--prompt") a.prompt = need(arg);
        else if (arg == "--save-dir") a.save_dir = need(arg);
        else if (arg == "--output-csv") a.output_csv = need(arg);  // Legacy support

        // Agent configuration
        else if (arg == "--model") a.model = need(arg);
        else if (arg == "--data") a.data_path = need(arg);
        else if (arg == "--goal") a.goal = need(arg);
        else if (arg == "--python") a.python = need(arg);
        else if (arg == "--agent-module") a.agent_module = need(arg);

        // Agent behavior
        else if (arg == "--lookup-only") a.lookup_only = true;
        else if (arg == "--no-lookup-only") a.lookup_only = false;
        else if (arg == "--no-vis") a.no_vis = true;

        // Best-of-n
        else if (arg == "--best-of-n") a.best_of_n = std::max(1, std::atoi(need(arg).c_str()));
        else if (arg == "--temp") a.temp = need(arg);
        else if (arg == "--temp-max") a.temp_max = need(arg);

        // Legacy temperature arguments (for backward compatibility)
        else if (arg == "--best-of-n-temp-min") a.temp = need(arg);
        else if (arg == "--best-of-n-temp-max") a.temp_max = need(arg);

        // Ground truth
        else if (arg == "--expected-csv") a.expected_csv = need(arg);
        else if (arg == "--expected-text") a.expected_text = need(arg);

        // CSV evaluation
        else if (arg == "--py-csv-eval") a.py_csv_eval = true;
        else if (arg == "--cpp-csv-eval") a.cpp_csv_eval = true;
        else if (arg == "--evaluator-exe") a.evaluator_exe = need(arg);
        else if (arg == "--eval-keys") a.eval_keys = need(arg);
        else if (arg == "--iou-type") a.iou_type = need(arg);

        // Text evaluation
        else if (arg == "--bleu-text-eval") a.bleu_text_eval = true;
        else if (arg == "--bleu-nltk") a.bleu_nltk = true;
        else if (arg == "--spice-text-eval") a.spice_text_eval = true;
        else if (arg == "--spice-jar") a.spice_jar = need(arg);
        else if (arg == "--spice-java-bin") a.spice_java_bin = need(arg);
        else if (arg == "--llm-text-eval") a.llm_text_eval = true;

        // Comparison
        else if (arg == "--keys") a.keys = need(arg);
        else if (arg == "--resultcmp") a.resultcmp_path = need(arg);
        else if (arg == "--no-compare") a.compare = false;

        // Tracing/tracking
        else if (arg == "--enable-tracing") a.enable_tracing = true;
        else if (arg == "--phoenix-endpoint") a.phoenix_endpoint = need(arg);
        else if (arg == "--project-name") a.project_name = need(arg);
        else if (arg == "--enable-codecarbon") a.enable_codecarbon = true;

        // Help
        else if (arg == "-h" || arg == "--help") return false;
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }

    if (a.prompt.empty()) {
        std::cerr << "Missing required arg: --prompt\n";
        return false;
    }

    if (a.save_dir.empty() && a.output_csv.empty()) {
        std::cerr << "Missing required arg: --save-dir (or legacy --output-csv)\n";
        return false;
    }

    // If output_csv was given (legacy), derive save_dir
    if (!a.output_csv.empty() && a.save_dir.empty()) {
        a.save_dir = std::filesystem::path(a.output_csv).parent_path().string();
        if (a.save_dir.empty()) a.save_dir = ".";
    }

    // Derive output_csv from save_dir if not explicitly set
    if (a.output_csv.empty()) {
        a.output_csv = (std::filesystem::path(a.save_dir) / "run_data.csv").string();
    }

    if (a.compare && a.expected_csv.empty() && !a.py_csv_eval && !a.cpp_csv_eval) {
        // If compare is enabled but no evaluation method specified, just use resultcmp
        // This is fine, we'll skip agent evaluation but still do resultcmp
    }

    return true;
}

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) {
        print_usage();
        return 2;
    }

    // Build Python command (mirrors new Agent/data_agent.py CLI)
    std::ostringstream py;
    py << a.python << " -m " << a.agent_module << " " << quote_arg(a.prompt);

    // Core arguments
    if (!a.data_path.empty()) py << " --data " << quote_arg(a.data_path);
    if (!a.goal.empty()) py << " --goal " << quote_arg(a.goal);
    if (!a.model.empty()) py << " --model " << quote_arg(a.model);
    py << " --save_dir " << quote_arg(a.save_dir);

    // Agent behavior
    if (a.lookup_only) py << " --lookup_only";
    if (a.no_vis) py << " --no_vis";

    // Best-of-n
    if (a.best_of_n > 1) {
        py << " --best_of_n " << a.best_of_n;
        if (!a.temp.empty()) py << " --temp " << quote_arg(a.temp);
        if (!a.temp_max.empty()) py << " --temp-max " << quote_arg(a.temp_max);
    }

    // Ground truth
    if (!a.expected_csv.empty()) py << " --gt_csv " << quote_arg(a.expected_csv);
    if (!a.expected_text.empty()) py << " --gt_text " << quote_arg(a.expected_text);

    // CSV evaluation
    if (a.py_csv_eval) py << " --py_csv_eval";
    if (a.cpp_csv_eval) {
        py << " --cpp_csv_eval";
        if (!a.evaluator_exe.empty()) py << " --evaluator_exe " << quote_arg(a.evaluator_exe);
        if (!a.eval_keys.empty()) py << " --eval_keys " << quote_arg(a.eval_keys);
    }
    if (!a.iou_type.empty()) py << " --iou_type " << quote_arg(a.iou_type);

    // Text evaluation
    if (a.bleu_text_eval) py << " --bleu_text_eval";
    if (a.bleu_nltk) py << " --bleu_nltk";
    if (a.spice_text_eval) {
        py << " --spice_text_eval";
        if (!a.spice_jar.empty()) py << " --spice_jar " << quote_arg(a.spice_jar);
        if (!a.spice_java_bin.empty()) py << " --spice_java_bin " << quote_arg(a.spice_java_bin);
    }
    if (a.llm_text_eval) py << " --llm_text_eval";

    // Tracing/tracking
    if (a.enable_tracing) {
        py << " --enable_tracing";
        if (!a.phoenix_endpoint.empty()) py << " --phoenix_endpoint " << quote_arg(a.phoenix_endpoint);
        if (!a.project_name.empty()) py << " --project_name " << quote_arg(a.project_name);
    }
    if (a.enable_codecarbon) py << " --enable_codecarbon";

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

    // Optional: run resultcmp for additional verification (if expected_csv provided)
    if (!a.expected_csv.empty() && !a.resultcmp_path.empty()) {
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
        if (cmp_rc == 0) {
            std::cerr << "[orchestrator] resultcmp: CSVs are equal\n";
            return 0;
        }
        if (cmp_rc == 1) {
            std::cerr << "[orchestrator] resultcmp: CSVs differ\n";
            return 1;
        }
        std::cerr << "[orchestrator] resultcmp failed with error code " << cmp_rc << "\n";
        return 11;
    }

    std::cerr << "[orchestrator] Done (no resultcmp comparison requested)\n";
    return 0;
}
