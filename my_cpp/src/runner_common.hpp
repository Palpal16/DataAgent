#pragma once

#include <functional>
#include <string>
#include <vector>

namespace runner_common {

// -----------------------------
// Simple YAML (flat + 1-level sections)
// -----------------------------

using YamlCallback = std::function<void(const std::string& key, const std::string& value)>;

// Iterate key/value pairs from a very small YAML subset used by this project.
// Supports:
// - "key: value"
// - "section:" followed by indented children "child: value" -> emits "section.child"
// - strips comments (# ...)
// - ignores empty values and "null"
void for_each_yaml_kv(
    const std::string& filename,
    const std::vector<std::string>& section_headers,
    const YamlCallback& on_kv
);

// -----------------------------
// JSON test cases
// -----------------------------

struct JsonTestCase {
    std::string prompt;
    std::string gt_data;
    std::string gt_analysis;
    std::string gt_vis_json;
    std::string gt_sql;
    std::string difficulty;
    bool visualization = false;
};

std::vector<JsonTestCase> parse_json_test_cases(const std::string& filename);

// -----------------------------
// File + string helpers
// -----------------------------

std::string determine_temp_dir(const std::string& python_bin);
std::string quote_arg(const std::string& s);
std::string replace_all(std::string str, const std::string& from, const std::string& to);

int write_file(const std::string& filepath, const std::string& content);

// Convert a simple text table (like pandas printout) into CSV rows.
std::vector<std::vector<std::string>> text_table_to_rows(const std::string& text);
int write_csv_from_rows(const std::string& path, const std::vector<std::vector<std::string>>& rows);

// -----------------------------
// CLI command builder (Agent.data_agent)
// -----------------------------

struct AgentCliConfig {
    std::string prompt;
    std::string data_path;
    std::string visualization_goal;
    std::string python_bin;
    std::string model;
    std::string ollama_url;
    double temperature = 0.1;
    std::string agent_mode = "analysis";
    int best_of_n = 1;
    std::string temperature_max;
    std::string save_dir;
    std::string gt_csv;
    std::string gt_text;
    std::string gt_vis;

    bool enable_csv_eval = false;
    std::string csv_eval_method = "python";
    std::string csv_iou_type = "rows";
    std::string cpp_evaluator_exe;
    std::string cpp_evaluator_keys;
    int cpp_evaluator_threads = 0;
    bool cpp_evaluator_benchmark = false;
    int cpp_evaluator_benchmark_iters = 3;

    bool enable_text_eval = false;
    std::string text_eval_method = "bleu";
    bool bleu_use_nltk = false;
    std::string spice_jar_path;
    std::string spice_java_bin = "java";
    std::string llm_judge_model;

    bool enable_tracing = false;
    std::string phoenix_endpoint = "http://localhost:6006/v1/traces";
    std::string phoenix_project_name = "evaluating-agent";

    bool enable_codecarbon = false;

    // Two-stage plan -> final
    bool two_stage_cot = false;
    int cot_max_bullets = 8;
    bool cot_print_plan = false;
    bool cot_store_plan = false;

    // Output shaping
    bool emit_viz_placeholders = false;
};

std::string build_agent_command(
    const AgentCliConfig& cfg,
    const std::string& prompt_override = "",
    const std::string& gt_csv_override = "",
    const std::string& gt_text_override = "",
    const std::string& gt_vis_override = ""
);

} // namespace runner_common

