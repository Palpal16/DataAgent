#pragma once

#include <string>

namespace agent_config_runner {

// Configuration loaded from agent_config.yaml (for the direct/sweep runner).
struct Config {
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
    std::string gt_visualization;

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

    std::string test_cases_json;
    bool run_batch = false;

    bool enable_tracing = false;
    std::string phoenix_endpoint = "http://localhost:6006/v1/traces";
    std::string phoenix_project_name = "evaluating-agent";
    std::string phoenix_api_key;
    bool phoenix_auto_start = true;

    bool enable_codecarbon = false;

    // Two-stage plan -> final
    bool two_stage_cot = false;
    int cot_max_bullets = 8;
    bool cot_print_plan = false;
    bool cot_store_plan = false;

    // Sweep (parameter grid)
    bool sweep_enabled = false;
    std::string sweep_best_of_n;       // e.g. "1,2,3,5"
    std::string sweep_temperature;     // e.g. "0.0,0.1"
    std::string sweep_temperature_max; // e.g. "0.6" (or "null")
    int sweep_repetitions = 1;
};

// Parse a YAML file into Config (implementation in agent_config_runner_config.cpp).
Config parse_config_file(const std::string& filename);

} // namespace agent_config_runner

