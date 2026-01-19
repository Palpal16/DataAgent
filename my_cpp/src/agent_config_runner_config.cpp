#include "agent_config_runner.hpp"

#include "runner_common.hpp"

#include <cstdlib>

namespace agent_config_runner {

Config parse_config_file(const std::string& filename) {
    Config cfg;

#ifdef _WIN32
    cfg.python_bin = "python";
#else
    cfg.python_bin = "python3";
#endif

    runner_common::for_each_yaml_kv(
        filename,
        {"cpp_evaluator", "bleu", "spice", "llm_judge", "phoenix", "sweep"},
        [&](const std::string& key, const std::string& value) {
            if (key == "prompt") cfg.prompt = value;
            else if (key == "data_path") cfg.data_path = value;
            else if (key == "visualization_goal") cfg.visualization_goal = value;
            else if (key == "python_bin") cfg.python_bin = value;
            else if (key == "model") cfg.model = value;
            else if (key == "ollama_url") cfg.ollama_url = value;
            else if (key == "temperature") cfg.temperature = std::stod(value);
            else if (key == "agent_mode") cfg.agent_mode = value;
            else if (key == "best_of_n") cfg.best_of_n = std::stoi(value);
            else if (key == "temperature_max") cfg.temperature_max = value;
            else if (key == "save_dir") cfg.save_dir = value;
            else if (key == "gt_csv") cfg.gt_csv = value;
            else if (key == "gt_text") cfg.gt_text = value;
            else if (key == "gt_visualization") cfg.gt_visualization = value;

            else if (key == "enable_csv_eval") cfg.enable_csv_eval = (value == "true");
            else if (key == "csv_eval_method") cfg.csv_eval_method = value;
            else if (key == "csv_iou_type") cfg.csv_iou_type = value;
            else if (key == "cpp_evaluator.executable") cfg.cpp_evaluator_exe = value;
            else if (key == "cpp_evaluator.keys") cfg.cpp_evaluator_keys = value;
            else if (key == "cpp_evaluator.threads") cfg.cpp_evaluator_threads = std::stoi(value);
            else if (key == "cpp_evaluator.benchmark") cfg.cpp_evaluator_benchmark = (value == "true");
            else if (key == "cpp_evaluator.benchmark_iters") cfg.cpp_evaluator_benchmark_iters = std::stoi(value);

            else if (key == "enable_text_eval") cfg.enable_text_eval = (value == "true");
            else if (key == "text_eval_method") cfg.text_eval_method = value;
            else if (key == "bleu.use_nltk") cfg.bleu_use_nltk = (value == "true");
            else if (key == "spice.jar_path") cfg.spice_jar_path = value;
            else if (key == "spice.java_bin") cfg.spice_java_bin = value;
            else if (key == "llm_judge.model") cfg.llm_judge_model = value;

            else if (key == "test_cases_json") cfg.test_cases_json = value;
            else if (key == "run_batch") cfg.run_batch = (value == "true");

            else if (key == "enable_tracing") cfg.enable_tracing = (value == "true");
            else if (key == "phoenix.endpoint") cfg.phoenix_endpoint = value;
            else if (key == "phoenix.project_name") cfg.phoenix_project_name = value;
            else if (key == "phoenix.api_key") cfg.phoenix_api_key = value;
            else if (key == "phoenix.auto_start") cfg.phoenix_auto_start = (value == "true");

            else if (key == "enable_codecarbon") cfg.enable_codecarbon = (value == "true");

            else if (key == "two_stage_cot") cfg.two_stage_cot = (value == "true");
            else if (key == "cot_max_bullets") cfg.cot_max_bullets = std::stoi(value);
            else if (key == "cot_print_plan") cfg.cot_print_plan = (value == "true");
            else if (key == "cot_store_plan") cfg.cot_store_plan = (value == "true");

            else if (key == "sweep.enabled") cfg.sweep_enabled = (value == "true");
            else if (key == "sweep.best_of_n") cfg.sweep_best_of_n = value;
            else if (key == "sweep.temperature") cfg.sweep_temperature = value;
            else if (key == "sweep.temperature_max") cfg.sweep_temperature_max = value;
            else if (key == "sweep.repetitions") cfg.sweep_repetitions = std::stoi(value);
        }
    );

    return cfg;
}

} // namespace agent_config_runner

