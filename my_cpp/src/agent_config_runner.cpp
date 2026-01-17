// agent_config_runner.cpp
// Loads agent_config.yaml and executes data_agent.py with the specified configuration.
// Supports both single query and batch processing modes.

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <filesystem>
#include <cctype>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <algorithm>
#include <cstdlib>

#include "util.hpp"
#include "runner_common.hpp"
#include "agent_config_runner.hpp"

static pid_t phoenix_pid = 0;
static std::string temp_dir = "/tmp";

static std::string determine_temp_dir(const std::string& python_bin) {
    return runner_common::determine_temp_dir(python_bin);
}

void cleanup_phoenix() {
    if (phoenix_pid > 0) {
        std::cout << "[Cleanup] Stopping Phoenix server (PID: " << phoenix_pid << ")" << std::endl;
        kill(phoenix_pid, SIGTERM);
        waitpid(phoenix_pid, NULL, 0);
        phoenix_pid = 0;
    }
}

void signal_handler(int signum) {
    std::cout << "\n[Signal] Caught signal " << signum << ", cleaning up..." << std::endl;
    cleanup_phoenix();
    exit(signum);
}

using RunnerConfig = agent_config_runner::Config;

class JSONTestCase {
public:
    std::string prompt;
    std::string gt_data;
    std::string gt_analysis;
    std::string gt_sql;
};

class SimpleJSONParser {
public:
    static std::vector<runner_common::JsonTestCase> parse(const std::string& filename) {
        return runner_common::parse_json_test_cases(filename);
    }
};

[[maybe_unused]] static std::string quote_arg(const std::string& s) {
    return runner_common::quote_arg(s);
}

static std::vector<std::string> split_csv_list(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream iss(s);
    while (std::getline(iss, cur, ',')) {
        // trim
        auto is_space = [](unsigned char c){ return std::isspace(c); };
        cur.erase(cur.begin(), std::find_if(cur.begin(), cur.end(), [&](unsigned char c){ return !is_space(c); }));
        cur.erase(std::find_if(cur.rbegin(), cur.rend(), [&](unsigned char c){ return !is_space(c); }).base(), cur.end());
        if (!cur.empty()) out.push_back(cur);
    }
    return out;
}

static std::vector<int> parse_int_list(const std::string& s, int fallback) {
    if (s.empty()) return {fallback};
    std::vector<int> out;
    for (const auto& tok : split_csv_list(s)) {
        try { out.push_back(std::stoi(tok)); } catch (...) {}
    }
    if (out.empty()) out.push_back(fallback);
    return out;
}

static std::vector<double> parse_double_list(const std::string& s, double fallback) {
    if (s.empty()) return {fallback};
    std::vector<double> out;
    for (const auto& tok : split_csv_list(s)) {
        try { out.push_back(std::stod(tok)); } catch (...) {}
    }
    if (out.empty()) out.push_back(fallback);
    return out;
}

static std::vector<std::string> parse_string_list_allow_null(const std::string& s, const std::string& fallback) {
    if (s.empty()) return {fallback};
    auto out = split_csv_list(s);
    if (out.empty()) out.push_back(fallback);
    return out;
}

static bool extract_json_number(const std::string& json_text, const std::string& key, double& out) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = json_text.find(needle);
    if (pos == std::string::npos) return false;
    pos = json_text.find(':', pos);
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < json_text.size() && std::isspace(static_cast<unsigned char>(json_text[pos]))) pos++;
    const char* start = json_text.c_str() + pos;
    char* end = nullptr;
    out = std::strtod(start, &end);
    if (end == start) return false;
    return true;
}

static std::string read_file_to_string(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static std::string csv_escape(const std::string& s);

static void append_sweep_row(
    const std::string& csv_path,
    int sweep_id,
    int repetition,
    int test_case_idx,
    const std::string& prompt,
    int best_of_n,
    double temperature,
    const std::string& temperature_max,
    long long duration_ms,
    const std::string& test_dir
) {
    namespace fs = std::filesystem;
    const bool exists = fs::exists(csv_path);
    std::ofstream out(csv_path, std::ios::binary | std::ios::app);
    if (!out.is_open()) {
        std::cerr << "[ConfigRunner] Warning: failed to write sweep CSV: " << csv_path << std::endl;
        return;
    }
    if (!exists) {
        out << "sweep_id,repetition,test_case,prompt,best_of_n,temperature,temperature_max,"
               "csv_score,bleu,spice,text_score,best_score,duration_ms,total_emissions_kg,test_dir\n";
    }

    double csv_score = 0.0, text_score = 0.0, best_score = 0.0, bleu = 0.0, spice = 0.0, emissions = 0.0;
    const std::string best_path = test_dir + "/best_result.json";
    const std::string best_txt = read_file_to_string(best_path);
    if (!best_txt.empty()) {
        extract_json_number(best_txt, "csv_score", csv_score);
        extract_json_number(best_txt, "text_score", text_score);
        extract_json_number(best_txt, "best_score", best_score);
        extract_json_number(best_txt, "bleu", bleu);
        extract_json_number(best_txt, "spice", spice);
        extract_json_number(best_txt, "total_emissions_kg", emissions);
    }

    out << sweep_id << ","
        << repetition << ","
        << test_case_idx << ","
        << csv_escape(prompt) << ","
        << best_of_n << ","
        << temperature << ","
        << csv_escape(temperature_max) << ","
        << csv_score << ","
        << bleu << ","
        << spice << ","
        << text_score << ","
        << best_score << ","
        << duration_ms << ","
        << emissions << ","
        << csv_escape(test_dir)
        << "\n";
}

pid_t start_phoenix(const RunnerConfig& cfg) {
    if (!cfg.enable_tracing || !cfg.phoenix_auto_start) {
        return 0;
    }

    std::cout << "[Phoenix] Starting phoenix serve in background..." << std::endl;

    pid_t pid = fork();
    if (pid == 0) {
        setsid();
        if (!freopen("/dev/null", "w", stdout)) {
            std::cerr << "[Phoenix] Warning: failed to redirect stdout" << std::endl;
        }
        if (!freopen("/dev/null", "w", stderr)) {
            std::cerr << "[Phoenix] Warning: failed to redirect stderr" << std::endl;
        }
        execl("/bin/sh", "sh", "-c", "phoenix serve", (char*)NULL);
        exit(1);
    } else if (pid > 0) {
        std::cout << "[Phoenix] Started with PID: " << pid << std::endl;
        std::cout << "[Phoenix] Waiting 3 seconds for server to initialize..." << std::endl;
        sleep(3);
        return pid;
    } else {
        std::cerr << "[Phoenix] Failed to start" << std::endl;
        return 0;
    }
}

std::string build_command(const RunnerConfig& cfg, const std::string& prompt = "",
                         const std::string& gt_csv = "", const std::string& gt_text = "") {
    runner_common::AgentCliConfig c;
    c.prompt = cfg.prompt;
    c.data_path = cfg.data_path;
    c.visualization_goal = cfg.visualization_goal;
    c.python_bin = cfg.python_bin;
    c.model = cfg.model;
    c.ollama_url = cfg.ollama_url;
    c.temperature = cfg.temperature;
    c.agent_mode = cfg.agent_mode;
    c.best_of_n = cfg.best_of_n;
    c.temperature_max = cfg.temperature_max;
    c.save_dir = cfg.save_dir;
    c.gt_csv = cfg.gt_csv;
    c.gt_text = cfg.gt_text;
    c.enable_csv_eval = cfg.enable_csv_eval;
    c.csv_eval_method = cfg.csv_eval_method;
    c.csv_iou_type = cfg.csv_iou_type;
    c.cpp_evaluator_exe = cfg.cpp_evaluator_exe;
    c.cpp_evaluator_keys = cfg.cpp_evaluator_keys;
    c.cpp_evaluator_threads = cfg.cpp_evaluator_threads;
    c.cpp_evaluator_benchmark = cfg.cpp_evaluator_benchmark;
    c.cpp_evaluator_benchmark_iters = cfg.cpp_evaluator_benchmark_iters;
    c.enable_text_eval = cfg.enable_text_eval;
    c.text_eval_method = cfg.text_eval_method;
    c.bleu_use_nltk = cfg.bleu_use_nltk;
    c.spice_jar_path = cfg.spice_jar_path;
    c.spice_java_bin = cfg.spice_java_bin;
    c.llm_judge_model = cfg.llm_judge_model;
    c.enable_tracing = cfg.enable_tracing;
    c.phoenix_endpoint = cfg.phoenix_endpoint;
    c.phoenix_project_name = cfg.phoenix_project_name;
    c.enable_codecarbon = cfg.enable_codecarbon;
    c.two_stage_cot = cfg.two_stage_cot;
    c.cot_max_bullets = cfg.cot_max_bullets;
    c.cot_print_plan = cfg.cot_print_plan;
    c.cot_store_plan = cfg.cot_store_plan;
    return runner_common::build_agent_command(c, prompt, gt_csv, gt_text);
}

int write_file(const std::string& filepath, const std::string& content) {
    return runner_common::write_file(filepath, content);
}

// Convert a simple text table (like pandas printout) into CSV rows.
// This mirrors the project’s Python helper (text_to_csv) but stays dependency-free.
static std::vector<std::vector<std::string>> text_table_to_rows(const std::string& text) {
    return runner_common::text_table_to_rows(text);
}

static int write_csv_from_rows(const std::string& filepath, const std::vector<std::vector<std::string>>& rows) {
    return runner_common::write_csv_from_rows(filepath, rows);
}

static std::string csv_escape(const std::string& s);

static std::string csv_escape(const std::string& s) {
    // Wrap in quotes if it contains special chars, and escape quotes by doubling them.
    bool needs_quotes = false;
    for (char c : s) {
        if (c == ',' || c == '"' || c == '\n' || c == '\r') { needs_quotes = true; break; }
    }
    if (!needs_quotes) return s;
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out.push_back(c);
    }
    out.push_back('"');
    return out;
}

static std::vector<std::string> read_all_lines(const std::string& path) {
    std::vector<std::string> lines;
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return lines;
    std::string line;
    while (std::getline(in, line)) lines.push_back(line);
    return lines;
}

static void append_codecarbon_to_batch_summary(
    const std::string& summary_csv,
    const std::string& emissions_csv,
    int test_case_idx,
    const std::string& prompt
) {
    namespace fs = std::filesystem;
    if (!fs::exists(emissions_csv)) return;

    auto lines = read_all_lines(emissions_csv);
    if (lines.size() < 2) return; // header + at least one row

    const std::string header = lines[0];
    const bool summary_exists = fs::exists(summary_csv);

    std::ofstream out(summary_csv, std::ios::binary | std::ios::app);
    if (!out.is_open()) {
        std::cerr << "[ConfigRunner] Warning: failed to open CodeCarbon summary: " << summary_csv << std::endl;
        return;
    }

    if (!summary_exists) {
        out << "test_case,prompt," << header << "\n";
    }

    const std::string tc = std::to_string(test_case_idx);
    const std::string pr = csv_escape(prompt);
    for (size_t i = 1; i < lines.size(); ++i) {
        const std::string& row = lines[i];
        if (row.empty()) continue;
        out << tc << "," << pr << "," << row << "\n";
    }
}

static void append_timing_to_batch_summary(
    const std::string& summary_csv,
    int test_case_idx,
    const std::string& prompt,
    long long duration_ms,
    int exit_code
) {
    namespace fs = std::filesystem;
    const bool summary_exists = fs::exists(summary_csv);
    std::ofstream out(summary_csv, std::ios::binary | std::ios::app);
    if (!out.is_open()) {
        std::cerr << "[ConfigRunner] Warning: failed to open timing summary: " << summary_csv << std::endl;
        return;
    }
    if (!summary_exists) {
        out << "test_case,prompt,duration_ms,exit_code\n";
    }
    out << test_case_idx << "," << csv_escape(prompt) << "," << duration_ms << "," << exit_code << "\n";
}

static void write_per_test_timing_json(
    const std::string& save_dir,
    const std::string& prompt,
    long long duration_ms,
    int exit_code
) {
    const std::string path = save_dir + "/timing.json";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "[ConfigRunner] Warning: failed to write timing.json in " << save_dir << std::endl;
        return;
    }
    // Minimal JSON escaping for quotes/backslashes/newlines
    auto json_escape = [](const std::string& s) -> std::string {
        std::string o;
        o.reserve(s.size() + 8);
        for (char c : s) {
            switch (c) {
                case '\\': o += "\\\\"; break;
                case '"':  o += "\\\""; break;
                case '\n': o += "\\n"; break;
                case '\r': o += "\\r"; break;
                case '\t': o += "\\t"; break;
                default:   o.push_back(c); break;
            }
        }
        return o;
    };

    out
        << "{\n"
        << "  \"prompt\": \"" << json_escape(prompt) << "\",\n"
        << "  \"duration_ms\": " << duration_ms << ",\n"
        << "  \"exit_code\": " << exit_code << "\n"
        << "}\n";
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    std::string config_file = "agent_config.yaml";

    if (argc > 1) {
        config_file = argv[1];
    }

    std::cout << "[ConfigRunner] Loading configuration from: " << config_file << std::endl;
    auto cfg = agent_config_runner::parse_config_file(config_file);
    temp_dir = determine_temp_dir(cfg.python_bin);

    phoenix_pid = start_phoenix(cfg);

    int exit_code = 0;

    if (cfg.sweep_enabled && cfg.run_batch && !cfg.test_cases_json.empty()) {
        std::cout << "[ConfigRunner] Running in sweep mode" << std::endl;
        auto test_cases = SimpleJSONParser::parse(cfg.test_cases_json);
        std::cout << "[ConfigRunner] Loaded " << test_cases.size() << " test cases" << std::endl;

        const auto bon_list = parse_int_list(cfg.sweep_best_of_n, cfg.best_of_n);
        const auto temp_list = parse_double_list(cfg.sweep_temperature, cfg.temperature);
        const auto tmax_list = parse_string_list_allow_null(cfg.sweep_temperature_max, cfg.temperature_max);
        const int reps = std::max(1, cfg.sweep_repetitions);

        const std::string sweep_csv = (cfg.save_dir.empty() ? "./output" : cfg.save_dir) + "/sweep_results.csv";

        int sweep_id = 0;
        for (int rep = 1; rep <= reps; ++rep) {
            for (int bon : bon_list) {
                for (double t : temp_list) {
                    for (const auto& tmax : tmax_list) {
                        ++sweep_id;
                        std::cout << "\n" << std::string(80, '=') << std::endl;
                        std::cout << "[Sweep] setting " << sweep_id
                                  << " | rep=" << rep
                                  << " | best_of_n=" << bon
                                  << " | temp=" << t
                                  << " | temp_max=" << (tmax.empty() ? "null" : tmax)
                                  << std::endl;

                        // For each sweep setting, create a dedicated folder to avoid overwriting outputs.
                        const std::string sweep_root = (cfg.save_dir.empty() ? "./output" : cfg.save_dir) + "/sweep_" + std::to_string(sweep_id);
                        std::filesystem::create_directories(sweep_root);

                        for (size_t i = 0; i < test_cases.size(); ++i) {
                            std::string test_dir = sweep_root + "/test_" + std::to_string(i + 1);
                            std::filesystem::create_directories(test_dir);

                            // Write per-test GT artifacts
                            std::string gt_csv_file, gt_text_file;
                            if (!test_cases[i].gt_data.empty()) {
                                gt_csv_file = test_dir + "/gt_data.csv";
                                auto rows = text_table_to_rows(test_cases[i].gt_data);
                                if (!rows.empty()) write_csv_from_rows(gt_csv_file, rows);
                                else write_file(gt_csv_file, test_cases[i].gt_data);
                            }
                            if (!test_cases[i].gt_analysis.empty()) {
                                gt_text_file = test_dir + "/gt_analysis.txt";
                                write_file(gt_text_file, test_cases[i].gt_analysis);
                            }

                            RunnerConfig run_cfg = cfg;
                            run_cfg.save_dir = test_dir;
                            run_cfg.best_of_n = bon;
                            run_cfg.temperature = t;
                            run_cfg.temperature_max = (tmax == "null" ? "" : tmax);

                            const std::string cmd = build_command(run_cfg, test_cases[i].prompt, gt_csv_file, gt_text_file);
                            std::cout << "[Sweep] Executing: " << cmd << std::endl;
                            const auto t0 = std::chrono::steady_clock::now();
                            int rc = system(cmd.c_str());
                            const auto t1 = std::chrono::steady_clock::now();
                            const long long duration_ms =
                                std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

                            // Aggregate per-setting results into a single CSV
                            append_sweep_row(
                                sweep_csv,
                                sweep_id,
                                rep,
                                static_cast<int>(i + 1),
                                test_cases[i].prompt,
                                bon,
                                t,
                                (tmax == "null" ? "" : tmax),
                                duration_ms,
                                test_dir
                            );

                            // Also aggregate timing + CodeCarbon summaries as before
                            {
                                const std::string times_csv = (cfg.save_dir.empty() ? "./output" : cfg.save_dir) + "/batch_times.csv";
                                write_per_test_timing_json(test_dir, test_cases[i].prompt, duration_ms, rc);
                                append_timing_to_batch_summary(times_csv, static_cast<int>(i + 1), test_cases[i].prompt, duration_ms, rc);
                            }
                            if (cfg.enable_codecarbon) {
                                const std::string emissions_csv = test_dir + "/codecarbon/emissions.csv";
                                const std::string summary_csv = (cfg.save_dir.empty() ? "./output" : cfg.save_dir) + "/codecarbon_batch.csv";
                                append_codecarbon_to_batch_summary(summary_csv, emissions_csv, static_cast<int>(i + 1), test_cases[i].prompt);
                            }
                        }
                    }
                }
            }
        }

        cleanup_phoenix();
        return 0;

    } else if (cfg.run_batch && !cfg.test_cases_json.empty()) {
        std::cout << "[ConfigRunner] Running in batch mode" << std::endl;
        auto test_cases = SimpleJSONParser::parse(cfg.test_cases_json);
        std::cout << "[ConfigRunner] Loaded " << test_cases.size() << " test cases" << std::endl;

        int success_count = 0;
        int fail_count = 0;

        for (size_t i = 0; i < test_cases.size(); ++i) {
            std::cout << "\n" << std::string(80, '=') << std::endl;
            std::cout << "[ConfigRunner] Test case " << (i + 1) << "/" << test_cases.size() << std::endl;
            std::cout << "[ConfigRunner] Prompt: " << test_cases[i].prompt.substr(0, 60)
                      << (test_cases[i].prompt.size() > 60 ? "..." : "") << std::endl;

            std::string save_dir = cfg.save_dir;
            if (save_dir.empty()) {
                save_dir = "./output";  // Use current directory as fallback
            }
            save_dir = save_dir + "/test_" + std::to_string(i + 1);

            // Create directory (portable, no shell dependency)
            try {
                std::filesystem::create_directories(save_dir);
            } catch (const std::exception& e) {
                std::cerr << "[ConfigRunner] Failed to create directory: " << save_dir << std::endl;
                std::cerr << "[ConfigRunner] Reason: " << e.what() << std::endl;
                fail_count++;
                continue;  // Skip this test case
            }

            std::string gt_csv_file, gt_text_file;

            if (!test_cases[i].gt_data.empty()) {
                gt_csv_file = save_dir + "/gt_data.csv";
                auto rows = text_table_to_rows(test_cases[i].gt_data);
                if (!rows.empty()) {
                    if (write_csv_from_rows(gt_csv_file, rows) != 0) {
                        std::cerr << "[ConfigRunner] Failed to create gt CSV file" << std::endl;
                        fail_count++;
                        continue;
                    }
                } else if (write_file(gt_csv_file, test_cases[i].gt_data)  != 0) {
                    std::cerr << "[ConfigRunner] Failed to create temp CSV file" << std::endl;
                    fail_count++;
                    continue;
                }
            }

            if (!test_cases[i].gt_analysis.empty()) {
                gt_text_file = save_dir + "/gt_analysis.txt";
                if (write_file(gt_text_file, test_cases[i].gt_analysis)!= 0) {
                    std::cerr << "[ConfigRunner] Failed to create temp text file" << std::endl;
                    fail_count++;
                    continue;
                }
            }

            RunnerConfig test_cfg = cfg;
            test_cfg.save_dir = save_dir;

            std::string cmd = build_command(test_cfg, test_cases[i].prompt, gt_csv_file, gt_text_file);

            std::cout << "[ConfigRunner] Executing: " << cmd << std::endl;
            long long duration_ms = 0;
            int result = 0;
            {
                util::ScopedTimer<long long> timer(duration_ms);
                result = system(cmd.c_str());
            }

            // Save per-test timing + aggregate batch timing summary
            {
                const std::string times_csv = (cfg.save_dir.empty() ? "./output" : cfg.save_dir) + "/batch_times.csv";
                write_per_test_timing_json(save_dir, test_cases[i].prompt, duration_ms, result);
                append_timing_to_batch_summary(times_csv, static_cast<int>(i + 1), test_cases[i].prompt, duration_ms, result);
            }

            // If CodeCarbon is enabled, the Python agent writes:
            //   <save_dir>/codecarbon/emissions.csv
            // We also aggregate all per-test emissions into a batch summary CSV.
            if (cfg.enable_codecarbon) {
                const std::string emissions_csv = save_dir + "/codecarbon/emissions.csv";
                const std::string summary_csv = (cfg.save_dir.empty() ? "./output" : cfg.save_dir) + "/codecarbon_batch.csv";
                append_codecarbon_to_batch_summary(summary_csv, emissions_csv, static_cast<int>(i + 1), test_cases[i].prompt);
            }

            // Keep ground-truth artifacts in each test folder for reproducibility.
            // (Previously this was treated as a temp file and removed.)

            if (result == 0) {
                std::cout << "[ConfigRunner] Test case " << (i + 1) << " PASSED" << std::endl;
                success_count++;
            } else {
                std::cout << "[ConfigRunner] Test case " << (i + 1) << " FAILED (exit code: "
                          << result << ")" << std::endl;
                fail_count++;
            }
        }

        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "[ConfigRunner] Batch processing complete" << std::endl;
        std::cout << "[ConfigRunner] Success: " << success_count << " / " << test_cases.size() << std::endl;
        std::cout << "[ConfigRunner] Failed: " << fail_count << " / " << test_cases.size() << std::endl;

        exit_code = (fail_count > 0) ? 1 : 0;

    } else {
        std::cout << "[ConfigRunner] Running in single query mode" << std::endl;

        if (cfg.prompt.empty()) {
            std::cerr << "Error: No prompt specified in config" << std::endl;
            cleanup_phoenix();
            return 2;
        }

        std::string cmd = build_command(cfg);
        std::cout << "[ConfigRunner] Executing: " << cmd << std::endl;

        int result = system(cmd.c_str());

        if (result == 0) {
            std::cout << "[ConfigRunner] Execution completed successfully" << std::endl;
        } else {
            std::cout << "[ConfigRunner] Execution failed with exit code: " << result << std::endl;
        }

        exit_code = result;
    }

    cleanup_phoenix();
    return exit_code;
}
