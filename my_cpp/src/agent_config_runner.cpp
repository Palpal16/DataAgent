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
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>

static pid_t phoenix_pid = 0;

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

class SimpleYAML {
public:
    struct Config {
        std::string prompt;
        std::string data_path;
        std::string visualization_goal;
        std::string model;
        std::string ollama_url;
        double temperature;
        std::string agent_mode;
        int best_of_n;
        std::string temperature_max;
        std::string save_dir;
        std::string gt_csv;
        std::string gt_text;
        bool enable_csv_eval;
        std::string csv_eval_method;
        std::string csv_iou_type;
        std::string cpp_evaluator_exe;
        std::string cpp_evaluator_keys;
        bool enable_text_eval;
        std::string text_eval_method;
        bool bleu_use_nltk;
        std::string spice_jar_path;
        std::string spice_java_bin;
        std::string llm_judge_model;
        std::string test_cases_json;
        bool run_batch;
        bool enable_tracing;
        std::string phoenix_endpoint;
        std::string phoenix_project_name;
        std::string phoenix_api_key;
        bool phoenix_auto_start;
        bool enable_codecarbon;
    };

    static Config parse(const std::string& filename) {
        Config cfg;
        cfg.temperature = 0.1;
        cfg.best_of_n = 1;
        cfg.enable_csv_eval = false;
        cfg.enable_text_eval = false;
        cfg.bleu_use_nltk = false;
        cfg.run_batch = false;
        cfg.enable_tracing = false;
        cfg.phoenix_auto_start = true;
        cfg.enable_codecarbon = false;
        cfg.model = "llama3.2:3b";
        cfg.ollama_url = "http://localhost:11434";
        cfg.agent_mode = "analysis";
        cfg.csv_eval_method = "python";
        cfg.csv_iou_type = "rows";
        cfg.text_eval_method = "bleu";
        cfg.spice_java_bin = "java";
        cfg.phoenix_endpoint = "http://localhost:6006/v1/traces";
        cfg.phoenix_project_name = "evaluating-agent";

        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open " << filename << std::endl;
            return cfg;
        }

        std::string line, current_section;
        int base_indent = -1;

        while (std::getline(file, line)) {
            std::string original = line;
            size_t comment_pos = line.find('#');
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }

            line = trim(line);
            if (line.empty()) continue;

            int indent = getIndent(original);

            if (line.find(':') != std::string::npos) {
                auto pos = line.find(':');
                std::string key = trim(line.substr(0, pos));
                std::string value = trim(line.substr(pos + 1));
                value = unquote(value);

                if (base_indent == -1 || indent <= base_indent) {
                    base_indent = indent;
                    current_section = "";
                } else if (indent > base_indent) {
                    if (value.empty() || key == "cpp_evaluator" || key == "bleu" || 
                        key == "spice" || key == "llm_judge" || key == "phoenix") {
                        current_section = key;
                        continue;
                    }

                    if (!current_section.empty()) {
                        key = current_section + "." + key;
                    }
                }

                if (value == "null" || value.empty()) continue;

                if (key == "prompt") cfg.prompt = value;
                else if (key == "data_path") cfg.data_path = value;
                else if (key == "visualization_goal") cfg.visualization_goal = value;
                else if (key == "model") cfg.model = value;
                else if (key == "ollama_url") cfg.ollama_url = value;
                else if (key == "temperature") cfg.temperature = std::stod(value);
                else if (key == "agent_mode") cfg.agent_mode = value;
                else if (key == "best_of_n") cfg.best_of_n = std::stoi(value);
                else if (key == "temperature_max") cfg.temperature_max = value;
                else if (key == "save_dir") cfg.save_dir = value;
                else if (key == "gt_csv") cfg.gt_csv = value;
                else if (key == "gt_text") cfg.gt_text = value;
                else if (key == "enable_csv_eval") cfg.enable_csv_eval = (value == "true");
                else if (key == "csv_eval_method") cfg.csv_eval_method = value;
                else if (key == "csv_iou_type") cfg.csv_iou_type = value;
                else if (key == "cpp_evaluator.executable") cfg.cpp_evaluator_exe = value;
                else if (key == "cpp_evaluator.keys") cfg.cpp_evaluator_keys = value;
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
            }
        }
        return cfg;
    }

private:
    static int getIndent(const std::string& s) {
        int count = 0;
        for (char c : s) {
            if (c == ' ') count++;
            else if (c == '\t') count += 4;
            else break;
        }
        return count;
    }

    static std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        size_t end = s.find_last_not_of(" \t\r\n");
        return s.substr(start, end - start + 1);
    }

    static std::string unquote(const std::string& s) {
        std::string result = trim(s);
        if (result.size() >= 2 && result[0] == '"' && result.back() == '"') {
            return result.substr(1, result.size() - 2);
        }
        return result;
    }
};

class JSONTestCase {
public:
    std::string prompt;
    std::string gt_data;
    std::string gt_analysis;
    std::string gt_sql;
};

class SimpleJSONParser {
public:
    static std::vector<JSONTestCase> parse(const std::string& filename) {
        std::vector<JSONTestCase> cases;
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open " << filename << std::endl;
            return cases;
        }

        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();

        size_t pos = 0;
        while ((pos = content.find("{", pos)) != std::string::npos) {
            size_t end = findMatchingBrace(content, pos);
            if (end == std::string::npos) break;

            std::string obj = content.substr(pos, end - pos + 1);
            JSONTestCase tc;
            tc.prompt = extractValue(obj, "prompt");
            tc.gt_data = extractValue(obj, "gt_data");
            tc.gt_analysis = extractValue(obj, "gt_analysis");
            tc.gt_sql = extractValue(obj, "gt_sql");

            if (!tc.prompt.empty()) {
                cases.push_back(tc);
            }
            pos = end + 1;
        }
        return cases;
    }

private:
    static size_t findMatchingBrace(const std::string& s, size_t start) {
        int depth = 0;
        for (size_t i = start; i < s.size(); ++i) {
            if (s[i] == '{') depth++;
            else if (s[i] == '}') {
                depth--;
                if (depth == 0) return i;
            }
        }
        return std::string::npos;
    }

    static std::string extractValue(const std::string& obj, const std::string& key) {
        std::string searchKey = "\"" + key + "\"";
        size_t pos = obj.find(searchKey);
        if (pos == std::string::npos) return "";

        pos = obj.find(':', pos);
        if (pos == std::string::npos) return "";
        pos++;

        while (pos < obj.size() && (obj[pos] == ' ' || obj[pos] == '\t')) pos++;

        if (pos >= obj.size() || obj[pos] != '"') return "";
        pos++;

        std::string value;
        bool escaped = false;
        while (pos < obj.size()) {
            if (escaped) {
                if (obj[pos] == 'n') value += '\n';
                else if (obj[pos] == 't') value += '\t';
                else if (obj[pos] == 'r') value += '\r';
                else value += obj[pos];
                escaped = false;
            } else if (obj[pos] == '\\') {
                escaped = true;
            } else if (obj[pos] == '"') {
                break;
            } else {
                value += obj[pos];
            }
            pos++;
        }
        return value;
    }
};

static std::string quote_arg(const std::string& s) {
    if (s.empty()) return "\"\"";
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else if (c == '\t') out += "\\t";
        else out.push_back(c);
    }
    out.push_back('"');
    return out;
}

pid_t start_phoenix(const SimpleYAML::Config& cfg) {
    if (!cfg.enable_tracing || !cfg.phoenix_auto_start) {
        return 0;
    }

    std::cout << "[Phoenix] Starting phoenix serve in background..." << std::endl;

    pid_t pid = fork();
    if (pid == 0) {
        setsid();
        freopen("/dev/null", "w", stdout);
        freopen("/dev/null", "w", stderr);
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

std::string build_command(const SimpleYAML::Config& cfg, const std::string& prompt = "",
                         const std::string& gt_csv = "", const std::string& gt_text = "") {
    std::ostringstream cmd;
    cmd << "python -m Agent.data_agent";

    std::string use_prompt = prompt.empty() ? cfg.prompt : prompt;
    std::string use_gt_csv = gt_csv.empty() ? cfg.gt_csv : gt_csv;
    std::string use_gt_text = gt_text.empty() ? cfg.gt_text : gt_text;

    cmd << " " << quote_arg(use_prompt);

    if (!cfg.data_path.empty()) cmd << " --data " << quote_arg(cfg.data_path);
    if (!cfg.visualization_goal.empty()) cmd << " --goal " << quote_arg(cfg.visualization_goal);
    if (!cfg.model.empty()) cmd << " --model " << quote_arg(cfg.model);
    if (!cfg.ollama_url.empty()) cmd << " --ollama_url " << quote_arg(cfg.ollama_url);

    cmd << " --temp " << cfg.temperature;

    if (cfg.agent_mode == "lookup_only") {
        cmd << " --lookup_only";
    } else if (cfg.agent_mode == "analysis") {
        cmd << " --no_vis";
    }

    if (cfg.best_of_n > 1) {
        cmd << " --best_of_n " << cfg.best_of_n;
        if (!cfg.temperature_max.empty()) {
            cmd << " --temp-max " << cfg.temperature_max;
        }
    }

    if (!cfg.save_dir.empty()) cmd << " --save_dir " << quote_arg(cfg.save_dir);
    if (!use_gt_csv.empty()) cmd << " --gt_csv " << quote_arg(use_gt_csv);
    if (!use_gt_text.empty()) cmd << " --gt_text " << quote_arg(use_gt_text);

    if (cfg.enable_csv_eval) {
        if (cfg.csv_eval_method == "python") {
            cmd << " --py_csv_eval";
        } else if (cfg.csv_eval_method == "cpp") {
            cmd << " --cpp_csv_eval";
            if (!cfg.cpp_evaluator_exe.empty())
                cmd << " --evaluator_exe " << quote_arg(cfg.cpp_evaluator_exe);
            if (!cfg.cpp_evaluator_keys.empty())
                cmd << " --eval_keys " << quote_arg(cfg.cpp_evaluator_keys);
        }
        cmd << " --iou_type " << cfg.csv_iou_type;
    }

    if (cfg.enable_text_eval) {
        if (cfg.text_eval_method == "bleu") {
            cmd << " --bleu_text_eval";
            if (cfg.bleu_use_nltk) cmd << " --bleu_nltk";
        } else if (cfg.text_eval_method == "spice") {
            cmd << " --spice_text_eval";
            if (!cfg.spice_jar_path.empty())
                cmd << " --spice_jar " << quote_arg(cfg.spice_jar_path);
            if (!cfg.spice_java_bin.empty())
                cmd << " --spice_java_bin " << quote_arg(cfg.spice_java_bin);
        } else if (cfg.text_eval_method == "llm") {
            cmd << " --llm_text_eval";
            if (!cfg.llm_judge_model.empty())
                cmd << " --llm_judge_model " << quote_arg(cfg.llm_judge_model);
        }
    }

    if (cfg.enable_tracing) {
        cmd << " --enable_tracing";
        if (!cfg.phoenix_endpoint.empty())
            cmd << " --phoenix_endpoint " << quote_arg(cfg.phoenix_endpoint);
        if (!cfg.phoenix_project_name.empty())
            cmd << " --project_name " << quote_arg(cfg.phoenix_project_name);
    }

    if (cfg.enable_codecarbon) {
        cmd << " --enable_codecarbon";
    }

    return cmd.str();
}

int write_temp_file(const std::string& content, std::string& out_path) {
    char temp_template[] = "/tmp/agent_gt_XXXXXX";
    int fd = mkstemp(temp_template);
    if (fd == -1) {
        std::cerr << "Error: Could not create temporary file" << std::endl;
        return -1;
    }

    FILE* f = fdopen(fd, "w");
    if (!f) {
        close(fd);
        return -1;
    }

    fputs(content.c_str(), f);
    fclose(f);

    out_path = temp_template;
    return 0;
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    std::string config_file = "agent_config.yaml";

    if (argc > 1) {
        config_file = argv[1];
    }

    std::cout << "[ConfigRunner] Loading configuration from: " << config_file << std::endl;
    auto cfg = SimpleYAML::parse(config_file);

    phoenix_pid = start_phoenix(cfg);

    int exit_code = 0;

    if (cfg.run_batch && !cfg.test_cases_json.empty()) {
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

            std::string gt_csv_file, gt_text_file;

            if (!test_cases[i].gt_data.empty()) {
                if (write_temp_file(test_cases[i].gt_data, gt_csv_file) != 0) {
                    std::cerr << "[ConfigRunner] Failed to create temp CSV file" << std::endl;
                    fail_count++;
                    continue;
                }
            }

            if (!test_cases[i].gt_analysis.empty()) {
                if (write_temp_file(test_cases[i].gt_analysis, gt_text_file) != 0) {
                    std::cerr << "[ConfigRunner] Failed to create temp text file" << std::endl;
                    if (!gt_csv_file.empty()) remove(gt_csv_file.c_str());
                    fail_count++;
                    continue;
                }
            }

            std::string save_dir = cfg.save_dir;
            if (!save_dir.empty()) {
                save_dir = save_dir + "/test_" + std::to_string(i + 1);
                std::string mkdir_cmd = "mkdir -p " + quote_arg(save_dir);
                system(mkdir_cmd.c_str());
            }

            SimpleYAML::Config test_cfg = cfg;
            test_cfg.save_dir = save_dir;

            std::string cmd = build_command(test_cfg, test_cases[i].prompt, gt_csv_file, gt_text_file);

            std::cout << "[ConfigRunner] Executing: " << cmd << std::endl;
            int result = system(cmd.c_str());

            if (!gt_csv_file.empty()) remove(gt_csv_file.c_str());
            if (!gt_text_file.empty()) remove(gt_text_file.c_str());

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
