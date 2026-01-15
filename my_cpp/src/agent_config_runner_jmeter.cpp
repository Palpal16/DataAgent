// agent_config_runner.cpp
// Loads agent_config.yaml and executes data_agent.py with the specified configuration.
// Supports both single query and batch processing modes, plus JMeter integration.

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
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>

static pid_t phoenix_pid = 0;
static pid_t api_pid = 0;
static std::string temp_dir = "/tmp";

static std::string determine_temp_dir(const std::string& python_bin) {
    if (!python_bin.empty()) {
        if (python_bin.find(".exe") != std::string::npos || python_bin.find("/mnt/") != std::string::npos) {
            return "./tmp";
        }
    }
    return "/tmp";
}

static std::string make_temp_path(const std::string& dir, const std::string& prefix, const std::string& suffix) {
    static int counter = 0;
    auto ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::ostringstream path;
    path << dir;
    if (!dir.empty() && dir.back() != '/') {
        path << "/";
    }
    path << prefix << "_" << ts << "_" << counter++ << suffix;
    return path.str();
}

void cleanup_processes() {
    if (api_pid > 0) {
        std::cout << "[Cleanup] Stopping API server (PID: " << api_pid << ")" << std::endl;
        kill(api_pid, SIGTERM);
        waitpid(api_pid, NULL, 0);
        api_pid = 0;
    }
    if (phoenix_pid > 0) {
        std::cout << "[Cleanup] Stopping Phoenix server (PID: " << phoenix_pid << ")" << std::endl;
        kill(phoenix_pid, SIGTERM);
        waitpid(phoenix_pid, NULL, 0);
        phoenix_pid = 0;
    }
}

void signal_handler(int signum) {
    std::cout << "\n[Signal] Caught signal " << signum << ", cleaning up..." << std::endl;
    cleanup_processes();
    exit(signum);
}

class SimpleYAML {
public:
    struct Config {
        std::string prompt;
        std::string data_path;
        std::string visualization_goal;
        std::string python_bin;
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
        bool use_jmeter;
        std::string jmeter_bin;
        std::string jmx_folder;
        std::string jmeter_output_folder;
        std::string api_host;
        int api_port;
        bool jmeter_auto_start_api;
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
        cfg.use_jmeter = false;
        cfg.jmeter_auto_start_api = true;
        cfg.model = "llama3.2:3b";
        cfg.ollama_url = "http://localhost:11434";
        cfg.agent_mode = "analysis";
        cfg.csv_eval_method = "python";
        cfg.csv_iou_type = "rows";
        cfg.text_eval_method = "bleu";
        cfg.spice_java_bin = "java";
        cfg.phoenix_endpoint = "http://localhost:6006/v1/traces";
        cfg.phoenix_project_name = "evaluating-agent";
#ifdef _WIN32
        cfg.python_bin = "python";
#else
        cfg.python_bin = "python3";
#endif
        cfg.jmeter_bin = "./apache-jmeter-5.6.3/bin/jmeter";
        cfg.jmx_folder = "./jmx_files";
        cfg.jmeter_output_folder = "./output";
        cfg.api_host = "localhost";
        cfg.api_port = 5001;

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
                        key == "spice" || key == "llm_judge" || key == "phoenix" || key == "jmeter") {
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
                else if (key == "use_jmeter") cfg.use_jmeter = (value == "true");
                else if (key == "jmeter.bin") cfg.jmeter_bin = value;
                else if (key == "jmeter.jmx_folder") cfg.jmx_folder = value;
                else if (key == "jmeter.output_folder") cfg.jmeter_output_folder = value;
                else if (key == "jmeter.api_host") cfg.api_host = value;
                else if (key == "jmeter.api_port") cfg.api_port = std::stoi(value);
                else if (key == "jmeter.auto_start_api") cfg.jmeter_auto_start_api = (value == "true");
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

std::string replace_all(const std::string& str, const std::string& from, const std::string& to) {
    std::string result = str;
    size_t pos = 0;
    while ((pos = result.find(from, pos)) != std::string::npos) {
        result.replace(pos, from.length(), to);
        pos += to.length();
    }
    return result;
}

std::string modify_jmx_variables(const std::string& jmx_content, const SimpleYAML::Config& cfg) {
    std::string result = jmx_content;

    // Replace domain and port
    result = replace_all(result, "<stringProp name=\"HTTPSampler.domain\">localhost</stringProp>",
                        "<stringProp name=\"HTTPSampler.domain\">" + cfg.api_host + "</stringProp>");
    result = replace_all(result, "<stringProp name=\"HTTPSampler.port\">5001</stringProp>",
                        "<stringProp name=\"HTTPSampler.port\">" + std::to_string(cfg.api_port) + "</stringProp>");

    // Build the JSON payload
    std::ostringstream json;
    json << "{";
    json << "&quot;prompt&quot;:&quot;" << cfg.prompt << "&quot;,";
    json << "&quot;model&quot;:&quot;" << cfg.model << "&quot;,";

    // Handle agent_mode
    bool lookup_only = (cfg.agent_mode == "lookup_only");
    bool no_vis = (cfg.agent_mode == "analysis");
    json << "&quot;lookup_only&quot;:" << (lookup_only ? "true" : "false") << ",";
    json << "&quot;no_vis&quot;:" << (no_vis ? "true" : "false") << ",";

    // Basic parameters
    json << "&quot;best_of_n&quot;:" << cfg.best_of_n << ",";
    json << "&quot;temp&quot;:" << cfg.temperature << ",";

    // Optional temp_max
    if (!cfg.temperature_max.empty()) {
        json << "&quot;temp_max&quot;:" << cfg.temperature_max << ",";
    }

    // Save directory
    if (!cfg.save_dir.empty()) {
        json << "&quot;save_dir&quot;:&quot;" << cfg.save_dir << "&quot;,";
    }

    // Ground truth files
    if (!cfg.gt_csv.empty()) {
        json << "&quot;gt_csv&quot;:&quot;" << cfg.gt_csv << "&quot;,";
    }
    if (!cfg.gt_text.empty()) {
        json << "&quot;gt_text&quot;:&quot;" << cfg.gt_text << "&quot;,";
    }

    // CSV evaluation
    bool py_csv_eval = cfg.enable_csv_eval && (cfg.csv_eval_method == "python");
    bool cpp_csv_eval = cfg.enable_csv_eval && (cfg.csv_eval_method == "cpp");
    json << "&quot;py_csv_eval&quot;:" << (py_csv_eval ? "true" : "false") << ",";
    json << "&quot;cpp_csv_eval&quot;:" << (cpp_csv_eval ? "true" : "false") << ",";

    if (cpp_csv_eval) {
        if (!cfg.cpp_evaluator_exe.empty()) {
            json << "&quot;evaluator_exe&quot;:&quot;" << cfg.cpp_evaluator_exe << "&quot;,";
        }
        if (!cfg.cpp_evaluator_keys.empty()) {
            json << "&quot;eval_keys&quot;:&quot;" << cfg.cpp_evaluator_keys << "&quot;,";
        }
    }

    json << "&quot;iou_type&quot;:&quot;" << cfg.csv_iou_type << "&quot;,";

    // Text evaluation
    bool bleu_text_eval = cfg.enable_text_eval && (cfg.text_eval_method == "bleu");
    bool spice_text_eval = cfg.enable_text_eval && (cfg.text_eval_method == "spice");
    bool llm_text_eval = cfg.enable_text_eval && (cfg.text_eval_method == "llm");
    json << "&quot;bleu_text_eval&quot;:" << (bleu_text_eval ? "true" : "false") << ",";
    json << "&quot;spice_text_eval&quot;:" << (spice_text_eval ? "true" : "false") << ",";
    json << "&quot;llm_text_eval&quot;:" << (llm_text_eval ? "true" : "false") << ",";

    if (bleu_text_eval && cfg.bleu_use_nltk) {
        json << "&quot;bleu_nltk&quot;:true,";
    }

    if (spice_text_eval) {
        if (!cfg.spice_jar_path.empty()) {
            json << "&quot;spice_jar&quot;:&quot;" << cfg.spice_jar_path << "&quot;,";
        }
        if (!cfg.spice_java_bin.empty() && cfg.spice_java_bin != "java") {
            json << "&quot;spice_java_bin&quot;:&quot;" << cfg.spice_java_bin << "&quot;,";
        }
    }

    if (llm_text_eval && !cfg.llm_judge_model.empty()) {
        json << "&quot;llm_judge_model&quot;:&quot;" << cfg.llm_judge_model << "&quot;,";
    }

    // Tracing and monitoring
    json << "&quot;enable_tracing&quot;:" << (cfg.enable_tracing ? "true" : "false") << ",";

    if (cfg.enable_tracing) {
        if (!cfg.phoenix_endpoint.empty() && cfg.phoenix_endpoint != "http://localhost:6006/v1/traces") {
            json << "&quot;phoenix_endpoint&quot;:&quot;" << cfg.phoenix_endpoint << "&quot;,";
        }
        if (!cfg.phoenix_project_name.empty() && cfg.phoenix_project_name != "evaluating-agent") {
            json << "&quot;project_name&quot;:&quot;" << cfg.phoenix_project_name << "&quot;,";
        }
    }

    json << "&quot;enable_codecarbon&quot;:" << (cfg.enable_codecarbon ? "true" : "false");
    json << "}";

    // Replace the default JSON payload
    std::string default_json = "{&quot;prompt&quot;:&quot;What were the sales in November 2021?&quot;,&quot;model&quot;:&quot;llama3.2:3b&quot;,&quot;lookup_only&quot;:false,&quot;no_vis&quot;:true,&quot;best_of_n&quot;:1,&quot;temp&quot;:0.1,&quot;save_dir&quot;:&quot;./output&quot;,&quot;py_csv_eval&quot;:false,&quot;bleu_text_eval&quot;:false,&quot;iou_type&quot;:&quot;rows&quot;,&quot;enable_tracing&quot;:false,&quot;enable_codecarbon&quot;:false}";
    result = replace_all(result, default_json, json.str());

    return result;
}


pid_t start_background_process(const std::string& command, const std::string& name) {
    std::cout << "[Startup] Starting " << name << "..." << std::endl;
    std::cout << "[Startup] Command: " << command << std::endl;

    pid_t pid = fork();
    if (pid == 0) {
        setsid();
        if (!freopen("/dev/null", "w", stdout)) {
            std::cerr << "[Startup] Warning: failed to redirect stdout" << std::endl;
        }
        if (!freopen("/dev/null", "w", stderr)) {
            std::cerr << "[Startup] Warning: failed to redirect stderr" << std::endl;
        }
        execl("/bin/sh", "sh", "-c", command.c_str(), (char*)NULL);
        exit(1);
    } else if (pid > 0) {
        std::cout << "[Startup] " << name << " started with PID: " << pid << std::endl;
        sleep(3);
        return pid;
    } else {
        std::cerr << "[Error] Failed to start " << name << std::endl;
        return 0;
    }
}

pid_t start_phoenix(const SimpleYAML::Config& cfg) {
    if (!cfg.enable_tracing || !cfg.phoenix_auto_start) {
        return 0;
    }

    std::cout << "[Phoenix] Starting phoenix serve in background..." << std::endl;
    return start_background_process("phoenix serve", "Phoenix server");
}

int run_jmeter(const SimpleYAML::Config& cfg) {
    if (cfg.phoenix_auto_start && cfg.enable_tracing) {
        phoenix_pid = start_phoenix(cfg);
        if (phoenix_pid == 0) {
            std::cerr << "[Warning] Failed to start Phoenix server" << std::endl;
        }
    }

    if (cfg.jmeter_auto_start_api) {
        api_pid = start_background_process("python -m Agent.agent_api", "API server");
        if (api_pid == 0) {
            cleanup_processes();
            return -1;
        }
    }

    std::string jmx_file = cfg.run_batch ? "agent_api_batch_test.jmx" : "agent_api_single_test.jmx";
    std::string template_path = cfg.jmx_folder + "/" + jmx_file;

    std::ifstream input(template_path);
    if (!input.is_open()) {
        std::cerr << "[JMeter] Error: Could not open template " << template_path << std::endl;
        cleanup_processes();
        return -1;
    }

    std::string jmx_content((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    input.close();

    jmx_content = modify_jmx_variables(jmx_content, cfg);

    char temp_jmx[] = "/tmp/agent_test_XXXXXX.jmx";
    int fd = mkstemps(temp_jmx, 4);
    if (fd == -1) {
        std::cerr << "[JMeter] Error: Could not create temp JMX file" << std::endl;
        cleanup_processes();
        return -1;
    }

    FILE* f = fdopen(fd, "w");
    if (!f) {
        close(fd);
        cleanup_processes();
        return -1;
    }

    fputs(jmx_content.c_str(), f);
    fclose(f);

    std::string mkdir_cmd = "mkdir -p " + quote_arg(cfg.jmeter_output_folder);
    int mkdir_rc = system(mkdir_cmd.c_str());
    if (mkdir_rc != 0) {
        std::cerr << "[JMeter] Warning: mkdir failed (code " << mkdir_rc << ")" << std::endl;
    }

    time_t now = time(0);
    struct tm* ltm = localtime(&now);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", ltm);

    std::string log_file = cfg.jmeter_output_folder + "/jmeter_log_" + timestamp + ".jtl";
    std::string report_dir = cfg.jmeter_output_folder + "/jmeter_report_" + timestamp;

    std::ostringstream cmd;
    cmd << cfg.jmeter_bin << " -n -t " << temp_jmx << " -l " << quote_arg(log_file) << " -e -o " << quote_arg(report_dir);

    std::cout << "[JMeter] Running JMeter test..." << std::endl;
    std::cout << "[JMeter] Command: " << cmd.str() << std::endl;
    std::cout << "[JMeter] Log: " << log_file << std::endl;
    std::cout << "[JMeter] Report: " << report_dir << std::endl;

    int result = system(cmd.str().c_str());

    remove(temp_jmx);
    cleanup_processes();

    if (result == 0) {
        std::cout << "[JMeter] Test completed successfully" << std::endl;
        std::cout << "[JMeter] Open " << report_dir << "/index.html for detailed report" << std::endl;
    } else {
        std::cout << "[JMeter] Test failed with exit code: " << result << std::endl;
    }

    return result;
}

std::string build_command(const SimpleYAML::Config& cfg, const std::string& prompt = "",
                         const std::string& gt_csv = "", const std::string& gt_text = "") {
    std::ostringstream cmd;
    cmd << (cfg.python_bin.empty() ? "python" : cfg.python_bin) << " -m Agent.data_agent";

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

int write_file(const std::string& filepath, const std::string& content) {
    std::ofstream out(filepath, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "Error: Could not create file: " << filepath << std::endl;
        return -1;
    }
    out << content;
    out.close();
    return 0;
}

int run_direct_python(const SimpleYAML::Config& cfg) {
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
            std::cout << "[ConfigRunner] Prompt: " << test_cases[i].prompt.substr(0, 60);
            if (test_cases[i].prompt.size() > 60) std::cout << "...";
            std::cout << std::endl;

            std::string save_dir = cfg.save_dir;
            if (save_dir.empty()) {
                save_dir = "./output";  // Use current directory as fallback
            }
            save_dir = save_dir + "/test_" + std::to_string(i + 1);

            // Create directory
            std::string mkdir_cmd = "mkdir -p " + quote_arg(save_dir);
            int mkdir_rc = system(mkdir_cmd.c_str());
            if (mkdir_rc != 0) {
                std::cerr << "[ConfigRunner] Failed to create directory: " << save_dir << std::endl;
                fail_count++;
                continue;  // Skip this test case
            }

            std::string gt_csv_file, gt_text_file;

            if (!test_cases[i].gt_data.empty()) {
                gt_csv_file = save_dir + "/gt_data.txt";
                if (write_file(gt_csv_file, test_cases[i].gt_data)  != 0) {
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

            SimpleYAML::Config test_cfg = cfg;
            test_cfg.save_dir = save_dir;

            std::string cmd = build_command(test_cfg, test_cases[i].prompt, gt_csv_file, gt_text_file);

            std::cout << "[ConfigRunner] Executing: " << cmd << std::endl;
            int result = system(cmd.c_str());

            if (result == 0) {
                std::cout << "[ConfigRunner] Test case " << (i + 1) << " PASSED" << std::endl;
                success_count++;
            } else {
                std::cout << "[ConfigRunner] Test case " << (i + 1) << " FAILED (exit code: " << result << ")" << std::endl;
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
            cleanup_processes();
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

    cleanup_processes();
    return exit_code;
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
    temp_dir = determine_temp_dir(cfg.python_bin);

    if (cfg.use_jmeter) {
        std::cout << "[ConfigRunner] Using JMeter mode" << std::endl;
        return run_jmeter(cfg);
    } else {
        std::cout << "[ConfigRunner] Using direct Python execution mode" << std::endl;
        return run_direct_python(cfg);
    }
}
