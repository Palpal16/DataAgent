#include "runner_common.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

namespace runner_common {

static int get_indent(const std::string& s) {
    int count = 0;
    for (char c : s) {
        if (c == ' ') count++;
        else if (c == '\t') count += 4;
        else break;
    }
    return count;
}

static std::string trim_copy(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static std::string unquote_copy(const std::string& s) {
    std::string result = trim_copy(s);
    if (result.size() >= 2 && result.front() == '"' && result.back() == '"') {
        return result.substr(1, result.size() - 2);
    }
    return result;
}

void for_each_yaml_kv(
    const std::string& filename,
    const std::vector<std::string>& section_headers,
    const YamlCallback& on_kv
) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return;
    }

    auto is_section = [&](const std::string& key) {
        return std::find(section_headers.begin(), section_headers.end(), key) != section_headers.end();
    };

    std::string line;
    std::string current_section;
    int base_indent = -1;

    while (std::getline(file, line)) {
        std::string original = line;
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        line = trim_copy(line);
        if (line.empty()) continue;

        const int indent = get_indent(original);

        auto pos = line.find(':');
        if (pos == std::string::npos) continue;

        std::string key = trim_copy(line.substr(0, pos));
        std::string value = unquote_copy(trim_copy(line.substr(pos + 1)));

        // Section header: "section:" with empty value.
        if (value.empty() && is_section(key)) {
            current_section = key;
            base_indent = indent;
            continue;
        }

        // If indentation resets, drop section.
        if (base_indent == -1 || indent <= base_indent) {
            base_indent = indent;
            current_section.clear();
        } else if (indent > base_indent && !current_section.empty()) {
            key = current_section + "." + key;
        }

        if (value.empty() || value == "null") continue;
        if (on_kv) on_kv(key, value);
    }
}

// -----------------------------
// JSON parsing (very small subset)
// -----------------------------

static size_t find_matching_brace(const std::string& s, size_t start) {
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

static std::string extract_json_string(const std::string& obj, const std::string& key) {
    std::string search_key = "\"" + key + "\"";
    size_t pos = obj.find(search_key);
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

std::vector<JsonTestCase> parse_json_test_cases(const std::string& filename) {
    std::vector<JsonTestCase> cases;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return cases;
    }

    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    size_t pos = 0;
    while ((pos = content.find("{", pos)) != std::string::npos) {
        size_t end = find_matching_brace(content, pos);
        if (end == std::string::npos) break;

        std::string obj = content.substr(pos, end - pos + 1);
        JsonTestCase tc;
        tc.prompt = extract_json_string(obj, "prompt");
        tc.gt_data = extract_json_string(obj, "gt_data");
        tc.gt_analysis = extract_json_string(obj, "gt_analysis");
        tc.gt_sql = extract_json_string(obj, "gt_sql");

        if (!tc.prompt.empty()) cases.push_back(std::move(tc));
        pos = end + 1;
    }
    return cases;
}

// -----------------------------
// Utilities
// -----------------------------

std::string determine_temp_dir(const std::string& python_bin) {
    if (!python_bin.empty()) {
        if (python_bin.find(".exe") != std::string::npos || python_bin.find("/mnt/") != std::string::npos) {
            return "./tmp";
        }
    }
    return "/tmp";
}

std::string quote_arg(const std::string& s) {
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

std::string replace_all(std::string str, const std::string& from, const std::string& to) {
    if (from.empty()) return str;
    size_t pos = 0;
    while ((pos = str.find(from, pos)) != std::string::npos) {
        str.replace(pos, from.length(), to);
        pos += to.length();
    }
    return str;
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

// -----------------------------
// Text-table to CSV helpers
// -----------------------------

static std::vector<std::string> split_ws(const std::string& line) {
    std::vector<std::string> parts;
    std::istringstream iss(line);
    std::string tok;
    while (iss >> tok) parts.push_back(tok);
    return parts;
}

std::vector<std::vector<std::string>> text_table_to_rows(const std::string& text) {
    std::vector<std::vector<std::string>> rows;
    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line)) {
        // trim
        auto is_space = [](unsigned char c){ return std::isspace(c); };
        line.erase(line.begin(), std::find_if(line.begin(), line.end(), [&](unsigned char c){ return !is_space(c); }));
        line.erase(std::find_if(line.rbegin(), line.rend(), [&](unsigned char c){ return !is_space(c); }).base(), line.end());
        if (line.empty()) continue;

        std::vector<std::string> parts;
        // Prefer pipe-separated
        if (line.find('|') != std::string::npos) {
            std::string cur;
            std::istringstream ls(line);
            while (std::getline(ls, cur, '|')) {
                // trim token
                cur.erase(cur.begin(), std::find_if(cur.begin(), cur.end(), [&](unsigned char c){ return !is_space(c); }));
                cur.erase(std::find_if(cur.rbegin(), cur.rend(), [&](unsigned char c){ return !is_space(c); }).base(), cur.end());
                if (!cur.empty()) parts.push_back(cur);
            }
        } else {
            // Whitespace split (works for typical pandas tables)
            std::istringstream ls(line);
            std::string tok;
            while (ls >> tok) parts.push_back(tok);
        }

        if (!parts.empty()) rows.push_back(std::move(parts));
    }
    return rows;
}

int write_csv_from_rows(const std::string& path, const std::vector<std::vector<std::string>>& rows) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) return -1;
    for (const auto& r : rows) {
        for (size_t i = 0; i < r.size(); ++i) {
            if (i) out << ",";
            // minimal escaping: wrap if contains comma
            const auto& v = r[i];
            if (v.find(',') != std::string::npos) out << "\"" << v << "\"";
            else out << v;
        }
        out << "\n";
    }
    return 0;
}

// -----------------------------
// Agent CLI command builder
// -----------------------------

std::string build_agent_command(
    const AgentCliConfig& cfg,
    const std::string& prompt_override,
    const std::string& gt_csv_override,
    const std::string& gt_text_override
) {
    std::ostringstream cmd;
    cmd << (cfg.python_bin.empty() ? "python" : cfg.python_bin) << " -m Agent.data_agent";

    const std::string use_prompt = prompt_override.empty() ? cfg.prompt : prompt_override;
    const std::string use_gt_csv = gt_csv_override.empty() ? cfg.gt_csv : gt_csv_override;
    const std::string use_gt_text = gt_text_override.empty() ? cfg.gt_text : gt_text_override;

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
            if (cfg.cpp_evaluator_threads > 0)
                cmd << " --evaluator_threads " << cfg.cpp_evaluator_threads;
            if (cfg.cpp_evaluator_benchmark)
                cmd << " --evaluator_benchmark";
            if (cfg.cpp_evaluator_benchmark_iters > 0)
                cmd << " --evaluator_benchmark_iters " << cfg.cpp_evaluator_benchmark_iters;
        }
        cmd << " --iou_type " << cfg.csv_iou_type;
    }

    if (cfg.enable_text_eval) {
        if (cfg.text_eval_method == "bleu") {
            cmd << " --bleu_text_eval";
            if (cfg.bleu_use_nltk) cmd << " --bleu_nltk";
        } else if (cfg.text_eval_method == "bleu+spice" || cfg.text_eval_method == "bleu_spice") {
            cmd << " --bleu_text_eval";
            if (cfg.bleu_use_nltk) cmd << " --bleu_nltk";
            cmd << " --spice_text_eval";
            if (!cfg.spice_jar_path.empty())
                cmd << " --spice_jar " << quote_arg(cfg.spice_jar_path);
            if (!cfg.spice_java_bin.empty())
                cmd << " --spice_java_bin " << quote_arg(cfg.spice_java_bin);
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

    if (cfg.two_stage_cot) {
        cmd << " --two_stage_cot";
        if (cfg.cot_max_bullets > 0) cmd << " --cot_max_bullets " << cfg.cot_max_bullets;
        if (cfg.cot_print_plan) cmd << " --cot_print_plan";
        if (cfg.cot_store_plan) cmd << " --cot_store_plan";
    }

    return cmd.str();
}

} // namespace runner_common

