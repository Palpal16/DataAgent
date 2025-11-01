#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

struct Options {
    std::string actual_path;
    std::string expected_path;
    std::vector<std::string> keys; // join keys for order-insensitive compare
    double float_abs = 1e-8;
    double float_rel = 1e-6;
    bool json = true;
    bool case_insensitive = false;
};

static inline std::string trim(const std::string &s) {
    size_t start = 0, end = s.size();
    while (start < end && std::isspace(static_cast<unsigned char>(s[start]))) start++;
    while (end > start && std::isspace(static_cast<unsigned char>(s[end-1]))) end--;
    return s.substr(start, end - start);
}

static inline std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return s;
}

static inline bool parse_double(const std::string &s, double &out) {
    char *end = nullptr;
    std::string ts = trim(s);
    if (ts.empty()) return false;
    out = std::strtod(ts.c_str(), &end);
    if (end == ts.c_str() || *end != '\0') return false;
    return true;
}

static inline bool equals_numeric_or_string(const std::string &a, const std::string &b, const Options &opt) {
    // Null/empty equal
    if (trim(a).empty() && trim(b).empty()) return true;
    // NaN handling
    auto la = lower_copy(trim(a));
    auto lb = lower_copy(trim(b));
    if ((la == "nan" || la == "na" || la == "null") && (lb == "nan" || lb == "na" || lb == "null")) return true;

    double da = 0.0, db = 0.0;
    bool na = parse_double(a, da);
    bool nb = parse_double(b, db);
    if (na && nb) {
        double diff = std::fabs(da - db);
        if (diff <= opt.float_abs) return true;
        double denom = std::max(std::fabs(da), std::fabs(db));
        if (denom == 0.0) return diff == 0.0;
        return (diff / denom) <= opt.float_rel;
    }
    // String compare
    if (opt.case_insensitive) {
        return lower_copy(trim(a)) == lower_copy(trim(b));
    }
    return trim(a) == trim(b);
}

// Very simple CSV splitter (no embedded commas in quotes). Good enough for evaluator data.
static std::vector<std::string> split_csv_line(const std::string &line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            in_quotes = !in_quotes;
            continue; // drop quotes
        }
        if (c == ',' && !in_quotes) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}

struct Table {
    std::vector<std::string> columns;
    std::vector< std::vector<std::string> > rows; // rows[i][col_index]
    std::unordered_map<std::string, size_t> col_index;
};

static bool read_csv(const std::string &path, Table &t, std::string &err) {
    std::ifstream f(path);
    if (!f.is_open()) { err = "Cannot open file: " + path; return false; }
    std::string line;
    if (!std::getline(f, line)) { err = "Empty file: " + path; return false; }
    t.columns = split_csv_line(line);
    for (size_t i = 0; i < t.columns.size(); ++i) t.col_index[t.columns[i]] = i;
    while (std::getline(f, line)) {
        auto parts = split_csv_line(line);
        // pad/truncate to header size
        parts.resize(t.columns.size());
        t.rows.push_back(std::move(parts));
    }
    return true;
}

static std::string make_key(const std::vector<std::string> &keys, const Table &t, const std::vector<std::string> &row) {
    if (keys.empty()) return {};
    std::ostringstream oss;
    for (size_t i = 0; i < keys.size(); ++i) {
        auto it = t.col_index.find(keys[i]);
        if (it == t.col_index.end()) continue; // skip missing key silently
        if (i) oss << "\x1f"; // unit separator
        oss << row[it->second];
    }
    return oss.str();
}

struct DiffSummary {
    bool equal = false;
    size_t row_count_actual = 0;
    size_t row_count_expected = 0;
    size_t mismatched_rows = 0;
    std::set<std::string> mismatched_columns;
    long long duration_ms = 0;
    std::string error;
};

static DiffSummary compare_tables(const Table &a, const Table &b, const Options &opt) {
    auto start = std::chrono::steady_clock::now();
    DiffSummary s;
    s.row_count_actual = a.rows.size();
    s.row_count_expected = b.rows.size();

    // Column set equality
    std::set<std::string> cols_a(a.columns.begin(), a.columns.end());
    std::set<std::string> cols_b(b.columns.begin(), b.columns.end());
    if (cols_a != cols_b) {
        s.equal = false;
        s.mismatched_rows = std::max(a.rows.size(), b.rows.size());
        s.mismatched_columns.insert("SCHEMA");
        s.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        return s;
    }

    std::vector<std::string> columns(a.columns.begin(), a.columns.end());
    // If keys provided, build maps by key
    if (!opt.keys.empty()) {
        std::unordered_map<std::string, const std::vector<std::string>*> map_a;
        std::unordered_map<std::string, const std::vector<std::string>*> map_b;
        map_a.reserve(a.rows.size()*2);
        map_b.reserve(b.rows.size()*2);
        for (const auto &r : a.rows) map_a[make_key(opt.keys, a, r)] = &r;
        for (const auto &r : b.rows) map_b[make_key(opt.keys, b, r)] = &r;

        if (map_a.size() != map_b.size()) {
            s.mismatched_rows += std::llabs((long long)map_a.size() - (long long)map_b.size());
        }

        // Compare intersection of keys
        for (const auto &kv : map_a) {
            auto it = map_b.find(kv.first);
            if (it == map_b.end()) { s.mismatched_rows++; continue; }
            const auto &ra = *kv.second;
            const auto &rb = *it->second;
            for (const auto &col : columns) {
                size_t idx = a.col_index.at(col);
                const std::string &va = ra[idx];
                const std::string &vb = rb[idx];
                if (!equals_numeric_or_string(va, vb, opt)) s.mismatched_columns.insert(col);
            }
        }
    } else {
        // Order-sensitive row-by-row compare
        size_t n = std::min(a.rows.size(), b.rows.size());
        s.mismatched_rows += std::llabs((long long)a.rows.size() - (long long)b.rows.size());
        for (size_t i = 0; i < n; ++i) {
            const auto &ra = a.rows[i];
            const auto &rb = b.rows[i];
            for (const auto &col : columns) {
                size_t idx = a.col_index.at(col);
                const std::string &va = ra[idx];
                const std::string &vb = rb[idx];
                if (!equals_numeric_or_string(va, vb, opt)) s.mismatched_columns.insert(col);
            }
        }
    }

    s.equal = (s.mismatched_rows == 0) && s.mismatched_columns.empty();
    s.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    return s;
}

static void print_json(const DiffSummary &s) {
    std::cout << "{\n";
    std::cout << "  \"equal\": " << (s.equal ? "true" : "false") << ",\n";
    std::cout << "  \"row_count_actual\": " << s.row_count_actual << ",\n";
    std::cout << "  \"row_count_expected\": " << s.row_count_expected << ",\n";
    std::cout << "  \"mismatched_rows\": " << s.mismatched_rows << ",\n";
    std::cout << "  \"mismatched_columns\": [";
    bool first = true;
    for (const auto &c : s.mismatched_columns) {
        if (!first) std::cout << ", ";
        std::cout << "\"" << c << "\"";
        first = false;
    }
    std::cout << "],\n";
    std::cout << "  \"duration_ms\": " << s.duration_ms;
    if (!s.error.empty()) {
        std::cout << ",\n  \"error\": \"" << s.error << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";
}

static void print_usage() {
    std::cerr << "Usage: resultcmp --actual A.csv --expected E.csv [--key k1,k2] [--float-abs 1e-8] [--float-rel 1e-6] [--case-insensitive]\n";
}

int main(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() -> const char* { if (i+1 < argc) return argv[++i]; print_usage(); std::exit(2); };
        if (arg == "--actual") opt.actual_path = next();
        else if (arg == "--expected") opt.expected_path = next();
        else if (arg == "--key") {
            std::string ks = next();
            std::stringstream ss(ks);
            std::string item;
            while (std::getline(ss, item, ',')) opt.keys.push_back(item);
        }
        else if (arg == "--float-abs") opt.float_abs = std::stod(next());
        else if (arg == "--float-rel") opt.float_rel = std::stod(next());
        else if (arg == "--case-insensitive") opt.case_insensitive = true;
        else if (arg == "--help" || arg == "-h") { print_usage(); return 0; }
        else { std::cerr << "Unknown arg: " << arg << "\n"; print_usage(); return 2; }
    }

    if (opt.actual_path.empty() || opt.expected_path.empty()) { print_usage(); return 2; }

    Table ta, te; std::string err;
    if (!read_csv(opt.actual_path, ta, err)) {
        DiffSummary s; s.error = err; print_json(s); return 2;
    }
    if (!read_csv(opt.expected_path, te, err)) {
        DiffSummary s; s.error = err; print_json(s); return 2;
    }
    auto summary = compare_tables(ta, te, opt);
    print_json(summary);
    return summary.equal ? 0 : 1;
}


