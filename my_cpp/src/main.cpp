// resultcmp: CSV comparator CLI
// Compares two CSV files (actual vs expected) with optional key-based row alignment
// and numeric tolerances. Emits a compact JSON summary and exits 0 on equality, 1 otherwise.
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
#include <unordered_set>
#include <vector>
#include <cstdlib>
#include <iterator>
#include <limits>

#include "util.hpp"

#if defined(RESULTCMP_HAS_OPENMP)
#include <omp.h>
#endif

// Command-line options controlling comparison behavior
struct Options {
    std::string actual_path;
    std::string expected_path;
    std::vector<std::string> keys; // join keys for order-insensitive compare
    bool json = true;
    bool case_insensitive = false;
    bool debug = false;
    int threads = 0; // 0 = default
    bool benchmark = false;
    int benchmark_iters = 3;
};

static void maybe_set_threads(const Options& opt) {
#if defined(RESULTCMP_HAS_OPENMP)
    if (opt.threads > 0) {
        omp_set_num_threads(opt.threads);
    }
#else
    (void)opt;
#endif
}

// Numeric comparison tolerances (constants)
static constexpr double FLOAT_ABS_TOLERANCE = 1e-8;
static constexpr double FLOAT_REL_TOLERANCE = 1e-6;

// Trim leading/trailing whitespace
static inline std::string trim(const std::string &s) {
    size_t start = 0, end = s.size();
    while (start < end && std::isspace(static_cast<unsigned char>(s[start]))) start++;
    while (end > start && std::isspace(static_cast<unsigned char>(s[end-1]))) end--;
    return s.substr(start, end - start);
}

// Lowercase a copy (ASCII)
static inline std::string lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return s;
}

// Best-effort parse of a string to double; returns false on failure
static inline bool parse_double(const std::string &s, double &out) {
    char *end = nullptr;
    std::string ts = trim(s);
    if (ts.empty()) return false;
    out = std::strtod(ts.c_str(), &end);
    if (end == ts.c_str() || *end != '\0') return false;
    return true;
}

// Tolerant cell comparison (numbers with abs/rel tolerance; optional case-insensitive strings; NaN/NULL normalization)
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
        if (diff <= FLOAT_ABS_TOLERANCE) return true;
        double denom = std::max(std::fabs(da), std::fabs(db));
        if (denom == 0.0) return diff == 0.0;
        return (diff / denom) <= FLOAT_REL_TOLERANCE;
    }
    // String compare
    if (opt.case_insensitive) {
        return lower_copy(trim(a)) == lower_copy(trim(b));
    }
    return trim(a) == trim(b);
}

// Very simple CSV splitter (no embedded commas in quotes). Good enough for evaluator data.
// Split a single CSV line into fields (minimal handling of quotes)
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
    // Always push the final field (may be empty when line ends with a comma)
    out.push_back(cur);
    // Trim whitespace/CR from fields (important for Windows CRLF files)
    for (auto &f : out) f = trim(f);
    return out;
}

static inline bool looks_like_index_header(const std::string& h) {
    const auto lh = lower_copy(trim(h));
    return lh.empty() || lh == "unnamed: 0" || lh == "index";
}

static inline bool looks_like_index_value(const std::string& v) {
    // Heuristic: common pandas index is an integer like 0,1,2,...
    std::string s = trim(v);
    if (s.empty()) return false;
    // allow leading +/-
    size_t i = (s[0] == '+' || s[0] == '-') ? 1 : 0;
    if (i >= s.size()) return false;
    for (; i < s.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    }
    return true;
}

// Parsed CSV in memory
struct Table {
    std::vector<std::string> columns;
    std::vector< std::vector<std::string> > rows; // rows[i][col_index]
    std::unordered_map<std::string, size_t> col_index;
};

// Read a CSV file into Table (first line as header)
static bool read_csv(const std::string &path, Table &t, std::string &err) {
    std::ifstream f(path);
    if (!f.is_open()) { err = "Cannot open file: " + path; return false; }
    std::string line;
    if (!std::getline(f, line)) { err = "Empty file: " + path; return false; }
    // Parse header. Some pandas CSVs include an index column (often "Unnamed: 0").
    auto header = split_csv_line(line);
    if (header.empty()) { err = "Malformed header in file: " + path; return false; }

    // If header explicitly includes an index column, drop it from both header + rows.
    bool drop_index_header = looks_like_index_header(header.front());
    if (drop_index_header) header.erase(header.begin());
    t.columns = std::move(header);

    t.col_index.clear();
    for (size_t i = 0; i < t.columns.size(); ++i) t.col_index[t.columns[i]] = i;

    // Some CSVs don't have an index in the header, but do include it in rows (e.g., pandas.to_csv with index=True).
    // Detect this using the first data row.
    bool drop_index_rows = drop_index_header;
    std::string first_data_line;
    if (std::getline(f, first_data_line)) {
        auto parts = split_csv_line(first_data_line);
        if (!drop_index_rows && parts.size() == t.columns.size() + 1 && looks_like_index_value(parts.front())) {
            drop_index_rows = true;
        }
        if (drop_index_rows && !parts.empty()) parts.erase(parts.begin());
        parts.resize(t.columns.size());
        t.rows.push_back(std::move(parts));
    }

    while (std::getline(f, line)) {
        auto parts = split_csv_line(line);
        if (drop_index_rows && !parts.empty()) parts.erase(parts.begin());
        parts.resize(t.columns.size());
        t.rows.push_back(std::move(parts));
    }
    return true;
}

// Build composite key from selected columns in a row
static std::string make_key(const std::vector<std::string> &keys, const Table &t, const std::vector<std::string> &row, const Options &opt) {
    if (keys.empty()) return {};
    std::ostringstream oss;
    oss << "(";
    bool first = true;
    for (size_t i = 0; i < keys.size(); ++i) {
        auto it = t.col_index.find(keys[i]);
        if (it == t.col_index.end()) continue; // skip missing key silently
        if (!first) oss << ",";
        oss << row[it->second];
        first = false;
    }
    oss << ")";
    if (opt.debug) std::cerr << "Key: " << oss.str() << std::endl;
    return oss.str();
}

// Aggregate result for a comparison
struct DiffSummary {
    bool equal = false;
    size_t row_count_actual = 0;
    size_t row_count_expected = 0;
    size_t mismatched_rows = 0;
    std::set<std::string> mismatched_columns;
    long long duration_ms = 0;
    std::string error;
    // IoU metrics:
    // rows_iou:    Jaccard similarity of unique rows restricted to common columns
    // columns_iou: Jaccard similarity of column name sets
    // iou:         overall score = rows_iou * columns_iou
    double rows_iou = 0.0;
    double columns_iou = 0.0;
    double _overall_iou = 0.0;
    // Row set sizes (for debugging): |A|, |B|, |A∩B|, |A∪B|
    size_t rows_set_a_size = 0;
    size_t rows_set_b_size = 0;
    size_t rows_intersection_size = 0;
    size_t rows_union_size = 0;

    // Optional benchmark (serial vs OpenMP)
    bool benchmark_ran = false;
    long long benchmark_serial_ms = 0;
    long long benchmark_parallel_ms = 0;
    int benchmark_serial_threads = 1;
    int benchmark_parallel_threads = 0;
    int benchmark_iters = 0;
};

template <typename K, typename V>
void print_map(const std::unordered_map<K, V> &m, const std::string& name = "map") {
    std::cerr << name << " (size: " << m.size() << "): {";
    bool first = true;
    for (const auto& kv : m) {
        if (!first) std::cerr << ", ";
        std::cerr << kv.first << ": " << kv.second;
        first = false;
    }
    std::cerr << "}" << std::endl;
}

// Overload specifically for maps whose values are pointers to vector<string> (prints contents)
template <typename K>
void print_map(const std::unordered_map<K, const std::vector<std::string>*> &m, const std::string& name = "map") {
    std::cerr << name << " (size: " << m.size() << "): {";
    bool first = true;
    for (const auto& kv : m) {
        if (!first) std::cerr << ", ";
        std::cerr << kv.first << ": [";
        const auto& vec = *(kv.second);
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i) std::cerr << ", ";
            std::cerr << vec[i];
        }
        std::cerr << "]";
        first = false;
    }
    std::cerr << "}" << std::endl;
}

// Pretty-print a Table (like a simple dataframe preview) to stderr
[[maybe_unused]] static void print_table(const Table &t, const std::string &name, size_t max_rows) {
    std::cerr << name << " (rows: " << t.rows.size() << ", cols: " << t.columns.size() << ")\n";
    std::cerr << "columns: [";
    for (size_t i = 0; i < t.columns.size(); ++i) {
        if (i) std::cerr << ", ";
        std::cerr << t.columns[i];
    }
    std::cerr << "]\n";
    size_t n = std::min(max_rows, t.rows.size());
    for (size_t r = 0; r < n; ++r) {
        std::cerr << "row " << r << ": {";
        const auto &row = t.rows[r];
        for (size_t c = 0; c < t.columns.size(); ++c) {
            if (c) std::cerr << ", ";
            std::cerr << t.columns[c] << ": " << row[c];
        }
        std::cerr << "}\n";
    }
    if (t.rows.size() > n) {
        std::cerr << "... (" << (t.rows.size() - n) << " more rows)\n";
    }
}

// Function determine the intersection of two sets
static std::set<std::string> intersection_set(const std::set<std::string> &a, const std::set<std::string> &b) {
    std::set<std::string> intsect;
    std::set_intersection(a.begin(), a.end(),
                          b.begin(), b.end(),
                          std::inserter(intsect, intsect.begin()));
    return intsect;
}

//Function determine the union of two sets
static std::set<std::string> union_set(const std::set<std::string> &a, const std::set<std::string> &b) {
    std::set<std::string> uni;
    std::set_union(a.begin(), a.end(),
                          b.begin(), b.end(),
                          std::inserter(uni, uni.begin()));
    return uni;
}

static std::multiset<std::string> intersection_multiset(std::multiset<std::string> a, std::multiset<std::string> b) {
    std::multiset<std::string> intersection;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                          std::inserter(intersection, intersection.begin()));
    return intersection;
}

// Returns the union of two multisets: for each distinct string element,
// includes it as many times as its maximum count in either input multiset.
// for example if a = {a, b, b, c} and b = {a, b, b, d}, then the union is {a, b, b, c, d}
static std::multiset<std::string> union_multiset(const std::multiset<std::string>& a, const std::multiset<std::string>& b) {
    // Map to hold the max count of each unique string from both multisets
    std::map<std::string, int> count_map;
    // Count occurrences in 'a'
    for (const auto& x : a) ++count_map[x];
    // Update counts to the max of 'a' and 'b' for each string in 'b'
    for (const auto& x : b) count_map[x] = std::max(count_map[x], static_cast<int>(b.count(x)));
    // Build the resulting multiset with each string inserted 'max_count' times
    std::multiset<std::string> result;
    for (const auto& kv : count_map) {
        int max_count = std::max(static_cast<int>(a.count(kv.first)), static_cast<int>(b.count(kv.first)));
        for (int i = 0; i < max_count; ++i)
            result.insert(kv.first);
    }
    return result;
}

static DiffSummary compare_tables_multiset(const Table &a, const Table &b, const Options &opt) {
    maybe_set_threads(opt);
    auto start = std::chrono::steady_clock::now();
    DiffSummary s;
    s.row_count_actual = a.rows.size();
    s.row_count_expected = b.rows.size();
    
    // 1) Columns IoU = |A ∩ B| / |A ∪ B| where sets are column names
    std::multiset<std::string> cols_a(a.columns.begin(), a.columns.end());
    std::multiset<std::string> cols_b(b.columns.begin(), b.columns.end());
    std::multiset<std::string> cols_intersection = intersection_multiset(cols_a, cols_b);
    // Just rename cols_intersection to common_cols for clarity
    std::vector<std::string> common_cols(cols_intersection.begin(), cols_intersection.end());
    if (opt.debug) {
        std::cerr << "Common columns:\n";
        for (const auto &c : common_cols) std::cerr << c << "\n";
    }
    std::multiset<std::string> cols_union = union_multiset(cols_a, cols_b);
    s.columns_iou = !cols_union.empty()
        ? static_cast<double>(cols_intersection.size()) / static_cast<double>(cols_union.size())
        : 0.0;
    if (opt.debug) {
        std::cerr << "columns_iou=" << s.columns_iou << "\n";
    }
    // 2) Rows IoU computed on the intersection of columns:
    //    - Build a canonical string key per row using only common columns (trimmed; optional lowercase)
    //    - Compare sets of unique row keys across the two tables
    if (!common_cols.empty()) {
        // Build row keys (dominant cost). Parallelized when OpenMP is available.
        std::vector<std::string> keys_a(a.rows.size());
        std::vector<std::string> keys_b(b.rows.size());

#if defined(RESULTCMP_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < static_cast<int>(a.rows.size()); ++i) {
            keys_a[static_cast<size_t>(i)] = make_key(common_cols, a, a.rows[static_cast<size_t>(i)], opt);
        }

#if defined(RESULTCMP_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < static_cast<int>(b.rows.size()); ++i) {
            keys_b[static_cast<size_t>(i)] = make_key(common_cols, b, b.rows[static_cast<size_t>(i)], opt);
        }

        std::sort(keys_a.begin(), keys_a.end());
        std::sort(keys_b.begin(), keys_b.end());
        s.rows_set_a_size = keys_a.size();
        s.rows_set_b_size = keys_b.size();

        // Multiset intersection/union sizes (count duplicates)
        size_t i = 0, j = 0;
        size_t inter = 0, uni = 0;
        while (i < keys_a.size() && j < keys_b.size()) {
            if (keys_a[i] == keys_b[j]) {
                inter++; uni++; i++; j++;
            } else if (keys_a[i] < keys_b[j]) {
                uni++; i++;
            } else {
                uni++; j++;
            }
        }
        uni += (keys_a.size() - i) + (keys_b.size() - j);
        s.rows_intersection_size = inter;
        s.rows_union_size = uni;
        s.rows_iou = (uni > 0) ? static_cast<double>(inter) / static_cast<double>(uni) : 0.0;
    }


    // 3) Overall IoU = product (matches Python reference behavior)
    s._overall_iou = s.columns_iou * s.rows_iou;

    // Consider tables equal only under perfect match across columns and rows
    s.equal = (s.columns_iou == 1.0) && (s.rows_iou == 1.0);
    s.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    return s;
}


// Compare two tables:
// - Validate column set equality (schema)
// - With keys: align rows by key, then compare values
// - Without keys: compare rows positionally (order-sensitive)
[[maybe_unused]] static DiffSummary compare_tables(const Table &a, const Table &b, const Options &opt) {
    maybe_set_threads(opt);
    auto start = std::chrono::steady_clock::now();
    DiffSummary s;
    s.row_count_actual = a.rows.size();
    s.row_count_expected = b.rows.size();

    // 1) Columns IoU = |A ∩ B| / |A ∪ B| where sets are column names
    std::set<std::string> cols_a(a.columns.begin(), a.columns.end());
    std::set<std::string> cols_b(b.columns.begin(), b.columns.end());
    std::set<std::string> cols_intersection = intersection_set(cols_a, cols_b);
    // Just rename cols_intersection to common_cols for clarity
    std::vector<std::string> common_cols(cols_intersection.begin(), cols_intersection.end());
    if (opt.debug) {
        std::cerr << "Common columns:\n";
        for (const auto &c : common_cols) std::cerr << c << "\n";
    }

    std::set<std::string> cols_union = union_set(cols_a, cols_b);
    s.columns_iou = !cols_union.empty()
        ? static_cast<double>(cols_intersection.size()) / static_cast<double>(cols_union.size())
        : 0.0;
    
    // 2) Rows IoU computed on the intersection of columns:
    //    - Build a canonical string key per row using only common columns (trimmed; optional lowercase)
    //    - Compare sets of unique row keys across the two tables
    if (!common_cols.empty()) {
        std::vector<std::string> keys_a(a.rows.size());
        std::vector<std::string> keys_b(b.rows.size());

#if defined(RESULTCMP_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < static_cast<int>(a.rows.size()); ++i) {
            keys_a[static_cast<size_t>(i)] = make_key(common_cols, a, a.rows[static_cast<size_t>(i)], opt);
        }

#if defined(RESULTCMP_HAS_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < static_cast<int>(b.rows.size()); ++i) {
            keys_b[static_cast<size_t>(i)] = make_key(common_cols, b, b.rows[static_cast<size_t>(i)], opt);
        }

        std::sort(keys_a.begin(), keys_a.end());
        keys_a.erase(std::unique(keys_a.begin(), keys_a.end()), keys_a.end());
        std::sort(keys_b.begin(), keys_b.end());
        keys_b.erase(std::unique(keys_b.begin(), keys_b.end()), keys_b.end());
        s.rows_set_a_size = keys_a.size();
        s.rows_set_b_size = keys_b.size();

        // Set intersection/union sizes
        size_t i = 0, j = 0;
        size_t inter = 0;
        while (i < keys_a.size() && j < keys_b.size()) {
            if (keys_a[i] == keys_b[j]) { inter++; i++; j++; }
            else if (keys_a[i] < keys_b[j]) i++;
            else j++;
        }
        const size_t uni = keys_a.size() + keys_b.size() - inter;
        s.rows_intersection_size = inter;
        s.rows_union_size = uni;
        s.rows_iou = (uni > 0) ? static_cast<double>(inter) / static_cast<double>(uni) : 0.0;
    }


    // 3) Overall IoU = product (matches Python reference behavior)
    s._overall_iou = s.columns_iou * s.rows_iou;

    // Consider tables equal only under perfect match across columns and rows
    s.equal = (s.columns_iou == 1.0) && (s.rows_iou == 1.0);
    s.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    return s;
}

// Emit comparison result as JSON on stdout
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
    std::cout << "  \"rows_iou\": " << s.rows_iou << ",\n";
    std::cout << "  \"columns_iou\": " << s.columns_iou << ",\n";
    std::cout << "  \"iou\": " << s._overall_iou << ",\n";
    std::cout << "  \"rows_set_sizes\": {\n";
    std::cout << "    \"A\": " << s.rows_set_a_size << ",\n";
    std::cout << "    \"B\": " << s.rows_set_b_size << ",\n";
    std::cout << "    \"intersection\": " << s.rows_intersection_size << ",\n";
    std::cout << "    \"union\": " << s.rows_union_size << "\n";
    std::cout << "  },\n";
    std::cout << "  \"duration_ms\": " << s.duration_ms;

    if (!s.error.empty()) {
        std::cout << ",\n  \"error\": \"" << s.error << "\"";
    }
    if (s.benchmark_ran) {
        std::cout << ",\n  \"benchmark\": {\n";
#if defined(RESULTCMP_HAS_OPENMP)
        std::cout << "    \"openmp\": true,\n";
#else
        std::cout << "    \"openmp\": false,\n";
#endif
        std::cout << "    \"iters\": " << s.benchmark_iters << ",\n";
        std::cout << "    \"serial\": {\"threads\": " << s.benchmark_serial_threads << ", \"ms\": " << s.benchmark_serial_ms << "},\n";
        std::cout << "    \"parallel\": {\"threads\": " << s.benchmark_parallel_threads << ", \"ms\": " << s.benchmark_parallel_ms << "},\n";
        const double speedup = (s.benchmark_parallel_ms > 0)
            ? (static_cast<double>(s.benchmark_serial_ms) / static_cast<double>(s.benchmark_parallel_ms))
            : 0.0;
        std::cout << "    \"speedup\": " << speedup << "\n";
        std::cout << "  }\n";
    }
    std::cout << "}\n";
}

// Print CLI usage to stderr
static void print_usage() {
    std::cerr << "Usage: resultcmp --actual A.csv --expected E.csv [--key k1,k2] [--case-insensitive] [--debug] [--threads N] [--benchmark] [--benchmark-iters K]\n";
}

// CLI entrypoint:
// - Parse args (actual/expected paths, keys, tolerances)
// - Load CSVs, compare, print JSON
// Exit code: 0 equal, 1 not equal, 2 usage/error
int main(int argc, char **argv) {
    std::cerr << "Starting C++ comparator" << std::endl;
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
        else if (arg == "--case-insensitive") opt.case_insensitive = true;
        else if (arg == "--debug") opt.debug = true;
        else if (arg == "--threads") opt.threads = std::stoi(next());
        else if (arg == "--benchmark") opt.benchmark = true;
        else if (arg == "--benchmark-iters") opt.benchmark_iters = std::max(1, std::stoi(next()));
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

    auto summary = compare_tables_multiset(ta, te, opt);

    if (opt.benchmark) {
        // Benchmark comparison only (CSV already loaded). We run multiple iterations and take the minimum
        // to reduce noise from scheduling/CPU frequency changes.
        const int iters = std::max(1, opt.benchmark_iters);

        Options serial_opt = opt;
        serial_opt.debug = false;
        serial_opt.threads = 1;

        Options par_opt = opt;
        par_opt.debug = false;
#if defined(RESULTCMP_HAS_OPENMP)
        if (par_opt.threads <= 0) {
            par_opt.threads = omp_get_max_threads();
        }
#else
        par_opt.threads = 1;
#endif

        // Warm-up (avoid first-run effects like allocation/caches)
        (void)compare_tables_multiset(ta, te, serial_opt);
        (void)compare_tables_multiset(ta, te, par_opt);

        const long long best_serial = util::measure_min_ms(iters, [&] {
            (void)compare_tables_multiset(ta, te, serial_opt);
        });
        const long long best_parallel = util::measure_min_ms(iters, [&] {
            (void)compare_tables_multiset(ta, te, par_opt);
        });

        summary.benchmark_ran = true;
        summary.benchmark_iters = iters;
        summary.benchmark_serial_threads = 1;
        summary.benchmark_parallel_threads = par_opt.threads;
        summary.benchmark_serial_ms = best_serial;
        summary.benchmark_parallel_ms = best_parallel;
        // If the dataset is tiny, millisecond resolution may be 0; fall back to at least 1ms for speedup readability.
        if (summary.benchmark_serial_ms == 0) summary.benchmark_serial_ms = 1;
        if (summary.benchmark_parallel_ms == 0) summary.benchmark_parallel_ms = 1;
    }

    print_json(summary);
    return summary.equal ? 0 : 1;
}


