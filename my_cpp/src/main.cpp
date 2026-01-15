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

// Command-line options controlling comparison behavior
struct Options {
    std::string actual_path;
    std::string expected_path;
    std::vector<std::string> keys; // join keys for order-insensitive compare
    bool json = true;
    bool case_insensitive = false;
};

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
    return out;
}

// Helper: split a CSV line and drop the first field (used to skip index columns)
static std::vector<std::string> split_csv_line_skip_first(const std::string &line) {
    std::vector<std::string> out = split_csv_line(line);
    if (!out.empty()) out.erase(out.begin());
    return out;
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
    // Parse header and skip first field (index)
    t.columns = split_csv_line(line);
    if (t.columns.empty()) { err = "Malformed header in file: " + path; return false; }
    t.col_index.clear();
    for (size_t i = 0; i < t.columns.size(); ++i) t.col_index[t.columns[i]] = i;
    while (std::getline(f, line)) {
        auto parts = split_csv_line_skip_first(line);
        // pad/truncate to header size
        parts.resize(t.columns.size());
        t.rows.push_back(std::move(parts));
    }
    return true;
}

// Build composite key from selected columns in a row
static std::string make_key(const std::vector<std::string> &keys, const Table &t, const std::vector<std::string> &row) {
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
    std::cerr << "Key: " << oss.str() << std::endl;
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
    (void)opt;
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
    std::cerr << "Common columns: " << std::endl;
    for (const auto &c : common_cols) std::cerr << c << std::endl;
    std::multiset<std::string> cols_union = union_multiset(cols_a, cols_b);
    s.columns_iou = !cols_union.empty()
        ? static_cast<double>(cols_intersection.size()) / static_cast<double>(cols_union.size())
        : 0.0;
    // Print cols a and b in the same line, comma separated
    std::cerr << "Cols A: ";
    for (auto it = cols_a.begin(); it != cols_a.end(); ++it) {
        if (it != cols_a.begin()) std::cerr << ",";
        std::cerr << *it;
    }
    std::cerr << std::endl;

    std::cerr << "Cols B: ";
    for (auto it = cols_b.begin(); it != cols_b.end(); ++it) {
        if (it != cols_b.begin()) std::cerr << ",";
        std::cerr << *it;
    }
    std::cerr << std::endl;

    std::cerr << "Cols Intersection: ";
    for (auto it = cols_intersection.begin(); it != cols_intersection.end(); ++it) {
        if (it != cols_intersection.begin()) std::cerr << ",";
        std::cerr << *it;
    }
    std::cerr << std::endl;

    std::cerr << "Cols Union: ";
    for (auto it = cols_union.begin(); it != cols_union.end(); ++it) {
        if (it != cols_union.begin()) std::cerr << ",";
        std::cerr << *it;
    }
    std::cerr << std::endl;
    // 2) Rows IoU computed on the intersection of columns:
    //    - Build a canonical string key per row using only common columns (trimmed; optional lowercase)
    //    - Compare sets of unique row keys across the two tables
    if (!common_cols.empty()) {
        std::multiset<std::string> set_a;
        std::multiset<std::string> set_b;
        std::cerr << "Set A: " << std::endl;
        for (const auto &ra : a.rows) set_a.insert(make_key(common_cols, a, ra));
        for (const auto &rb : b.rows) set_b.insert(make_key(common_cols, b, rb));
        std::cerr << "Set B: " << std::endl;
        for (const auto &rb : set_b) std::cerr << rb << std::endl;
        std::multiset<std::string> rows_intersection = intersection_multiset(set_a, set_b);
        std::multiset<std::string> rows_union = union_multiset(set_a, set_b);
        s.rows_intersection_size = rows_intersection.size();
        s.rows_union_size = rows_union.size();
        s.rows_iou = (rows_union.size() > 0) ? static_cast<double>(rows_intersection.size()) / static_cast<double>(rows_union.size()) : 0.0;
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
    (void)opt;
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
    std::cerr << "Common columns: " << std::endl;
    for (const auto &c : common_cols) std::cerr << c << std::endl;

    std::set<std::string> cols_union = union_set(cols_a, cols_b);
    s.columns_iou = !cols_union.empty()
        ? static_cast<double>(cols_intersection.size()) / static_cast<double>(cols_union.size())
        : 0.0;
    
    // 2) Rows IoU computed on the intersection of columns:
    //    - Build a canonical string key per row using only common columns (trimmed; optional lowercase)
    //    - Compare sets of unique row keys across the two tables
    if (!common_cols.empty()) {
        std::set<std::string> set_a;
        std::set<std::string> set_b;
        std::cerr << "Set A: " << std::endl;
        for (const auto &ra : a.rows) set_a.insert(make_key(common_cols, a, ra));
        for (const auto &rb : b.rows) set_b.insert(make_key(common_cols, b, rb));
        std::cerr << "Set B: " << std::endl;
        for (const auto &rb : set_b) std::cerr << rb << std::endl;
        std::set<std::string> rows_intersection = intersection_set(set_a, set_b);
        std::set<std::string> rows_union = union_set(set_a, set_b);
        s.rows_intersection_size = rows_intersection.size();
        s.rows_union_size = rows_union.size();
        s.rows_iou = (rows_union.size() > 0) ? static_cast<double>(rows_intersection.size()) / static_cast<double>(rows_union.size()) : 0.0;
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
        std::cout << ",\n  \"error\": \"" << s.error << "\"\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "}\n";
}

// Print CLI usage to stderr
static void print_usage() {
    std::cerr << "Usage: resultcmp --actual A.csv --expected E.csv [--key k1,k2] [--case-insensitive]\n";
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
    print_json(summary);
    return summary.equal ? 0 : 1;
}


