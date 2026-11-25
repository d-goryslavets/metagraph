#ifndef __READ_EXTRACTION_HPP__
#define __READ_EXTRACTION_HPP__

#include <cstdint>
#include <functional>
#include <map>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "graph/annotated_dbg.hpp"
#include "annotation/int_matrix/base/int_matrix.hpp"


namespace mtg {
namespace graph {

struct ReadExtractionConfig {
    // number of nodes for which the annotations are decompressed at once
    uint64_t traversal_batch_size = 500;
    // number of columns (e.g., samples) to extract the reads from at once, 0 for all
    uint64_t column_batch_size = 0;
    // number of k-mers to add on each side of the span covering all queries
    uint64_t flank_length = 25'000;
    // maximum distance between the first and the last k-mer of the span covering all queries
    uint64_t max_query_distance = 25'000;
};


// A read (contig, genome) containing all queries, or a window of it if it is longer than the flanks
struct ExtractedRead {
    struct Match {
        uint64_t query_pos; // position of the first k-mer of the query found in the graph
        uint64_t read_pos;  // position of that k-mer in |sequence|
    };

    std::string sequence;
    // the sequence header if a CoordToHeader mapping is loaded, the annotation label otherwise
    std::string label;
    // the first match of each query in |sequence|, in the order of the queries
    std::vector<Match> matches;
};


/**
 * Extracts the reads that contain the queries from a graph with indexed k-mer coordinates.
 *
 * The graph is traversed from the k-mers of the queries in both directions, and the read is
 * reconstructed from the k-mers whose coordinates are consecutive.
 */
class ReadExtractor {
  public:
    // Throws std::invalid_argument if |anno_graph| has no k-mer coordinates or its graph is
    // not in the basic mode. |anno_graph| must outlive this object.
    ReadExtractor(const AnnotatedDBG &anno_graph, const ReadExtractionConfig &config = {});

    /**
     * Calls |callback| for each read that contains all |queries|.
     * The reads are passed one by one and are not kept in memory.
     *
     * K-mers missing in the graph are only allowed at the ends of a query, and are ignored.
     * Thread-safe.
     */
    void call_reads(std::span<const std::string_view> queries,
                    const std::function<void(ExtractedRead&&)> &callback) const;

  private:
    using Row = annot::matrix::BinaryMatrix::Row;
    using Column = annot::matrix::BinaryMatrix::Column;

    // rows (k-mers) of a single label, sorted by their coordinate
    using CoordMap = std::map<uint64_t, Row>;
    using Paths = std::unordered_map<Column, CoordMap>;
    using Boundaries = std::unordered_map<Column, std::set<uint64_t>>;

    // a query mapped to the graph
    struct Query {
        std::vector<Row> rows;
        uint64_t pos; // position of the first mapped k-mer
    };

    // coordinates from |start| to |end|, both included
    struct Interval {
        uint64_t start;
        uint64_t end;
    };
    using AdmissibleRanges = std::unordered_map<Column, std::vector<Interval>>;

    // the query is empty (no rows) if it can't be mapped to a path in the graph
    Query map_query(std::string_view sequence) const;

    // columns with k-mers that match the entire query and have consecutive coordinates
    std::vector<Column> get_columns_containing_query(const std::vector<Row> &rows) const;

    void extract_from_columns(const std::vector<Query> &queries,
                              std::span<const Column> columns,
                              const std::function<void(ExtractedRead&&)> &callback) const;

    Paths initialise_paths(const std::vector<Query> &queries,
                           const std::unordered_set<Column> &columns) const;

    // minimal spans covering at least one match of each query
    std::vector<Interval> find_query_clusters(const CoordMap &coords,
                                              const std::vector<Query> &queries) const;

    AdmissibleRanges build_admissible_ranges(const Paths &paths,
                                             const std::vector<Query> &queries) const;

    void compute_initial_boundaries(const Paths &paths,
                                    const AdmissibleRanges &valid_ranges,
                                    Boundaries *starts,
                                    Boundaries *ends) const;

    // |visited| is the set of rows already added to |paths|
    void traverse_direction(bool forward,
                            const AdmissibleRanges &valid_ranges,
                            std::unordered_set<Row> visited,
                            Boundaries *boundaries,
                            Paths *paths) const;

    void refine_boundaries(const Paths &paths,
                           const std::unordered_set<Column> &active_columns,
                           bool forward,
                           Boundaries *boundaries) const;

    // add the unvisited neighbors of the boundary k-mers to |queue| and
    // remove the columns without such neighbors from |active_columns|
    template <class Queue>
    void collect_frontier(const Paths &paths,
                          const Boundaries &boundaries,
                          const std::unordered_set<Row> &visited,
                          bool forward,
                          Queue *queue,
                          std::unordered_set<Column> *active_columns) const;

    void call_column_reads(Column column,
                           const CoordMap &coords,
                           const std::vector<Query> &queries,
                           const std::function<void(ExtractedRead&&)> &callback) const;

    bool has_edge(Row from, Row to) const;

    const AnnotatedDBG &anno_graph_;
    const DeBruijnGraph &graph_;
    const annot::matrix::MultiIntMatrix &matrix_;
    const ReadExtractionConfig config_;
};

} // namespace graph
} // namespace mtg

#endif // __READ_EXTRACTION_HPP__
