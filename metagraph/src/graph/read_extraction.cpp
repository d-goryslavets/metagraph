#include "read_extraction.hpp"

#include <algorithm>
#include <cassert>
#include <deque>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <tuple>

#include <fmt/format.h>

#include "annotation/coord_to_header.hpp"
#include "common/logger.hpp"
#include "graph/alignment/alignment.hpp"
#include "graph/representation/succinct/boss.hpp"


namespace mtg {
namespace graph {

using mtg::common::logger;
using mtg::annot::matrix::MultiIntMatrix;

using node_index = DeBruijnGraph::node_index;

namespace {

node_index to_node(annot::matrix::BinaryMatrix::Row row) {
    return AnnotatedDBG::anno_to_graph_index(row);
}

const MultiIntMatrix& get_coordinates(const AnnotatedDBG &anno_graph) {
    const auto *matrix = dynamic_cast<const MultiIntMatrix *>(
            &anno_graph.get_annotator().get_matrix());
    if (!matrix)
        throw std::invalid_argument("Annotation does not support k-mer coordinate queries");
    return *matrix;
}

// merge overlapping intervals
template <class Interval>
void merge_intervals(std::vector<Interval> *intervals) {
    std::sort(intervals->begin(), intervals->end(),
              [](const Interval &a, const Interval &b) { return a.start < b.start; });
    std::vector<Interval> merged;
    for (const Interval &interval : *intervals) {
        if (merged.size() && interval.start <= merged.back().end) {
            merged.back().end = std::max(merged.back().end, interval.end);
        } else {
            merged.push_back(interval);
        }
    }
    intervals->swap(merged);
}

// check if |coord| is in one of the sorted non-overlapping |ranges|
template <class Interval>
bool is_admissible(const std::vector<Interval> &ranges, uint64_t coord) {
    // the first interval that does not end before |coord|
    auto it = std::lower_bound(ranges.begin(), ranges.end(), coord,
        [](const Interval &interval, uint64_t coord) { return interval.end < coord; });
    return it != ranges.end() && coord >= it->start;
}

// check if the k-mers of the query have consecutive coordinates starting from |start|
template <class CoordMap, class Row>
bool query_matches_at(const CoordMap &coords, const std::vector<Row> &rows, uint64_t start) {
    auto it = coords.find(start);
    for (size_t i = 0; i < rows.size(); ++i, ++it) {
        if (it == coords.end() || it->first != start + i || it->second != rows[i])
            return false;
    }
    return true;
}

} // namespace


ReadExtractor::ReadExtractor(const AnnotatedDBG &anno_graph, const ReadExtractionConfig &config)
      : anno_graph_(anno_graph),
        graph_(anno_graph.get_graph()),
        matrix_(get_coordinates(anno_graph)),
        config_(config) {
    // In the other modes, the k-mers of a read are indexed in their canonical form,
    // and hence they don't form a path in the graph.
    if (graph_.get_mode() != DeBruijnGraph::BASIC)
        throw std::invalid_argument("Reads can only be extracted from graphs in the basic mode");

    if (!config_.traversal_batch_size)
        throw std::invalid_argument("The traversal batch size must be positive");

    const auto *coord_to_header = anno_graph_.get_coord_to_header();
    if (coord_to_header && coord_to_header->num_columns() != matrix_.get_binary_matrix().num_columns()) {
        throw std::invalid_argument(fmt::format(
            "Incompatible number of columns in CoordToHeader and annotation matrix: {} != {}",
            coord_to_header->num_columns(), matrix_.get_binary_matrix().num_columns()));
    }
}

void ReadExtractor::call_reads(std::span<const std::string_view> sequences,
                               const std::function<void(ExtractedRead&&)> &callback) const {
    if (sequences.empty())
        return;

    std::vector<Query> queries;
    queries.reserve(sequences.size());
    std::vector<Column> columns;

    for (std::string_view sequence : sequences) {
        queries.push_back(map_query(sequence));
        if (queries.back().rows.empty())
            return;

        auto query_columns = get_columns_containing_query(queries.back().rows);
        if (queries.size() == 1) {
            columns = std::move(query_columns);
        } else {
            std::vector<Column> common;
            std::set_intersection(columns.begin(), columns.end(),
                                  query_columns.begin(), query_columns.end(),
                                  std::back_inserter(common));
            columns = std::move(common);
        }
        if (columns.empty())
            return;
    }

    logger->trace("Extracting reads from {} labels matching the query", columns.size());

    const size_t batch_size = config_.column_batch_size
                                ? std::min<uint64_t>(config_.column_batch_size, columns.size())
                                : columns.size();
    for (size_t i = 0; i < columns.size(); i += batch_size) {
        extract_from_columns(queries,
                             std::span<const Column>(columns).subspan(
                                    i, std::min(batch_size, columns.size() - i)),
                             callback);
    }
}

ReadExtractor::Query ReadExtractor::map_query(std::string_view sequence) const {
    if (sequence.size() < graph_.get_k())
        return {};

    const std::vector<node_index> nodes = map_to_nodes(graph_, sequence);
    auto is_mapped = [](node_index node) { return node != DeBruijnGraph::npos; };
    auto begin = std::find_if(nodes.begin(), nodes.end(), is_mapped);
    if (begin == nodes.end())
        return {};

    auto end = std::find_if(nodes.rbegin(), nodes.rend(), is_mapped).base();

    // no read can contain a query with missing k-mers in the middle
    if (!std::all_of(begin, end, is_mapped))
        return {};

    Query query { .rows = {}, .pos = static_cast<uint64_t>(begin - nodes.begin()) };
    query.rows.reserve(end - begin);
    for (auto it = begin; it != end; ++it) {
        query.rows.push_back(AnnotatedDBG::graph_to_anno_index(*it));
    }
    return query;
}

std::vector<ReadExtractor::Column>
ReadExtractor::get_columns_containing_query(const std::vector<Row> &rows) const {
    const auto row_tuples = matrix_.get_row_tuples(rows);

    // the coordinates of the last k-mer for all matches of the query prefix
    std::unordered_map<Column, std::vector<uint64_t>> matches;
    for (const auto &[j, tuple] : row_tuples.front()) {
        matches.emplace(j, std::vector<uint64_t>(tuple.begin(), tuple.end()));
    }

    for (size_t i = 1; i < rows.size() && matches.size(); ++i) {
        std::unordered_map<Column, std::vector<uint64_t>> extended;
        for (const auto &[j, tuple] : row_tuples[i]) {
            auto it = matches.find(j);
            if (it == matches.end())
                continue;

            std::vector<uint64_t> coords;
            for (uint64_t coord : tuple) {
                if (coord && std::binary_search(it->second.begin(), it->second.end(), coord - 1))
                    coords.push_back(coord);
            }
            if (coords.size())
                extended.emplace(j, std::move(coords));
        }
        matches = std::move(extended);
    }

    std::vector<Column> columns;
    columns.reserve(matches.size());
    for (const auto &[j, coords] : matches) {
        columns.push_back(j);
    }
    std::sort(columns.begin(), columns.end());
    return columns;
}

void ReadExtractor::extract_from_columns(const std::vector<Query> &queries,
                                         std::span<const Column> columns,
                                         const std::function<void(ExtractedRead&&)> &callback) const {
    Paths paths = initialise_paths(queries, { columns.begin(), columns.end() });

    const AdmissibleRanges valid_ranges = build_admissible_ranges(paths, queries);
    if (valid_ranges.empty())
        return;

    Boundaries starts;
    Boundaries ends;
    compute_initial_boundaries(paths, valid_ranges, &starts, &ends);

    std::unordered_set<Row> query_rows;
    for (const Query &query : queries) {
        query_rows.insert(query.rows.begin(), query.rows.end());
    }

    logger->trace("Traversing the graph forwards");
    traverse_direction(true, valid_ranges, query_rows, &ends, &paths);

    logger->trace("Traversing the graph backwards");
    traverse_direction(false, valid_ranges, std::move(query_rows), &starts, &paths);

    logger->trace("Reconstructing the paths");
    for (Column j : columns) {
        // the k-mers of a column are not needed after its reads are called
        auto column_path = paths.extract(j);
        if (column_path && valid_ranges.count(j))
            call_column_reads(j, column_path.mapped(), queries, callback);
    }
}

ReadExtractor::Paths
ReadExtractor::initialise_paths(const std::vector<Query> &queries,
                                const std::unordered_set<Column> &columns) const {
    std::vector<Row> rows;
    for (const Query &query : queries) {
        rows.insert(rows.end(), query.rows.begin(), query.rows.end());
    }

    const auto row_tuples = matrix_.get_row_tuples_in_columns(rows, columns);

    Paths paths;
    for (size_t i = 0; i < rows.size(); ++i) {
        for (const auto &[j, coords] : row_tuples[i]) {
            CoordMap &path = paths[j];
            for (uint64_t coord : coords) {
                path.emplace(coord, rows[i]);
            }
        }
    }
    return paths;
}

std::vector<ReadExtractor::Interval>
ReadExtractor::find_query_clusters(const CoordMap &coords,
                                   const std::vector<Query> &queries) const {
    struct Occurrence {
        Interval interval;
        size_t query_id;
    };

    std::vector<Occurrence> occurrences;
    for (size_t q = 0; q < queries.size(); ++q) {
        const std::vector<Row> &rows = queries[q].rows;
        for (const auto &[coord, row] : coords) {
            if (row == rows.front() && query_matches_at(coords, rows, coord))
                occurrences.push_back({ { coord, coord + rows.size() - 1 }, q });
        }
    }

    std::sort(occurrences.begin(), occurrences.end(), [](const auto &a, const auto &b) {
        return std::tie(a.interval.start, a.query_id) < std::tie(b.interval.start, b.query_id);
    });

    // slide a window over the occurrences and report it whenever it covers all queries
    std::vector<Interval> clusters;
    std::vector<size_t> count(queries.size(), 0);
    std::multiset<uint64_t> ends;
    size_t num_covered = 0;
    auto left = occurrences.begin();

    for (const auto &[interval, query_id] : occurrences) {
        if (!count[query_id]++)
            ++num_covered;
        ends.insert(interval.end);

        while (num_covered == queries.size()) {
            const Interval span { left->interval.start, *ends.rbegin() };
            if (span.end - span.start <= config_.max_query_distance)
                clusters.push_back(span);

            ends.erase(ends.find(left->interval.end));
            if (!--count[left->query_id])
                --num_covered;
            ++left;
        }
    }
    return clusters;
}

ReadExtractor::AdmissibleRanges
ReadExtractor::build_admissible_ranges(const Paths &paths,
                                       const std::vector<Query> &queries) const {
    const uint64_t flank = config_.flank_length;

    AdmissibleRanges valid_ranges;
    for (const auto &[j, coords] : paths) {
        std::vector<Interval> ranges = find_query_clusters(coords, queries);
        if (ranges.empty())
            continue;

        for (auto &[start, end] : ranges) {
            start -= std::min(start, flank);
            end += std::min(std::numeric_limits<uint64_t>::max() - end, flank);
        }
        merge_intervals(&ranges);
        valid_ranges.emplace(j, std::move(ranges));
    }
    return valid_ranges;
}

void ReadExtractor::compute_initial_boundaries(const Paths &paths,
                                               const AdmissibleRanges &valid_ranges,
                                               Boundaries *starts,
                                               Boundaries *ends) const {
    for (const auto &[j, coords] : paths) {
        if (!valid_ranges.count(j))
            continue;

        // a segment breaks wherever the coordinates are not consecutive or
        // there is no edge between the k-mers (they belong to different reads)
        auto prev = coords.begin();
        (*starts)[j].insert(prev->first);
        for (auto it = std::next(prev); it != coords.end(); prev = it++) {
            if (it->first != prev->first + 1 || !has_edge(prev->second, it->second)) {
                (*ends)[j].insert(prev->first);
                (*starts)[j].insert(it->first);
            }
        }
        (*ends)[j].insert(prev->first);
    }
}

void ReadExtractor::traverse_direction(bool forward,
                                       const AdmissibleRanges &valid_ranges,
                                       std::unordered_set<Row> visited,
                                       Boundaries *boundaries,
                                       Paths *paths) const {
    std::unordered_set<Column> active_columns;
    for (const auto &[j, _] : valid_ranges) {
        active_columns.insert(j);
    }

    std::deque<Row> queue;
    collect_frontier(*paths, *boundaries, visited, forward, &queue, &active_columns);

    while (queue.size()) {
        std::vector<Row> batch;
        while (queue.size() && batch.size() < config_.traversal_batch_size) {
            Row row = queue.front();
            queue.pop_front();
            if (visited.insert(row).second)
                batch.push_back(row);
        }

        if (batch.size()) {
            const auto row_tuples = matrix_.get_row_tuples_in_columns(batch, active_columns);

            for (size_t i = 0; i < batch.size(); ++i) {
                for (const auto &[j, coords] : row_tuples[i]) {
                    const auto &ranges = valid_ranges.at(j);
                    CoordMap &path = paths->at(j);
                    for (uint64_t coord : coords) {
                        if (is_admissible(ranges, coord))
                            path.emplace(coord, batch[i]);
                    }
                }
            }

            refine_boundaries(*paths, active_columns, forward, boundaries);
        }

        collect_frontier(*paths, *boundaries, visited, forward, &queue, &active_columns);
    }
}

void ReadExtractor::refine_boundaries(const Paths &paths,
                                      const std::unordered_set<Column> &active_columns,
                                      bool forward,
                                      Boundaries *boundaries) const {
    for (auto &[j, coords] : *boundaries) {
        if (!active_columns.count(j))
            continue;

        // move each boundary along the path as far as possible
        const CoordMap &path = paths.at(j);
        std::set<uint64_t> refined;
        for (uint64_t coord : coords) {
            auto it = path.find(coord);
            assert(it != path.end());
            if (forward) {
                for (auto next = std::next(it); next != path.end()
                                                && next->first == it->first + 1
                                                && has_edge(it->second, next->second); ++next) {
                    it = next;
                }
            } else {
                while (it != path.begin()) {
                    auto prev = std::prev(it);
                    if (prev->first + 1 != it->first || !has_edge(prev->second, it->second))
                        break;
                    it = prev;
                }
            }
            refined.insert(it->first);
        }
        coords.swap(refined);
    }
}

template <class Queue>
void ReadExtractor::collect_frontier(const Paths &paths,
                                     const Boundaries &boundaries,
                                     const std::unordered_set<Row> &visited,
                                     bool forward,
                                     Queue *queue,
                                     std::unordered_set<Column> *active_columns) const {
    for (const auto &[j, coords] : boundaries) {
        if (!active_columns->count(j))
            continue;

        const CoordMap &path = paths.at(j);
        bool finished = true;
        for (uint64_t coord : coords) {
            const node_index node = to_node(path.at(coord));
            for (auto [next, c] : forward ? graph_.outgoing_kmers(node)
                                          : graph_.incoming_kmers(node)) {
                if (c == boss::BOSS::kSentinel)
                    continue;

                const Row next_row = AnnotatedDBG::graph_to_anno_index(next);
                if (!visited.count(next_row)) {
                    // the read possibly continues in this k-mer
                    queue->push_front(next_row);
                    finished = false;
                }
            }
        }

        // if all neighbors of the read ends are visited and don't continue the reads,
        // the annotations of this column don't have to be decompressed anymore
        if (finished)
            active_columns->erase(j);
    }
}

void ReadExtractor::call_column_reads(Column column,
                                      const CoordMap &coords,
                                      const std::vector<Query> &queries,
                                      const std::function<void(ExtractedRead&&)> &callback) const {
    const auto *coord_to_header = anno_graph_.get_coord_to_header();
    const std::string &column_label = anno_graph_.get_annotator().get_label_encoder().decode(column);

    std::set<std::pair<uint64_t, uint64_t>> called;

    for (const Interval &cluster : find_query_clusters(coords, queries)) {
        // coordinates of the read (if known)
        std::pair<uint64_t, uint64_t> read { 0, std::numeric_limits<uint64_t>::max() };
        const std::string *label = &column_label;
        if (coord_to_header) {
            const size_t seq_id = coord_to_header->map_single_coord(column, cluster.start).first;
            read = coord_to_header->sequence_coords(column, seq_id);
            label = &coord_to_header->get_headers(column)[seq_id];
        }

        // The last k-mer of a read and the first k-mer of the next read in the same column
        // have consecutive coordinates as well. If the coordinates of the read are not known,
        // the reads are told apart by there being no edge between these k-mers.
        auto in_same_read = [&](auto it, auto next) {
            return next->first == it->first + 1 && has_edge(it->second, next->second);
        };

        // all queries must match in the same read
        auto begin = coords.find(cluster.start);
        auto end = std::next(begin);
        bool same_read = cluster.end <= read.second;
        while (same_read && std::prev(end)->first < cluster.end) {
            same_read = end != coords.end() && in_same_read(std::prev(end), end);
            if (same_read)
                ++end;
        }
        if (!same_read)
            continue;

        // add the flanks, up to the ends of the read
        while (begin != coords.begin() && begin->first > read.first
                && cluster.start - begin->first < config_.flank_length
                && in_same_read(std::prev(begin), begin)) {
            --begin;
        }
        while (end != coords.end() && end->first <= read.second
                && end->first - cluster.end <= config_.flank_length
                && in_same_read(std::prev(end), end)) {
            ++end;
        }

        const uint64_t window_start = begin->first;
        const uint64_t window_end = std::prev(end)->first;
        if (!called.emplace(window_start, window_end).second)
            continue;

        std::vector<node_index> nodes;
        nodes.reserve(window_end - window_start + 1);
        for (auto it = begin; it != end; ++it) {
            nodes.push_back(to_node(it->second));
        }

        std::vector<ExtractedRead::Match> matches;
        matches.reserve(queries.size());
        for (const Query &query : queries) {
            // each query has a match in the window, as it covers the cluster
            auto it = begin;
            while (it->second != query.rows.front()
                    || window_end - it->first + 1 < query.rows.size()
                    || !query_matches_at(coords, query.rows, it->first)) {
                ++it;
                assert(it != end);
            }
            matches.push_back({ query.pos, it->first - window_start });
        }

        callback(ExtractedRead { align::spell_path(graph_, nodes), *label, std::move(matches) });
    }
}

bool ReadExtractor::has_edge(Row from, Row to) const {
    const auto next_nodes = graph_.adjacent_outgoing_nodes(to_node(from));
    return std::find(next_nodes.begin(), next_nodes.end(), to_node(to)) != next_nodes.end();
}

} // namespace graph
} // namespace mtg
