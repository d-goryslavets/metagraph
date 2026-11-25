#ifndef __TUPLE_ROW_DIFF_HPP__
#define __TUPLE_ROW_DIFF_HPP__

#include <algorithm>
#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <queue>
#include <deque>
#include <span>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include <utility>

#include "common/vectors/bit_vector_adaptive.hpp"
#include "common/vector_map.hpp"
#include "common/vector.hpp"
#include "common/logger.hpp"
#include "common/utils/template_utils.hpp"
#include "graph/annotated_dbg.hpp"
#include "graph/representation/succinct/boss.hpp"
#include "graph/representation/succinct/dbg_succinct.hpp"
#include "annotation/binary_matrix/row_diff/row_diff.hpp"
#include "annotation/int_matrix/base/int_matrix.hpp"


namespace mtg {
namespace annot {
namespace matrix {


// Detection idiom to check at compile time if class has get_row_tuples_labelled method
template <typename T, typename = void>
struct has_get_row_tuples_labelled : std::false_type {};

template <typename T>
struct has_get_row_tuples_labelled<
    T,
    std::void_t<decltype(std::declval<const T&>().get_row_tuples_labelled(
        std::declval<std::span<const BinaryMatrix::Row>>(),
        std::declval<const std::unordered_set<BinaryMatrix::Column>&>(),
        std::declval<size_t>()))>> : std::true_type {};


class ITupleRowDiff {
  public:
    virtual ~ITupleRowDiff() = default;

    virtual std::unordered_set<BinaryMatrix::Column> get_samples_containing_query(const std::vector<BinaryMatrix::Row> &i) const = 0;

    virtual std::vector<std::tuple<std::vector<BinaryMatrix::Row>, BinaryMatrix::Column, uint64_t>> get_traces_with_row_labelled(
        const std::vector<std::vector<BinaryMatrix::Row>>& queries,
        const std::unordered_set<BinaryMatrix::Column>& samples_with_query,
        uint64_t traversal_batch_size,
        uint64_t flank_length,
        uint64_t max_query_distance) const = 0;

};

template <class BaseMatrix>
class TupleRowDiff : public ITupleRowDiff, public IRowDiff, public BinaryMatrix, public MultiIntMatrix {
  public:
    // 1. Define the structures inside the class (Public so the caller can use them)
    struct Interval {
        uint64_t start;
        uint64_t end;
    };

    static_assert(std::is_convertible<BaseMatrix*, MultiIntMatrix*>::value);
    static const int SHIFT = 1; // coordinates increase by 1 at each edge

    // check graph traversal in batches
    static const uint64_t TRAVERSAL_BATCH_SIZE = 500; //  50'000

    // TODO: implement this as a configurable command line parameter
    // preferably optional. Then, if no value is passed, the fallback is 
    // to assume that the sequences can be of arbitrary length
    static const uint64_t MAX_READ_LENGTH = 25'000; // PacBio HIFI reads

    // TupleRowDiff() {}

    template <typename... Args>
    TupleRowDiff(const graph::DeBruijnGraph *graph = nullptr, Args&&... args)
        : diffs_(std::forward<Args>(args)...) { graph_ = graph; }

    std::vector<Row> get_column(Column column) const override {
        return get_column_from_rows(*this, column);
    }
    std::vector<SetBitPositions> get_rows(std::span<const Row> rows) const override;
    RowTuples get_row_tuples(Row i) const;

    /** Returns all labeled traces that pass through a given row.
     * 
     * @param i Index of the row.
     * @param auto_labels If true, the read labels will be derived based on k-mer coordinates.
     * Use if the graph doesn't have distinct labels for different reads.
     * 
     * @return Vector of pairs (path, column), where path is
     * a vector of Row indices and column is a corresponding Label index.
     */


    // auto_labels means that the labels will be derived based on coordinates
    // this is needed for the graphs where reads are not marked with different labels

    // returns reads for a set of columns (samples)
    std::vector<std::tuple<std::vector<Row>, Column, uint64_t>> get_traces_with_row_labelled(
        const std::vector<Row>& i,
        const std::unordered_set<Column>& samples_with_query,
        uint64_t traversal_batch_size) const;

    std::vector<std::tuple<std::vector<Row>, Column, uint64_t>> get_traces_with_row_labelled(
        const std::vector<std::vector<Row>>& queries,
        const std::unordered_set<Column>& samples_with_query,
        uint64_t traversal_batch_size,
        uint64_t flank_length,
        uint64_t max_query_distance) const override;

    // pre-traverse graph to find samples with reads containing full query sequence
    std::unordered_set<Column> get_samples_containing_query(const std::vector<Row> &i) const override;

    std::vector<std::unordered_set<uint64_t>> get_labels_of_rows(const std::vector<Row> &i, size_t num_threads = 1) const;
    
    // no deduplication: see class comment on RowDiff::get_rows_dict (speed vs limited size win)
    std::vector<SetBitPositions>
    get_rows_dict(std::vector<Row> *rows, size_t num_threads) const override;
    std::vector<RowValues> get_row_values(std::span<const Row> rows,
                                          size_t num_threads = 1) const override;
    std::vector<RowTuples> get_row_tuples(std::span<const Row> rows,
                                          size_t num_threads = 1) const override;
    bool get_row_tuples_cached(std::span<const Row> rows, RowDiffCache<RowTuples> &cache,
                               std::vector<RowTuples> *out) const override {
        *out = decode_rows_cached(rows, cache,
            [this](std::span<const Row> rd_ids) { return diffs_.get_row_tuples(rd_ids, 1); },
            add_diff, decode_diffs);
        return true;
    }

    // used in read extraction query to limit decompression to samples of interest only
    std::vector<RowTuples> get_row_tuples_labelled(std::span<const Row> rows, 
                                                  const std::unordered_set<Column> &labels_of_interest, 
                                                  size_t num_threads = 1) const;


    uint64_t num_columns() const override { return diffs_.num_columns(); }
    uint64_t num_relations() const override { return diffs_.num_relations(); }
    uint64_t num_attributes() const override { return diffs_.num_attributes(); }
    uint64_t num_rows() const override { return diffs_.num_rows(); }

    bool load(std::istream &in) override;
    void serialize(std::ostream &out) const override;

    const BaseMatrix& diffs() const { return diffs_; }
    BaseMatrix& diffs() { return diffs_; }

    const BinaryMatrix& get_binary_matrix() const override { return *this; }

  private:
    static void decode_diffs(RowTuples *diffs);
    static void add_diff(const RowTuples &diff, RowTuples *row);
    static void add_diff_labelled(const RowTuples &diff, RowTuples *row, const std::unordered_set<Column> &labels_of_interest);

    using CoordMap = std::map<uint64_t, Row>;
    using AdmissibleRanges = std::unordered_map<Column, std::vector<Interval>>;

    struct QueryCluster {
        uint64_t min_start;
        uint64_t max_end;
    };


    // 2. Function declarations
    bool query_matches_at(
        const CoordMap& map,
        const std::vector<Row>& query,
        uint64_t start
    ) const;

    void find_query_clusters(
        const CoordMap& map,
        const std::vector<std::vector<Row>>& queries,
        uint64_t max_query_distance,
        std::vector<QueryCluster>& clusters
    ) const;

    void merge_intervals(std::vector<Interval>& intervals) const;

    void build_admissible_ranges(
        const std::unordered_map<Column, CoordMap>& paths,
        const std::vector<std::vector<Row>>& queries,
        uint64_t flank_length,
        uint64_t max_query_distance,
        AdmissibleRanges& col_to_ranges
    ) const;

    bool is_admissible(
        const std::vector<Interval>& ranges, 
        uint64_t x
    ) const;

    void initialise_paths(
        const std::vector<std::vector<Row>>& queries,
        const std::unordered_set<Column>& samples,
        std::unordered_map<Column, CoordMap>& paths
    ) const;

    void compute_initial_boundaries(
        const std::unordered_map<Column, CoordMap>& paths,
        const AdmissibleRanges& valid_ranges,
        std::unordered_map<Column, std::set<uint64_t>>& starts,
        std::unordered_map<Column, std::set<uint64_t>>& ends
    ) const;

    void traverse_direction(
        bool forward,
        const std::vector<Row>& seeds,
        std::unordered_map<Column, CoordMap>& paths,
        std::unordered_map<Column, std::set<uint64_t>>& boundaries,
        std::unordered_set<Column>& active_samples,
        uint64_t batch_size, 
        AdmissibleRanges& valid_ranges,
        std::unordered_set<Row> visited
    ) const;

    void refine_boundaries(
        const std::unordered_map<Column, CoordMap>& paths,
        std::unordered_map<Column, std::set<uint64_t>>& boundaries,
        const std::unordered_set<Column>& active_samples,
        bool forward
    ) const;

    void collect_frontier(
        const std::unordered_map<Column, CoordMap>& paths,
        std::unordered_map<Column, std::set<uint64_t>>& boundaries,
        const std::unordered_set<Row>& visited,
        std::deque<Row>& queue,
        std::unordered_set<Column>& active_samples,
        bool forward, 
        AdmissibleRanges& valid_ranges
    ) const;

    void build_result(
        const std::unordered_map<Column, CoordMap>& paths,
        const std::vector<std::vector<Row>>& queries,
        uint64_t flank_length,
        uint64_t max_query_distance,
        std::vector<std::tuple<std::vector<Row>, Column, uint64_t>>& result
    ) const;

    BaseMatrix diffs_;
};


template <class BaseMatrix>
std::vector<BinaryMatrix::SetBitPositions>
TupleRowDiff<BaseMatrix>::get_rows(std::span<const Row> row_ids) const {
    std::vector<SetBitPositions> rows(row_ids.size());
    call_rows(row_ids,
        [this](std::span<const Row> rd_ids, size_t num_threads) {
            return diffs_.get_row_tuples(rd_ids, num_threads);
        },
        add_diff, decode_diffs,
        [&](size_t i, const RowTuples &row) { rows[i] = utils::get_firsts<SetBitPositions>(row); },
        1
    );
    return rows;
}

template <class BaseMatrix>
std::vector<BinaryMatrix::SetBitPositions>
TupleRowDiff<BaseMatrix>::get_rows_dict(std::vector<Row> *rows, size_t num_threads) const {
    std::vector<SetBitPositions> rows_dict(rows->size());
    call_rows(*rows,
        [this](std::span<const Row> rd_ids, size_t num_threads) {
            return diffs_.get_row_tuples(rd_ids, num_threads);
        },
        add_diff, decode_diffs,
        [&](size_t i, const RowTuples &row) {
            rows_dict[i] = utils::get_firsts<SetBitPositions>(row);
            (*rows)[i] = i;
        },
        num_threads
    );
    return rows_dict;
}

template <class BaseMatrix>
std::vector<MultiIntMatrix::RowValues>
TupleRowDiff<BaseMatrix>::get_row_values(std::span<const Row> row_ids, size_t num_threads) const {
    std::vector<RowValues> rows(row_ids.size());
    call_rows(row_ids,
        [this](std::span<const Row> rd_ids, size_t num_threads) {
            return diffs_.get_row_tuples(rd_ids, num_threads);
        },
        add_diff, decode_diffs,
        [&](size_t i, const RowTuples &row) {
            RowValues &row_values = rows[i];
            row_values.reserve(row.size());
            for (const auto &[j, tuple] : row) {
                row_values.emplace_back(j, tuple.size());
            }
        },
        num_threads
    );
    return rows;
}

template <class BaseMatrix>
MultiIntMatrix::RowTuples TupleRowDiff<BaseMatrix>::get_row_tuples(Row row) const {
    return get_row_tuples(std::vector<Row>{ row })[0];
}

template <class BaseMatrix>
std::vector<MultiIntMatrix::RowTuples>
TupleRowDiff<BaseMatrix>::get_row_tuples(std::span<const Row> row_ids, size_t num_threads) const {
    std::vector<RowTuples> rows(row_ids.size());
    call_rows(row_ids,
        [this](std::span<const Row> rd_ids, size_t num_threads) {
            return diffs_.get_row_tuples(rd_ids, num_threads);
        },
        add_diff, decode_diffs,
        [&](size_t i, const RowTuples &row) { rows[i] = row; },
        num_threads
    );
    return rows;
}

template <class BaseMatrix>
std::vector<MultiIntMatrix::RowTuples>
TupleRowDiff<BaseMatrix>::get_row_tuples_labelled(
        std::span<const Row> row_ids,
        const std::unordered_set<Column> &labels_of_interest,
        size_t num_threads) const {
    if constexpr (has_get_row_tuples_labelled<BaseMatrix>::value) {
        std::vector<RowTuples> rows(row_ids.size());
        call_rows(row_ids,
            [this, &labels_of_interest](std::span<const Row> rd_ids, size_t num_threads) {
                return diffs_.get_row_tuples_labelled(rd_ids, labels_of_interest, num_threads);
            },
            [&labels_of_interest](const RowTuples &diff, RowTuples *row) {
                return add_diff_labelled(diff, row, labels_of_interest);
            },
            decode_diffs,
            [&](size_t i, const RowTuples &row) { rows[i] = row; },
            num_threads
        );
        return rows;
    } else {
        throw std::runtime_error("get_row_tuples_labelled is not supported for this BaseMatrix");
    }
}


template <class BaseMatrix>
bool TupleRowDiff<BaseMatrix>::load(std::istream &in) {
    std::string version(4, '\0');
    in.read(version.data(), 4);
    auto anchor_start = in.tellg();
    if (!anchor_.load(in) || !fork_succ_.load(in))
        return false;
    // anchor_ / fork_succ_ are accessed randomly, hint the kernel.
    utils::madvise_random_range(in, anchor_start, in.tellg() - anchor_start);
    return diffs_.load(in);
}

template <class BaseMatrix>
void TupleRowDiff<BaseMatrix>::serialize(std::ostream &out) const {
    out.write("v2.0", 4);
    anchor_.serialize(out);
    fork_succ_.serialize(out);
    diffs_.serialize(out);
}

template <class BaseMatrix>
void TupleRowDiff<BaseMatrix>::decode_diffs(RowTuples *diffs) {
    std::ignore = diffs;
    // no encoding
}

template <class BaseMatrix>
void TupleRowDiff<BaseMatrix>::add_diff(const RowTuples &diff, RowTuples *row) {
    assert(std::is_sorted(row->begin(), row->end()));
    assert(std::is_sorted(diff.begin(), diff.end()));

    if (diff.size()) {
        RowTuples result;
        result.reserve(row->size() + diff.size());

        auto it = row->begin();
        auto it2 = diff.begin();
        while (it != row->end() && it2 != diff.end()) {
            if (it->first < it2->first) {
                result.push_back(*it);
                ++it;
            } else if (it->first > it2->first) {
                result.push_back(*it2);
                ++it2;
            } else {
                if (it2->second.size()) {
                    result.emplace_back(it->first, Tuple{});
                    std::set_symmetric_difference(it->second.begin(), it->second.end(),
                                                  it2->second.begin(), it2->second.end(),
                                                  std::back_inserter(result.back().second));
                    // just for safety, normally rows without coordinates shouldn't be annotated
                    if (result.back().second.empty())
                        result.pop_back();
                }
                ++it;
                ++it2;
            }
        }
        std::copy(it, row->end(), std::back_inserter(result));
        std::copy(it2, diff.end(), std::back_inserter(result));

        row->swap(result);
    }

    assert(std::is_sorted(row->begin(), row->end()));
    assert(std::all_of(row->begin(), row->end(),
                       [](auto &p) { return p.second.size(); }));
    for (auto &[j, tuple] : *row) {
        for (uint64_t &c : tuple) {
            c -= SHIFT;
        }
        assert(std::is_sorted(tuple.begin(), tuple.end()));
    }
}

template <class BaseMatrix>
void TupleRowDiff<BaseMatrix>::add_diff_labelled(const RowTuples &diff, RowTuples *row, const std::unordered_set<Column> &labels_of_interest) {
    assert(std::is_sorted(row->begin(), row->end()));
    assert(std::is_sorted(diff.begin(), diff.end()));

    if (diff.size()) {
        RowTuples result;
        result.reserve(row->size() + diff.size());

        auto it = row->begin();
        auto it2 = diff.begin();
        while (it != row->end() && it2 != diff.end()) {
            if (it->first < it2->first) {
                if (labels_of_interest.count(it->first))
                    result.push_back(*it);
                ++it;
            } else if (it->first > it2->first) {
                if (labels_of_interest.count(it2->first))
                    result.push_back(*it2);
                ++it2;
            } else if (it->first == it2->first && labels_of_interest.count(it2->first)) {
                if (it2->second.size()) {
                    result.emplace_back(it->first, Tuple{});
                    std::set_symmetric_difference(it->second.begin(), it->second.end(),
                                                  it2->second.begin(), it2->second.end(),
                                                  std::back_inserter(result.back().second));
                    // just for safety, normally rows without coordinates shouldn't be annotated
                    if (result.back().second.empty())
                        result.pop_back();
                }
                ++it;
                ++it2;
            } else {
                ++it;
                ++it2;
            }
        }
        for (; it != row->end(); ++it) {
            if (labels_of_interest.count(it->first))
                result.push_back(*it);
        }
        for (; it2 != diff.end(); ++it2) {
            if (labels_of_interest.count(it2->first))
                result.push_back(*it2);
        }

        row->swap(result);
    }

    assert(std::is_sorted(row->begin(), row->end()));
    assert(std::all_of(row->begin(), row->end(),
                       [](auto &p) { return p.second.size(); }));
    for (auto &[j, tuple] : *row) {
        for (uint64_t &c : tuple) {
            c -= SHIFT;
        }
        assert(std::is_sorted(tuple.begin(), tuple.end()));
    }
}

template <class BaseMatrix>
std::vector<std::unordered_set<uint64_t>> TupleRowDiff<BaseMatrix>
::get_labels_of_rows(const std::vector<Row> &i, size_t num_threads) const {
    std::vector<std::unordered_set<uint64_t>> result;
    auto row_tuples = get_row_tuples(i, num_threads);

    for (auto &rowt : row_tuples) {
        std::unordered_set<uint64_t> labels_set;
        for (auto &[j, tuple] : rowt)
            labels_set.insert(j);
        result.push_back(labels_set);
    }

    return result;
}


template <class BaseMatrix>
std::unordered_set<BinaryMatrix::Column> TupleRowDiff<BaseMatrix>
::get_samples_containing_query(const std::vector<Row> &i) const {
    // assert(graph_ && "graph must be loaded");
    // assert(anchor_.size() == diffs_.num_rows() && "anchors must be loaded");
    // assert(!fork_succ_.size() || fork_succ_.size() == graph_->get_boss().get_last().size());

    // find a path in the graph consistent with the query k-mers

    for (size_t query_row_i = 0; query_row_i < i.size() - 1; ++query_row_i) {
        graph::AnnotatedSequenceGraph::node_index row_to_graph = graph::AnnotatedSequenceGraph::anno_to_graph_index(i[query_row_i]);
        bool edge_to_next = false;
        for (auto [next, c] : graph_->outgoing_kmers(row_to_graph)) {
            if (c == graph::boss::BOSS::kSentinel)
                continue;
            Row next_to_anno = graph::AnnotatedSequenceGraph::graph_to_anno_index(next);
            if (next_to_anno == i[query_row_i + 1]) {
                edge_to_next = true;
                break;
            }
        }
        if (!edge_to_next) return {}; // no path in the graph matching the query
    }

    // find samples containing the full query 
    // TODO: try making it in batches
    std::vector<RowTuples> query_annot = get_row_tuples(i);
    std::unordered_map<Column, std::unordered_set<uint64_t>> labels_matching_query;
    std::unordered_set<Column> labels_matching_query_result;

    size_t annot_i = 0;
    for (auto & [j, tuple] : query_annot[annot_i]) {
        std::unordered_set<uint64_t> coord_set(tuple.begin(), tuple.end());
        labels_matching_query[j] = coord_set;
        labels_matching_query_result.insert(j);
    }

    for (annot_i = 1; annot_i < query_annot.size(); ++annot_i) {
        // search for at least one coord increase in every sample

        // Use explicit iterators to allow safe erasing during traversal
        for (auto it = labels_matching_query.begin(); it != labels_matching_query.end(); ) {
            const Column& j = it->first;
            const std::unordered_set<uint64_t>& tuple = it->second;
            
            bool found_match = false;

            for (const auto& [j_next, tuple_next] : query_annot[annot_i]) {
                if (j_next != j) continue;

                std::unordered_set<uint64_t> intersection_with_succ;

                for (const uint64_t& x : tuple_next) {
                    if (x == 0) continue;
                    uint64_t transformed = x - SHIFT;
                    if (tuple.count(transformed)) {
                        intersection_with_succ.insert(x);
                    }
                }

                if (!intersection_with_succ.empty()) {
                    // Update the mapped value in place using move semantics
                    it->second = std::move(intersection_with_succ);
                    found_match = true;
                }
                break; // Stop inner loop once we found the matching 'j'
            }

            // If the intersection was empty, OR if 'j' wasn't in query_annot at all
            if (!found_match) {
                labels_matching_query_result.erase(j);
                // erase() returns the iterator to the next element
                it = labels_matching_query.erase(it); 
            } else {
                // Only increment manually if we didn't erase
                ++it;
            }
        }
    }

    return labels_matching_query_result;
}

template <class BaseMatrix>
bool TupleRowDiff<BaseMatrix>::query_matches_at(
    const CoordMap& map,
    const std::vector<Row>& query,
    uint64_t start
) const {
    if (query.empty())
        return false;

    for (size_t i = 0; i < query.size(); ++i) {
        auto it = map.find(start + static_cast<uint64_t>(i) * static_cast<uint64_t>(SHIFT));
        if (it == map.end() || it->second != query[i])
            return false;
    }
    return true;
}

template <class BaseMatrix>
void TupleRowDiff<BaseMatrix>::find_query_clusters(
    const CoordMap& map,
    const std::vector<std::vector<Row>>& queries,
    uint64_t max_query_distance,
    std::vector<QueryCluster>& clusters
) const {
    clusters.clear();
    if (queries.empty() || map.empty())
        return;

    struct Occ {
        uint64_t start;
        uint64_t end;
        size_t query_id;
    };

    std::vector<Occ> occs;
    for (size_t q = 0; q < queries.size(); ++q) {
        const auto &query = queries[q];
        if (query.empty())
            return;

        const uint64_t span = static_cast<uint64_t>(query.size() - 1) * static_cast<uint64_t>(SHIFT);
        for (const auto &[c, row] : map) {
            if (row != query.front())
                continue;
            if (query_matches_at(map, query, c))
                occs.push_back({ c, c + span, q });
        }
    }

    if (occs.empty())
        return;

    std::sort(occs.begin(), occs.end(), [](const Occ &a, const Occ &b) {
        return a.start < b.start || (a.start == b.start && a.query_id < b.query_id);
    });

    std::vector<size_t> count(queries.size(), 0);
    std::multiset<uint64_t> ends;
    size_t have = 0;
    size_t left = 0;

    auto add = [&](const Occ &o) {
        if (count[o.query_id]++ == 0)
            ++have;
        ends.insert(o.end);
    };
    auto remove = [&](const Occ &o) {
        ends.erase(ends.find(o.end));
        if (--count[o.query_id] == 0)
            --have;
    };

    for (size_t right = 0; right < occs.size(); ++right) {
        add(occs[right]);
        while (have == queries.size()) {
            const uint64_t min_start = occs[left].start;
            const uint64_t max_end = *ends.rbegin();
            if (max_end >= min_start && max_end - min_start <= max_query_distance)
                clusters.push_back({ min_start, max_end });
            remove(occs[left]);
            ++left;
        }
    }
}

template <class BaseMatrix>
void TupleRowDiff<BaseMatrix>::merge_intervals(std::vector<Interval>& intervals) const {
    if (intervals.size() < 2)
        return;

    std::sort(intervals.begin(), intervals.end(), [](const Interval &a, const Interval &b) {
        return a.start < b.start;
    });

    std::vector<Interval> merged;
    merged.reserve(intervals.size());
    merged.push_back(intervals.front());
    for (size_t i = 1; i < intervals.size(); ++i) {
        if (intervals[i].start <= merged.back().end)
            merged.back().end = std::max(merged.back().end, intervals[i].end);
        else
            merged.push_back(intervals[i]);
    }
    intervals.swap(merged);
}

template <class BaseMatrix>
void TupleRowDiff<BaseMatrix>
::initialise_paths(
    const std::vector<std::vector<Row>>& queries,
    const std::unordered_set<Column>& samples,
    std::unordered_map<Column, CoordMap>& paths
) const {
    std::vector<Row> all_rows;
    for (const auto &query : queries)
        all_rows.insert(all_rows.end(), query.begin(), query.end());

    if (all_rows.empty())
        return;

    auto tuples = get_row_tuples_labelled(all_rows, samples);

    for (size_t i = 0; i < all_rows.size(); ++i) {
        Row r = all_rows[i];

        for (auto& [col, coords] : tuples[i]) {
            auto& map = paths[col];

            for (uint64_t c : coords)
                map.emplace(c, r);
        }
    }
}

// template <class BaseMatrix>
// void TupleRowDiff<BaseMatrix>
// ::get_coordinate_range(
//     const std::unordered_map<Column, CoordMap>& paths, 
//     std::unordered_map<Column, std::pair<uint64_t, uint64_t>>& sample_to_min_max_coord
// ) const {
//     // get the smallest and largest possible coordinate to store during decompression
//     // for each sample
//     for (const auto& [col, coords] : paths) {
//         if (coords.empty())
//             continue;
    
//         // std::map is automatically sorted, so first and last elements are min and max
//         uint64_t min_c = coords.begin()->first;
//         uint64_t max_c = coords.rbegin()->first;

//         // Calculate boundaries, protecting against unsigned integer underflow
//         uint64_t min_boundary = (min_c > MAX_READ_LENGTH) ? (min_c - MAX_READ_LENGTH) : 0;
//         uint64_t max_boundary = max_c + MAX_READ_LENGTH;

//         sample_to_min_max_coord[col] = {min_boundary, max_boundary};
//     }
// }

template <class BaseMatrix>
void TupleRowDiff<BaseMatrix>::build_admissible_ranges(
    const std::unordered_map<Column, CoordMap>& paths,
    const std::vector<std::vector<Row>>& queries,
    uint64_t flank_length,
    uint64_t max_query_distance,
    AdmissibleRanges& col_to_ranges
) const {
    for (const auto& [col, coords] : paths) {
        std::vector<QueryCluster> clusters;
        find_query_clusters(coords, queries, max_query_distance, clusters);
        if (clusters.empty())
            continue;

        std::vector<Interval> ranges;
        ranges.reserve(clusters.size());
        for (const auto &cluster : clusters) {
            uint64_t start = (cluster.min_start > flank_length)
                ? (cluster.min_start - flank_length) : 0;
            ranges.push_back({ start, cluster.max_end + flank_length });
        }
        merge_intervals(ranges);
        col_to_ranges[col] = std::move(ranges);
    }
}


template <class BaseMatrix>
bool TupleRowDiff<BaseMatrix>::is_admissible(
    const std::vector<Interval>& ranges, 
    uint64_t x
) const {
    if (ranges.empty()) {
        return false;
    }

    // Binary search for the first interval where interval.end >= x
    auto it = std::lower_bound(ranges.begin(), ranges.end(), x, 
        [](const Interval& interval, uint64_t val) {
            return interval.end < val;
        });

    // If we found such an interval, check if x is also >= interval.start
    if (it != ranges.end() && x >= it->start) {
        return true;
    }

    return false;
}

template <class BaseMatrix>
void TupleRowDiff<BaseMatrix>
::compute_initial_boundaries(
    const std::unordered_map<Column, CoordMap>& paths,
    const AdmissibleRanges& valid_ranges,
    std::unordered_map<Column, std::set<uint64_t>>& starts,
    std::unordered_map<Column, std::set<uint64_t>>& ends
) const {

    // mtg::common::logger->trace("Getting initial starts and ends of the reads");
    for (const auto& [col, coords] : paths) {
        if (!valid_ranges.count(col) || coords.empty())
            continue;

        auto it = coords.begin();
        uint64_t prev_coord = it->first;
        auto prev_node = 
            graph::AnnotatedSequenceGraph::anno_to_graph_index(it->second);
        
        ++it;

        for (; it != coords.end(); ++it) {
            uint64_t cur_coord = it->first;
            
            auto cur_node = graph::AnnotatedSequenceGraph::anno_to_graph_index(it->second);

            // check if an edge exists between the current and next node
            const auto next_nodes = graph_->adjacent_outgoing_nodes(prev_node);
            bool edge_exists = std::ranges::find(next_nodes, cur_node) != next_nodes.end();

            bool contiguous = 
                (cur_coord == (prev_coord + SHIFT)) && edge_exists;

            // if edge does not exist, then this is the end of the read
            if (!contiguous)
                ends[col].insert(prev_coord);

            prev_node = cur_node;
            prev_coord = cur_coord;
        }

        // last coordinate always ends a read
        ends[col].insert(prev_coord);

        // compute starts

        auto rit = coords.rbegin();
        uint64_t prev_coord_rev = rit->first;
        auto prev_node_rev = 
            graph::AnnotatedSequenceGraph::anno_to_graph_index(rit->second);

        ++rit;

        for (; rit != coords.rend(); ++rit) {
            uint64_t cur_coord = rit->first;
            
            auto cur_node = 
                graph::AnnotatedSequenceGraph::anno_to_graph_index(rit->second);

            // check if an edge exists between the current and next rev node
            const auto prev_nodes = graph_->adjacent_incoming_nodes(prev_node_rev);
            bool edge_exists = std::ranges::find(prev_nodes, cur_node) != prev_nodes.end();

            bool contiguous = 
                (cur_coord == (prev_coord_rev - SHIFT)) && edge_exists;

            // if edge does not exist, then this is the end of the read
            if (!contiguous)
                starts[col].insert(prev_coord_rev);

            prev_node_rev = cur_node;
            prev_coord_rev = cur_coord;
        }

        // the smallest coordinate always marks the start of a read
        starts[col].insert(prev_coord_rev);
    }
}

template <class BaseMatrix>
void TupleRowDiff<BaseMatrix>
::refine_boundaries(
    const std::unordered_map<Column, CoordMap>& paths,
    std::unordered_map<Column, std::set<uint64_t>>& boundaries,
    const std::unordered_set<Column>& active_samples,
    bool forward
) const {

    // coordinates increase during forward graph traversal
    // and decrease when traversing the graph backwards 
    int shift_direction = forward ? 1 : -1;

    std::unordered_map<Column, std::vector<std::pair<uint64_t, uint64_t>>> boundaries_to_update;

    for (const auto& [col, coords] : boundaries) {
        if (!active_samples.count(col))
            continue;
        auto& map = paths.at(col);

        for (uint64_t c : coords) {

            // if traversing backward and the current k-mer is at the start of the read
            // it is a final read start
            if (!forward && c == 0)
                continue;

            uint64_t cur_c = c;
            bool update_boundary = false;
            auto boundary_node = graph::AnnotatedSequenceGraph::anno_to_graph_index(map.at(cur_c));

            // find a node with a larger (smaller) coordinate among the visited nodes
            while (map.count(cur_c + shift_direction * SHIFT)) {
                auto potential_boundary_node = graph::AnnotatedSequenceGraph::anno_to_graph_index(map.at(cur_c + shift_direction * SHIFT));

                const auto adj_nodes = forward
                    ? graph_->adjacent_outgoing_nodes(boundary_node)
                    : graph_->adjacent_incoming_nodes(boundary_node);

                if (std::ranges::find(adj_nodes, potential_boundary_node) != adj_nodes.end()) {
                    boundary_node = potential_boundary_node;
                    update_boundary = true;
                    cur_c += shift_direction * SHIFT;
                } else {
                    break;
                }
            }

            if (update_boundary)
                boundaries_to_update[col].push_back(std::make_pair(c, cur_c));
        }
    }

    // update boundaries map with new k-mers
    for (auto & [col, boundary_coords_to_update] : boundaries_to_update) {
        for (auto & [coord_old, coord_new] : boundary_coords_to_update) {
            boundaries[col].erase(coord_old);
            boundaries[col].insert(coord_new);
        }
    }
}

template <class BaseMatrix>
void TupleRowDiff<BaseMatrix>
::collect_frontier(
    const std::unordered_map<Column, CoordMap>& paths,
    std::unordered_map<Column, std::set<uint64_t>>& boundaries,
    const std::unordered_set<Row>& visited,
    std::deque<Row>& queue,
    std::unordered_set<Column>& active_samples,
    bool forward, 
    AdmissibleRanges& valid_ranges
) const {
    /*
    Determine if the graph needs to be traversed more by exploring 
    coordinate increase at boundary nodes' 
    */
    for (auto & [col, coords]: boundaries) {
        if (!active_samples.count(col))
            continue;
        bool traverse_more = false;
        for (uint64_t c : coords) {
            auto boundary_node = graph::AnnotatedSequenceGraph::anno_to_graph_index(paths.at(col).at(c));

            const auto adj_kmers = forward
                ? graph_->outgoing_kmers(boundary_node)
                : graph_->incoming_kmers(boundary_node);

            for (auto [next, edge_char] : adj_kmers) {
                if (edge_char == graph::boss::BOSS::kSentinel)
                    continue;

                Row next_anno = graph::AnnotatedSequenceGraph::graph_to_anno_index(next);

                // if there is at least one unvisited node at the branching point
                // we will traverse the graph more
                if (!visited.count(next_anno)) {
                    queue.push_front(next_anno);
                    traverse_more = true; // the read possibly is not fully traversed yet
                }
            }
        }
        
        // if all nodes at the branching point of the read end (start) were visited 
        // and verified to be not read continuations
        // we don't need to decompress the annotations for the current sample
        if (!traverse_more) {
            mtg::common::logger->trace("Finished processing sample {}", col);
            active_samples.erase(col);

            // // note, this is incorrect as this loop processed individual boundaries
            // // but valid ranges contains ranges centered around each position of each k-mer in the query
            // // which means that 
            // valid_ranges.erase(col);

        }
    }
}

template <class BaseMatrix>
void TupleRowDiff<BaseMatrix>
::build_result(
    const std::unordered_map<Column, CoordMap>& paths,
    const std::vector<std::vector<Row>>& queries,
    uint64_t flank_length,
    uint64_t max_query_distance,
    std::vector<std::tuple<std::vector<Row>, Column, uint64_t>>& result
) const {

    mtg::common::logger->trace("Reconstructing the paths...");

    if (queries.empty())
        return;

    for (const auto &[col, coords] : paths) {
        if (coords.empty())
            continue;

        std::vector<QueryCluster> clusters;
        find_query_clusters(coords, queries, max_query_distance, clusters);

        // Check that coordinates |c| and |c + 1| belong to the same read. The last k-mer of
        // a read and the first k-mer of the next one have consecutive coordinates as well,
        // but normally there is no edge between them.
        auto continues = [&, &coords=coords](uint64_t c) {
            auto it = coords.find(c);
            auto next = coords.find(c + SHIFT);
            if (it == coords.end() || next == coords.end())
                return false;
            const auto next_nodes = graph_->adjacent_outgoing_nodes(
                graph::AnnotatedSequenceGraph::anno_to_graph_index(it->second));
            return std::ranges::find(next_nodes,
                                     graph::AnnotatedSequenceGraph::anno_to_graph_index(next->second))
                        != next_nodes.end();
        };

        std::set<std::pair<uint64_t, uint64_t>> emitted;
        for (const auto &cluster : clusters) {
            // the queries may match in different reads of the same label
            bool same_read = true;
            for (uint64_t c = cluster.min_start; same_read && c < cluster.max_end; ++c) {
                same_read = continues(c);
            }
            if (!same_read)
                continue;

            // add the flanks, up to the ends of the read
            uint64_t window_start = cluster.min_start;
            while (window_start && cluster.min_start - window_start < flank_length
                    && continues(window_start - 1)) {
                --window_start;
            }
            uint64_t window_end = cluster.max_end;
            while (window_end - cluster.max_end < flank_length && continues(window_end)) {
                ++window_end;
            }

            if (!emitted.insert({ window_start, window_end }).second)
                continue;

            std::vector<Row> trace;
            trace.reserve(window_end - window_start + 1);
            for (uint64_t c = window_start; c <= window_end; ++c) {
                trace.push_back(coords.at(c));
            }

            uint64_t first_query_start = cluster.min_start;
            for (uint64_t c = window_start; c <= window_end; ++c) {
                if (query_matches_at(coords, queries.front(), c)) {
                    first_query_start = c;
                    break;
                }
            }

            result.emplace_back(std::move(trace), col, first_query_start - window_start);
        }
    }
}


template <class BaseMatrix>
void TupleRowDiff<BaseMatrix>
::traverse_direction(
    bool forward,
    const std::vector<Row>& seeds,
    std::unordered_map<Column, CoordMap>& paths,
    std::unordered_map<Column, std::set<uint64_t>>& boundaries,
    std::unordered_set<Column>& active_samples,
    uint64_t batch_size, 
    AdmissibleRanges& valid_ranges,
    std::unordered_set<Row> visited
) const {

    if (!batch_size)
        batch_size = 1;

    std::deque<Row> queue(seeds.begin(), seeds.end());
    collect_frontier(paths, boundaries, visited, queue, active_samples, forward, valid_ranges);

    while (!queue.empty()) {

        std::vector<Row> batch;
        batch.reserve(batch_size);

        while (!queue.empty() && batch.size() < batch_size) {
            Row r = queue.front();
            queue.pop_front();

            if (!visited.insert(r).second)
                continue;

            batch.push_back(r);
        }

        if (!batch.empty()) {
            auto annotations = get_row_tuples_labelled(batch, active_samples);

            for (size_t i = 0; i < batch.size(); ++i) {
                Row r = batch[i];

                for (auto& [col, coords] : annotations[i]) {
                    if (!active_samples.count(col))
                        continue;

                    auto path_it = paths.find(col);
                    auto range_it = valid_ranges.find(col);
                    if (path_it == paths.end() || range_it == valid_ranges.end())
                        continue;

                    auto& map = path_it->second;

                    for (uint64_t c : coords) {
                        if (is_admissible(range_it->second, c))
                            map.emplace(c, r);
                    }
                }
            }

            refine_boundaries(paths, boundaries, active_samples, forward);
        }

        collect_frontier(paths, boundaries, visited, queue, active_samples, forward, valid_ranges);
    }
}

template <class BaseMatrix>
std::vector<std::tuple<std::vector<BinaryMatrix::Row>, BinaryMatrix::Column, uint64_t>> TupleRowDiff<BaseMatrix>
::get_traces_with_row_labelled(
    const std::vector<Row>& query_rows,
    const std::unordered_set<Column>& samples,
    uint64_t batch_size
) const {
    return get_traces_with_row_labelled(
        std::vector<std::vector<Row>>{ query_rows },
        samples, batch_size, MAX_READ_LENGTH, MAX_READ_LENGTH);
}

template <class BaseMatrix>
std::vector<std::tuple<std::vector<BinaryMatrix::Row>, BinaryMatrix::Column, uint64_t>> TupleRowDiff<BaseMatrix>
::get_traces_with_row_labelled(
    const std::vector<std::vector<Row>>& queries,
    const std::unordered_set<Column>& samples,
    uint64_t batch_size,
    uint64_t flank_length,
    uint64_t max_query_distance
) const {

    assert(graph_ && "graph must be loaded");
    assert(diffs_.num_rows() == graph_->max_index());
    assert(anchor_.size() == diffs_.num_rows() && "anchors must be loaded");

    assert(!fork_succ_.size() || fork_succ_.size() == graph_->max_index() + 1);

    std::vector<std::tuple<std::vector<Row>, Column, uint64_t>> result;

    if (queries.empty() || samples.empty())
        return result;

    std::vector<Row> forward_seeds;
    std::vector<Row> backward_seeds;
    std::unordered_set<Row> query_rows_visited;
    forward_seeds.reserve(queries.size());
    backward_seeds.reserve(queries.size());

    for (const auto &query : queries) {
        if (query.empty())
            return result;
        forward_seeds.push_back(query.back());
        backward_seeds.push_back(query.front());
        query_rows_visited.insert(query.begin(), query.end());
    }

    std::unordered_map<Column, CoordMap> paths;

    mtg::common::logger->trace("Initialising paths");
    initialise_paths(queries, samples, paths);
    if (paths.empty())
        return result;

    AdmissibleRanges valid_ranges;
    build_admissible_ranges(paths, queries, flank_length, max_query_distance, valid_ranges);
    if (valid_ranges.empty())
        return result;

    std::unordered_set<Column> active_samples;
    for (const auto &[col, _] : valid_ranges)
        active_samples.insert(col);

    mtg::common::logger->trace("Computing initial boundaries");
    std::unordered_map<Column, std::set<uint64_t>> starts, ends;
    compute_initial_boundaries(paths, valid_ranges, starts, ends);

    mtg::common::logger->trace("Traversing the graph forwards");
    traverse_direction(true, forward_seeds, paths, ends, active_samples, batch_size,
                       valid_ranges, query_rows_visited);

    active_samples.clear();
    for (const auto &[col, _] : valid_ranges)
        active_samples.insert(col);

    mtg::common::logger->trace("Traversing the graph backwards");
    traverse_direction(false, backward_seeds, paths, starts, active_samples, batch_size,
                       valid_ranges, query_rows_visited);

    build_result(paths, queries, flank_length, max_query_distance, result);

    return result;
}

} // namespace matrix
} // namespace annot
} // namespace mtg

#endif // __TUPLE_ROW_DIFF_HPP__
