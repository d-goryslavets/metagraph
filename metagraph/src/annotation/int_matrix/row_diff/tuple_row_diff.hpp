#ifndef __TUPLE_ROW_DIFF_HPP__
#define __TUPLE_ROW_DIFF_HPP__

#include <algorithm>
#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <queue>
#include <deque>
#include <unordered_map>
#include <unordered_set>

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

template <class BaseMatrix>
class TupleRowDiff : public IRowDiff, public BinaryMatrix, public MultiIntMatrix {
  public:
    static_assert(std::is_convertible<BaseMatrix*, MultiIntMatrix*>::value);
    static const int SHIFT = 1; // coordinates increase by 1 at each edge

    // check graph traversal in batches
    static const uint64_t TRAVERSAL_BATCH_SIZE = 20'000;

    // TupleRowDiff() {}

    template <typename... Args>
    TupleRowDiff(const graph::DBGSuccinct *graph = nullptr, Args&&... args)
        : diffs_(std::forward<Args>(args)...) { graph_ = graph; }

    TupleRowDiff() {}

    // TupleRowDiff(const graph::DBGSuccinct *graph, BaseMatrix&& diff)
    //     : diffs_(std::move(diff)) { graph_ = graph; }

    std::vector<Row> get_column(Column j) const override;
    std::vector<SetBitPositions> get_rows(const std::vector<Row> &rows) const override;
    std::vector<RowTuples> get_row_tuples(const std::vector<Row> &rows) const override;
    RowTuples get_row_tuples(Row i) const;
    std::vector<RowTuples> get_row_tuples_labeled(const std::vector<Row> &rows, std::unordered_set<Column> labels_of_interest) const;

    RowTuples get_row_tuples(Row i, std::vector<Row> &rd_ids,
    VectorMap<Row, size_t> &node_to_rd, std::vector<RowTuples> &rd_rows,
    std::unordered_map<Row, RowTuples> &rows_annotations,
    const graph::boss::BOSS &boss, const bit_vector &rd_succ) const;

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

    // new approach attempt
    std::vector<std::tuple<std::vector<Row>, Column, uint64_t>> get_traces_with_row_reborn(std::vector<Row> i) const;

    // pre-traverse graph to find samples with reads containing full query sequence
    std::unordered_set<Column> get_samples_containing_query(const std::vector<Row> &i) const;

    std::vector<std::unordered_set<uint64_t>> get_labels_of_rows(std::vector<Row> i) const;
    

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
    static void add_diff_labeled(const RowTuples &diff, RowTuples *row, std::unordered_set<Column> labels_of_interest);

    BaseMatrix diffs_;
};


template <class BaseMatrix>
std::vector<BinaryMatrix::Row> TupleRowDiff<BaseMatrix>::get_column(Column j) const {
    assert(graph_ && "graph must be loaded");
    assert(anchor_.size() == diffs_.num_rows() && "anchors must be loaded");

    const graph::boss::BOSS &boss = graph_->get_boss();
    assert(!fork_succ_.size() || fork_succ_.size() == boss.get_last().size());

    // TODO: implement a more efficient algorithm
    std::vector<Row> result;
    for (Row i = 0; i < num_rows(); ++i) {
        auto edge = graph_->kmer_to_boss_index(
            graph::AnnotatedSequenceGraph::anno_to_graph_index(i)
        );

        if (!boss.get_W(edge))
            continue;

        SetBitPositions set_bits = get_rows({ i })[0];
        if (std::binary_search(set_bits.begin(), set_bits.end(), j))
            result.push_back(i);
    }
    return result;
}

template <class BaseMatrix>
std::vector<BinaryMatrix::SetBitPositions>
TupleRowDiff<BaseMatrix>::get_rows(const std::vector<Row> &row_ids) const {
    std::vector<SetBitPositions> rows;
    rows.reserve(row_ids.size());
    for (const auto &row : get_row_tuples(row_ids)) {
        rows.emplace_back();
        rows.back().reserve(row.size());
        for (const auto &[j, _] : row) {
            rows.back().push_back(j);
        }
    }
    return rows;
}

template <class BaseMatrix>
MultiIntMatrix::RowTuples TupleRowDiff<BaseMatrix>::get_row_tuples(Row row) const {
    return get_row_tuples(std::vector<Row>{ row })[0];
}

template <class BaseMatrix>
std::vector<MultiIntMatrix::RowTuples>
TupleRowDiff<BaseMatrix>::get_row_tuples(const std::vector<Row> &row_ids) const {
    assert(graph_ && "graph must be loaded");
    assert(anchor_.size() == diffs_.num_rows() && "anchors must be loaded");
    assert(!fork_succ_.size() || fork_succ_.size() == graph_->get_boss().get_last().size());

    // get row-diff paths
    auto [rd_ids, rd_paths_trunc, times_traversed] = get_rd_ids(row_ids);

    std::vector<RowTuples> rd_rows = diffs_.get_row_tuples(rd_ids);
    for (auto &row : rd_rows) {
        decode_diffs(&row);
    }

    rd_ids = std::vector<Row>();

    // reconstruct annotation rows from row-diff
    std::vector<RowTuples> rows(row_ids.size());

    for (size_t i = 0; i < row_ids.size(); ++i) {
        RowTuples &result = rows[i];

        auto it = rd_paths_trunc[i].rbegin();
        std::sort(rd_rows[*it].begin(), rd_rows[*it].end());
        result = rd_rows[*it];
        // propagate back and reconstruct full annotations for predecessors
        for (++it ; it != rd_paths_trunc[i].rend(); ++it) {
            std::sort(rd_rows[*it].begin(), rd_rows[*it].end());
            add_diff(rd_rows[*it], &result);
            // replace diff row with full reconstructed annotation
            if (--times_traversed[*it]) {
                rd_rows[*it] = result;
            } else {
                // free memory
                rd_rows[*it] = {};
            }
        }
        assert(std::all_of(result.begin(), result.end(),
                           [](auto &p) { return p.second.size(); }));
    }

    return rows;
}

template <class BaseMatrix>
std::vector<MultiIntMatrix::RowTuples>
TupleRowDiff<BaseMatrix>::get_row_tuples_labeled(const std::vector<Row> &row_ids, std::unordered_set<Column> labels_of_interest) const {
    assert(graph_ && "graph must be loaded");
    assert(anchor_.size() == diffs_.num_rows() && "anchors must be loaded");
    assert(!fork_succ_.size() || fork_succ_.size() == graph_->get_boss().get_last().size());

    // get row-diff paths
    auto [rd_ids, rd_paths_trunc, times_traversed] = get_rd_ids(row_ids);

    std::vector<RowTuples> rd_rows = diffs_.get_row_tuples_labeled(rd_ids, labels_of_interest);
    for (auto &row : rd_rows) {
        decode_diffs(&row);
    }

    rd_ids = std::vector<Row>();

    // reconstruct annotation rows from row-diff
    std::vector<RowTuples> rows(row_ids.size());

    for (size_t i = 0; i < row_ids.size(); ++i) {
        RowTuples &result = rows[i];

        auto it = rd_paths_trunc[i].rbegin();
        std::sort(rd_rows[*it].begin(), rd_rows[*it].end());
        result = rd_rows[*it];
        // propagate back and reconstruct full annotations for predecessors
        for (++it ; it != rd_paths_trunc[i].rend(); ++it) {
            std::sort(rd_rows[*it].begin(), rd_rows[*it].end());
            add_diff_labeled(rd_rows[*it], &result, labels_of_interest);
            // replace diff row with full reconstructed annotation
            if (--times_traversed[*it]) {
                rd_rows[*it] = result;
            } else {
                // free memory
                rd_rows[*it] = {};
            }
        }
        assert(std::all_of(result.begin(), result.end(),
                           [](auto &p) { return p.second.size(); }));
    }

    return rows;
}

template <class BaseMatrix>
MultiIntMatrix::RowTuples TupleRowDiff<BaseMatrix>::get_row_tuples(Row i, std::vector<Row> &rd_ids,
VectorMap<Row, size_t> &node_to_rd, std::vector<RowTuples> &rd_rows,
std::unordered_map<Row, RowTuples> &rows_annotations,
const graph::boss::BOSS &boss, const bit_vector &rd_succ) const {
    assert(graph_ && "graph must be loaded");
    assert(anchor_.size() == diffs_.num_rows() && "anchors must be loaded");
    assert(!fork_succ_.size() || fork_succ_.size() == graph_->get_boss().get_last().size());

    // get row-diff paths
    auto [rd_ids_current, rd_path_trunc] = get_rd_ids(i, rd_ids, node_to_rd, boss, rd_succ);

    std::vector<RowTuples> rd_rows_current = diffs_.get_row_tuples(rd_ids_current);
    for (auto &row : rd_rows_current) {
        decode_diffs(&row);
    }

    rd_rows.insert(rd_rows.end(), rd_rows_current.begin(), rd_rows_current.end());

    RowTuples result;     
    auto it = rd_path_trunc.rbegin();
    std::sort(rd_rows[*it].begin(), rd_rows[*it].end());

    result = rd_rows[*it];
    rows_annotations[rd_ids[*it]] = result;

    // propagate back and reconstruct full annotations for predecessors
    for (++it ; it != rd_path_trunc.rend(); ++it) {
        std::sort(rd_rows[*it].begin(), rd_rows[*it].end());
        add_diff(rd_rows[*it], &result);
        // replace diff row with full reconstructed annotation
        rd_rows[*it] = result;

        // keep the decompressed annotations for the Rows
        // along that row-diff path
        rows_annotations[rd_ids[*it]] = result;
    }
    assert(std::all_of(result.begin(), result.end(),
                        [](auto &p) { return p.second.size(); }));

    return result;
}

template <class BaseMatrix>
bool TupleRowDiff<BaseMatrix>::load(std::istream &in) {
    std::string version(4, '\0');
    in.read(version.data(), 4);
    return anchor_.load(in) && fork_succ_.load(in) && diffs_.load(in);
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
    for (auto &[j, tuple] : *row) {
        assert(std::is_sorted(tuple.begin(), tuple.end()));
        for (uint64_t &c : tuple) {
            c -= SHIFT;
        }
    }
}

template <class BaseMatrix>
void TupleRowDiff<BaseMatrix>::add_diff_labeled(const RowTuples &diff, RowTuples *row, std::unordered_set<Column> labels_of_interest) {
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
                }
                ++it;
                ++it2;
            } else {
                ++it;
                ++it2;
            }
        }
        std::copy(it, row->end(), std::back_inserter(result));
        std::copy(it2, diff.end(), std::back_inserter(result));

        row->swap(result);
    }

    assert(std::is_sorted(row->begin(), row->end()));
    for (auto &[j, tuple] : *row) {
        assert(std::is_sorted(tuple.begin(), tuple.end()));
        for (uint64_t &c : tuple) {
            c -= SHIFT;
        }
    }
}

template <class BaseMatrix>
std::vector<std::unordered_set<uint64_t>> TupleRowDiff<BaseMatrix>
::get_labels_of_rows(std::vector<Row> i) const {
    std::vector<std::unordered_set<uint64_t>> result;
    auto row_tuples = get_row_tuples(i);

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
    assert(graph_ && "graph must be loaded");
    assert(anchor_.size() == diffs_.num_rows() && "anchors must be loaded");
    assert(!fork_succ_.size() || fork_succ_.size() == graph_->get_boss().get_last().size());

    // find a path in the graph consistent with the query k-mers

    for (size_t query_row_i = 0; query_row_i < i.size() - 1; ++query_row_i) {
        graph::AnnotatedSequenceGraph::node_index row_to_graph = graph::AnnotatedSequenceGraph::anno_to_graph_index(i[query_row_i]);
        bool edge_to_next = false;
        graph_->call_outgoing_kmers(row_to_graph, [&](auto next, char c) {
                if (c == graph::boss::BOSS::kSentinel)
                    return;
                // add adjacent outgoing nodes to the stack for further traversal
                Row next_to_anno = graph::AnnotatedSequenceGraph::graph_to_anno_index(next);
                if (next_to_anno == i[query_row_i + 1]) {
                    edge_to_next = true;
                    return;
                }
            } );
        if (!edge_to_next) return {}; // no path in the graph matching the query
    }

    // find samples containing the full query 
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
        for (auto & [j, tuple] : query_annot[annot_i]) {
            bool found_coord_increase_in_child = false;
            for (uint64_t &c : tuple) {
                if (c == 0) continue;
                if (labels_matching_query[j].count(c - SHIFT)) {
                    found_coord_increase_in_child = true;
                    break;
                }
            }
            // no path consistent with the query found
            if (!found_coord_increase_in_child) {
                labels_matching_query.erase(j);
                labels_matching_query_result.erase(j);
            } else {
                labels_matching_query[j] = std::unordered_set<uint64_t>(tuple.begin(), tuple.end());
            }       
        }
    }

    return labels_matching_query_result;
}

template <class BaseMatrix>
std::vector<std::tuple<std::vector<BinaryMatrix::Row>, BinaryMatrix::Column, uint64_t>> TupleRowDiff<BaseMatrix>
::get_traces_with_row_reborn(std::vector<Row> i) const {
    assert(graph_ && "graph must be loaded");
    assert(anchor_.size() == diffs_.num_rows() && "anchors must be loaded");
    assert(!fork_succ_.size() || fork_succ_.size() == graph_->get_boss().get_last().size());

    // vector of reads <vector<Row>, Column, coord_of_query_1st_kmer>
    std::vector<std::tuple<std::vector<Row>, Column, uint64_t>> result;

    auto samples_with_query = get_samples_containing_query(i);

    mtg::common::logger->trace("samples matching query: ");
    for (const auto & sample_col_id : samples_with_query) {
        mtg::common::logger->trace("{}", sample_col_id);        
    }

    std::vector<RowTuples> initial_row_tuples = get_row_tuples_labeled(i, samples_with_query);

    // keep visited rows, and their coords for each label separately
    std::unordered_map<Column, std::map<uint64_t, Row>> reconstructed_paths;

    for (size_t query_row_i = 0; query_row_i < i.size(); query_row_i++) {
        for (auto & [j, coords] : initial_row_tuples[query_row_i]) {
            for (auto &c : coords) {
                reconstructed_paths[j][c] = i[query_row_i];
            }
        }
    }

    // find start and end coords of the reads to reconstruct
    std::unordered_map<Column, std::set<uint64_t>> ends_of_reads;
    std::unordered_map<Column, std::set<uint64_t>> starts_of_reads;

    std::unordered_map<Column, std::set<uint64_t>> ends_of_reads_final;
    std::unordered_map<Column, std::set<uint64_t>> starts_of_reads_final;

    mtg::common::logger->trace("Getting initial starts and ends of the reads");
    for (auto & [j, coords] : reconstructed_paths) {
        auto node_1 = graph::AnnotatedSequenceGraph::anno_to_graph_index(coords.begin()->second);
        auto coord_1 = coords.begin()->first;
        for (auto iter_map = std::next(coords.begin()); iter_map != coords.end(); ++iter_map) {
            auto node_2 = graph::AnnotatedSequenceGraph::anno_to_graph_index(iter_map->second);
            auto coord_2 = iter_map->first;

            // check if an edge exists between the current and next node
            bool edge_exists = false;
            graph_->adjacent_outgoing_nodes(node_1, [&](auto adj_outg_node) {
                if (adj_outg_node == node_2) {
                    edge_exists = true;
                    return;
                }
            });

            // if edge does not exist, then this is the end of the read
            if (!((coord_2 == (coord_1 + SHIFT)) && edge_exists)) {
                ends_of_reads[j].insert(coord_1);
            }

            node_1 = node_2;
            coord_1 = coord_2;
        }
        ends_of_reads[j].insert(coord_1);

        // similarly find initial starts of the reads
        auto node_1_start = graph::AnnotatedSequenceGraph::anno_to_graph_index(coords.rbegin()->second);
        auto coord_1_start = coords.rbegin()->first;
        for (auto iter_map = std::next(coords.rbegin()); iter_map != coords.rend(); ++iter_map) {
            auto node_2 = graph::AnnotatedSequenceGraph::anno_to_graph_index(iter_map->second);
            auto coord_2 = iter_map->first;

            // check if an edge exists between the current and next node
            bool edge_exists = false;
            graph_->adjacent_incoming_nodes(node_1_start, [&](auto adj_inc_node) {
                if (adj_inc_node == node_2) {
                    edge_exists = true;
                    return;
                }
            });

            // if edge does not exist, then this is the end of the read
            if (!((coord_2 == (coord_1_start - SHIFT)) && edge_exists)) {
                starts_of_reads[j].insert(coord_1_start);
            }

            node_1_start = node_2;
            coord_1_start = coord_2;
        }
        starts_of_reads[j].insert(coord_1_start);
    }

    // traverse graph forwards
    std::deque<Row> to_visit;
    to_visit.push_back(*(i.rbegin()));

    std::unordered_set<Row> visited_nodes_forward;
    uint64_t batches_count = 0;
    std::unordered_map<Column, std::set<uint64_t>> all_discovered_coordinates;
    mtg::common::logger->trace("Traversing the graph forwards");

    while (true) {
        uint64_t traversed_nodes_count = 0;
        std::vector<Row> batch_of_rows;

        while (!to_visit.empty()) {
            // Row current_parent = to_visit.front();
            // to_visit.pop();
            Row current_parent = to_visit.front();
            to_visit.pop_front();

            if (visited_nodes_forward.count(current_parent))
                continue;
            
            batch_of_rows.push_back(current_parent);
            traversed_nodes_count++;

            graph::AnnotatedSequenceGraph::node_index current_parent_to_graph = graph::AnnotatedSequenceGraph::anno_to_graph_index(current_parent);

            graph_->call_outgoing_kmers(current_parent_to_graph, [&](auto next, char c) {
                if (c == graph::boss::BOSS::kSentinel)
                    return;
                // add adjacent outgoing nodes to the stack for further traversal
                Row next_to_anno = graph::AnnotatedSequenceGraph::graph_to_anno_index(next);
                // to_visit.push(next_to_anno);
                to_visit.push_back(next_to_anno);
            } );

            visited_nodes_forward.insert(current_parent);

            // if we reached the batch size then stop the current batch traversal
            if (traversed_nodes_count >= TRAVERSAL_BATCH_SIZE) {
                break;
            }
        }

        batches_count++;

        mtg::common::logger->trace("traversed batch {}", batches_count);


        mtg::common::logger->trace("decompress the annotations for the current batch of rows");
        // decompress the annotations for the current batch of rows
        std::vector<RowTuples> batch_annotations = get_row_tuples_labeled(batch_of_rows, samples_with_query);


        mtg::common::logger->trace("collect all new discovered coordinates");
        // collect all new discovered coordinates
        for (size_t batch_i = 0; batch_i < batch_of_rows.size(); ++batch_i) {
            for (auto & [j_next, tuple_next] : batch_annotations[batch_i]) {
                if (!samples_with_query.count(j_next)) {
                    tuple_next = Tuple();
                    continue;
                }

                for (uint64_t &c_next : tuple_next) {
                    all_discovered_coordinates[j_next].insert(c_next);
                    reconstructed_paths[j_next][c_next] = batch_of_rows[batch_i];
                }

                tuple_next = Tuple();
            }
        }

        batch_annotations = std::vector<RowTuples>();
        batch_of_rows = std::vector<Row>();

        mtg::common::logger->trace("update reads ends");
        // update reads ends
        std::unordered_map<Column, std::vector<std::pair<uint64_t, uint64_t>>> ends_to_update;
        for (auto & [j_end, coords_ends] : ends_of_reads) {
            for (auto & c_end : coords_ends) {
                auto c_end_cur = c_end;
                bool replace_read_end = false;

                auto end_node_old = graph::AnnotatedSequenceGraph::anno_to_graph_index(reconstructed_paths[j_end][c_end_cur]);
                while (all_discovered_coordinates[j_end].count(c_end_cur + SHIFT)) {
                    auto end_node_new = graph::AnnotatedSequenceGraph::anno_to_graph_index(reconstructed_paths[j_end][c_end_cur + SHIFT]);

                    bool edge_exists = false;
                    graph_->adjacent_outgoing_nodes(end_node_old, [&](auto adj_outg_node) {
                        if (adj_outg_node == end_node_new) {
                            edge_exists = true;
                            return;
                        }
                    });

                    // if edge exists, then there is row with bigger coord (potential read end)
                    if (edge_exists) {
                        end_node_old = end_node_new;
                        replace_read_end = true;
                        c_end_cur += SHIFT;
                    } else {
                        break;
                    }
                }

                if (replace_read_end) {
                    ends_to_update[j_end].push_back(std::make_pair(c_end, c_end_cur));
                }
            }
        }

        for (auto & [j_to_update, old_new_ends_to_update] : ends_to_update) {
            for (auto & [c_end_old, c_end_updated] : old_new_ends_to_update) {
                ends_of_reads[j_to_update].erase(c_end_old);
                ends_of_reads[j_to_update].insert(c_end_updated);
            }
        }


        mtg::common::logger->trace("check if graph has to be traversed more");
        // check if graph has to be traversed more 
        // by checking if there are continuations of the reads after the known ends

        to_visit = std::deque<Row>(); // collect in the queue only the known ends of the reads

        std::unordered_set<Row> batch_of_rows_to_check_if_traverse_more_set;
        for (auto & [j_end, coords_ends] : ends_of_reads) {
            for (auto & c_end : coords_ends) {
                auto end_node_old = graph::AnnotatedSequenceGraph::anno_to_graph_index(reconstructed_paths[j_end][c_end]);

                graph_->call_outgoing_kmers(end_node_old, [&](auto adj_outg_node, char c) {
                    // std::ignore = c;
                    if (c == graph::boss::BOSS::kSentinel)
                        return;
                    
                    Row adj_outg_to_anno = graph::AnnotatedSequenceGraph::graph_to_anno_index(adj_outg_node);
                    batch_of_rows_to_check_if_traverse_more_set.insert(adj_outg_to_anno);
                });
            }
        }

        std::vector<Row> batch_of_rows_to_check_if_traverse_more;
        batch_of_rows_to_check_if_traverse_more.reserve(batch_of_rows_to_check_if_traverse_more_set.size());
        batch_of_rows_to_check_if_traverse_more.insert(batch_of_rows_to_check_if_traverse_more.end(),
        batch_of_rows_to_check_if_traverse_more_set.begin(), batch_of_rows_to_check_if_traverse_more_set.end());
        batch_of_rows_to_check_if_traverse_more_set = std::unordered_set<Row>();

        std::unordered_map<Column, std::unordered_set<uint64_t>> coordinates_of_the_successors;
        std::unordered_map<Column, std::unordered_map<uint64_t, Row>> successors_rows_and_coords;
        size_t batch_vec_size = batch_of_rows_to_check_if_traverse_more.size();

        for (size_t cur_part = 0; cur_part < batch_vec_size; cur_part += TRAVERSAL_BATCH_SIZE) {
            size_t cur_end = std::min(cur_part + TRAVERSAL_BATCH_SIZE, batch_vec_size);

            std::vector<Row> batch_part(batch_of_rows_to_check_if_traverse_more.begin() + cur_part,
            batch_of_rows_to_check_if_traverse_more.begin() + cur_end);

            std::vector<RowTuples> batch_part_annotations = get_row_tuples_labeled(batch_part, samples_with_query);
            // std::vector<RowTuples> batch_part_annotations = get_row_tuples(batch_part);

            for (size_t rowt_to_check = 0; rowt_to_check < batch_part.size(); ++rowt_to_check) {
                for (auto & [j_to_check, tuple_to_check] : batch_part_annotations[rowt_to_check]) {
                    if (!samples_with_query.count(j_to_check)) {
                        tuple_to_check = Tuple();
                        continue;
                    }

                    for (auto & c_to_check : tuple_to_check) {
                        coordinates_of_the_successors[j_to_check].insert(c_to_check);
                        successors_rows_and_coords[j_to_check][c_to_check] = batch_part[rowt_to_check];
                    }
                    tuple_to_check = Tuple();
                }
            }
            batch_part_annotations = std::vector<RowTuples>();
        }

        for (auto & [j_end, coords_ends] : ends_of_reads) {
            for (auto it_end_coords = coords_ends.begin(); it_end_coords != coords_ends.end(); ) {
                if (coordinates_of_the_successors[j_end].count(*it_end_coords + SHIFT)) {
                    to_visit.push_front(successors_rows_and_coords[j_end][*it_end_coords + SHIFT]);
                    visited_nodes_forward.erase(successors_rows_and_coords[j_end][*it_end_coords + SHIFT]);
                    ++it_end_coords;
                } else {
                    ends_of_reads_final[j_end].insert(*it_end_coords);
                    it_end_coords = coords_ends.erase(it_end_coords);
                }
            }
        }

        if (to_visit.empty())
            break;
    }

    ends_of_reads = ends_of_reads_final;
    ends_of_reads_final = std::unordered_map<Column, std::set<uint64_t>>();

    mtg::common::logger->trace("Traversed batches forwards count {}", batches_count);


    // similarly traverse the graph backwards
    to_visit = std::deque<Row>();
    to_visit.push_back(*(i.begin()));

    std::unordered_set<Row> visited_nodes_backward;
    uint64_t batches_count_backwards = 0;

    mtg::common::logger->trace("Traversing the graph backwards");
    while (true) {
        uint64_t traversed_nodes_count = 0;

        // set of rows in the current batch to decompress the annotations for
        std::vector<Row> batch_of_rows;
        while (!to_visit.empty()) {
            // Row current_child = to_visit.front();
            // to_visit.pop();
            Row current_child = to_visit.front();
            to_visit.pop_front();

            if (visited_nodes_backward.count(current_child)) {
                continue;
            }

            batch_of_rows.push_back(current_child);
            traversed_nodes_count++;

            graph::AnnotatedSequenceGraph::node_index current_child_to_graph = graph::AnnotatedSequenceGraph::anno_to_graph_index(current_child);

            graph_->call_incoming_kmers(current_child_to_graph, [&](auto prev, char c) {
                if (c == graph::boss::BOSS::kSentinel)
                    return;
                // add adjacent outgoing nodes to the stack for further traversal
                Row prev_to_anno = graph::AnnotatedSequenceGraph::graph_to_anno_index(prev);
                // to_visit.push(prev_to_anno);
                to_visit.push_back(prev_to_anno);
            } );

            visited_nodes_backward.insert(current_child);

            if (traversed_nodes_count >= TRAVERSAL_BATCH_SIZE) {
                break;
            }
        }
        
        batches_count_backwards++;

        std::vector<RowTuples> batch_annotations = get_row_tuples_labeled(batch_of_rows, samples_with_query);
        // std::vector<RowTuples> batch_annotations = get_row_tuples(batch_of_rows);

        // collect all new discovered coordinates
        for (size_t batch_i = 0; batch_i < batch_of_rows.size(); ++batch_i) {
            for (auto & [j_prev, tuple_prev] : batch_annotations[batch_i]) {
                if (!samples_with_query.count(j_prev)) {
                    tuple_prev = Tuple();
                    continue;
                }

                for (uint64_t &c_prev : tuple_prev) {
                    all_discovered_coordinates[j_prev].insert(c_prev);
                    reconstructed_paths[j_prev][c_prev] = batch_of_rows[batch_i];
                }

                tuple_prev = Tuple();
            }
        }

        batch_annotations = std::vector<RowTuples>();
        batch_of_rows = std::vector<Row>();

        // update reads starts
        std::unordered_map<Column, std::vector<std::pair<uint64_t, uint64_t>>> starts_to_update;
        for (auto & [j_start, coords_starts] : starts_of_reads) {
            for (auto & c_start : coords_starts) {
                if (c_start == 0)
                    continue;
                auto c_end_cur = c_start;
                bool replace_read_end = false;

                auto end_node_old = graph::AnnotatedSequenceGraph::anno_to_graph_index(reconstructed_paths[j_start][c_end_cur]);
                while (all_discovered_coordinates[j_start].count(c_end_cur - SHIFT)) {
                    auto end_node_new = graph::AnnotatedSequenceGraph::anno_to_graph_index(reconstructed_paths[j_start][c_end_cur - SHIFT]);

                    bool edge_exists = false;
                    graph_->adjacent_incoming_nodes(end_node_old, [&](auto adj_inc_node) {
                        if (adj_inc_node == end_node_new) {
                            edge_exists = true;
                            return;
                        }
                    });

                    // if edge exists, then read has its continuation => we have to traverse the graph more
                    if (edge_exists) {
                        end_node_old = end_node_new;
                        replace_read_end = true;
                        c_end_cur -= SHIFT;
                    } else {
                        break;
                    }
                }

                if (replace_read_end) {
                    starts_to_update[j_start].push_back(std::make_pair(c_start, c_end_cur));
                }
            }
        }

        for (auto & [j_to_update, old_new_ends_to_update] : starts_to_update) {
            for (auto & [c_end_old, c_end_updated] : old_new_ends_to_update) {
                starts_of_reads[j_to_update].erase(c_end_old);
                starts_of_reads[j_to_update].insert(c_end_updated);
            }
        }

        // check if graph has to be traversed more 
        // by checking if there are continuations of the reads after the known ends

        to_visit = std::deque<Row>();
        std::unordered_set<Row> batch_of_rows_to_check_if_traverse_more_set;

        for (auto & [j_end, coords_ends] : starts_of_reads) {
            for (auto & c_end : coords_ends) {
                auto end_node_old = graph::AnnotatedSequenceGraph::anno_to_graph_index(reconstructed_paths[j_end][c_end]);

                graph_->call_incoming_kmers(end_node_old, [&](auto adj_outg_node, char c) {
                    // std::ignore = c;
                    if (c == graph::boss::BOSS::kSentinel)
                        return;
                    
                    Row adj_outg_to_anno = graph::AnnotatedSequenceGraph::graph_to_anno_index(adj_outg_node);
                    batch_of_rows_to_check_if_traverse_more_set.insert(adj_outg_to_anno);
                });
            }
        }

        std::vector<Row> batch_of_rows_to_check_if_traverse_more;
        batch_of_rows_to_check_if_traverse_more.reserve(batch_of_rows_to_check_if_traverse_more_set.size());
        batch_of_rows_to_check_if_traverse_more.insert(batch_of_rows_to_check_if_traverse_more.end(),
        batch_of_rows_to_check_if_traverse_more_set.begin(), batch_of_rows_to_check_if_traverse_more_set.end());
        batch_of_rows_to_check_if_traverse_more_set = std::unordered_set<Row>();

        std::unordered_map<Column, std::unordered_set<uint64_t>> coordinates_of_the_successors;
        std::unordered_map<Column, std::unordered_map<uint64_t, Row>> successors_rows_and_coords;
        size_t batch_vec_size = batch_of_rows_to_check_if_traverse_more.size();

        for (size_t cur_part = 0; cur_part < batch_vec_size; cur_part += TRAVERSAL_BATCH_SIZE) {
            size_t cur_end = std::min(cur_part + TRAVERSAL_BATCH_SIZE, batch_vec_size);

            std::vector<Row> batch_part(batch_of_rows_to_check_if_traverse_more.begin() + cur_part,
            batch_of_rows_to_check_if_traverse_more.begin() + cur_end);

            std::vector<RowTuples> batch_part_annotations = get_row_tuples_labeled(batch_part, samples_with_query);
            // std::vector<RowTuples> batch_part_annotations = get_row_tuples(batch_part);

            for (size_t rowt_to_check = 0; rowt_to_check < batch_part.size(); ++rowt_to_check) {
                for (auto & [j_to_check, tuple_to_check] : batch_part_annotations[rowt_to_check]) {
                    if (!samples_with_query.count(j_to_check)) {
                        tuple_to_check = Tuple();
                        continue;
                    }

                    for (auto & c_to_check : tuple_to_check) {
                        coordinates_of_the_successors[j_to_check].insert(c_to_check);
                        successors_rows_and_coords[j_to_check][c_to_check] = batch_part[rowt_to_check];
                    }
                    tuple_to_check = Tuple();
                }
            }
            batch_part_annotations = std::vector<RowTuples>();
        }

        for (auto & [j_end, coords_ends] : starts_of_reads) {
            for (auto it_end_coords = coords_ends.begin(); it_end_coords != coords_ends.end(); ) {
                if (*it_end_coords == 0) {
                    starts_of_reads_final[j_end].insert(*it_end_coords);
                    it_end_coords = coords_ends.erase(it_end_coords);
                } else if (coordinates_of_the_successors[j_end].count(*it_end_coords - SHIFT)) {
                    to_visit.push_front(successors_rows_and_coords[j_end][*it_end_coords - SHIFT]);
                    visited_nodes_backward.erase(successors_rows_and_coords[j_end][*it_end_coords - SHIFT]);
                    ++it_end_coords;
                } else {
                    starts_of_reads_final[j_end].insert(*it_end_coords);
                    it_end_coords = coords_ends.erase(it_end_coords);
                }
            }
        }

        if (to_visit.empty())
            break;
    }

    starts_of_reads = starts_of_reads_final;
    starts_of_reads_final = std::unordered_map<Column, std::set<uint64_t>>();

    // reconstruct the paths
    mtg::common::logger->trace("Traversed batches backwards count {}", batches_count_backwards);
    mtg::common::logger->trace("Reconstructing the paths...");

    for (auto & [j_st, coords_st] : starts_of_reads) {
        // assert(coords_st.size() == ends_of_reads[j_st].size());
        auto it_start = coords_st.begin();
        auto it_end = ends_of_reads[j_st].begin();

        uint64_t cur_read_start_coord = *it_start;
        std::vector<Row> curr_read_trace_no_coords;

        while (it_start != coords_st.end() && it_end != ends_of_reads[j_st].end()) {
            curr_read_trace_no_coords.clear();
            cur_read_start_coord = *it_start;

            bool contains_first_input_kmer = false;
            bool contains_last_input_kmer = false;
            uint64_t input_row_coord_in_read = 0;
            bool found_first_occurence_of_the_first_kmer = false; // TODO: DEBUG
            for (uint64_t cur_read_coord_ind = cur_read_start_coord; cur_read_coord_ind <= *it_end; ++cur_read_coord_ind) {
                if (reconstructed_paths[j_st][cur_read_coord_ind] == *(i.begin()) &&
                    !found_first_occurence_of_the_first_kmer) {
                    found_first_occurence_of_the_first_kmer = true;

                    input_row_coord_in_read = cur_read_coord_ind;
                    contains_first_input_kmer = true;
                }
                // query can consist of a single k-mer (or end with the same as the first one)
                // so these conditions must be separate 
                if (reconstructed_paths[j_st][cur_read_coord_ind] == *(i.rbegin())) {
                    // TODO: clarify the neccessity of this condition
                    contains_last_input_kmer = true;
                }
                curr_read_trace_no_coords.push_back(reconstructed_paths[j_st][cur_read_coord_ind]);
            }

            if (contains_first_input_kmer && contains_last_input_kmer)
                result.push_back(std::make_tuple(curr_read_trace_no_coords, j_st, input_row_coord_in_read - cur_read_start_coord));

            it_start++;
            it_end++;
        }
    }


    return result;
}


} // namespace matrix
} // namespace annot
} // namespace mtg

#endif // __TUPLE_ROW_DIFF_HPP__
