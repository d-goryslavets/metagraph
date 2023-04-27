#ifndef __TUPLE_ROW_DIFF_HPP__
#define __TUPLE_ROW_DIFF_HPP__

#include <algorithm>
#include <iostream>
#include <cassert>
#include <string>
#include <vector>
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
class TupleRowDiff : public binmat::IRowDiff, public MultiIntMatrix {
  public:
    static_assert(std::is_convertible<BaseMatrix*, MultiIntMatrix*>::value);
    static const int SHIFT = 1; // coordinates increase by 1 at each edge

    TupleRowDiff() {}

    TupleRowDiff(const graph::DBGSuccinct *graph, BaseMatrix&& diff)
        : diffs_(std::move(diff)) { graph_ = graph; }

    bool get(Row i, Column j) const override;
    std::vector<Row> get_column(Column j) const override;
    RowTuples get_row_tuples(Row i) const override;
    std::vector<RowTuples> get_row_tuples(const std::vector<Row> &rows) const override;

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
    std::vector<std::tuple<std::vector<Row>, Column, uint64_t>> get_traces_with_row_auto_labels(Row i) const;
    std::vector<std::tuple<std::vector<Row>, Column, uint64_t>> get_traces_with_row(Row i) const;

    // auto_labels means that the labels will be derived based on coordinates
    // this is needed for the graphs where reads are not marked with different labels
    std::pair<std::unordered_map<Column, std::vector<Row>>, std::vector<std::vector<std::pair<Column, uint64_t>>>> get_traces_with_row(std::vector<Row> i) const;
    std::vector<std::pair<Column, uint64_t>> get_traces_with_row(Row i, std::unordered_map<Row, RowTuples> &rows_annotations, 
    std::vector<Row> &rd_ids, VectorMap<Row, size_t> &node_to_rd, std::vector<RowTuples> &rd_rows, std::unordered_map<Column, std::vector<Row>> &reconstructed_reads) const;

    std::vector<std::vector<std::tuple<std::vector<Row>, Column, uint64_t>>> get_traces_with_row_auto_labels(std::vector<Row> i) const;
    std::vector<std::tuple<std::vector<Row>, Column, uint64_t>> get_traces_with_row_auto_labels(Row i, std::unordered_map<Row, RowTuples> &rows_annotations, 
    std::vector<Row> &rd_ids, VectorMap<Row, size_t> &node_to_rd, std::vector<RowTuples> &rd_rows) const;
    

    uint64_t num_columns() const override { return diffs_.num_columns(); }
    uint64_t num_relations() const override { return diffs_.num_relations(); }
    uint64_t num_attributes() const override { return diffs_.num_attributes(); }
    uint64_t num_rows() const override { return diffs_.num_rows(); }

    bool load(std::istream &in) override;
    void serialize(std::ostream &out) const override;

    const BaseMatrix& diffs() const { return diffs_; }
    BaseMatrix& diffs() { return diffs_; }

  private:
    static void decode_diffs(RowTuples *diffs);
    static void add_diff(const RowTuples &diff, RowTuples *row);

    BaseMatrix diffs_;
};


template <class BaseMatrix>
bool TupleRowDiff<BaseMatrix>::get(Row i, Column j) const {
    SetBitPositions set_bits = get_row(i);
    auto v = std::lower_bound(set_bits.begin(), set_bits.end(), j);
    return v != set_bits.end() && *v == j;
}

template <class BaseMatrix>
std::vector<MultiIntMatrix::Row> TupleRowDiff<BaseMatrix>::get_column(Column j) const {
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

        if (boss.get_W(edge) && get(i, j))
            result.push_back(i);
    }
    return result;
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
    auto [rd_ids, rd_paths_trunc] = get_rd_ids(row_ids);

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
            rd_rows[*it] = result;
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
std::pair<std::unordered_map<MultiIntMatrix::Column, std::vector<MultiIntMatrix::Row>>, std::vector<std::vector<std::pair<MultiIntMatrix::Column, uint64_t>>>> TupleRowDiff<BaseMatrix>
::get_traces_with_row(std::vector<Row> i) const {
    assert(graph_ && "graph must be loaded");
    assert(anchor_.size() == diffs_.num_rows() && "anchors must be loaded");
    assert(!fork_succ_.size() || fork_succ_.size() == graph_->get_boss().get_last().size());

    // a map that stores decompressed annotations for the rows
    std::unordered_map<Row, RowTuples> rows_annotations;

    // diff rows annotating nodes along the row-diff paths
    std::vector<Row> rd_ids;
    // map row index to its index in |rd_rows|
    VectorMap<Row, size_t> node_to_rd;

    std::vector<RowTuples> rd_rows;
    std::unordered_map<Column, std::vector<Row>> reconstructed_reads;

    std::vector<std::vector<std::pair<Column, uint64_t>>> result;

    for (Row &row_i : i) {
        auto curr_res = get_traces_with_row(row_i, rows_annotations, rd_ids, node_to_rd, rd_rows, reconstructed_reads);
        result.push_back(curr_res);
    }

    return std::make_pair(std::move(reconstructed_reads), std::move(result));
}

template <class BaseMatrix>
std::vector<std::vector<std::tuple<std::vector<MultiIntMatrix::Row>, MultiIntMatrix::Column, uint64_t>>> TupleRowDiff<BaseMatrix>
::get_traces_with_row_auto_labels(std::vector<Row> i) const {
    assert(graph_ && "graph must be loaded");
    assert(anchor_.size() == diffs_.num_rows() && "anchors must be loaded");
    assert(!fork_succ_.size() || fork_succ_.size() == graph_->get_boss().get_last().size());

    // a map that stores decompressed annotations for the rows
    std::unordered_map<Row, RowTuples> rows_annotations;

    // diff rows annotating nodes along the row-diff paths
    std::vector<Row> rd_ids;
    // map row index to its index in |rd_rows|
    VectorMap<Row, size_t> node_to_rd;

    std::vector<RowTuples> rd_rows;
    std::unordered_map<Column, std::vector<Row>> reconstructed_reads;

    std::vector<std::vector<std::tuple<std::vector<Row>, Column, uint64_t>>> result;

    for (Row &row_i : i) {
        auto curr_res = get_traces_with_row_auto_labels(row_i, rows_annotations, rd_ids, node_to_rd, rd_rows);
        result.push_back(curr_res);
    }

    return result;
}


template <class BaseMatrix>
std::vector<std::tuple<std::vector<MultiIntMatrix::Row>, MultiIntMatrix::Column, uint64_t>> TupleRowDiff<BaseMatrix>
::get_traces_with_row_auto_labels(Row i) const {
    return get_traces_with_row_auto_labels(std::vector<Row>{ i })[0];
}

template <class BaseMatrix>
std::vector<std::tuple<std::vector<MultiIntMatrix::Row>, MultiIntMatrix::Column, uint64_t>> TupleRowDiff<BaseMatrix>
::get_traces_with_row(Row i) const {
    auto [labeled_paths, overlapping_reads] = get_traces_with_row(std::vector<Row>{ i });
    std::vector<std::tuple<std::vector<Row>, Column, uint64_t>> result;

    for (auto & [j, c] : overlapping_reads[0]) {
        result.push_back(std::make_tuple(labeled_paths[j], j, c));
    }

    return result;
}

template <class BaseMatrix>
std::vector<std::tuple<std::vector<MultiIntMatrix::Row>, MultiIntMatrix::Column, uint64_t>> TupleRowDiff<BaseMatrix>
::get_traces_with_row_auto_labels(Row i, std::unordered_map<Row, RowTuples> &rows_annotations, std::vector<Row> &rd_ids, VectorMap<Row, size_t> &node_to_rd, 
std::vector<RowTuples> &rd_rows) const {
    assert(graph_ && "graph must be loaded");
    assert(anchor_.size() == diffs_.num_rows() && "anchors must be loaded");
    assert(!fork_succ_.size() || fork_succ_.size() == graph_->get_boss().get_last().size());    

    const graph::boss::BOSS &boss = graph_->get_boss();
    const bit_vector &rd_succ = fork_succ_.size() ? fork_succ_ : boss.get_last();

    std::unordered_map<Column, std::map<uint64_t, Row>> reconstucted_paths;

    // part 1. Graph traversal

    // step 1. Build row-diff path for the input Row
    RowTuples input_row_tuples;

    if (!rows_annotations.count(i)) {
        input_row_tuples = get_row_tuples(i, rd_ids, node_to_rd, rd_rows,
        rows_annotations, boss, rd_succ);
    } else {
        // if we already decompressed the annotations for this node
        // then simply retrieve them from the map
        input_row_tuples = rows_annotations[i];
    }

    // step 2.2 Save labels of the input Row in a set
    std::unordered_set<Column> labels_of_the_start_node;

    std::vector<std::tuple<std::vector<Row>, Column, uint64_t>> result;

    std::unordered_map<Column, uint64_t> input_row_position_in_ref_seq;

    for (auto &[j, tuple] : input_row_tuples) {

        // if (!reconstructed_reads.count(j)) {
        //     labels_of_the_start_node.insert(j);

        //     // initialize adjacency lists for each label
        //     labeled_parents_map[j];
        //     labeled_children_map[j];

        //     labeled_parents_map_backward[j];
        //     labeled_children_map_backward[j];

        //     input_row_position_in_ref_seq[j] = tuple.front();
        // } else {
        //     // return this as well
        //     result.push_back(std::make_tuple(reconstructed_reads[j], j, tuple.front()));
        // }
        

        labels_of_the_start_node.insert(j);

        // initialize adjacency lists for each label
        // labeled_parents_map[j];
        // labeled_children_map[j];

        // labeled_parents_map_backward[j];
        // labeled_children_map_backward[j];

        input_row_position_in_ref_seq[j] = tuple.front();
    }


    // step 3. Traverse the graph using DFS 
    // and fill adjacency lists along the way
    std::vector<Row> to_visit;
    to_visit.push_back(i);

    std::unordered_set<Row> visited_nodes_forward;
    while (!to_visit.empty()) {
        // pick a node from the stack
        Row current_parent = to_visit.back();
        to_visit.pop_back();
    
        // if it was already visited, continue
        if (visited_nodes_forward.count(current_parent))
            continue;

        // collect the current node's labels into a set
        std::unordered_set<Column> current_parent_labels;
        for (auto &rowt_parent : rows_annotations[current_parent]) {
            current_parent_labels.insert(rowt_parent.first);
        }
        
        visited_nodes_forward.insert(current_parent);

        graph::AnnotatedSequenceGraph::node_index current_parent_to_graph = graph::AnnotatedSequenceGraph::anno_to_graph_index(current_parent);
        
        // for each adjacent outgoing k-mer 
        graph_->call_outgoing_kmers(current_parent_to_graph, [&](auto next, char c) {
            // std::ignore = c;
            if (c == graph::boss::BOSS::kSentinel)
                return;
            
            Row next_to_anno = graph::AnnotatedSequenceGraph::graph_to_anno_index(next);
            RowTuples next_annotations;

            // if the annotations for the child node are not decompressed,
            // build rd-path and decompress the annotations
            // (repeat steps 1-2.1)
            if (!rows_annotations.count(next_to_anno)) {
                next_annotations = get_row_tuples(next_to_anno, rd_ids, node_to_rd, rd_rows,
                rows_annotations, boss, rd_succ);
            } else {
                // if we already decompressed the annotations for this node
                // then simply retrieve them from the map
                next_annotations = rows_annotations[next_to_anno];
            }

            // acceptable node is a node whose labels match
            // the labels of the start node
            bool acceptable_node = false;
            
            for (auto &[j_next, tuple_next] : next_annotations) {
                // check if the label matches the start row labels
                // and add the child-parent relation to the corresponding map
                if (labels_of_the_start_node.count(j_next) && current_parent_labels.count(j_next)) {
                    acceptable_node = true;
                }
            }

            // if the row has at least one label that matches
            // the start row's labels then add it to the stack
            if (acceptable_node) {
                to_visit.push_back(next_to_anno);
            }

        } );
    }

    // DFS backwards is similar to the DFS forwards
    to_visit.clear();
    to_visit.push_back(i);


    std::unordered_set<Row> visited_nodes_backward;
    while (!to_visit.empty()) {
        Row current_child = to_visit.back();
        to_visit.pop_back();

        if (visited_nodes_backward.count(current_child))
            continue;

        std::unordered_set<Column> current_child_labels;
        for (auto &rowt_parent : rows_annotations[current_child]) {
            current_child_labels.insert(rowt_parent.first);
        }
        
        visited_nodes_backward.insert(current_child);

        graph::AnnotatedSequenceGraph::node_index current_child_to_graph = graph::AnnotatedSequenceGraph::anno_to_graph_index(current_child);

        graph_->call_incoming_kmers(current_child_to_graph, [&](auto previous, char c) {

            if (c == graph::boss::BOSS::kSentinel)
                return;
            
            Row previous_to_anno = graph::AnnotatedSequenceGraph::graph_to_anno_index(previous);
            RowTuples previous_annotations;

            if (!rows_annotations.count(previous_to_anno)) {
                previous_annotations = get_row_tuples(previous_to_anno, rd_ids, node_to_rd, rd_rows,
                rows_annotations, boss, rd_succ);
            } else {
                previous_annotations = rows_annotations[previous_to_anno];
            }

            // aceptable node is a node whose labels match
            // the labels of the start node
            bool acceptable_node = false;

            for (auto &[j_prev, tuple_prev] : previous_annotations) {
                if (labels_of_the_start_node.count(j_prev) && current_child_labels.count(j_prev)) {
                    acceptable_node = true;
                }
            }
            if (acceptable_node) {
                to_visit.push_back(previous_to_anno);
            }
                        
        } );

    }

    for (auto & [row, row_tuples] : rows_annotations) {
        for (auto & [j, tuple] : row_tuples) {
            if (!labels_of_the_start_node.count(j)) {
                continue;
            }
            for (uint64_t &c : tuple) {
                reconstucted_paths[j][c] = row;
            }
        }
    }
    
    // part 2. Paths reconstruction

    // new path reconstruction for graphs with unlabeled reads

    std::unordered_map<Column, std::vector<std::pair<Row, uint64_t>>> traces_unlabeled;

    for (auto & [j, coords_map] : reconstucted_paths) {
        for (auto & [ccoord, rrow] : coords_map) {
            traces_unlabeled[j].push_back(std::make_pair(rrow, ccoord));
        }
    }

    // extact reads from walk that has jumps in coordinates

    std::vector<std::tuple<std::vector<std::pair<Row, uint64_t>>, Column, uint64_t>> result_coords;

    for (auto & [j, trace_with_jumps] : traces_unlabeled) {

        uint64_t curr_read_start_coord = trace_with_jumps.front().second;
        uint64_t curr_read_curr_coord = curr_read_start_coord;

        std::vector<Row> curr_read_trace;
        curr_read_trace.push_back(trace_with_jumps.front().first);

        auto curr_edge_node = graph::AnnotatedSequenceGraph::anno_to_graph_index(trace_with_jumps.front().first);


        std::vector<std::pair<Row, uint64_t>> curr_read_trace_coords;
        curr_read_trace_coords.push_back(trace_with_jumps.front());

        for (size_t cur_pos = 1; cur_pos < trace_with_jumps.size(); ++cur_pos) {

            auto curr_next_edge_node = graph::AnnotatedSequenceGraph::anno_to_graph_index(trace_with_jumps[cur_pos].first);
            
            bool edge_exists = false;

            graph_->adjacent_outgoing_nodes(curr_edge_node, [&](auto adj_outg_node) {
                if (adj_outg_node == curr_next_edge_node) {
                    edge_exists = true;
                    return;
                }
            });


            if ((trace_with_jumps[cur_pos].second == (curr_read_curr_coord + SHIFT)) && edge_exists) {
                curr_read_trace.push_back(trace_with_jumps[cur_pos].first);
                curr_read_trace_coords.push_back(trace_with_jumps[cur_pos]);

                curr_read_curr_coord += SHIFT;                

            } else {
                result.push_back(std::make_tuple(curr_read_trace, j, curr_read_start_coord));

                result_coords.push_back(std::make_tuple(curr_read_trace_coords, j, curr_read_start_coord));

                curr_read_start_coord = trace_with_jumps[cur_pos].second;
                curr_read_curr_coord = curr_read_start_coord;

                // curr_read_trace.clear();
                // curr_read_trace.push_back(trace_with_jumps[cur_pos].first);

                // curr_read_trace_coords.clear();
                // curr_read_trace_coords.push_back(trace_with_jumps[cur_pos]);

                if (edge_exists) {
                    size_t delta_coord = curr_read_curr_coord - trace_with_jumps[cur_pos].second;

                    std::vector<Row> trace_slice;
                    trace_slice = std::vector<Row>(curr_read_trace.begin() + (curr_read_trace.size() - delta_coord), curr_read_trace.end());
                    std::vector<std::pair<Row, uint64_t>> trace_slice_coords;
                    trace_slice_coords = std::vector<std::pair<Row, uint64_t>>(curr_read_trace_coords.begin() + (curr_read_trace_coords.size() - delta_coord), curr_read_trace_coords.end());

                    curr_read_trace.clear();
                    curr_read_trace_coords.clear();

                    curr_read_trace.insert(curr_read_trace.begin(), trace_slice.begin(), trace_slice.end());
                    curr_read_trace_coords.insert(curr_read_trace_coords.begin(), trace_slice_coords.begin(), trace_slice_coords.end());

                    curr_read_trace.push_back(trace_with_jumps[cur_pos].first);
                    curr_read_trace_coords.push_back(trace_with_jumps[cur_pos]);
                } else {
                    curr_read_trace.clear();
                    curr_read_trace_coords.clear();

                    curr_read_trace.push_back(trace_with_jumps[cur_pos].first);
                    curr_read_trace_coords.push_back(trace_with_jumps[cur_pos]);
                }
            }

            curr_edge_node = curr_next_edge_node;
        }
    }


    std::vector<std::tuple<std::vector<std::pair<Row, uint64_t>>, Column, uint64_t>> final_result;
    
    for (auto & [read_trace, read_label, read_start_coord] : result_coords) { 
        for (size_t cur_pos = 0; cur_pos < read_trace.size(); ++cur_pos) {
            if (read_trace[cur_pos].first == i) {
                final_result.push_back(std::make_tuple(read_trace, read_label, cur_pos));
                break;
            }
        }
    }

    std::vector<std::tuple<std::vector<Row>, Column, uint64_t>> final_final_result;


    for (auto & [read_trace, read_label, cur_pos] : final_result) {
        std::vector<Row> cur_trace_rows;
        for (auto & el : read_trace) {
            cur_trace_rows.push_back(el.first);
        }

        final_final_result.push_back(std::make_tuple(cur_trace_rows, read_label, cur_pos));
    }

    return final_final_result;

    // // concatenate all labeled paths
    // for (Column const &j : labels_of_the_start_node) {

    //     std::vector<std::pair<Row, uint64_t>> path_to_start = traces_backward_sep_by_labels[j];
    //     std::vector<std::pair<Row, uint64_t>> path_to_end = traces_forward_sep_by_labels[j];

    //     if (path_to_start.empty()) {
            
    //         std::vector<Row> path_to_add_to_result;
    //         path_to_add_to_result.reserve(path_to_end.size());
    //         for (auto & [row, c] : path_to_end) { 
    //             path_to_add_to_result.push_back(row);
    //         }

    //         reconstructed_reads[j] = path_to_add_to_result;
    //         result.push_back(std::make_tuple(path_to_add_to_result, j, input_row_position_in_ref_seq[j]));

    //     } else if (path_to_end.empty()) {

    //         std::vector<Row> path_to_add_to_result;
    //         path_to_add_to_result.reserve(path_to_start.size());
    //         for (auto & [row, c] : path_to_start) { 
    //             path_to_add_to_result.push_back(row);
    //         }

    //         reconstructed_reads[j] = path_to_add_to_result;
    //         result.push_back(std::make_tuple(path_to_add_to_result, j, input_row_position_in_ref_seq[j]));

    //     } else {

    //         // if graph contains cycles then forward and backward paths may overlap
    //         std::vector<std::pair<Row, uint64_t>> starting_path;
    //         std::vector<std::pair<Row, uint64_t>> ending_path;

    //         // find which path has a starting node (i.e. first node with lower coordinate)
    //         if (path_to_start.front().second < path_to_end.front().second)
    //             starting_path = path_to_start;
    //         else
    //             starting_path = path_to_end;

    //         // find which path has an starting node (i.e. last node with bigger coordinate)
    //         if (path_to_start.back().second > path_to_end.back().second)
    //             ending_path = path_to_start;
    //         else
    //             ending_path = path_to_end;

    //         uint64_t last_coord_start_path = starting_path.back().second;
    //         std::vector<Row> path_to_add_to_result;

    //         for (auto it = ending_path.rbegin(); it < ending_path.rend(); ++it) {
    //             if (it->second > last_coord_start_path)
    //                 path_to_add_to_result.insert(path_to_add_to_result.begin(), it->first);
    //             else
    //                 break;
    //         }

    //         std::vector<Row> rows_start_path;
    //         rows_start_path.reserve(starting_path.size());
    //         for (auto & [row, c] : starting_path) {
    //             rows_start_path.push_back(row);
    //         }            

    //         path_to_add_to_result.insert(path_to_add_to_result.begin(),
    //                                      rows_start_path.begin(),
    //                                      rows_start_path.end());

    //         reconstructed_reads[j] = path_to_add_to_result;
    //         result.push_back(std::make_tuple(path_to_add_to_result, j, input_row_position_in_ref_seq[j]));
    //     }
    // }    

    // return result;
}

template <class BaseMatrix>
std::vector<std::pair<MultiIntMatrix::Column, uint64_t>> TupleRowDiff<BaseMatrix>
::get_traces_with_row(Row i, std::unordered_map<Row, RowTuples> &rows_annotations, std::vector<Row> &rd_ids, VectorMap<Row, size_t> &node_to_rd, 
std::vector<RowTuples> &rd_rows, std::unordered_map<Column, std::vector<Row>> &reconstructed_reads) const {
    assert(graph_ && "graph must be loaded");
    assert(anchor_.size() == diffs_.num_rows() && "anchors must be loaded");
    assert(!fork_succ_.size() || fork_succ_.size() == graph_->get_boss().get_last().size());   

    const graph::boss::BOSS &boss = graph_->get_boss();
    const bit_vector &rd_succ = fork_succ_.size() ? fork_succ_ : boss.get_last();

    std::unordered_map<Column, std::map<uint64_t, Row>> reconstructed_paths;

    // get annotations for the start node to obtain labels of paths to reconstuct
    RowTuples input_row_tuples;
    if (!rows_annotations.count(i)) {
        input_row_tuples = get_row_tuples(i, rd_ids, node_to_rd, rd_rows,
        rows_annotations, boss, rd_succ);
    } else {
        // if we already decompressed the annotations for this node
        // then simply retrieve them from the map
        input_row_tuples = rows_annotations[i];
    }

    std::unordered_set<Column> labels_of_the_start_node;

    // initialize the result 
    std::vector<std::pair<Column, uint64_t>> result;

    // positions of the input k-mer in reconstucted reads
    std::unordered_map<Column, uint64_t> input_row_position_in_ref_seq;

    for (auto &[j, tuple] : input_row_tuples) {
        if (!reconstructed_reads.count(j)) {
            labels_of_the_start_node.insert(j);
            input_row_position_in_ref_seq[j] = tuple.front();
            for (uint64_t &c : tuple)
                reconstructed_paths[j][c] = i;
        } else {
            // add already reconstucted path to the result
            result.push_back(std::make_pair(j, tuple.front()));
        }
    }

    if (labels_of_the_start_node.empty())
        return result;

    // DFS traversal forwards
    std::vector<Row> to_visit;
    to_visit.push_back(i);

    std::unordered_set<Row> visited_nodes_forward;
    while (!to_visit.empty()) {
        Row current_parent = to_visit.back();
        to_visit.pop_back();

        if (visited_nodes_forward.count(current_parent))
            continue;

        visited_nodes_forward.insert(current_parent);

        graph::AnnotatedSequenceGraph::node_index current_parent_to_graph = graph::AnnotatedSequenceGraph::anno_to_graph_index(
            current_parent);

        // for each adjacent outgoing k-mer get the annotations
        graph_->call_outgoing_kmers(current_parent_to_graph, [&](auto next, char c) {
            if (c == graph::boss::BOSS::kSentinel)
                return;
            
            Row next_to_anno = graph::AnnotatedSequenceGraph::graph_to_anno_index(next);
            RowTuples next_annotations;

            // if the annotations for the child node are not decompressed,
            // build rd-path and decompress the annotations
            if (!rows_annotations.count(next_to_anno)) {
                next_annotations = get_row_tuples(next_to_anno, rd_ids, node_to_rd, rd_rows,
                rows_annotations, boss, rd_succ);
            } else {
                // if we already decompressed the annotations for this node
                // then simply retrieve them from the map
                next_annotations = rows_annotations[next_to_anno];
            }

            // acceptable node is a node whose labels match
            // the labels of the start node
            bool acceptable_node = false;
            
            for (auto &[j_next, tuple_next] : next_annotations) {
                // check if the label matches the start row labels
                if (labels_of_the_start_node.count(j_next)) {
                    acceptable_node = true;
                    for (uint64_t &c : tuple_next)
                        reconstructed_paths[j_next][c] = next_to_anno;
                }
                    
            }

            // if the row has at least one label that matches
            // the start row's labels then add it to the stack
            if (acceptable_node) {
                to_visit.push_back(next_to_anno);
            }

        } );   
    }

    // DFS backwards is similar to the DFS forwards
    to_visit.clear();
    to_visit.push_back(i);

    std::unordered_set<Row> visited_nodes_backward;
    while (!to_visit.empty()) {
        Row current_child = to_visit.back();
        to_visit.pop_back();

        if (visited_nodes_backward.count(current_child))
            continue;
        
        visited_nodes_backward.insert(current_child);
        graph::AnnotatedSequenceGraph::node_index current_child_to_graph = graph::AnnotatedSequenceGraph::anno_to_graph_index(
            current_child);

        graph_->call_incoming_kmers(current_child_to_graph, [&](auto previous, char c) {
            if (c == graph::boss::BOSS::kSentinel)
                return;
            
            Row previous_to_anno = graph::AnnotatedSequenceGraph::graph_to_anno_index(previous);
            RowTuples previous_annotations;

            if (!rows_annotations.count(previous_to_anno)) {
                previous_annotations = get_row_tuples(previous_to_anno, rd_ids, node_to_rd, rd_rows,
                rows_annotations, boss, rd_succ);
            } else {
                previous_annotations = rows_annotations[previous_to_anno];
            }

            // aceptable node is a node whose labels match
            // the labels of the start node
            bool acceptable_node = false;

            for (auto &[j_prev, tuple_prev] : previous_annotations) {
                if (labels_of_the_start_node.count(j_prev)) {
                    acceptable_node = true;
                    for (uint64_t &c : tuple_prev)
                        reconstructed_paths[j_prev][c] = previous_to_anno;
                }
            }

            if (acceptable_node)
                to_visit.push_back(previous_to_anno);
        } );
    }
    
    for (auto & [j, coords_map] : reconstructed_paths) {
        std::vector<Row> labeled_trace;
        for (auto & [c, row_c] : coords_map) {
            labeled_trace.push_back(row_c);
        }
        reconstructed_reads[j] = labeled_trace;
        result.push_back(std::make_pair(j, input_row_position_in_ref_seq[j]));        
    }

    return result;
}

} // namespace matrix
} // namespace annot
} // namespace mtg

#endif // __TUPLE_ROW_DIFF_HPP__
