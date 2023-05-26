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
class TupleRowDiff : public binmat::IRowDiff, public MultiIntMatrix {
  public:
    static_assert(std::is_convertible<BaseMatrix*, MultiIntMatrix*>::value);
    static const int SHIFT = 1; // coordinates increase by 1 at each edge

    static const size_t MAX_ROWS_WITH_DECOMPRESSED_ANNOTATIONS = 1'000;

    // check graph traversal in batches
    static const uint64_t TRAVERSAL_BATCH_SIZE = 1'000;

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
    // std::vector<std::tuple<std::vector<Row>, Column, uint64_t>> get_traces_with_row_auto_labels(Row i, std::unordered_map<Row, RowTuples> &rows_annotations,
    // std::vector<Row> &rd_ids, VectorMap<Row, size_t> &node_to_rd, std::vector<RowTuples> &rd_rows) const;

    std::vector<std::vector<std::tuple<std::vector<Row>, Column, uint64_t>>> get_traces_with_row_auto_labels(std::vector<Row> i, std::vector<std::unordered_set<uint64_t>> manifest_labels) const;
    std::vector<std::tuple<std::vector<Row>, Column, uint64_t>> get_traces_with_row_auto_labels(Row i, std::unordered_map<Row, RowTuples> &rows_annotations, 
    std::vector<Row> &rd_ids, VectorMap<Row, size_t> &node_to_rd, std::vector<RowTuples> &rd_rows, std::unordered_set<uint64_t> labels_of_interest, std::unordered_map<Column, std::vector<std::vector<std::pair<Row, uint64_t>>>> &reconstructed_reads) const;

    std::vector<std::unordered_set<uint64_t>> get_labels_of_rows(std::vector<Row> i) const;
    

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
    std::unordered_map<Column, std::vector<std::vector<std::pair<Row, uint64_t>>>> reconstructed_reads;

    std::vector<std::vector<std::tuple<std::vector<Row>, Column, uint64_t>>> result;

    auto initial_labels = get_labels_of_rows(i);

    for (size_t j = 0; j < i.size(); ++j) {
        auto curr_res = get_traces_with_row_auto_labels(i[j], rows_annotations, rd_ids, node_to_rd, rd_rows, initial_labels[j], reconstructed_reads);
        result.push_back(curr_res);
    }

    // for (Row &row_i : i) {
    //     auto curr_res = get_traces_with_row_auto_labels(row_i, rows_annotations, rd_ids, node_to_rd, rd_rows, {});
    //     result.push_back(curr_res);
    // }

    return result;
}

template <class BaseMatrix>
std::vector<std::vector<std::tuple<std::vector<MultiIntMatrix::Row>, MultiIntMatrix::Column, uint64_t>>> TupleRowDiff<BaseMatrix>
::get_traces_with_row_auto_labels(std::vector<Row> i, std::vector<std::unordered_set<uint64_t>> manifest_labels) const {
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
    std::unordered_map<Column, std::vector<std::vector<std::pair<Row, uint64_t>>>> reconstructed_reads;

    std::vector<std::vector<std::tuple<std::vector<Row>, Column, uint64_t>>> result;

    for (size_t j = 0; j < i.size(); ++j) {
        std::unordered_set<uint64_t> labels_of_interest = manifest_labels[j];
        auto curr_res = get_traces_with_row_auto_labels(i[j], rows_annotations, rd_ids, node_to_rd, rd_rows, labels_of_interest, reconstructed_reads);
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
std::vector<RowTuples> &rd_rows, std::unordered_set<uint64_t> labels_of_interest, std::unordered_map<Column, std::vector<std::vector<std::pair<Row, uint64_t>>>> &reconstructed_reads) const {
    assert(graph_ && "graph must be loaded");
    assert(anchor_.size() == diffs_.num_rows() && "anchors must be loaded");
    assert(!fork_succ_.size() || fork_succ_.size() == graph_->get_boss().get_last().size());

    std::ignore = rows_annotations;
    std::ignore = rd_ids;
    std::ignore = rd_rows;
    std::ignore = node_to_rd;

    std::vector<std::tuple<std::vector<std::pair<Row, uint64_t>>, Column, uint64_t>> final_result;
    std::unordered_map<Column, std::set<std::pair<uint64_t, uint64_t>>> already_reconstruced_reads_ends;
    std::unordered_map<Column, std::unordered_set<uint64_t>> added_reads_before_traversal_start_coordinates;

    // const graph::boss::BOSS &boss = graph_->get_boss();
    // const bit_vector &rd_succ = fork_succ_.size() ? fork_succ_ : boss.get_last();

    std::unordered_map<Column, std::map<uint64_t, Row>> reconstucted_paths;
    // std::unordered_map<Column, std::set<uint64_t>> coordinates_to_check_if_traverse_more;
    std::unordered_map<Column, std::map<uint64_t, Row>> reconstucted_paths_backwards;

    mtg::common::logger->trace("Getting annotations of starts node");

    RowTuples input_row_tuples = get_row_tuples(i);

    // collect the coordinates of the start node in two structures
    for (auto &[j, tuple] : input_row_tuples) {
        if (!labels_of_interest.count(j))
            continue;

        for (uint64_t &c : tuple) {

            bool already_in_the_read = false;

            for (auto & already_reconstruced_read : reconstructed_reads[j]) {
                uint64_t first_coord_of_read = already_reconstruced_read.front().second;
                uint64_t last_coord_of_read = already_reconstruced_read.back().second;

                // if current coordinates falls in the already reconstructed read coordinate range
                if ((c >= first_coord_of_read && c <= last_coord_of_read)) {
                    already_in_the_read = true;

                    if (!already_reconstruced_reads_ends[j].count(std::make_pair(first_coord_of_read, last_coord_of_read))) {
                        final_result.push_back(std::make_tuple(already_reconstruced_read, j, c - first_coord_of_read));
                        already_reconstruced_reads_ends[j].insert(std::make_pair(first_coord_of_read, last_coord_of_read));

                        added_reads_before_traversal_start_coordinates[j].insert(already_reconstruced_read.front().second);
                    }

                    // if we found the read, no need to iterate more (maybe :))
                    break;
                }
            }

            if (already_in_the_read)
                continue;

            // is used to check if the graph has to be traversed more
            // based on the coordinates shift
            // upd 26 may we don't need that as the check must 
            // be performed based on the all discovered coordinates (not the ones from the current batch)
            // coordinates_to_check_if_traverse_more[j].insert(c);

            // is used to collect all visited nodes and trace the result paths
            reconstucted_paths[j][c] = i;
            reconstucted_paths_backwards[j][c] = i;
        }
    }

    if (reconstucted_paths.empty()) {
        std::vector<std::tuple<std::vector<Row>, Column, uint64_t>> final_final_result;
        for (auto & [read_trace, read_label, cur_pos] : final_result) {
            std::vector<Row> cur_trace_rows;
            for (auto & el : read_trace) {
                cur_trace_rows.push_back(el.first);
            }
            final_final_result.push_back(std::make_tuple(cur_trace_rows, read_label, cur_pos));
        }

        return final_final_result;
    }

    std::unordered_map<Column, std::set<uint64_t>> ends_of_reads;

    mtg::common::logger->trace("Getting initial ends of reads");

    for (auto & [j, coords] : reconstucted_paths) {
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
    }


    // std::cout << "Starting DFS forwards\n";

    // stack for the DFS traversal
    // (std::vector is used because it has to be cleared after forward traversal)
    // std::queue<Row> to_visit;
    // to_visit.push(i);
    std::deque<Row> to_visit;
    to_visit.push_back(i);

    std::unordered_set<Row> visited_nodes_forward;
    uint64_t batches_count = 0;

    std::unordered_map<Column, std::set<uint64_t>> all_discovered_coordinates;

    mtg::common::logger->trace("Traversing graph forwards");

    while (true) {
        uint64_t traversed_nodes_count = 0;

        // set of rows in the current batch to decompresse the annotations for
        std::unordered_set<Row> batch_of_rows_set;

        // std::cout << "Starting traversing batch " << batches_count << '\n';
        // std::cout << "to_visit.size() = " << to_visit.size() << '\n';

        while (!to_visit.empty()) {
            // Row current_parent = to_visit.front();
            // to_visit.pop();
            Row current_parent = to_visit.front();
            to_visit.pop_front();

            if (visited_nodes_forward.count(current_parent))
                continue;
            
            batch_of_rows_set.insert(current_parent);
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

        // convert a set of visited rows to vector
        std::vector<Row> batch_of_rows;
        batch_of_rows.reserve(batch_of_rows_set.size());
        batch_of_rows.insert(batch_of_rows.end(),
        batch_of_rows_set.begin(), batch_of_rows_set.end());

        // std::cout << "batch size = " << batch_of_rows.size() << '\n';

        // std::cout << "Getting annotations for rows in a batch" << '\n';

        // decompress the annotations for the current batch of nodes
        std::vector<RowTuples> batch_annotations = get_row_tuples(batch_of_rows);

        // new coordinates discovered in the current batch of nodes
        // std::unordered_map<Column, std::set<uint64_t>> new_coordinates_batch;

        // std::cout << "Collecting discovered coordinates" << '\n';
        // collect all new discovered coordinates
        for (size_t qqq = 0; qqq < batch_of_rows.size(); ++qqq) {
            for (auto & [j_next, tuple_next] : batch_annotations[qqq]) {
                if (!labels_of_interest.count(j_next))
                    continue;

                for (uint64_t &c_next : tuple_next) {
                    // new_coordinates_batch[j_next].insert(c_next);
                    all_discovered_coordinates[j_next].insert(c_next);
                    reconstucted_paths[j_next][c_next] = batch_of_rows[qqq];
                }
            }
        }

        // std::cout << "New coordinates from batch : ";
        // for (auto & [j_qq, set_qq] : new_coordinates_batch) {
        //     std::cout << " Column = " << j_qq << "  new coord size = " << set_qq.size() << " ; "; 
        // }
        // std::cout << std::endl;
        
        // std::cout << "Updating reads ends" << '\n';

        // update reads ends
        for (auto & [j_end, coords_ends] : ends_of_reads) {
            for (auto & c_end : coords_ends) {
                auto c_end_cur = c_end;
                bool replace_read_end = false;

                auto end_node_old = graph::AnnotatedSequenceGraph::anno_to_graph_index(reconstucted_paths[j_end][c_end_cur]);
                while (all_discovered_coordinates[j_end].count(c_end_cur + SHIFT)) {
                    auto end_node_new = graph::AnnotatedSequenceGraph::anno_to_graph_index(reconstucted_paths[j_end][c_end_cur + SHIFT]);

                    bool edge_exists = false;
                    graph_->adjacent_outgoing_nodes(end_node_old, [&](auto adj_outg_node) {
                        if (adj_outg_node == end_node_new) {
                            edge_exists = true;
                            return;
                        }
                    });

                    // if edge exists, then read has its continuation => we have to traverse the graph more
                    if (edge_exists) {
                        end_node_old = end_node_new;
                        replace_read_end = true;
                        c_end_cur += SHIFT;
                    } else {
                        break;
                    }
                }

                if (replace_read_end) {
                    ends_of_reads[j_end].erase(c_end);
                    ends_of_reads[j_end].insert(c_end_cur);
                }
            }
        }

        // check if graph has to be traversed more 
        // by checking if there are continuations of the reads after the known ends

        // std::cout << "Checking if traverse more" << '\n';

        // std::cout << "Ends of reads size : ";
        // for (auto & [j_qq, ends_qq] : ends_of_reads) {
        //     std::cout << " COlumn = " << j_qq << "  ends size = " << ends_qq.size() << " ; ";
        // }
        // std::cout << '\n';

        bool has_to_be_traversed_more = false;

        for (auto & [j_end, coords_ends] : ends_of_reads) {
            for (auto & c_end : coords_ends) {
                auto end_node_old = graph::AnnotatedSequenceGraph::anno_to_graph_index(reconstucted_paths[j_end][c_end]);

                graph_->call_outgoing_kmers(end_node_old, [&](auto adj_outg_node, char c) {
                    // std::ignore = c;
                    if (c == graph::boss::BOSS::kSentinel)
                        return;
                    
                    Row adj_outg_to_anno = graph::AnnotatedSequenceGraph::graph_to_anno_index(adj_outg_node);
                    RowTuples outg_annotataions = get_row_tuples(adj_outg_to_anno);

                    std::set<uint64_t> coordinates_of_the_adj_outg_node;
                    for (auto &[j_next, tuple_next] : outg_annotataions) {
                        if (!labels_of_interest.count(j_next) || j_next != j_end)
                            continue;

                        for (uint64_t &c_next : tuple_next)
                            coordinates_of_the_adj_outg_node.insert(c_next);
                    }

                    if (coordinates_of_the_adj_outg_node.count(c_end + SHIFT) && !visited_nodes_forward.count(c_end + SHIFT)) {
                        has_to_be_traversed_more = true;
                        // reconstucted_paths[j_end][c_end + SHIFT] = adj_outg_to_anno;
                        to_visit.push_front(adj_outg_to_anno);
                        return;
                    }
                });

                if (has_to_be_traversed_more)
                    break;
            }

            if (has_to_be_traversed_more)
                break;
        }

        if (!has_to_be_traversed_more)
            break;  
    }


    mtg::common::logger->trace("Traversed batches count {}", batches_count);

    // std::cout << "Preparing for the backwards traversal" << '\n';

    // do this at the very forward traversal stage
    // std::unordered_map<Column, std::map<uint64_t, Row>> reconstucted_paths_backwards;
    // // collect the coordinates of the start node in two structures
    // for (auto &[j, tuple] : input_row_tuples) {
    //     if (!labels_of_interest.count(j))
    //         continue;

    //     for (uint64_t &c : tuple) {
    //         // is used to collect all visited nodes and trace the result paths
    //         reconstucted_paths_backwards[j][c] = i;
    //     }
    // }


    // std::cout << "Initializing starts of the reads" << '\n';

    mtg::common::logger->trace("Getting initial starts of the reads");

    std::unordered_map<Column, std::set<uint64_t>> starts_of_reads;


    for (auto & [j, coords] : reconstucted_paths_backwards) {
        auto node_1 = graph::AnnotatedSequenceGraph::anno_to_graph_index(coords.rbegin()->second);
        auto coord_1 = coords.rbegin()->first;

        for (auto iter_map = std::next(coords.rbegin()); iter_map != coords.rend(); ++iter_map) {
            auto node_2 = graph::AnnotatedSequenceGraph::anno_to_graph_index(iter_map->second);
            auto coord_2 = iter_map->first;

            // check if an edge exists between the current and next node
            bool edge_exists = false;
            graph_->adjacent_incoming_nodes(node_1, [&](auto adj_inc_node) {
                if (adj_inc_node == node_2) {
                    edge_exists = true;
                    return;
                }
            });


            // if edge does not exist, then this is the end of the read
            if (!((coord_2 == (coord_1 - SHIFT)) && edge_exists)) {
                starts_of_reads[j].insert(coord_1);
            }

            node_1 = node_2;
            coord_1 = coord_2;           
        }

        starts_of_reads[j].insert(coord_1);
    }

    // std::cout << "Starting DFS backwards\n";

    // traverse the graph backwards
    // to_visit = std::queue<Row>();
    // to_visit.push(i);
    to_visit = std::deque<Row>();
    to_visit.push_back(i);

    std::unordered_set<Row> visited_nodes_backward;
    uint64_t batches_count_backwards = 0;

    mtg::common::logger->trace("Traversing graph backwards");

    while (true) {

        uint64_t traversed_nodes_count = 0;

        // set of rows in the current batch to decompresse the annotations for
        std::unordered_set<Row> batch_of_rows_set;

        // std::cout << "Started traversing a new batch " << batches_count_backwards << '\n';
        // std::cout << "to_visit.size() = " << to_visit.size() << '\n';

        while (!to_visit.empty()) {
            // Row current_child = to_visit.front();
            // to_visit.pop();
            Row current_child = to_visit.front();
            to_visit.pop_front();

            if (visited_nodes_backward.count(current_child)) {
                continue;
            }

            batch_of_rows_set.insert(current_child);
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

        // convert a set of visited rows to vector
        std::vector<Row> batch_of_rows;
        batch_of_rows.reserve(batch_of_rows_set.size());
        batch_of_rows.insert(batch_of_rows.end(),
        batch_of_rows_set.begin(), batch_of_rows_set.end());

        std::vector<RowTuples> batch_annotations = get_row_tuples(batch_of_rows);

        // std::unordered_map<Column, std::set<uint64_t>> new_coordinates_batch;


        // std::cout << "Collecting discovered coordinates" << '\n';

        // collect all new discovered coordinates
        for (size_t qqq = 0; qqq < batch_of_rows.size(); ++qqq) {
            for (auto & [j_prev, tuple_prev] : batch_annotations[qqq]) {
                if (!labels_of_interest.count(j_prev))
                    continue;

                for (uint64_t &c_prev : tuple_prev) {
                    // new_coordinates_batch[j_prev].insert(c_prev);
                    all_discovered_coordinates[j_prev].insert(c_prev);
                    reconstucted_paths[j_prev][c_prev] = batch_of_rows[qqq];
                }
            }
        }

        // std::cout << "Discovered coordinates sizes : \n";
        // for (auto & [qweqwe, qweqweqwe] : new_coordinates_batch) {
        //     std::cout << " Column = " << qweqwe << " new coord set size() = " << qweqweqwe.size() << '\n';
        // }
        // std::cout << '\n';


        // std::cout << "Updating reads starts" << '\n';

        // update reads starts
        for (auto & [j_start, coords_starts] : starts_of_reads) {
            for (auto & c_start : coords_starts) {
                if (c_start == 0)
                    continue;
                auto c_end_cur = c_start;
                bool replace_read_end = false;

                auto end_node_old = graph::AnnotatedSequenceGraph::anno_to_graph_index(reconstucted_paths[j_start][c_end_cur]);
                while (all_discovered_coordinates[j_start].count(c_end_cur - SHIFT)) {
                    auto end_node_new = graph::AnnotatedSequenceGraph::anno_to_graph_index(reconstucted_paths[j_start][c_end_cur - SHIFT]);

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
                    starts_of_reads[j_start].erase(c_start);
                    starts_of_reads[j_start].insert(c_end_cur);
                }
            }
        }


        
        // check if graph has to be traversed more 
        // by checking if there are continuations of the reads after the known ends

        // std::cout << "Checking if graph has to be traversed more backwards" << '\n';

        bool has_to_be_traversed_more = false;
        for (auto & [j_end, coords_ends] : starts_of_reads) {
            for (auto & c_end : coords_ends) {
                if (c_end == 0)
                    continue;

                auto end_node_old = graph::AnnotatedSequenceGraph::anno_to_graph_index(reconstucted_paths[j_end][c_end]);

                graph_->call_incoming_kmers(end_node_old, [&](auto adj_outg_node, char c) {
                    // std::ignore = c;
                    if (c == graph::boss::BOSS::kSentinel)
                        return;
                    
                    Row adj_outg_to_anno = graph::AnnotatedSequenceGraph::graph_to_anno_index(adj_outg_node);
                    RowTuples outg_annotataions = get_row_tuples(adj_outg_to_anno);

                    std::set<uint64_t> coordinates_of_the_adj_outg_node;
                    for (auto &[j_next, tuple_next] : outg_annotataions) {
                        if (!labels_of_interest.count(j_next) || j_next != j_end)
                            continue;

                        for (uint64_t &c_next : tuple_next)
                            coordinates_of_the_adj_outg_node.insert(c_next);
                    }

                    if (coordinates_of_the_adj_outg_node.count(c_end - SHIFT) && !visited_nodes_backward.count(c_end - SHIFT)) {
                        has_to_be_traversed_more = true;
                        to_visit.push_front(adj_outg_to_anno);
                        return;
                    }
                });

                if (has_to_be_traversed_more)
                    break;
            }

            if (has_to_be_traversed_more)
                break;
        }

        if (!has_to_be_traversed_more)
            break;

    }


    // std::cout << "Reads starts : \n";
    // for (auto & [j, weqwe] : starts_of_reads) {
    //     std::cout << "Column = " << j << " : ";
    //     for (auto & qqq : weqwe) {
    //         std::cout << qqq << ", ";
    //     }
    //     std::cout << '\n';
    // }

    // std::cout << "Reads ends : \n";
    // for (auto & [j, weqwe] : ends_of_reads) {
    //     std::cout << "Column = " << j << " : ";
    //     for (auto & qqq : weqwe) {
    //         std::cout << qqq << ", ";
    //     }
    //     std::cout << '\n';
    // }

    // reconstruct the paths

    // not sure about that
    // std::cout << "reads starts and ends: \n";
    // for (auto & [j_reads, reads_ends] : ends_of_reads) {
    //     std::cout << "Column = " << j_reads << '\n';
    //     std::cout << " ends : ";
    //     for (auto & qwe : reads_ends) {
    //         std::cout << qwe << ", ";
    //     }
    //     std::cout << "\nstarts : ";
    //     for (auto & qwe : starts_of_reads[j_reads]) {
    //         std::cout << qwe << ", ";
    //     }
    //     std::cout << '\n';
    //     // assert(reads_ends.size() == starts_of_reads[j_reads].size());
    // }

    // std::cout << "Explicitly check incoming nodes of the start node : \n";

    // for (auto & [j_qwe, j_starts] : starts_of_reads) {
    //     std::cout << " Column = " << j_qwe << " : ";
    //     for (auto & qqqq : j_starts) {

    //         auto end_node_old = graph::AnnotatedSequenceGraph::anno_to_graph_index(reconstucted_paths[j_qwe][qqqq]);

    //         bool exists_prev = false;

    //         graph_->call_incoming_kmers(end_node_old, [&](auto adj_outg_node, char c) {
    //             if (c == graph::boss::BOSS::kSentinel)
    //                 return;
                
    //             Row adj_outg_to_anno = graph::AnnotatedSequenceGraph::graph_to_anno_index(adj_outg_node);
    //             RowTuples outg_annotataions = get_row_tuples(adj_outg_to_anno);

    //             std::set<uint64_t> coordinates_of_the_adj_outg_node;
    //             for (auto &[j_next, tuple_next] : outg_annotataions) {
    //                 if (!labels_of_interest.count(j_next) || j_next != j_qwe)
    //                     continue;

    //                 for (uint64_t &c_next : tuple_next)
    //                     coordinates_of_the_adj_outg_node.insert(c_next);
    //             }
                
    //             if (coordinates_of_the_adj_outg_node.count(qqqq - SHIFT) && !visited_nodes_backward.count(adj_outg_to_anno))
    //                 exists_prev = true;
    //         });
            
    //         std::cout << "For coord = " << qqqq << " exists prev = " << exists_prev << '\n';
    //     }
    // }

    // std::cout << '\n';


    mtg::common::logger->trace("Traversed batches backwards {}", batches_count_backwards);


    std::cout << "Number of visited nodes forwards = " << visited_nodes_forward.size() << "   ;   backward = " << visited_nodes_backward.size() << '\n';


    mtg::common::logger->trace("Reconstructing paths...");

    // traces where separate reads don't have distinct labels
    std::unordered_map<Column, std::vector<std::pair<Row, uint64_t>>> traces_unlabeled;
    for (auto & [j, coords_map] : reconstucted_paths) {
        for (auto & [ccoord, rrow] : coords_map) {
            traces_unlabeled[j].push_back(std::make_pair(rrow, ccoord));
        }
    }

    // extract reads from walks which have jumps in coordinates
    std::vector<std::tuple<std::vector<std::pair<Row, uint64_t>>, Column, uint64_t>> result_coords;
    

    // for (auto & [j_st, coords_st] : starts_of_reads) {
    //     assert(coords_st.size() == ends_of_reads[j_st].size());
    //     auto it_start = coords_st.begin();
    //     auto it_end = ends_of_reads[j_st].begin();

    //     uint64_t cur_read_start_coord = *it_start;
    //     std::vector<std::pair<Row, uint64_t>> curr_read_trace_coords;

    //     while (it_start != coords_st.end() && it_end != ends_of_reads[j_st].end()) {
    //         curr_read_trace_coords.clear();
    //         cur_read_start_coord = *it_start;

    //         for (uint64_t cur_read_coord_ind = cur_read_start_coord; cur_read_coord_ind <= *it_end; ++cur_read_coord_ind) {
    //             curr_read_trace_coords.push_back(std::make_pair(reconstucted_paths[j_st][cur_read_coord_ind], cur_read_coord_ind));
    //         }

    //         result_coords.push_back(std::make_tuple(curr_read_trace_coords, j_st, cur_read_start_coord));

    //         it_start++;
    //         it_end++;
    //     }

    //     result_coords.push_back(std::make_tuple(curr_read_trace_coords, j_st, cur_read_start_coord));
    // }

    for (auto & [j, trace_with_jumps] : traces_unlabeled) {
        uint64_t curr_read_start_coord = trace_with_jumps.front().second;
        uint64_t curr_read_curr_coord = curr_read_start_coord;

        auto curr_edge_node = graph::AnnotatedSequenceGraph::anno_to_graph_index(trace_with_jumps.front().first);

        std::vector<std::pair<Row, uint64_t>> curr_read_trace_coords;
        curr_read_trace_coords.push_back(trace_with_jumps.front());

        for (size_t cur_pos = 1; cur_pos < trace_with_jumps.size(); ++cur_pos) {

            auto curr_next_edge_node = graph::AnnotatedSequenceGraph::anno_to_graph_index(trace_with_jumps[cur_pos].first);
            
            // check if an edge exists between the current and next node
            bool edge_exists = false;
            graph_->adjacent_outgoing_nodes(curr_edge_node, [&](auto adj_outg_node) {
                if (adj_outg_node == curr_next_edge_node) {
                    edge_exists = true;
                    return;
                }
            });

            // if edge exists, then we are still in the current read
            if ((trace_with_jumps[cur_pos].second == (curr_read_curr_coord + SHIFT)) && edge_exists) {
                curr_read_trace_coords.push_back(trace_with_jumps[cur_pos]);
                curr_read_curr_coord += SHIFT;
            } else {
                // otherwise, the read has ended and the new one started

                // add the current path to the result
                result_coords.push_back(std::make_tuple(curr_read_trace_coords, j, curr_read_start_coord));

                // and update the start coordinate of the read
                curr_read_start_coord = trace_with_jumps[cur_pos].second;
                curr_read_curr_coord = curr_read_start_coord;

                // start collecting nodes into a new path
                curr_read_trace_coords.clear();
                curr_read_trace_coords.push_back(trace_with_jumps[cur_pos]);
            }

            curr_edge_node = curr_next_edge_node;
        }

        // add the last path to the result
        result_coords.push_back(std::make_tuple(curr_read_trace_coords, j, curr_read_start_coord));
    }

    // save only reads that contain the input node
    // std::vector<std::tuple<std::vector<std::pair<Row, uint64_t>>, Column, uint64_t>> final_result;
    for (auto & [read_trace, read_label, read_start_coord] : result_coords) { 
        for (size_t cur_pos = 0; cur_pos < read_trace.size(); ++cur_pos) {
            if (read_trace[cur_pos].first == i && !added_reads_before_traversal_start_coordinates[read_label].count(read_trace.front().second)) {
                final_result.push_back(std::make_tuple(read_trace, read_label, cur_pos));
                reconstructed_reads[read_label].push_back(read_trace);
                break;
            }
        }
    }

    // discard the coordinates from the result as we don't longer need them
    // (we need only a vector of Rows representing a path)
    std::vector<std::tuple<std::vector<Row>, Column, uint64_t>> final_final_result;
    for (auto & [read_trace, read_label, cur_pos] : final_result) {
        std::vector<Row> cur_trace_rows;
        for (auto & el : read_trace) {
            cur_trace_rows.push_back(el.first);
        }
        final_final_result.push_back(std::make_tuple(cur_trace_rows, read_label, cur_pos));
    }

    return final_final_result;
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
