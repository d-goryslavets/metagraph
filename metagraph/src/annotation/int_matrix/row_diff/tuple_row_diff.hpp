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

    // check graph traversal in batches
    static const uint64_t TRAVERSAL_BATCH_SIZE = 1'000;

    TupleRowDiff() {}

    TupleRowDiff(const graph::DBGSuccinct *graph, BaseMatrix&& diff)
        : diffs_(std::move(diff)) { graph_ = graph; }

    bool get(Row i, Column j) const override;
    std::vector<Row> get_column(Column j) const override;
    RowTuples get_row_tuples(Row i) const override;
    std::vector<RowTuples> get_row_tuples(const std::vector<Row> &rows) const override;
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
    std::vector<std::tuple<std::vector<Row>, Column, uint64_t>> get_traces_with_row_auto_labels(Row i) const;
    std::vector<std::tuple<std::vector<Row>, Column, uint64_t>> get_traces_with_row(Row i) const;

    // auto_labels means that the labels will be derived based on coordinates
    // this is needed for the graphs where reads are not marked with different labels
    std::pair<std::unordered_map<Column, std::vector<Row>>, std::vector<std::vector<std::pair<Column, uint64_t>>>> get_traces_with_row(std::vector<Row> i) const;
    std::vector<std::pair<Column, uint64_t>> get_traces_with_row(Row i, std::unordered_map<Row, RowTuples> &rows_annotations, 
    std::vector<Row> &rd_ids, VectorMap<Row, size_t> &node_to_rd, std::vector<RowTuples> &rd_rows, std::unordered_map<Column, std::vector<Row>> &reconstructed_reads) const;

    std::vector<std::vector<std::tuple<std::vector<Row>, Column, uint64_t>>> get_traces_with_row_auto_labels(std::vector<Row> i) const;
    std::vector<std::vector<std::tuple<std::vector<Row>, Column, uint64_t>>> get_traces_with_row_auto_labels(std::vector<Row> i, std::vector<std::unordered_set<uint64_t>> manifest_labels) const;

    std::vector<std::tuple<std::vector<Row>, Column, uint64_t>> get_traces_with_row_auto_labels(Row i, std::unordered_set<uint64_t> labels_of_interest,
    std::unordered_map<Column, std::vector<std::tuple<std::vector<Row>, uint64_t, uint64_t>>> &reconstructed_reads) const;

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
    static void add_diff_labeled(const RowTuples &diff, RowTuples *row, std::unordered_set<Column> labels_of_interest);

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
std::vector<MultiIntMatrix::RowTuples>
TupleRowDiff<BaseMatrix>::get_row_tuples_labeled(const std::vector<Row> &row_ids, std::unordered_set<Column> labels_of_interest) const {
    assert(graph_ && "graph must be loaded");
    assert(anchor_.size() == diffs_.num_rows() && "anchors must be loaded");
    assert(!fork_succ_.size() || fork_succ_.size() == graph_->get_boss().get_last().size());

    // get row-diff paths
    auto [rd_ids, rd_paths_trunc] = get_rd_ids(row_ids);

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

    // keep the path and coordinates of its start and end
    std::unordered_map<Column, std::vector<std::tuple<std::vector<Row>, uint64_t, uint64_t>>> reconstructed_reads;
    std::vector<std::vector<std::tuple<std::vector<Row>, Column, uint64_t>>> result;

    auto initial_labels = get_labels_of_rows(i);

    for (size_t j = 0; j < i.size(); ++j) {
        auto curr_res = get_traces_with_row_auto_labels(i[j], initial_labels[j], reconstructed_reads);
        result.push_back(curr_res);
    }

    return result;
}

template <class BaseMatrix>
std::vector<std::vector<std::tuple<std::vector<MultiIntMatrix::Row>, MultiIntMatrix::Column, uint64_t>>> TupleRowDiff<BaseMatrix>
::get_traces_with_row_auto_labels(std::vector<Row> i, std::vector<std::unordered_set<uint64_t>> manifest_labels) const {
    assert(graph_ && "graph must be loaded");
    assert(anchor_.size() == diffs_.num_rows() && "anchors must be loaded");
    assert(!fork_succ_.size() || fork_succ_.size() == graph_->get_boss().get_last().size());

    // keep the path and coordinates of its start and end
    std::unordered_map<Column, std::vector<std::tuple<std::vector<Row>, uint64_t, uint64_t>>> reconstructed_reads;
    std::vector<std::vector<std::tuple<std::vector<Row>, Column, uint64_t>>> result;

    for (size_t j = 0; j < i.size(); ++j) {
        auto curr_res = get_traces_with_row_auto_labels(i[j], manifest_labels[j], reconstructed_reads);
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
::get_traces_with_row_auto_labels(Row i, std::unordered_set<uint64_t> labels_of_interest, std::unordered_map<Column, std::vector<std::tuple<std::vector<Row>, uint64_t, uint64_t>>> &reconstructed_reads) const {
    assert(graph_ && "graph must be loaded");
    assert(anchor_.size() == diffs_.num_rows() && "anchors must be loaded");
    assert(!fork_succ_.size() || fork_succ_.size() == graph_->get_boss().get_last().size());

    // result reads
    std::vector<std::tuple<std::vector<Row>, Column, uint64_t>> result;

    // starts of reads that were reconstructed for the previous k-mer in the query
    std::unordered_map<Column, std::unordered_set<uint64_t>> start_coords_of_already_reconstructed_reads;

    // map that keeps all visited rows and their coordinates in reads
    std::unordered_map<Column, std::map<uint64_t, Row>> reconstructed_paths;

    mtg::common::logger->trace("Getting annotations of the initial node");

    RowTuples input_row_tuples = get_row_tuples(i);
    // inspect coordinates of the start row
    for (auto &[j, tuple] : input_row_tuples) {
        if (!labels_of_interest.count(j))
            continue;

        for (uint64_t &c : tuple) {
            bool already_in_the_read = false;
            for (auto & [already_reconstructed_read, first_coord_of_read, last_coord_of_read] : reconstructed_reads[j]) {

                // if current coordinates falls in the already reconstructed read coordinate range
                if ((c >= first_coord_of_read && c <= last_coord_of_read)) {
                    already_in_the_read = true;

                    result.push_back(std::make_tuple(already_reconstructed_read, j, c - first_coord_of_read));
                    start_coords_of_already_reconstructed_reads[j].insert(first_coord_of_read);

                    // if we found the read, no need to iterate more
                    break;
                }
            }

            // if the read was previously reconstructed we will not reconstruct it now
            if (already_in_the_read)
                continue;

            // is used to collect all visited nodes and trace the result paths
            reconstructed_paths[j][c] = i;
        }
    }

    input_row_tuples = RowTuples();

    // case when all the reads that contain current row are already reconstructed
    if (reconstructed_paths.empty()) {
        return result;
    }

    std::unordered_map<Column, std::set<uint64_t>> ends_of_reads;
    std::unordered_map<Column, std::set<uint64_t>> starts_of_reads;

    std::unordered_map<Column, std::set<uint64_t>> ends_of_reads_final;
    std::unordered_map<Column, std::set<uint64_t>> starts_of_reads_final;

    mtg::common::logger->trace("Getting initial starts and ends of the reads");

    for (auto & [j, coords] : reconstructed_paths) {
        // find initial ends of the reads from the input row annotations
        // (they may not be actual ends though)
        // these will be updated during traversal
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

    // queue for the BFS traversal
    // std::queue<Row> to_visit;
    // to_visit.push(i);
    std::deque<Row> to_visit;
    to_visit.push_back(i);

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

        // decompress the annotations for the current batch of rows
        std::vector<RowTuples> batch_annotations = get_row_tuples_labeled(batch_of_rows, labels_of_interest);

        // collect all new discovered coordinates
        for (size_t batch_i = 0; batch_i < batch_of_rows.size(); ++batch_i) {
            for (auto & [j_next, tuple_next] : batch_annotations[batch_i]) {
                if (!labels_of_interest.count(j_next)) {
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

            std::vector<RowTuples> batch_part_annotations = get_row_tuples_labeled(batch_part, labels_of_interest);
            // std::vector<RowTuples> batch_part_annotations = get_row_tuples(batch_part);

            for (size_t rowt_to_check = 0; rowt_to_check < batch_part.size(); ++rowt_to_check) {
                for (auto & [j_to_check, tuple_to_check] : batch_part_annotations[rowt_to_check]) {
                    if (!labels_of_interest.count(j_to_check)) {
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

        // mtg::common::logger->trace("Reduced ends of reads {}; current found true ends {} ; current potential ends {}", ends_of_reads_size_before - ends_of_reads_size_after, found_true_ends_at_this_step, ends_of_reads_size_after);

        // clean up reconstructed paths:
        std::vector<Column> labels_to_clean_up;
        for (auto & [j_clean_up, coords_and_rows_clean_up] : reconstructed_paths) {
            if ((ends_of_reads[j_clean_up].empty() && ends_of_reads_final[j_clean_up].empty()) || starts_of_reads[j_clean_up].empty()) {
                labels_to_clean_up.push_back(j_clean_up);
                continue;
            }

            uint64_t max_end_of_read;
            uint64_t min_start_of_read = *(starts_of_reads[j_clean_up].begin());

            if (ends_of_reads_final[j_clean_up].empty()) {
                max_end_of_read = *(ends_of_reads[j_clean_up].rbegin());
            } else if (ends_of_reads[j_clean_up].empty()) {
                max_end_of_read = *(ends_of_reads_final[j_clean_up].rbegin());
            } else {
                max_end_of_read = std::max(*(ends_of_reads[j_clean_up].rbegin()), *(ends_of_reads_final[j_clean_up].rbegin()));
            }

            for (auto it_clean_up = coords_and_rows_clean_up.rbegin(); it_clean_up != coords_and_rows_clean_up.rend(); ) {
                if (it_clean_up->first > max_end_of_read) {
                    it_clean_up = decltype(it_clean_up)(coords_and_rows_clean_up.erase(std::next(it_clean_up).base()));
                } else {
                    break;
                }
            }
            for (auto it_clean_up = coords_and_rows_clean_up.begin(); it_clean_up != coords_and_rows_clean_up.end(); ) {
                if (it_clean_up->first < min_start_of_read) {
                    it_clean_up = coords_and_rows_clean_up.erase(it_clean_up);
                } else {
                    break;
                }
            }

        }

        for (auto & j_clean_up : labels_to_clean_up) {
            reconstructed_paths.erase(j_clean_up);
        }

        // clean up all discovered coordinates
        labels_to_clean_up = std::vector<Column>();
        for (auto & [j_to_clean_discovered_coords, discovered_coords] : all_discovered_coordinates) {
            if ((ends_of_reads[j_to_clean_discovered_coords].empty() && ends_of_reads_final[j_to_clean_discovered_coords].empty()) || starts_of_reads[j_to_clean_discovered_coords].empty()) {
                labels_to_clean_up.push_back(j_to_clean_discovered_coords);
                continue;
            }

            uint64_t max_end_of_read;
            uint64_t min_start_of_read = *(starts_of_reads[j_to_clean_discovered_coords].begin());

            if (ends_of_reads_final[j_to_clean_discovered_coords].empty()) {
                max_end_of_read = *(ends_of_reads[j_to_clean_discovered_coords].rbegin());
            } else if (ends_of_reads[j_to_clean_discovered_coords].empty()) {
                max_end_of_read = *(ends_of_reads_final[j_to_clean_discovered_coords].rbegin());
            } else {
                max_end_of_read = std::max(*(ends_of_reads[j_to_clean_discovered_coords].rbegin()), *(ends_of_reads_final[j_to_clean_discovered_coords].rbegin()));
            }

            for (auto it_clean_up = discovered_coords.rbegin(); it_clean_up != discovered_coords.rend(); ) {
                if (*it_clean_up > max_end_of_read) {
                    it_clean_up = decltype(it_clean_up)(discovered_coords.erase(std::next(it_clean_up).base()));
                } else {
                    break;
                }
            }
            for (auto it_clean_up = discovered_coords.begin(); it_clean_up != discovered_coords.end(); ) {
                if (*it_clean_up < min_start_of_read) {
                    it_clean_up = discovered_coords.erase(it_clean_up);
                } else {
                    break;
                }
            }
        }

        for (auto & j_clean_up : labels_to_clean_up) {
            all_discovered_coordinates.erase(j_clean_up);
        }


        if (to_visit.empty())
            break;
    }

    ends_of_reads = ends_of_reads_final;
    ends_of_reads_final = std::unordered_map<Column, std::set<uint64_t>>();

    mtg::common::logger->trace("Traversed batches forwards count {}", batches_count);

    // similarly traverse the graph backwards
    // to_visit = std::queue<Row>();
    // to_visit.push(i);
    to_visit = std::deque<Row>();
    to_visit.push_back(i);

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

        std::vector<RowTuples> batch_annotations = get_row_tuples_labeled(batch_of_rows, labels_of_interest);
        // std::vector<RowTuples> batch_annotations = get_row_tuples(batch_of_rows);

        // collect all new discovered coordinates
        for (size_t batch_i = 0; batch_i < batch_of_rows.size(); ++batch_i) {
            for (auto & [j_prev, tuple_prev] : batch_annotations[batch_i]) {
                if (!labels_of_interest.count(j_prev)) {
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

            std::vector<RowTuples> batch_part_annotations = get_row_tuples_labeled(batch_part, labels_of_interest);
            // std::vector<RowTuples> batch_part_annotations = get_row_tuples(batch_part);

            for (size_t rowt_to_check = 0; rowt_to_check < batch_part.size(); ++rowt_to_check) {
                for (auto & [j_to_check, tuple_to_check] : batch_part_annotations[rowt_to_check]) {
                    if (!labels_of_interest.count(j_to_check)) {
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

        // clean up reconstructed paths:
        std::vector<Column> labels_to_clean_up;
        for (auto & [j_clean_up, coords_and_rows_clean_up] : reconstructed_paths) {
            if ((starts_of_reads[j_clean_up].empty() && starts_of_reads_final[j_clean_up].empty()) || ends_of_reads[j_clean_up].empty()) {
                labels_to_clean_up.push_back(j_clean_up);
                continue;
            }

            uint64_t min_start_of_read;
            uint64_t max_end_of_read = *(ends_of_reads[j_clean_up].rbegin());

            if (starts_of_reads_final[j_clean_up].empty()) {
                min_start_of_read = *(starts_of_reads[j_clean_up].begin());
            } else if (starts_of_reads[j_clean_up].empty()) {
                min_start_of_read = *(starts_of_reads_final[j_clean_up].begin());
            } else {
                min_start_of_read = std::min(*(starts_of_reads[j_clean_up].begin()), *(starts_of_reads_final[j_clean_up].begin()));
            }
            for (auto it_clean_up = coords_and_rows_clean_up.rbegin(); it_clean_up != coords_and_rows_clean_up.rend(); ) {
                if (it_clean_up->first > max_end_of_read) {
                    it_clean_up = decltype(it_clean_up)(coords_and_rows_clean_up.erase(std::next(it_clean_up).base()));
                } else {
                    break;
                }
            }
            for (auto it_clean_up = coords_and_rows_clean_up.begin(); it_clean_up != coords_and_rows_clean_up.end(); ) {
                if (it_clean_up->first < min_start_of_read) {
                    it_clean_up = coords_and_rows_clean_up.erase(it_clean_up);
                } else {
                    break;
                }
            }
        }

        for (auto & j_clean_up : labels_to_clean_up) {
            reconstructed_paths.erase(j_clean_up);
        }

        // clean up all discovered coordinates
        labels_to_clean_up = std::vector<Column>();
        for (auto & [j_to_clean_discovered_coords, discovered_coords] : all_discovered_coordinates) {
            if (ends_of_reads[j_to_clean_discovered_coords].empty() || (starts_of_reads[j_to_clean_discovered_coords].empty() && starts_of_reads_final[j_to_clean_discovered_coords].empty())) {
                labels_to_clean_up.push_back(j_to_clean_discovered_coords);
                continue;
            }

            uint64_t min_start_of_read;
            uint64_t max_end_of_read = *(ends_of_reads[j_to_clean_discovered_coords].rbegin());

            if (starts_of_reads_final[j_to_clean_discovered_coords].empty()) {
                min_start_of_read = *(starts_of_reads[j_to_clean_discovered_coords].begin());
            } else if (starts_of_reads[j_to_clean_discovered_coords].empty()) {
                min_start_of_read = *(starts_of_reads_final[j_to_clean_discovered_coords].begin());
            } else {
                min_start_of_read = std::min(*(starts_of_reads[j_to_clean_discovered_coords].begin()), *(starts_of_reads_final[j_to_clean_discovered_coords].begin()));
            }

            for (auto it_clean_up = discovered_coords.rbegin(); it_clean_up != discovered_coords.rend(); ) {
                if (*it_clean_up > max_end_of_read) {
                    it_clean_up = decltype(it_clean_up)(discovered_coords.erase(std::next(it_clean_up).base()));
                } else {
                    break;
                }
            }
            for (auto it_clean_up = discovered_coords.begin(); it_clean_up != discovered_coords.end(); ) {
                if (*it_clean_up < min_start_of_read) {
                    it_clean_up = discovered_coords.erase(it_clean_up);
                } else {
                    break;
                }
            }
        }

        for (auto & j_clean_up : labels_to_clean_up) {
            all_discovered_coordinates.erase(j_clean_up);
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

            bool contains_input_row = false;
            uint64_t input_row_coord_in_read = 0;

            for (uint64_t cur_read_coord_ind = cur_read_start_coord; cur_read_coord_ind <= *it_end; ++cur_read_coord_ind) {
                if (reconstructed_paths[j_st][cur_read_coord_ind] == i) {
                    contains_input_row = true;
                    input_row_coord_in_read = cur_read_coord_ind;
                }

                curr_read_trace_no_coords.push_back(reconstructed_paths[j_st][cur_read_coord_ind]);
            }

            // if the read contains input row and was not added to the result before
            if (contains_input_row  && !start_coords_of_already_reconstructed_reads[j_st].count(cur_read_start_coord)) {
                result.push_back(std::make_tuple(curr_read_trace_no_coords, j_st, input_row_coord_in_read - cur_read_start_coord));
                reconstructed_reads[j_st].push_back(std::make_tuple(curr_read_trace_no_coords, cur_read_start_coord, *it_end));
            }

            it_start++;
            it_end++;
        }
    }

    return result;
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
