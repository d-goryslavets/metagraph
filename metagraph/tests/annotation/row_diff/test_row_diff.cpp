#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "annotation/binary_matrix/column_sparse/column_major.hpp"
#include "annotation/binary_matrix/row_diff/row_diff.hpp"
#include "common/vectors/bit_vector_sd.hpp"
#include "common/utils/file_utils.hpp"
#include "graph/representation/succinct/dbg_succinct.hpp"

namespace {
using namespace mtg;
using namespace testing;
using ::testing::_;
using mtg::annot::matrix::RowDiff;
using mtg::annot::matrix::ColumnMajor;

typedef RowDiff<ColumnMajor>::anchor_bv_type anchor_bv_type;

TEST(RowDiff, Empty) {
    RowDiff<ColumnMajor> rowdiff;
    EXPECT_EQ(0, rowdiff.diffs().num_columns());
    EXPECT_EQ(0, rowdiff.diffs().num_relations());
    EXPECT_EQ(0, rowdiff.diffs().num_rows());
    EXPECT_EQ(0, rowdiff.anchor().size());
    EXPECT_EQ(0, rowdiff.num_relations());
    EXPECT_EQ(0, rowdiff.num_columns());
    EXPECT_EQ(0, rowdiff.num_rows());
    EXPECT_EQ(nullptr, rowdiff.graph());
}

TEST(RowDiff, Serialize) {
    sdsl::bit_vector bterminal = { 0, 0, 0, 1 };
    anchor_bv_type terminal(bterminal);
    utils::TempFile fterm_temp;
    std::ofstream fterm(fterm_temp.name(), ios::binary);
    terminal.serialize(fterm);
    fterm.flush();

    std::vector<std::unique_ptr<bit_vector>> cols(2);
    cols[0] = std::make_unique<bit_vector_sd>(std::initializer_list<bool>({0,1,0,1}));
    cols[1] = std::make_unique<bit_vector_sd>(std::initializer_list<bool>({1,0,1,0}));

    utils::TempFile fmat;
    ColumnMajor mat(std::move(cols));

    RowDiff<ColumnMajor> annot(nullptr, std::move(mat));

    utils::TempFile tempfile;
    std::ofstream &out = tempfile.ofstream();
    annot.serialize(out);
    out.flush();

    RowDiff<ColumnMajor> loaded;
    ASSERT_TRUE(loaded.load(tempfile.ifstream()));
    loaded.load_anchor(fterm_temp.name());

    ASSERT_EQ(loaded.num_columns(), 2);
    ASSERT_EQ(loaded.num_rows(), 4);
    ASSERT_EQ(loaded.diffs().num_relations(), 4);
    ASSERT_EQ(loaded.anchor().size(), 4);

    for (uint32_t i = 0; i < 4; ++i) {
        ASSERT_THAT(loaded.diffs().get_column(0), ElementsAre(1,3));
        ASSERT_THAT(loaded.diffs().get_column(1), ElementsAre(0,2));
        ASSERT_EQ(loaded.anchor()[i], bterminal[i]);
    }
}

TEST(RowDiff, GetRows) {
    // build graph
    graph::DBGSuccinct graph(4);
    graph.add_sequence("ACTAGCTAGCTAGCTAGCTAGC");
    graph.add_sequence("ACTCTAG");

    // build annotation
    sdsl::bit_vector bterminal = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0 };
    anchor_bv_type terminal(bterminal);
    utils::TempFile fterm_temp;
    std::ofstream fterm(fterm_temp.name(), ios::binary);
    terminal.serialize(fterm);
    fterm.flush();

    std::vector<std::unique_ptr<bit_vector>> cols(2);
    cols[0] = std::make_unique<bit_vector_sd>(
            std::initializer_list<bool>({ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0  }));
    cols[1] = std::make_unique<bit_vector_sd>(
            std::initializer_list<bool>({ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1 }));

    ColumnMajor mat(std::move(cols));

    RowDiff<ColumnMajor> annot(&graph, std::move(mat));
    annot.load_anchor(fterm_temp.name());

    auto rows = annot.get_rows({ 3, 3, 3, 3, 5, 5, 6, 7, 8, 9, 10, 11 });
    EXPECT_EQ("CTAG", graph.get_node_sequence(4));
    ASSERT_THAT(rows[3], ElementsAre(0, 1));

    EXPECT_EQ("AGCT", graph.get_node_sequence(6));
    ASSERT_THAT(rows[5], ElementsAre(1));

    EXPECT_EQ("CTCT", graph.get_node_sequence(7));
    ASSERT_THAT(rows[6], ElementsAre(0));

    EXPECT_EQ("TAGC", graph.get_node_sequence(8));
    ASSERT_THAT(rows[7], ElementsAre(1));

    EXPECT_EQ("ACTA", graph.get_node_sequence(9));
    ASSERT_THAT(rows[8], ElementsAre(1));

    EXPECT_EQ("ACTC", graph.get_node_sequence(10));
    ASSERT_THAT(rows[9], ElementsAre(0));

    EXPECT_EQ("GCTA", graph.get_node_sequence(11));
    ASSERT_THAT(rows[10], ElementsAre(1));

    EXPECT_EQ("TCTA", graph.get_node_sequence(12));
    ASSERT_THAT(rows[11], ElementsAre(0));
}

/**
 * Tests annotations on the graph in
 * https://docs.google.com/document/d/1e0MFgZRJfmDUSvmDPuC_lvnnWA0VKm5hPdzM8mdrHMM/edit#bookmark=id.ciri4266pkc4
 */
TEST(RowDiff, GetAnnotation) {
    // build graph
    graph::DBGSuccinct graph(4);
    graph.add_sequence("ACTAGCTAGCTAGCTAGCTAGC");
    graph.add_sequence("ACTCTAG");

    // build annotation
    sdsl::bit_vector bterminal = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0 };
    anchor_bv_type terminal(bterminal);
    utils::TempFile fterm_temp;
    std::ofstream fterm(fterm_temp.name(), ios::binary);
    terminal.serialize(fterm);
    fterm.flush();

    std::vector<std::unique_ptr<bit_vector>> cols(2);
    cols[0] = std::make_unique<bit_vector_sd>(
            std::initializer_list<bool>({ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0  }));
    cols[1] = std::make_unique<bit_vector_sd>(
            std::initializer_list<bool>({ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1 }));

    ColumnMajor mat(std::move(cols));

    RowDiff<ColumnMajor> annot(&graph, std::move(mat));
    annot.load_anchor(fterm_temp.name());

    EXPECT_EQ("CTAG", graph.get_node_sequence(4));
    ASSERT_THAT(annot.get_rows({3})[0], ElementsAre(0, 1));

    EXPECT_EQ("AGCT", graph.get_node_sequence(6));
    ASSERT_THAT(annot.get_rows({5})[0], ElementsAre(1));

    EXPECT_EQ("CTCT", graph.get_node_sequence(7));
    ASSERT_THAT(annot.get_rows({6})[0], ElementsAre(0));

    EXPECT_EQ("TAGC", graph.get_node_sequence(8));
    ASSERT_THAT(annot.get_rows({7})[0], ElementsAre(1));

    EXPECT_EQ("ACTA", graph.get_node_sequence(9));
    ASSERT_THAT(annot.get_rows({8})[0], ElementsAre(1));

    EXPECT_EQ("ACTC", graph.get_node_sequence(10));
    ASSERT_THAT(annot.get_rows({9})[0], ElementsAre(0));

    EXPECT_EQ("GCTA", graph.get_node_sequence(11));
    ASSERT_THAT(annot.get_rows({10})[0], ElementsAre(1));

    EXPECT_EQ("TCTA", graph.get_node_sequence(12));
    ASSERT_THAT(annot.get_rows({11})[0], ElementsAre(0));
}

/**
 * Tests annotations on the graph in
 * https://docs.google.com/document/d/1e0MFgZRJfmDUSvmDPuC_lvnnWA0VKm5hPdzM8mdrHMM/edit#bookmark=id.ciri4266pkc4
 * after having removed dummy nodes.
 */
TEST(RowDiff, GetAnnotationMasked) {
    // build graph
    graph::DBGSuccinct graph(4);
    graph.add_sequence("ACTAGCTAGCTAGCTAGCTAGC");
    graph.add_sequence("ACTCTAG");
    graph.mask_dummy_kmers(1, false);

    // build annotation
    sdsl::bit_vector bterminal = { 0, 0, 0, 0, 1, 0, 1, 0 };
    anchor_bv_type terminal(bterminal);
    utils::TempFile fterm_temp;
    std::ofstream fterm(fterm_temp.name(), ios::binary);
    terminal.serialize(fterm);
    fterm.flush();

    std::vector<std::unique_ptr<bit_vector>> cols(2);
    cols[0] = std::make_unique<bit_vector_sd>(
            std::initializer_list<bool>({ 1, 0, 0, 0, 0, 0, 0, 0 }));
    cols[1] = std::make_unique<bit_vector_sd>(
            std::initializer_list<bool>({ 0, 0, 0, 0, 1, 0, 1, 1 }));

    ColumnMajor mat(std::move(cols));

    RowDiff<ColumnMajor> annot(&graph, std::move(mat));
    annot.load_anchor(fterm_temp.name());

    EXPECT_EQ("CTAG", graph.get_node_sequence(1));
    ASSERT_THAT(annot.get_rows({0})[0], ElementsAre(0, 1));

    EXPECT_EQ("AGCT", graph.get_node_sequence(2));
    ASSERT_THAT(annot.get_rows({1})[0], ElementsAre(1));

    EXPECT_EQ("CTCT", graph.get_node_sequence(3));
    ASSERT_THAT(annot.get_rows({2})[0], ElementsAre(0));

    EXPECT_EQ("TAGC", graph.get_node_sequence(4));
    ASSERT_THAT(annot.get_rows({3})[0], ElementsAre(1));

    EXPECT_EQ("ACTA", graph.get_node_sequence(5));
    ASSERT_THAT(annot.get_rows({4})[0], ElementsAre(1));

    EXPECT_EQ("ACTC", graph.get_node_sequence(6));
    ASSERT_THAT(annot.get_rows({5})[0], ElementsAre(0));

    EXPECT_EQ("GCTA", graph.get_node_sequence(7));
    ASSERT_THAT(annot.get_rows({6})[0], ElementsAre(1));

    EXPECT_EQ("TCTA", graph.get_node_sequence(8));
    ASSERT_THAT(annot.get_rows({7})[0], ElementsAre(0));
}

/**
 *  Tests that annotations for the graph in
 *  https://docs.google.com/document/d/1siWApHWBDtiYCsetb6vHPuT7WwBkVFJp5fgti_AJK-s/edit
 *  are correctly retrieved.
 */
TEST(RowDiff, GetAnnotationBifurcation) {
    // build graph
    graph::DBGSuccinct graph(4);
    graph.add_sequence("TACTAGCTAGCTAGCTAGCTAGC");
    graph.add_sequence("ACTCTAGCTAT");

    // build annotation
    sdsl::bit_vector bterminal = { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0 };
    anchor_bv_type terminal(bterminal);
    utils::TempFile fterm_temp;
    std::ofstream fterm(fterm_temp.name(), ios::binary);
    terminal.serialize(fterm);
    fterm.flush();

    std::vector<std::unique_ptr<bit_vector>> cols(2);
    cols[0] = std::make_unique<bit_vector_sd>(
            std::initializer_list<bool>({0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0 }));
    cols[1] = std::make_unique<bit_vector_sd>(
            std::initializer_list<bool>({0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0 }));

    ColumnMajor mat(std::move(cols));

    RowDiff<ColumnMajor> annot(&graph, std::move(mat));
    annot.load_anchor(fterm_temp.name());

    EXPECT_EQ("CTAG", graph.get_node_sequence(4));
    ASSERT_THAT(annot.get_rows({3})[0], ElementsAre(0, 1));

    EXPECT_EQ("CTAT", graph.get_node_sequence(5));
    ASSERT_THAT(annot.get_rows({4})[0], ElementsAre(1));

    EXPECT_EQ("TACT", graph.get_node_sequence(6));
    ASSERT_THAT(annot.get_rows({5})[0], ElementsAre(0));

    EXPECT_EQ("AGCT", graph.get_node_sequence(7));
    ASSERT_THAT(annot.get_rows({6})[0], ElementsAre(0, 1));

    EXPECT_EQ("CTCT", graph.get_node_sequence(8));
    ASSERT_THAT(annot.get_rows({7})[0], ElementsAre(1));

    EXPECT_EQ("TAGC", graph.get_node_sequence(9));
    ASSERT_THAT(annot.get_rows({8})[0], ElementsAre(0, 1));

    EXPECT_EQ("ACTA", graph.get_node_sequence(12));
    ASSERT_THAT(annot.get_rows({11})[0], ElementsAre(0));

    EXPECT_EQ("ACTC", graph.get_node_sequence(13));
    ASSERT_THAT(annot.get_rows({12})[0], ElementsAre(1));

    EXPECT_EQ("GCTA", graph.get_node_sequence(14));
    ASSERT_THAT(annot.get_rows({13})[0], ElementsAre(0, 1));

    EXPECT_EQ("TCTA", graph.get_node_sequence(15));
    ASSERT_THAT(annot.get_rows({14})[0], ElementsAre(1));
}

TEST(RowDiff, GetAnnotationBifurcationMasked) {
    // build graph
    graph::DBGSuccinct graph(4);
    graph.add_sequence("TACTAGCTAGCTAGCTAGCTAGC");
    graph.add_sequence("ACTCTAGCTAT");
    graph.mask_dummy_kmers(1, false);

    // build annotation
    sdsl::bit_vector bterminal = { 0, 1, 0, 0, 0, 0, 1, 0, 1, 0 };
    anchor_bv_type terminal(bterminal);
    utils::TempFile fterm_temp;
    std::ofstream fterm(fterm_temp.name(), ios::binary);
    terminal.serialize(fterm);
    fterm.flush();

    Vector<uint64_t> diffs = { 1, 0, 1, 0, 0, 1 };
    sdsl::bit_vector boundary = { 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1 };


    std::vector<std::unique_ptr<bit_vector>> cols(2);
    cols[0] = std::make_unique<bit_vector_sd>(
            std::initializer_list<bool>({0, 0, 1, 0, 0, 0, 1, 0, 1, 0 }));
    cols[1] = std::make_unique<bit_vector_sd>(
            std::initializer_list<bool>({0, 1, 1, 0, 0, 0, 0, 0, 1, 0 }));

    ColumnMajor mat(std::move(cols));

    RowDiff<ColumnMajor> annot(&graph, std::move(mat));
    annot.load_anchor(fterm_temp.name());

    EXPECT_EQ("CTAG", graph.get_node_sequence(1));
    ASSERT_THAT(annot.get_rows({0})[0], ElementsAre(0, 1));

    EXPECT_EQ("CTAT", graph.get_node_sequence(2));
    ASSERT_THAT(annot.get_rows({1})[0], ElementsAre(1));

    EXPECT_EQ("TACT", graph.get_node_sequence(3));
    ASSERT_THAT(annot.get_rows({2})[0], ElementsAre(0));

    EXPECT_EQ("AGCT", graph.get_node_sequence(4));
    ASSERT_THAT(annot.get_rows({3})[0], ElementsAre(0, 1));

    EXPECT_EQ("CTCT", graph.get_node_sequence(5));
    ASSERT_THAT(annot.get_rows({4})[0], ElementsAre(1));

    EXPECT_EQ("TAGC", graph.get_node_sequence(6));
    ASSERT_THAT(annot.get_rows({5})[0], ElementsAre(0, 1));

    EXPECT_EQ("ACTA", graph.get_node_sequence(7));
    ASSERT_THAT(annot.get_rows({6})[0], ElementsAre(0));

    EXPECT_EQ("ACTC", graph.get_node_sequence(8));
    ASSERT_THAT(annot.get_rows({7})[0], ElementsAre(1));

    EXPECT_EQ("GCTA", graph.get_node_sequence(9));
    ASSERT_THAT(annot.get_rows({8})[0], ElementsAre(0, 1));

    EXPECT_EQ("TCTA", graph.get_node_sequence(10));
    ASSERT_THAT(annot.get_rows({9})[0], ElementsAre(1));
}

} // namespace
