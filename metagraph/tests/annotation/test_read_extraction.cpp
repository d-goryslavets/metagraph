#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "../graph/all/test_dbg_helpers.hpp"
#include "test_annotated_dbg_helpers.hpp"

#include "annotation/representation/annotation_matrix/static_annotators_def.hpp"
#include "graph/representation/succinct/dbg_succinct.hpp"

namespace {

using namespace mtg;
using namespace mtg::graph;
using namespace mtg::test;

using Extracted = std::vector<std::tuple<std::string, std::string, uint64_t, uint64_t>>;

std::unique_ptr<AnnotatedDBG> make_coord_graph(uint64_t k,
                                               const std::vector<std::string> &sequences,
                                               const std::vector<std::string> &labels,
                                               const std::vector<uint64_t> &coord_starts = {}) {
    return build_anno_graph<DBGSuccinct, annot::RowDiffColumnAnnotator>(
        k, sequences, labels, DeBruijnGraph::BASIC, true, coord_starts);
}

Extracted sorted(Extracted result) {
    std::sort(result.begin(), result.end());
    return result;
}

TEST(ReadExtraction, LinearSharedKmer) {
    const uint64_t k = 4;
    const std::vector<std::string> sequences { "CAATGCTGCTAATGCTT", "CAATGCTT" };
    const std::vector<std::string> labels { "A", "B" };
    auto anno_graph = make_coord_graph(k, sequences, labels);

    auto obs = sorted(anno_graph->get_overlapping_reads("AATG", 50, 2));
    Extracted ref {
        { "CAATGCTGCTAATGCTT", "A", 0, 1 },
        { "CAATGCTT", "B", 0, 1 },
    };
    EXPECT_EQ(sorted(ref), obs);
}

TEST(ReadExtraction, MultiQueryAnd) {
    const uint64_t k = 4;
    const std::vector<std::string> sequences { "CAATGCTGCTAATGCTT", "CAATGCTT" };
    const std::vector<std::string> labels { "A", "B" };
    auto anno_graph = make_coord_graph(k, sequences, labels);

    const std::vector<std::string_view> queries { "GCTG", "AATG" };
    auto obs = anno_graph->get_overlapping_reads(queries, 50, 2);
    ASSERT_EQ(1u, obs.size());
    EXPECT_EQ(std::make_tuple(std::string("CAATGCTGCTAATGCTT"), std::string("A"),
                              uint64_t{0}, uint64_t{4}),
              obs[0]);
}

TEST(ReadExtraction, BranchUsesCoordinates) {
    const uint64_t k = 4;
    const std::vector<std::string> sequences { "ACGTAAAA", "ACGTCCCC" };
    const std::vector<std::string> labels { "A", "B" };
    auto anno_graph = make_coord_graph(k, sequences, labels);

    auto shared = sorted(anno_graph->get_overlapping_reads("ACGT", 8, 2));
    Extracted shared_ref {
        { "ACGTAAAA", "A", 0, 0 },
        { "ACGTCCCC", "B", 0, 0 },
    };
    EXPECT_EQ(sorted(shared_ref), shared);

    auto only_b = anno_graph->get_overlapping_reads("CCCC", 8, 2);
    ASSERT_EQ(1u, only_b.size());
    EXPECT_EQ(std::make_tuple(std::string("ACGTCCCC"), std::string("B"),
                              uint64_t{0}, uint64_t{4}),
              only_b[0]);
}

TEST(ReadExtraction, TandemRepeatFollowsCoordinates) {
    const uint64_t k = 3;
    const std::string seq = "ACACACACAC";
    auto anno_graph = make_coord_graph(k, { seq }, { "A" });

    auto obs = anno_graph->get_overlapping_reads("ACA", 20, 1);
    ASSERT_EQ(1u, obs.size());
    EXPECT_EQ(seq, std::get<0>(obs[0]));
    EXPECT_EQ("A", std::get<1>(obs[0]));
}

TEST(ReadExtraction, WindowCrop) {
    const uint64_t k = 4;
    const std::string seq = "ATGCGGATTACAATGCGTCCGGAT";
    auto anno_graph = make_coord_graph(k, { seq }, { "A" });

    auto obs = anno_graph->get_overlapping_reads("TACA", 20, 1, /*flank_length=*/2);
    ASSERT_EQ(1u, obs.size());
    EXPECT_EQ(std::make_tuple(std::string("ATTACAAT"), std::string("A"),
                              uint64_t{0}, uint64_t{2}),
              obs[0]);
}

TEST(ReadExtraction, MaxQueryDistance) {
    const uint64_t k = 4;
    const std::string source = "ATGCGGATCATTGCTAGATCGATGCTATGACCGGATTCATG";
    auto anno_graph = make_coord_graph(k, { source }, { "A" });

    const std::vector<std::string_view> queries { "CATT", "TGAC" };
    auto too_close = anno_graph->get_overlapping_reads(queries, 20, 1, 25000, 5);
    EXPECT_TRUE(too_close.empty());

    auto far_enough = anno_graph->get_overlapping_reads(queries, 20, 1, 25000, 20);
    ASSERT_EQ(1u, far_enough.size());
    EXPECT_EQ(source, std::get<0>(far_enough[0]));
    EXPECT_EQ("A", std::get<1>(far_enough[0]));
    EXPECT_EQ(8u, std::get<3>(far_enough[0]));
}

TEST(ReadExtraction, MultipleReadsPerLabel) {
    const uint64_t k = 4;
    // two reads containing the query and one without it, all with the same label
    const std::vector<std::string> sequences {
        "ATGCGGATTACAATG", "CCTGATTACAGCTTC", "GGTCAAGGC"
    };
    auto anno_graph = make_coord_graph(k, sequences, { "A", "A", "A" }, { 0, 12, 24 });

    Extracted ref {
        { sequences[0], "A", 0, 5 },
        { sequences[1], "A", 0, 3 },
    };
    EXPECT_EQ(sorted(ref), sorted(anno_graph->get_overlapping_reads("GATTACA", 20, 1)));

    // the flanks are cropped at the ends of each read
    EXPECT_EQ(sorted({ { "CGGATTACAAT", "A", 0, 2 }, { "CTGATTACAGC", "A", 0, 2 } }),
              sorted(anno_graph->get_overlapping_reads("GATTACA", 20, 1, /*flank_length=*/2)));
    EXPECT_EQ(sorted({ { "TGCGGATTACAATG", "A", 0, 4 }, { "CCTGATTACAGCTT", "A", 0, 3 } }),
              sorted(anno_graph->get_overlapping_reads("GATTACA", 20, 1, /*flank_length=*/4)));

    // queries must match in the same read
    const std::vector<std::string_view> same { "GCGG", "CAAT" };
    auto joint = anno_graph->get_overlapping_reads(same, 20, 1);
    ASSERT_EQ(1u, joint.size());
    EXPECT_EQ(sequences[0], std::get<0>(joint[0]));
    const std::vector<std::string_view> different { "GCGG", "CTTC" };
    EXPECT_TRUE(anno_graph->get_overlapping_reads(different, 20, 1).empty());
}

TEST(ReadExtraction, MissingQuery) {
    const uint64_t k = 4;
    auto anno_graph = make_coord_graph(k, { "CAATGCTT" }, { "A" });

    EXPECT_TRUE(anno_graph->get_overlapping_reads("GGGG", 20, 1).empty());

    const std::vector<std::string_view> queries { "AATG", "GGGG" };
    EXPECT_TRUE(anno_graph->get_overlapping_reads(queries, 20, 1).empty());
}

} // namespace
