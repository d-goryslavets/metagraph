#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "../graph/all/test_dbg_helpers.hpp"
#include "test_annotated_dbg_helpers.hpp"

#include "annotation/representation/annotation_matrix/static_annotators_def.hpp"
#include "annotation/representation/column_compressed/annotate_column_compressed.hpp"
#include "graph/read_extraction.hpp"
#include "graph/representation/hash/dbg_hash_fast.hpp"
#include "graph/representation/succinct/dbg_succinct.hpp"

namespace {

using namespace mtg;
using namespace mtg::graph;
using namespace mtg::test;

// read, label, positions of the queries in the queries and in the read
using Positions = std::vector<uint64_t>;
using Extracted = std::vector<std::tuple<std::string, std::string, Positions, Positions>>;

std::unique_ptr<AnnotatedDBG> make_coord_graph(uint64_t k,
                                               const std::vector<std::string> &sequences,
                                               const std::vector<std::string> &labels,
                                               const std::vector<uint64_t> &coord_starts = {}) {
    return build_anno_graph<DBGSuccinct, annot::RowDiffColumnAnnotator>(
        k, sequences, labels, DeBruijnGraph::BASIC, true, coord_starts);
}

// all reads in a sorted order
Extracted extract(const AnnotatedDBG &anno_graph,
                  const std::vector<std::string_view> &queries,
                  const ReadExtractionConfig &config = {}) {
    Extracted result;
    ReadExtractor(anno_graph, config).call_reads(queries, [&](ExtractedRead&& read) {
        Positions query_pos;
        Positions read_pos;
        for (const auto &match : read.matches) {
            query_pos.push_back(match.query_pos);
            read_pos.push_back(match.read_pos);
        }
        result.emplace_back(read.sequence, read.label, query_pos, read_pos);
    });
    std::sort(result.begin(), result.end());
    return result;
}

template <typename GraphAnnotationPair>
class ReadExtractionTest : public ::testing::Test {};

typedef ::testing::Types<std::pair<DBGSuccinct, annot::RowDiffColumnAnnotator>,
                         std::pair<DBGHashFast, annot::RowDiffColumnAnnotator>,
                         std::pair<DBGSuccinct, annot::ColumnCompressed<>>,
                         std::pair<DBGHashFast, annot::ColumnCompressed<>>> GraphAnnotationTypes;
TYPED_TEST_SUITE(ReadExtractionTest, GraphAnnotationTypes);

TYPED_TEST(ReadExtractionTest, LinearSharedKmer) {
    auto anno_graph = build_anno_graph<typename TypeParam::first_type,
                                       typename TypeParam::second_type>(
        4, { "CAATGCTGCTAATGCTT", "CAATGCTT" }, { "A", "B" }, DeBruijnGraph::BASIC, true);

    Extracted ref {
        { "CAATGCTGCTAATGCTT", "A", { 0 }, { 1 } },
        { "CAATGCTT", "B", { 0 }, { 1 } },
    };
    EXPECT_EQ(ref, extract(*anno_graph, { "AATG" }));
    // the batch sizes don't affect the result
    for (uint64_t traversal_batch_size : { 1, 2, 50 }) {
        for (uint64_t column_batch_size : { 1, 2, 3 }) {
            EXPECT_EQ(ref, extract(*anno_graph, { "AATG" },
                                   { .traversal_batch_size = traversal_batch_size,
                                     .column_batch_size = column_batch_size }));
        }
    }
}

TEST(ReadExtraction, MultiQueryAnd) {
    auto anno_graph = make_coord_graph(4, { "CAATGCTGCTAATGCTT", "CAATGCTT" }, { "A", "B" });

    // one read with the first match of each query, in the order of the queries
    EXPECT_EQ((Extracted {{ "CAATGCTGCTAATGCTT", "A", { 0, 0 }, { 4, 1 } }}),
              extract(*anno_graph, { "GCTG", "AATG" }));
    EXPECT_EQ((Extracted {{ "CAATGCTGCTAATGCTT", "A", { 0, 0 }, { 1, 4 } }}),
              extract(*anno_graph, { "AATG", "GCTG" }));
}

TEST(ReadExtraction, BranchUsesCoordinates) {
    auto anno_graph = make_coord_graph(4, { "ACGTAAAA", "ACGTCCCC" }, { "A", "B" });

    EXPECT_EQ((Extracted { { "ACGTAAAA", "A", { 0 }, { 0 } }, { "ACGTCCCC", "B", { 0 }, { 0 } } }),
              extract(*anno_graph, { "ACGT" }));
    EXPECT_EQ((Extracted {{ "ACGTCCCC", "B", { 0 }, { 4 } }}), extract(*anno_graph, { "CCCC" }));
}

TEST(ReadExtraction, TandemRepeatFollowsCoordinates) {
    auto anno_graph = make_coord_graph(3, { "ACACACACAC" }, { "A" });

    // the read is extracted once, for k-mers repeated in the read and in the query
    for (std::string_view query : { "ACA", "ACACA", "ACACACACAC" }) {
        EXPECT_EQ((Extracted {{ "ACACACACAC", "A", { 0 }, { 0 } }}), extract(*anno_graph, { query }));
    }
    EXPECT_EQ((Extracted {}), extract(*anno_graph, { "ACACACACACAC" }));
}

TEST(ReadExtraction, WindowCrop) {
    auto anno_graph = make_coord_graph(4, { "ATGCGGATTACAATGCGTCCGGAT" }, { "A" });

    EXPECT_EQ((Extracted {{ "ATTACAAT", "A", { 0 }, { 2 } }}),
              extract(*anno_graph, { "TACA" }, { .flank_length = 2 }));
    EXPECT_EQ((Extracted {{ "TACA", "A", { 0 }, { 0 } }}),
              extract(*anno_graph, { "TACA" }, { .flank_length = 0 }));
}

TEST(ReadExtraction, MaxQueryDistance) {
    const std::string source = "ATGCGGATCATTGCTAGATCGATGCTATGACCGGATTCATG";
    auto anno_graph = make_coord_graph(4, { source }, { "A" });

    EXPECT_EQ((Extracted {}), extract(*anno_graph, { "CATT", "TGAC" }, { .max_query_distance = 5 }));
    EXPECT_EQ((Extracted {{ source, "A", { 0, 0 }, { 8, 27 } }}),
              extract(*anno_graph, { "CATT", "TGAC" }, { .max_query_distance = 20 }));
}

TEST(ReadExtraction, MultipleReadsPerLabel) {
    // two reads containing the query and one without it, all with the same label
    const std::vector<std::string> sequences {
        "ATGCGGATTACAATG", "CCTGATTACAGCTTC", "GGTCAAGGC"
    };
    auto anno_graph = make_coord_graph(4, sequences, { "A", "A", "A" }, { 0, 12, 24 });

    EXPECT_EQ((Extracted { { sequences[0], "A", { 0 }, { 5 } }, { sequences[1], "A", { 0 }, { 3 } } }),
              extract(*anno_graph, { "GATTACA" }));

    // the flanks are cropped at the ends of each read
    EXPECT_EQ((Extracted { { "CGGATTACAAT", "A", { 0 }, { 2 } }, { "CTGATTACAGC", "A", { 0 }, { 2 } } }),
              extract(*anno_graph, { "GATTACA" }, { .flank_length = 2 }));
    EXPECT_EQ((Extracted { { "CCTGATTACAGCTT", "A", { 0 }, { 3 } }, { "TGCGGATTACAATG", "A", { 0 }, { 4 } } }),
              extract(*anno_graph, { "GATTACA" }, { .flank_length = 4 }));

    // queries must match in the same read
    EXPECT_EQ((Extracted {{ sequences[0], "A", { 0, 0 }, { 2, 10 } }}),
              extract(*anno_graph, { "GCGG", "CAAT" }));
    EXPECT_EQ((Extracted {}), extract(*anno_graph, { "GCGG", "CTTC" }));
}

TEST(ReadExtraction, SequenceHeaders) {
    // the second read starts with the last k-1 characters of the first one, hence there
    // is an edge between the last k-mer of the first read and the first k-mer of the second
    const std::vector<std::string> sequences { "ATGCGGATTACA", "ACAGCTTC" };
    auto anno_graph = make_coord_graph(4, sequences, { "A", "A" }, { 0, 9 });

    // without the header index, the reads can't be told apart
    EXPECT_EQ((Extracted {{ "ATGCGGATTACAGCTTC", "A", { 0 }, { 5 } }}),
              extract(*anno_graph, { "GATT" }));

    anno_graph = with_coord_to_header(std::move(anno_graph), {{ "read_1", "read_2" }}, {{ 9, 5 }});
    EXPECT_EQ((Extracted {{ sequences[0], "read_1", { 0 }, { 5 } }}), extract(*anno_graph, { "GATT" }));
    EXPECT_EQ((Extracted {{ sequences[1], "read_2", { 0 }, { 3 } }}), extract(*anno_graph, { "GCTT" }));
    EXPECT_EQ((Extracted {}), extract(*anno_graph, { "TTACAGC" }));
    EXPECT_EQ((Extracted {}), extract(*anno_graph, { "GATT", "GCTT" }));
}

TEST(ReadExtraction, MissingQuery) {
    auto anno_graph = make_coord_graph(4, { "CAATGCTT" }, { "A" });

    EXPECT_EQ((Extracted {}), extract(*anno_graph, { "GGGG" }));
    EXPECT_EQ((Extracted {}), extract(*anno_graph, { "AATG", "GGGG" }));
    // shorter than k
    EXPECT_EQ((Extracted {}), extract(*anno_graph, { "AAT" }));
    EXPECT_EQ((Extracted {}), extract(*anno_graph, {}));
}

TEST(ReadExtraction, MissingKmers) {
    auto anno_graph = make_coord_graph(4, { "GGCTAAAAAAATCCG" }, { "A" });

    // missing k-mers at the ends of the query are ignored
    EXPECT_EQ((Extracted {{ "GGCTAAAAAAATCCG", "A", { 1 }, { 1 } }}),
              extract(*anno_graph, { "CGCTAAC" }));
    // the k-mers around the missing ones are consecutive in the read, but not in the query
    EXPECT_EQ((Extracted {}), extract(*anno_graph, { "AAAACAAAA" }));
}

// the reads are passed to the callback as they are extracted, and the errors are passed through
TEST(ReadExtraction, Streaming) {
    auto anno_graph = make_coord_graph(4, { "CAATGCTGC", "CAATGCTT" }, { "A", "B" });

    size_t num_reads = 0;
    const std::vector<std::string_view> queries { "AATG" };
    EXPECT_THROW(ReadExtractor(*anno_graph).call_reads(queries, [&](ExtractedRead&&) {
        num_reads++;
        throw std::runtime_error("stop");
    }), std::runtime_error);
    EXPECT_EQ(1u, num_reads);
}

TEST(ReadExtraction, Unsupported) {
    // no coordinates
    auto binary = build_anno_graph<DBGSuccinct, annot::ColumnCompressed<>>(4, { "CAATGCTT" }, { "A" });
    EXPECT_THROW(ReadExtractor { *binary }, std::invalid_argument);

    auto canonical = build_anno_graph<DBGSuccinct, annot::ColumnCompressed<>>(
        5, { "CAATGCTT" }, { "A" }, DeBruijnGraph::CANONICAL, true);
    EXPECT_THROW(ReadExtractor { *canonical }, std::invalid_argument);

    auto anno_graph = make_coord_graph(4, { "CAATGCTT" }, { "A" });
    EXPECT_THROW((ReadExtractor { *anno_graph, { .traversal_batch_size = 0 } }),
                 std::invalid_argument);
}

} // namespace
