#ifndef __EXTRACT_READS_HPP__
#define __EXTRACT_READS_HPP__

#include <optional>
#include <string>
#include <vector>

#include "cli/query.hpp"
#include "graph/read_extraction.hpp"


namespace mtg {
namespace cli {

class Config;

// A query sequence, with its alignment if it was aligned to the graph
struct ReadQuery {
    QuerySequence sequence;
    std::optional<Alignment> alignment;
};

/**
 * Format a read that contains |queries| as a row
 * '<query id>\t<query name>\t<read>\t<label>\t<position in query>\t<position in read>'.
 * For multiple queries (--joint-query), their ids, names, and positions are comma-separated.
 */
std::string read_to_string(const graph::ExtractedRead &read,
                           const std::vector<ReadQuery> &queries);

// The same as a JSON object
Json::Value read_to_json(const graph::ExtractedRead &read,
                         const std::vector<ReadQuery> &queries);

/**
 * Extract the reads that contain the sequences from files |config.fnames|. Write them to
 * file |config.outfbase| if set, and to the standard output otherwise.
 *
 * Each read is written as soon as it is extracted and is not kept in memory.
 */
int extract_reads(const Config &config,
                  const graph::AnnotatedDBG &anno_graph,
                  const graph::align::DBGAlignerConfig *aligner_config);

} // namespace cli
} // namespace mtg

#endif // __EXTRACT_READS_HPP__
