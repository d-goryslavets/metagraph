#include "extract_reads.hpp"

#include <cassert>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "cli/config/config.hpp"
#include "common/logger.hpp"
#include "common/threads/threading.hpp"
#include "common/unix_tools.hpp"
#include "seq_io/sequence_io.hpp"


namespace mtg {
namespace cli {

using mtg::common::logger;
using mtg::graph::AnnotatedDBG;
using mtg::graph::ExtractedRead;

namespace {

// Writes each record immediately. Thread-safe.
class RecordWriter {
  public:
    // writes to the standard output if |filename| is empty
    explicit RecordWriter(const std::string &filename)
          : name_(filename.size() ? filename : "standard output") {
        if (filename.size()) {
            file_.open(filename);
            if (!file_)
                throw std::runtime_error(fmt::format("Cannot open file '{}' for writing", filename));
        }
    }

    void write(const std::string &record) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ostream &out = file_.is_open() ? file_ : std::cout;
        out << record << '\n' << std::flush;
        if (!out)
            throw std::runtime_error(fmt::format("Cannot write extracted reads to '{}'", name_));
    }

  private:
    std::mutex mutex_;
    std::string name_;
    std::ofstream file_;
};

} // namespace


std::string read_to_string(const ExtractedRead &read, const std::vector<ReadQuery> &queries) {
    assert(read.matches.size() == queries.size());

    std::vector<size_t> ids;
    std::vector<std::string> names;
    std::vector<uint64_t> query_pos;
    std::vector<uint64_t> read_pos;
    for (size_t i = 0; i < queries.size(); ++i) {
        ids.push_back(queries[i].sequence.id);
        names.push_back(get_sequence_name(queries[i].sequence, queries[i].alignment));
        query_pos.push_back(read.matches[i].query_pos);
        read_pos.push_back(read.matches[i].read_pos);
    }
    return fmt::format("{}\t{}\t{}\t{}\t{}\t{}", fmt::join(ids, ","), fmt::join(names, ","),
                       read.sequence, read.label,
                       fmt::join(query_pos, ","), fmt::join(read_pos, ","));
}

Json::Value read_to_json(const ExtractedRead &read, const std::vector<ReadQuery> &queries) {
    assert(read.matches.size() == queries.size());

    Json::Value root = get_label_as_json(read.label);
    root["read"] = read.sequence;
    root["queries"] = Json::Value(Json::arrayValue);
    for (size_t i = 0; i < queries.size(); ++i) {
        Json::Value &query = root["queries"].append(
                get_sequence_json(queries[i].sequence, queries[i].alignment));
        query["id"] = static_cast<Json::UInt64>(queries[i].sequence.id);
        query["query_pos"] = static_cast<Json::UInt64>(read.matches[i].query_pos);
        query["read_pos"] = static_cast<Json::UInt64>(read.matches[i].read_pos);
    }
    return root;
}

int extract_reads(const Config &config,
                  const AnnotatedDBG &anno_graph,
                  const graph::align::DBGAlignerConfig *aligner_config) {
    const graph::ReadExtractor extractor(anno_graph, {
        .traversal_batch_size = config.traversal_batch_size,
        .column_batch_size = config.column_batch_size,
        .flank_length = config.flank_length,
        .max_query_distance = config.max_query_distance,
    });

    RecordWriter writer(config.outfbase);

    Json::StreamWriterBuilder json_builder;
    json_builder["indentation"] = ""; // one object per line

    auto extract = [&](std::vector<ReadQuery> &queries) {
        std::vector<std::string_view> sequences;
        sequences.reserve(queries.size());
        for (auto &[sequence, alignment] : queries) {
            if (aligner_config)
                alignment = align_sequence(&sequence.sequence, anno_graph.get_graph(), *aligner_config);
            sequences.push_back(sequence.sequence);
        }

        size_t num_reads = 0;
        extractor.call_reads(sequences, [&](ExtractedRead&& read) {
            writer.write(config.output_json
                            ? Json::writeString(json_builder, read_to_json(read, queries))
                            : read_to_string(read, queries));
            ++num_reads;
        });
        logger->trace("Extracted {} reads for query '{}'{}", num_reads, queries[0].sequence.name,
                      queries.size() > 1 ? fmt::format(" and {} more", queries.size() - 1) : "");
    };

    // ids are unique across all files, as the reads are written to the same output
    size_t seq_count = 0;
    for (const auto &file : config.fnames) {
        logger->trace("Parsing sequences from file '{}'", file);
        Timer timer;
        size_t num_bp = 0;

        // a task that throws on a worker would terminate the process:
        // keep the first error and rethrow it
        std::exception_ptr error;
        std::mutex error_mutex;
        ThreadPool thread_pool(get_num_threads(), 1000);
        auto enqueue = [&](std::vector<ReadQuery>&& queries) {
            thread_pool.enqueue([&](std::vector<ReadQuery> &queries) {
                try {
                    {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (error)
                            return;
                    }
                    extract(queries);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!error)
                        error = std::current_exception();
                }
            }, std::move(queries));
        };

        std::vector<ReadQuery> queries;
        seq_io::FastaParser fasta_parser(file, config.forward_and_reverse);
        for (const seq_io::kseq_t &kseq : fasta_parser) {
            queries.push_back({ { seq_count++, kseq.name.s, kseq.seq.s }, {} });
            num_bp += kseq.seq.l;
            if (!config.joint_query)
                enqueue(std::exchange(queries, {}));
        }
        if (queries.size())
            enqueue(std::move(queries));

        thread_pool.join();
        if (error)
            std::rethrow_exception(error);

        logger->trace("File '{}' with {} base pairs was processed in {:.3f} sec, throughput: {:.1f} bp/s",
                      file, num_bp, timer.elapsed(), (double)num_bp / timer.elapsed());
    }

    return 0;
}

} // namespace cli
} // namespace mtg
