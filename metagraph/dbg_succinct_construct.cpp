#include "dbg_succinct_construct.hpp"

#include <ips4o.hpp>

#include "kmer.hpp"
#include "dbg_succinct_chunk.hpp"
#include "utils.hpp"
#include "unix_tools.hpp"
#include "reads_filtering.hpp"


const size_t kMaxKmersChunkSize = 30'000'000;


template <typename KMER>
void sort_and_remove_duplicates(Vector<KMER> *kmers,
                                size_t num_threads,
                                size_t end_sorted = 0) {
    if (num_threads <= 3) {
        // sort
        ips4o::parallel::sort(kmers->data() + end_sorted,
                              kmers->data() + kmers->size(),
                              std::less<KMER>(), num_threads);
        kmers->erase(std::unique(kmers->begin() + end_sorted, kmers->end()),
                     kmers->end());

        // merge two sorted arrays
#if 0
        std::__merge_without_buffer(kmers->data(),
                                    kmers->data() + end_sorted,
                                    kmers->data() + kmers->size(),
                                    end_sorted, kmers->size() - end_sorted,
                                    __gnu_cxx::__ops::__iter_less_iter());
#else
        std::inplace_merge(kmers->data(),
                           kmers->data() + end_sorted,
                           kmers->data() + kmers->size());
#endif
    } else {
        // sort
        ips4o::parallel::sort(kmers->data(), kmers->data() + kmers->size(),
                              std::less<KMER>(), num_threads);
    }
    // remove duplicates
    auto unique_end = std::unique(kmers->begin(), kmers->end());
    kmers->erase(unique_end, kmers->end());
}

template <typename KMER>
void shrink_kmers(Vector<KMER> *kmers,
                  size_t *end_sorted,
                  size_t num_threads,
                  bool verbose) {
    if (verbose) {
        std::cout << "Allocated capacity exceeded, filter out non-unique k-mers..."
                  << std::flush;
    }

    size_t prev_num_kmers = kmers->size();
    sort_and_remove_duplicates(kmers, num_threads, *end_sorted);
    *end_sorted = kmers->size();

    if (verbose) {
        std::cout << " done. Number of kmers reduced from " << prev_num_kmers
                                                  << " to " << *end_sorted << ", "
                  << (*end_sorted * sizeof(KMER) >> 20) << "Mb" << std::endl;
    }
}

template <class Array, class Vector>
void extend_kmer_storage(const Array &temp_storage,
                         Vector *kmers,
                         size_t *end_sorted,
                         size_t num_threads,
                         bool verbose,
                         std::mutex *mutex) {
    assert(mutex);

    // acquire the mutex to restrict the number of writing threads
    std::lock_guard<std::mutex> lock(*mutex);

    // shrink collected k-mers if the memory limit is exceeded
    if (kmers->size() + temp_storage.size() > kmers->capacity()) {
        shrink_kmers(kmers, end_sorted, num_threads, verbose);
        kmers->reserve(kmers->size()
                        + std::max(temp_storage.size(), kmers->size() / 2));
        if (kmers->size() + temp_storage.size() > kmers->capacity()) {
            std::cerr << "ERROR: Can't reallocate. Not enough memory" << std::endl;
        }
    }
    for (auto &kmer : temp_storage) {
        kmers->push_back(kmer);
    }
}

typedef std::function<void(const std::string&)> CallbackRead;

template <typename KMER>
void extract_kmers(std::function<void(CallbackRead)> generate_reads,
                   size_t k,
                   Vector<KMER> *kmers,
                   size_t *end_sorted,
                   const std::vector<TAlphabet> &suffix,
                   size_t num_threads,
                   bool verbose,
                   std::mutex *mutex,
                   bool remove_redundant = true) {
    Vector<KMER> temp_storage;
    temp_storage.reserve(1.1 * kMaxKmersChunkSize);

    generate_reads([&](const std::string &read) {
        utils::sequence_to_kmers(read, k, &temp_storage, suffix);

        if (temp_storage.size() < kMaxKmersChunkSize)
            return;

        if (remove_redundant) {
            sort_and_remove_duplicates(&temp_storage, 1);
        }

        if (temp_storage.size() > 0.9 * kMaxKmersChunkSize) {
            extend_kmer_storage(temp_storage, kmers, end_sorted,
                                num_threads, verbose, mutex);
            temp_storage.resize(0);
        }
    });

    if (temp_storage.size()) {
        if (remove_redundant) {
            sort_and_remove_duplicates(&temp_storage, 1);
        }
        extend_kmer_storage(temp_storage, kmers, end_sorted,
                            num_threads, verbose, mutex);
    }
}

// Although this function could be parallelized better,
// the experiments show it's already fast enough.
template <typename KMER>
void recover_source_dummy_nodes(size_t k,
                                Vector<KMER> *kmers,
                                size_t num_threads,
                                bool verbose) {
    // remove redundant dummy kmers inplace
    size_t cur_pos = 0;
    size_t end_sorted = kmers->size();

    for (size_t i = 0; i < end_sorted; ++i) {
        const KMER &kmer = kmers->at(i);
        // we never add reads shorter than k
        assert(kmer[1] != 0 || kmer[0] != 0 || kmer[k] == 0);

        TAlphabet edge_label;

        // check if it's not a source dummy kmer
        if (kmer[1] > 0 || (edge_label = kmer[0]) == 0) {
            kmers->at(cur_pos++) = kmer;
            continue;
        }

        bool redundant = false;
        for (size_t j = i + 1; j < end_sorted
                                && KMER::compare_suffix(kmer, kmers->at(j), 1); ++j) {
            if (edge_label == kmers->at(j)[0]) {
                // This source dummy kmer is redundant and has to be erased
                redundant = true;
                break;
            }
        }
        if (redundant)
            continue;

        // leave this dummy kmer in the list
        kmers->at(cur_pos++) = kmer;

        if (kmers->size() + k > kmers->capacity()) {
            if (verbose) {
                std::cout << "Allocated capacity exceeded,"
                          << " filter out non-unique k-mers..." << std::flush;
            }

            ips4o::parallel::sort(kmers->data() + end_sorted,
                                  kmers->data() + kmers->size(),
                                  std::less<KMER>(),
                                  num_threads);
            kmers->erase(
                std::unique(kmers->begin() + end_sorted, kmers->end()),
                kmers->end()
            );

            if (verbose) {
                std::cout << " done. Number of kmers: " << kmers->size() << ", "
                          << (kmers->size() * sizeof(KMER) >> 20) << "Mb" << std::endl;
            }
        }

        // anchor it to the dummy source node
        auto anchor_kmer = KMER::pack_kmer(std::vector<TAlphabet>(k + 1, 0), k + 1);
        for (size_t c = 2; c < k + 1; ++c) {
            KMER::update_kmer(k, kmers->at(i)[c], kmers->at(i)[c - 1], &anchor_kmer);

            kmers->emplace_back(anchor_kmer);
        }
    }
    std::copy(kmers->begin() + end_sorted, kmers->end(),
              kmers->begin() + cur_pos);
    kmers->resize(kmers->size() - end_sorted + cur_pos);

    sort_and_remove_duplicates(kmers, num_threads, cur_pos);
}


void KMerDBGSuccConstructor::build_graph(DBG_succ *graph) {
    // build a graph chunk from kmers
    auto chunk = constructor_->build_chunk();
    // initialize graph from the chunk built
    chunk->initialize_graph(graph);
    delete chunk;
}


IChunkConstructor*
IChunkConstructor::initialize(size_t k,
                              const std::string &filter_suffix,
                              size_t num_threads,
                              double memory_preallocated,
                              bool verbose) {
    if ((k + 1) * kBitsPerChar <= 64) {
        return new KMerDBGSuccChunkConstructor<KMer<uint64_t>>(
            k, filter_suffix, num_threads, memory_preallocated, verbose
        );
    } else if ((k + 1) * kBitsPerChar <= 128) {
        return new KMerDBGSuccChunkConstructor<KMer<sdsl::uint128_t>>(
            k, filter_suffix, num_threads, memory_preallocated, verbose
        );
    } else {
        return new KMerDBGSuccChunkConstructor<KMer<sdsl::uint256_t>>(
            k, filter_suffix, num_threads, memory_preallocated, verbose
        );
    }
}


template <typename KMER>
KMerDBGSuccChunkConstructor<KMER>
::KMerDBGSuccChunkConstructor(size_t k,
                              const std::string &filter_suffix,
                              size_t num_threads,
                              double memory_preallocated,
                              bool verbose)
      : k_(k),
        end_sorted_(0),
        num_threads_(num_threads),
        thread_pool_(std::max(static_cast<size_t>(1), num_threads_) - 1,
                     std::max(static_cast<size_t>(1), num_threads_)),
        stored_reads_size_(0),
        verbose_(verbose) {
    assert(num_threads_ > 0);

    filter_suffix_encoded_.resize(filter_suffix.size());
    std::transform(
        filter_suffix.begin(), filter_suffix.end(),
        filter_suffix_encoded_.begin(),
        [](char c) {
            return c == DBG_succ::kSentinel
                            ? DBG_succ::kSentinelCode
                            : DBG_succ::encode(c);
        }
    );

    kmers_.reserve(memory_preallocated / sizeof(KMER));

    if (filter_suffix == std::string(filter_suffix.size(), DBG_succ::kSentinel)) {
        kmers_.emplace_back(
            std::vector<TAlphabet>(k + 1, DBG_succ::kSentinelCode), k + 1
        );
    }
}

template <typename KMER>
void KMerDBGSuccChunkConstructor<KMER>::add_read(const std::string &sequence) {
    if (sequence.size() < k_)
        return;

    // put read into temporary storage
    stored_reads_size_ += sequence.size();
    reads_storage_.emplace_back(sequence);

    if (stored_reads_size_ < kMaxKmersChunkSize)
        return;

    // extract all k-mers from sequences accumulated in the temporary storage
    release_task_to_pool();

    assert(!stored_reads_size_);
    assert(!reads_storage_.size());
}

template <typename KMER>
void KMerDBGSuccChunkConstructor<KMER>::release_task_to_pool() {
    auto *current_reads_storage = new std::vector<std::string>();
    current_reads_storage->swap(reads_storage_);

    thread_pool_.enqueue(extract_kmers<KMER>,
                         [current_reads_storage](CallbackRead callback) {
                             for (auto &&read : *current_reads_storage) {
                                 callback(std::move(read));
                             }
                             delete current_reads_storage;
                         },
                         k_, &kmers_, &end_sorted_, filter_suffix_encoded_,
                         num_threads_, verbose_, &mutex_, true);
    stored_reads_size_ = 0;
}

/**
 * Initialize graph chunk from a list of sorted kmers.
 */
template <typename KMER>
DBG_succ::Chunk* chunk_from_kmers(size_t k, const KMER *kmers,
                                            uint64_t num_kmers) {
    assert(std::is_sorted(kmers, kmers + num_kmers));

    // the array containing edge labels
    std::vector<TAlphabet> W(1 + num_kmers);
    W[0] = 0;
    // the bit array indicating last outgoing edges for nodes
    std::vector<bool> last(1 + num_kmers, 1);
    last[0] = 0;
    // offsets
    std::vector<uint64_t> F(DBG_succ::alph_size, 0);
    F.at(0) = 0;

    size_t curpos = 1;
    TAlphabet lastF = 0;

    for (size_t i = 0; i < num_kmers; ++i) {
        TAlphabet curW = kmers[i][0];
        TAlphabet curF = kmers[i][k];

        assert(curW < DBG_succ::alph_size);

        // check redundancy and set last
        if (i + 1 < num_kmers && KMER::compare_suffix(kmers[i], kmers[i + 1])) {
            // skip redundant dummy edges
            if (curW == 0 && curF > 0)
                continue;

            last[curpos] = 0;
        }
        //set W
        if (i > 0) {
            for (size_t j = i - 1; KMER::compare_suffix(kmers[i], kmers[j], 1); --j) {
                if (curW > 0 && kmers[j][0] == curW) {
                    curW += DBG_succ::alph_size;
                    break;
                }
                if (j == 0)
                    break;
            }
        }
        W[curpos] = curW;

        while (lastF + 1 < DBG_succ::alph_size && curF != lastF) {
            F.at(++lastF) = curpos - 1;
        }
        curpos++;
    }
    while (++lastF < DBG_succ::alph_size) {
        F.at(lastF) = curpos - 1;
    }

    W.resize(curpos);
    last.resize(curpos);

    return new DBG_succ::Chunk(k, std::move(W), std::move(last), std::move(F));
}

template <typename KMER>
DBG_succ::Chunk* KMerDBGSuccChunkConstructor<KMER>::build_chunk() {
    release_task_to_pool();
    thread_pool_.join();

    if (verbose_) {
        std::cout << "Reading data has finished" << std::endl;
        get_RAM();
        std::cout << "Sorting kmers and appending succinct"
                  << " representation from current bin...\t" << std::flush;
    }
    Timer timer;

    sort_and_remove_duplicates(&kmers_, num_threads_, end_sorted_);

    if (verbose_)
        std::cout << timer.elapsed() << "sec" << std::endl;

    if (!filter_suffix_encoded_.size()) {
        if (verbose_) {
            std::cout << "Reconstructing all required dummy source k-mers...\t"
                      << std::flush;
        }
        timer.reset();

        recover_source_dummy_nodes(k_, &kmers_, num_threads_, verbose_);

        if (verbose_)
            std::cout << timer.elapsed() << "sec" << std::endl;
    }

    DBG_succ::Chunk *result = chunk_from_kmers(k_, kmers_.data(), kmers_.size());

    kmers_.clear();

    return result;
}

template <typename KMER>
void
KMerDBGSuccChunkConstructor<KMER>
::add_reads(std::function<void(CallbackRead)> generate_reads) {
    thread_pool_.enqueue(extract_kmers<KMER>, generate_reads,
                         k_, &kmers_, &end_sorted_,
                         filter_suffix_encoded_,
                         num_threads_, verbose_, &mutex_, true);
}


SuffixArrayDBGSuccConstructor::SuffixArrayDBGSuccConstructor(size_t k)
      : k_(k), data_("$") {}

void SuffixArrayDBGSuccConstructor::add_read(const std::string &read) {
    data_.append(read);
    data_.append("$");
}

// Implement SA construction and extract the kmers from the result
void SuffixArrayDBGSuccConstructor::build_graph(DBG_succ *graph) {
    // DBG_succ::Chunk result;
    // result.initialize_graph(graph);
    std::ignore = graph;
}
