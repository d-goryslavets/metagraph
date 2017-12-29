#include "dbg_succinct.hpp"

/**
 * This class contains a succinct representation of the de bruijn graph
 * following ideas and suggestions presented here:
 * http://link.springer.com/chapter/10.1007/978-3-642-33122-0_18
 *
 * There is also conceptual code available at
 * https://code.google.com/p/csalib/downloads/list
 * that has been used as a reference for this implementation.
 */

#include <assert.h>
#include <vector>
#include <map>
#include <stack>
#include <queue>
#include <deque>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <cstdio>
#include <parallel/algorithm>

/**
 * We use libmaus 2 for representing dynamic succint data structures
 * such as the dynamic bit array and the dynamic wavelet tree.
 */
#include <libmaus2/util/NumberSerialisation.hpp>
#include <zlib.h>

#include "serialization.hpp"


const std::string DBG_succ::alphabet = "$ACGTN$ACGTN";
const size_t DBG_succ::alph_size = 6;

const TAlphabet kCharToNucleotide[128] = {
    5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,
    5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,
    5, 5, 5, 5,  0, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,
    5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,
    5, 1, 5, 2,  5, 5, 5, 3,  5, 5, 5, 5,  5, 5, 5, 5,
    5, 5, 5, 5,  4, 4, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5,
    5, 1, 5, 2,  5, 5, 5, 3,  5, 5, 5, 5,  5, 5, 5, 5,
    5, 5, 5, 5,  4, 4, 5, 5,  5, 5, 5, 5,  5, 5, 5, 5
};


DBG_succ::DBG_succ(size_t k)
      : k_(k), last(new bit_vector_dyn()),
               F(alph_size, 0),
               W(new wavelet_tree_dyn(4)) {

    last->insertBit(0, false);
    W->insert(0, 0);

    // add the dummy source node
    last->insertBit(1, true);
    W->insert(0, 0);
    for (size_t j = 1; j < alph_size; j++) {
        F[j] = 1;
    }
}

DBG_succ::~DBG_succ() {
    delete W;
    delete last;
}

/**
* Given a pointer to a graph structures G1 and G2, the function compares their elements to the
* each other. It will perform an element wise comparison of the arrays W, last and
* F and will only check for identity. If any element differs, the function will return
* false and true otherwise.
*/
bool DBG_succ::operator==(const DBG_succ &other) const {
    // compare size
    if (W->size() != other.W->size()) {
        verbose_cout("sizes of graphs differ", "\n",
                     "1: ", W->size(), "\n",
                     "2: ", other.W->size(), "\n");
        return false;
    }

    if (F.size() != other.F.size()) {
        verbose_cout("sizes of F arrays differ", "\n",
                     "1: ", F.size(), "\n",
                     "2: ", other.F.size(), "\n");
        return false;
    }

    // compare last
    for (size_t i = 0; i < W->size(); ++i) {
        if (get_last(i) != other.get_last(i)) {
            verbose_cout("last differs at position ", i, "\n",
                         "1: last[", i, "] = ", get_last(i) , "\n",
                         "2: last[", i, "] = ", other.get_last(i), "\n");
            return false;
        }
    }

    // compare W
    for (size_t i = 0; i < W->size(); ++i) {
        if (get_W(i) != other.get_W(i)) {
            verbose_cout("W differs at position ", i, "\n",
                         "1: W[", i, "] = ", get_W(i) , "\n",
                         "2: W[", i, "] = ", other.get_W(i), "\n");
            return false;
        }
    }

    // compare F
    for (size_t i = 0; i < F.size(); ++i) {
        if (get_F(i) != other.get_F(i)) {
            verbose_cout("F differs at position ", i, "\n",
                         "1: F[", i, "] = ", get_F(i), "\n",
                         "2: F[", i, "] = ", other.get_F(i), "\n");
            return false;
        }
    }

    return true;
}

void DBG_succ::serialize(const std::string &outbase) const {
    // write Wavelet Tree
    std::ofstream outstream(outbase + ".W.dbg");
    W->serialise(outstream);
    outstream.close();

    // write last array
    outstream.open(outbase + ".l.dbg");
    last->serialise(outstream);
    outstream.close();

    // write F values and k
    outstream.open(outbase + ".F.dbg");
    outstream << ">F" << std::endl;
    for (size_t i = 0; i < F.size(); ++i) {
        outstream << F.at(i) << "\n";
    }
    outstream << ">k\n"
              << k_ << "\n"
              << ">s\n"
              << state << "\n";
    outstream.close();
}

bool DBG_succ::load(const std::string &infbase) {
    // if not specified in the file, the default for loading is dynamic
    state = Config::DYN;

    // load F and k and p
    F.resize(0);
    F.reserve(alph_size);

    try {
        std::ifstream instream(infbase + ".F.dbg");
        char mode = 0;
        std::string cur_line;

        while (std::getline(instream, cur_line)) {
            if (cur_line[0] == '>') {
                if (cur_line.length() < 2)
                    return false;
                mode = cur_line[1];
                continue;
            }
            switch (mode) {
                case 'F':
                    F.push_back(std::stoull(cur_line));
                    break;
                case 'k':
                    k_ = std::stoul(cur_line);
                    break;
                case 's':
                    state = static_cast<Config::StateType>(std::stoul(cur_line));
                    break;
                default:
                    return false;
            }
        }
        instream.close();

        if (F.size() != alph_size)
            return false;

        // load W and last arrays
        delete W;
        delete last;
        switch (state) {
            case Config::DYN:
                W = new wavelet_tree_dyn(4);
                last = new bit_vector_dyn();
                break;
            case Config::STAT:
                W = new wavelet_tree_stat(4);
                last = new bit_vector_stat();
                break;
        }
        std::ifstream instream_W(infbase + ".W.dbg");
        std::ifstream instream_l(infbase + ".l.dbg");
        return W->deserialise(instream_W) && last->deserialise(instream_l);
    } catch (...) {
        return false;
    }
}

//
//
// QUERY FUNCTIONS
//
//

/**
 * Uses the object's array W, a given position i in W and a character c
 * from the alphabet and returns the number of occurences of c in W up to
 * position i.
 */
uint64_t DBG_succ::rank_W(uint64_t i, TAlphabet c) const {
    assert(i < W->size());

    // deal with  border conditions
    if (i <= 0)
        return 0;
    return W->rank(c, std::min(i, W->size() - 1)) - (c == 0);
}

/**
 * Uses the array W and gets a count i and a character c from
 * the alphabet and returns the position of the i-th occurence of c in W.
 */
uint64_t DBG_succ::select_W(uint64_t i, TAlphabet c) const {

    // deal with  border conditions
    if (i <= 0)
        return 0;

    return i + (c == 0) <= W->rank(c, W->size() - 1)
                ? W->select(c, i + (c == 0))
                : W->size();
}

/**
 * This is a convenience function that returns for array W, a position i and
 * a character c the last index of a character c preceding in W[1..i].
 */
uint64_t DBG_succ::pred_W(uint64_t i, TAlphabet c) const {
    assert(i < W->size());

    return select_W(rank_W(i, c), c);
}

/**
 * This is a convenience function that returns for array W, a position i and
 * a character c the first index of a character c in W[i..N].
 */
uint64_t DBG_succ::succ_W(uint64_t i, TAlphabet c) const {
    assert(i < W->size());

    return select_W(rank_W(i - 1, c) + 1, c);
}

/**
 * Uses the object's array last and a position and
 * returns the number of set bits up to that postion.
 */
uint64_t DBG_succ::rank_last(uint64_t i) const {
    assert(i < last->size());

    // deal with  border conditions
    if (i <= 0)
        return 0;
    return last->rank1(i);
}

/**
 * Uses the object's array last and a given position i and
 * returns the position of the i-th set bit in last[1..i].
 */
uint64_t DBG_succ::select_last(uint64_t i) const {
    assert(i < last->size());

    // deal with  border conditions
    if (i <= 0)
        return 0;
    // for some reason the libmaus2 select is 0 based ...
    return std::min(last->select1(i), last->size());
}

/**
 * This is a convenience function that returns for the object's array last
 * and a given position i the position of the last set bit in last[1..i].
 */
uint64_t DBG_succ::pred_last(uint64_t i) const {
    assert(i < W->size());

    return select_last(rank_last(i));
}

/**
 * This is a convenience function that returns for the object's array last
 * and a given position i the position of the first set bit in last[i..N].
 */
uint64_t DBG_succ::succ_last(uint64_t i) const {
    assert(i < W->size());

    uint64_t next_rank = rank_last(i - 1) + 1;

    if (next_rank > rank_last(last->size() - 1))
        return last->size();

    return select_last(next_rank);
}

/**
 * This function gets a position i that reflects the i-th node and returns the
 * position in W that corresponds to the i-th node's last character.
 */
uint64_t DBG_succ::bwd(uint64_t i) const {
    assert(i < W->size());

    // get value of last position in node i
    TAlphabet c = get_node_last_char(i);
    // get the offset for the last position in node i
    uint64_t o = F[c];
    // compute the offset for this position in W and select it
    return select_W(rank_last(i) - rank_last(o), c);
}

/**
 * This functions gets a position i reflecting the r-th occurence of the corresponding
 * character c in W and returns the position of the r-th occurence of c in last.
 */
uint64_t DBG_succ::fwd(uint64_t i) const {
    assert(i < W->size());

    // get value of W at position i
    TAlphabet c = get_W(i) % alph_size;
    // get the offset for position c
    uint64_t o = F[c];
    // get the rank of c in W at position i
    uint64_t r = rank_W(i, c);
    // select the index of the position in last that is rank many positions after offset
    return select_last(rank_last(o) + r);
}


/**
 * Using the offset structure F this function returns the value of the last
 * position of node i.
 */
TAlphabet DBG_succ::get_node_last_char(uint64_t i) const {
    assert(i < W->size());

    if (i == 0)
        return 0;
    for (size_t j = 0; j < F.size(); j++) {
        if (F[j] >= i)
            return j - 1;
    }
    return F.size() - 1;
}


/**
 * Given index of node i, the function returns the
 * first character of the node.
 */
TAlphabet DBG_succ::get_node_begin_value(uint64_t i) const {
    assert(i < W->size());

    if (i == 1)
        return encode('$');

    for (size_t j = 0; j < k_ - 1; ++j) {
        i = bwd(succ_last(i));
        if (i == 1)
            return encode('$');
    }
    return get_node_last_char(i);
}

/**
 * Given a position i in W and an edge label c, this function returns the
 * index of the outgoing edge with label c.
 */
uint64_t DBG_succ::outgoing_edge_idx(uint64_t i, TAlphabet c) const {
    assert(i < W->size());
    assert(c < alph_size);

    if (i == 0 || i > W->size())
        return 0;

    uint64_t first_pos = pred_last(i - 1) + 1;
    uint64_t last_pos = succ_last(i);

    uint64_t j = std::max(pred_W(last_pos, c),
                          pred_W(last_pos, c + alph_size));

    if (j < first_pos || j >= W->size())
        return 0;

    return j;
}

/**
 * Given a position i in W and an edge label c, this function returns the
 * index of the node the edge is pointing to.
 */
uint64_t DBG_succ::outgoing(uint64_t i, TAlphabet c) const {
    assert(i < W->size());
    c = c % alph_size;

    uint64_t j = outgoing_edge_idx(i, c);
    if (j == 0)
        return 0;

    j = fwd(j);
    if (j == 0 || j == W->size())
        return 0;

    return j;
}

/**
 * Given a node index i and an edge label c, this function returns the
 * index of the node the incoming edge belongs to.
 */
uint64_t DBG_succ::incoming(uint64_t i, TAlphabet c) const {
    assert(i < W->size());

    if (i == 1)
        return 0;
    c %= alph_size;
    TAlphabet d = get_node_last_char(i);
    uint64_t x = bwd(i);
    if (get_node_begin_value(x) == c) {
        return succ_last(x);
    }
    uint64_t y = succ_W(x + 1, d);
    while (true) {
        x = succ_W(x + 1, d + alph_size);
        if (x >= y) {
            break;
        }
        if (get_node_begin_value(x) == c) {
            return succ_last(x);
        }
    }
    return 0;
}

/**
 * Given a node index i, this function returns the number of outgoing
 * edges from node i.
 */
uint64_t DBG_succ::outdegree(uint64_t i) const {
    assert(i < W->size());

    return (i < W->size()) ? succ_last(i) - pred_last(i - 1) : 0;
}


/**
 * Given a node index i, this function returns the number of incoming
 * edges to node i.
 */
uint64_t DBG_succ::indegree(uint64_t i) const {
    assert(i < W->size());

    if (i < 2)
        return 0;
    uint64_t x = bwd(succ_last(i));
    TAlphabet d = get_node_last_char(i);
    uint64_t y = succ_W(x + 1, d);
    return 1 + rank_W(y, d + alph_size) - rank_W(x, d + alph_size);
}



/**
 * Given a string str and a maximal number of edit operations
 * max_distance, this function returns all nodes with labels at most
 * max_distance many edits away from str.
 */
std::vector<HitInfo> DBG_succ::index_fuzzy(const std::string &str,
                                           uint64_t max_distance) const {

    std::vector<HitInfo> result;
    std::priority_queue<HitInfo, std::vector<HitInfo>, HitInfoCompare> hits;
    std::priority_queue<HitInfo, std::vector<HitInfo>, HitInfoCompare> hits2;
    uint64_t rl;
    uint64_t ru;

    // walk through pattern, thereby collecting possible partial matches
    // once the end of the pattern is reached, add match to results

    // init match/mismatch to first pattern position
    TAlphabet s = encode(str[0]);
    for (TAlphabet b = 1; b < 5; ++b) {
        rl = F[b] + 1 < W->size()
             ? succ_last(F[b] + 1)
             : W->size();
        ru = b + 1 < F.size()
             ? F[b + 1]
             : W->size() - 1;
        //std::cout << "pushing: rl " << rl << " ru " << ru << " str_pos 1 max_distance " << (uint64_t) (b != s) << std::endl;
        //std::cout << "s " << s << " b " << b << std::endl;
        std::vector<uint64_t> tmp;
        hits.push({ rl, ru, 1, 1, static_cast<uint64_t>(b != s),
                    std::string(1, decode(b)), tmp });

        // opening/extending a gap in the pattern starting with the first position
        if (max_distance > 0) {
            for (size_t p = 1; p < str.length() - 1; ++p) {
                TAlphabet ss = encode(str[p]);
                if ((p + (b != ss)) > max_distance)
                    break;
                hits.push({ rl, ru, p + 1, 1, p + (b != ss),
                            std::string(p, 'd') + std::string(1, decode(b)), tmp });
                //std::cout << "a) adding '-'" << std::endl;
            }
        }
    }

    // walk through pattern thereby extending all partial hits
    while (hits.size() > 0) {
        while (hits.size() > 0) {
            HitInfo curr_hit(hits.top());
            hits.pop();
            //std::cout << "loaded: rl " << curr_hit.rl << " ru " << curr_hit.ru << " dist " << curr_hit.distance << std::endl;

            if (curr_hit.str_pos < str.length()) {

                // opening/extending a gap in the graph, leaving current pattern position unmatched
                if (curr_hit.distance < max_distance) {
                    hits2.push({ curr_hit.rl, curr_hit.ru, curr_hit.str_pos + 1,
                                 curr_hit.graph_pos, curr_hit.distance + 1,
                                 curr_hit.cigar + 'd', curr_hit.path });
                    //std::cout << "b) " << curr_hit.cigar << " adding '-'" << std::endl;
                }

                s = encode(str[curr_hit.str_pos]);

                // has the number of matches exceeded the node length?
                // there are three possible scenarios for extension of the path:
                //  1) pattern is shorter than the node length --> get an interval of matching nodes
                //  2) pattern length exactly mathces the node length --> there is one correponding node
                //  3) pattern is longer than the node length --> we append to a path
                if (curr_hit.graph_pos >= k_) {
                //    std::cout << "push back tp path " << curr_hit.rl << std::endl;
                    curr_hit.path.push_back(curr_hit.rl);
                }

                // iterate through all possible extensions of current position
                for (TAlphabet b = 1; b < 5; ++b) {
                    if (curr_hit.distance <= max_distance) {

                        // we cannot afford any more mismatches
                        if ((curr_hit.distance + (b != s)) > max_distance)
                            continue;

                        // re-define range of nodes to check for outgoing nodes
                        rl = std::min(succ_W(pred_last(curr_hit.rl - 1) + 1, b),
                                      succ_W(pred_last(curr_hit.rl - 1) + 1, b + alph_size));
                        ru = std::max(pred_W(curr_hit.ru, b),
                                      pred_W(curr_hit.ru, b + alph_size));

                        // the current range in W does not contain our next symbol
                        if ((rl >= W->size()) || (ru >= W->size()) || (rl > ru))
                            continue;

                        // update the SA range with the current symbol b
                        rl = outgoing(rl, b);
                        ru = outgoing(ru, b);

                        // range is empty
                        if ((rl == 0) && (ru == 0))
                            continue;

                        // add hit for extension in next step
                        hits2.push({ rl, ru, curr_hit.str_pos + 1,
                                     curr_hit.graph_pos + 1, curr_hit.distance + (b != s),
                                     curr_hit.cigar + decode(b), curr_hit.path });

                        // opening/extending a gap in the pattern, leaving current graph position unmatched
                        // --> choose any available mismatching next edge
                        if (b != s) {
                            hits2.push({ rl, ru, curr_hit.str_pos,
                                         curr_hit.graph_pos + 1, curr_hit.distance + 1,
                                         curr_hit.cigar + 'i', curr_hit.path });
                        }
                    }
                }
            } else {
                // collect results
                //std::make_pair(curr_hit.rl < curr_hit.ru ? curr_hit.ru : curr_hit.rl, curr_hit.cigar));
                result.push_back(curr_hit);
            }
        }
        hits.swap(hits2);
    }

    return result;
}


/**
 * Given a node label s, this function returns the index
 * of the corresponding node or the closest predecessor, if no node
 * with the sequence is not found.
 */
uint64_t DBG_succ::pred_kmer(const std::deque<TAlphabet> &kmer) const {
    assert(kmer.size() == k_);

    // get first
    auto kmer_it = kmer.begin();

    uint64_t last = *kmer_it + 1 < F.size()
                    ? F.at(*kmer_it + 1)
                    : W->size() - 1;
    uint64_t shift = 0;

    // update range iteratively while scanning through s
    while (++kmer_it != kmer.end()) {
        TAlphabet s = *kmer_it;
        assert(s < alph_size);

        uint64_t last_target = std::max(pred_W(last, s),
                                        pred_W(last, s + alph_size));
        if (last_target > 0) {
            if (rank_last(last_target - 1) < rank_last(last - 1))
                shift = 0;
            last = succ_last(outgoing(last_target, s));
            continue;
        }
        assert(s > 0);

        last_target = std::min(succ_W(last, s),
                               succ_W(last, s + alph_size));

        if (last_target < W->size()) {
            last = succ_last(outgoing(last_target, s));
            shift = 1;
        } else {
            last = F[s];
            shift = 0;
        }
    }

    assert(pred_last(last - shift) > 0);
    return pred_last(last - shift);
}


/**
 * Given index i of a node and a value k, this function
 * will return the k-th last character of node i.
 */
std::pair<TAlphabet, uint64_t> DBG_succ::get_minus_k_value(uint64_t i, uint64_t k) const {
    assert(i < W->size());

    for (; k > 0; --k) {
        i = bwd(succ_last(i));
    }
    return std::make_pair(get_node_last_char(i), bwd(succ_last(i)));
}



/**
 * This function gets two node indices and returns whether the
 * node labels share a k-1 suffix.
 */
bool DBG_succ::compare_node_suffix(uint64_t i1, uint64_t i2) const {
    for (size_t ii = 0; ii < k_ - 1; ++ii) {
        if (get_node_last_char(i1) != get_node_last_char(i2)) {
            return false;
        }
        i1 = bwd(succ_last(i1));
        i2 = bwd(succ_last(i2));
    }
    return true;
}

bool DBG_succ::compare_node_suffix(TAlphabet *ref, uint64_t i2) const {
    TAlphabet *i1 = &ref[k_ - 1];
    for (size_t ii=0; ii < k_ - 1; ii++) {
        if (*i1 != get_node_last_char(i2)) {
            return false;
        }
        i1 = &ref[k_ - 2 - ii];
        i2 = bwd(succ_last(i2));
    }
    return true;
}

/**
 * This function returns true if node i is a terminal node.
 */
bool DBG_succ::is_terminal_node(uint64_t i) const {
    for (size_t ii = 0; ii < k_ - 1; ii++) {
        if (get_node_last_char(i) % alph_size != 0) {
            return false;
        }
        i = bwd(i);
    }
    return true;
}

/**
* Given a node index k_node, this function returns the k-mer sequence of the
* node in a deque data structure.
*/
std::deque<TAlphabet> DBG_succ::get_node_seq(uint64_t k_node) const {
    assert(k_node < W->size());

    std::deque<TAlphabet> ret;

    for (uint64_t curr_k = 0; curr_k < k_; ++curr_k) {
        assert(k_node < W->size());
        auto k_val = get_minus_k_value(k_node, 0);
        ret.push_front(k_val.first);
        k_node = k_val.second;
    }

    return ret;
}

/**
* Given a node index k_node, this function returns the k-mer sequence of the
* node as a string.
*/
std::string DBG_succ::get_node_str(uint64_t k_node) const {
    assert(k_node < W->size());

    std::string node_string(k_, 0);

    auto node_encoding = get_node_seq(k_node);

    std::transform(node_encoding.begin(), node_encoding.end(),
                   node_string.begin(), decode);
    return node_string;
}

std::vector<uint64_t> DBG_succ::align(const std::string &sequence,
                                      uint64_t alignment_length) const {
    std::vector<uint64_t> indices;

    if (alignment_length == 0)
        alignment_length = k_;

    std::vector<HitInfo> curr_result;
    for (uint64_t i = 0; i < sequence.size() - alignment_length + 1; ++i) {
        std::string kmer(sequence.data() + i, sequence.data() + i + alignment_length);
        indices.push_back(this->index(kmer));
    }

    return indices;
}

std::vector<std::vector<HitInfo>> DBG_succ::align_fuzzy(const std::string &sequence,
                                                        uint64_t alignment_length,
                                                        uint64_t max_distance) const {
    std::vector<std::vector<HitInfo>> hit_list;

    if (alignment_length == 0) {

    } else {
        alignment_length = alignment_length < 2 ? 2 : alignment_length;
        alignment_length = alignment_length < sequence.size()
                                    ? alignment_length
                                    : sequence.size();
        for (uint64_t i = 0; i < sequence.size() - alignment_length + 1; ++i) {
            std::string kmer(sequence.data() + i, sequence.data() + i + alignment_length);
            hit_list.push_back(this->index_fuzzy(kmer, max_distance));
        }
    }
    return hit_list;
}

/**
 * Returns the number of nodes on the current graph.
 */
uint64_t DBG_succ::num_nodes() const {
    return rank_last(last->size() - 1);
}

/**
 * Return the number of edges in the current graph.
 */
uint64_t DBG_succ::num_edges() const {
    return W->size() - 1;
}

/**
 * This function gets a value of the alphabet c and updates the offset of
 * all following values by +1 is positive is true and by -1 otherwise.
 */
void DBG_succ::update_F(TAlphabet c, int value) {
    assert(c < alph_size);
    assert(std::abs(value) == 1);

    for (TAlphabet i = c + 1; i < F.size(); i++) {
        F[i] += value;
    }
}

/**
 * This function gets a local range in W from lower bound l
 * to upper bound u and swaps the inserted element to the
 * righ location.
 */
//TODO: this function can be improved
//TODO: fix_order_in_W_range
void DBG_succ::sort_W_locally(uint64_t l, uint64_t u) {
    assert(l < W->size());
    assert(u < W->size());
    assert(state == Config::DYN);

    for (uint64_t s = u; s > l; --s) {
        auto first = get_W(s - 1);
        auto second = get_W(s);
        if ((second % alph_size) < (first % alph_size)) {
            W->set(s - 1, second);
            W->set(s, first);
        }
    }
    for (uint64_t s = l; s < u; ++s) {
        auto first = get_W(s);
        auto second = get_W(s + 1);
        if ((first % alph_size) > (second % alph_size)) {
            W->set(s + 1, first);
            W->set(s, second);
        }
    }
}

TAlphabet DBG_succ::encode(char s) {
    return kCharToNucleotide[static_cast<int>(s)];
}

char DBG_succ::decode(TAlphabet c) {
    assert(c < alphabet.size());
    return alphabet[c];
}

void DBG_succ::switch_state(Config::StateType new_state) {

    //std::cerr << "switching state from " << this->state << " to " << state << std::endl;
    if (state == new_state)
        return;

    switch (new_state) {
        case Config::DYN: {
            wavelet_tree *W_new = new wavelet_tree_dyn(4, *W);
            delete W;
            W = W_new;

            bit_vector *last_new = new bit_vector_dyn(*last);
            delete last;
            last = last_new;

            break;
        }
        case Config::STAT: {
            wavelet_tree *W_new = new wavelet_tree_stat(4, *W);
            delete W;
            W = W_new;

            bit_vector *last_new = new bit_vector_stat(*last);
            delete last;
            last = last_new;

            break;
        }
    }
    state = new_state;
}

void DBG_succ::print_state(std::ostream &os) const {
    os << "Index" << "\t" << "L"
                  << "\t" << "Vertex"
                  << "\t" << "W" << std::endl;

    for (uint64_t i = 1; i < W->size(); i++) {
        os << i << "\t" << get_last(i)
                << "\t" << get_node_str(i)
                << "\t" << decode(get_W(i))
                        << (get_W(i) >= alph_size
                                ? "-"
                                : "")
                        << std::endl;
    }
}

/*
 * Returns the sequence stored in W and prints the node
 * information in an overview.
 * Useful for debugging purposes.
 */
void DBG_succ::print_seq() const {

    const uint64_t linelen = 80;

    for (uint64_t start = 1; start < W->size(); start += linelen) {
        uint64_t end = start + linelen < W->size() ? start + linelen : W->size();

        for (uint64_t i = start; i < end; i++) {
            if (i % 10 == 0)
                fprintf(stdout, "%d", static_cast<int>((i / 10) % 10));
            else
                fprintf(stdout, " ");
        }
        fprintf(stdout, "\n");

        for (uint64_t i = start; i < end; i++) {
            if (get_W(i) >= alph_size)
                fprintf(stdout, "|");
            else
                fprintf(stdout, " ");
        }
        fprintf(stdout, "\n");

        for (uint64_t i = start; i < end; i++) {
            fprintf(stdout, "%c", decode(get_W(i)));
        }
        fprintf(stdout, "\n");

        std::vector<std::string> nodes;
        for (uint64_t i = start; i < end; i++) {
            nodes.push_back(get_node_str(i));
        }
        for (size_t l = 0; l < k_; l++) {
            for (uint64_t i = start; i < end; i++) {
                fprintf(stdout, "%c", nodes[i - start][k_ - l - 1]);
            }
            fprintf(stdout, "\n");
        }
        fprintf(stdout, "\n");

        for (uint64_t i = start; i < end; i++) {
            fprintf(stdout, "%d", get_last(i));
        }
        fprintf(stdout, "\n\n");

        for (uint64_t i = start; i < end; ++i) {
            fprintf(stdout, "%d", static_cast<int>(outdegree(i)));
        }
        fprintf(stdout, "\n");

        for (uint64_t i = start; i < end; ++i) {
            fprintf(stdout, "%d", static_cast<int>(indegree(i)));
        }
        fprintf(stdout, "\n\n");
    }
}

void DBG_succ::print_adj_list(const std::string &filename) const {
    std::ofstream of;
    std::ostream &outstream = filename != ""
                                       ? (of = std::ofstream(filename))
                                       : std::cout;

    for (uint64_t edge = 1; edge < W->size(); ++edge) {
        outstream << rank_last(succ_last(edge))
                  << "\t"
                  << rank_last(outgoing(edge, get_W(edge)))
                  << "\n";
    }
}

///////////////
// Construct //
///////////////

// add a full sequence to the graph
void DBG_succ::add_sequence(const std::string &seq, bool try_extend) {
    if (!seq.size())
        return;

    std::vector<TAlphabet> sequence(seq.size());
    std::transform(seq.begin(), seq.end(), sequence.begin(), encode);

    uint64_t source;

    if (!try_extend || seq.size() <= k_ || !(source = index(sequence))) {
        sequence.insert(sequence.begin(), k_, encode('$'));
        source = kDummySource;
    }

    for (size_t i = 0; i + k_ < sequence.size(); ++i) {
        // print the process
        if (i > 0 && i % 1'000 == 0) {
            std::cout << "." << std::flush;
            if (i % 10'000 == 0)
                verbose_cout(i, " - edges ", num_edges(), " / nodes ", num_nodes(), "\n");
        }

        source = append_pos(sequence[i + k_], source, &sequence[i]);
    }

    verbose_cout("edges ", num_edges(), " / nodes ", num_nodes(), "\n");
}


bool equal_encodings(const char first, const char second) {
    return DBG_succ::encode(first) == DBG_succ::encode(second);
}

void DBG_succ::add_sequence_fast(const std::string &seq,
                                 bool add_bridge, unsigned int parallel,
                                 std::string suffix) {
    // ther is nothing to parse
    if (!seq.size())
        return;

    if (add_bridge) {
        std::deque<char> bridge(k_, '$');
        bridge.push_back(seq[0]);
        for (size_t i = 0; i < std::min(k_, seq.length()); ++i) {
            if (std::equal(suffix.rbegin(), suffix.rend(), bridge.rbegin() + 1,
                           equal_encodings)) {
                kmers.emplace_back(bridge, DBG_succ::encode);
            }
            bridge.pop_front();
            bridge.push_back(i + 1 < seq.length() ? seq[i + 1] : '$');
        }
    }
    if (k_ < seq.length()) {
        #pragma omp parallel num_threads(parallel)
        {
            std::vector<KMer> kmer_priv;
            #pragma omp for nowait
            for (size_t i = 0; i < seq.length() - k_; ++i) {
                if (std::equal(suffix.begin(), suffix.end(),
                               seq.c_str() + i + k_ - suffix.length(),
                               equal_encodings)) {
                    kmer_priv.emplace_back(
                        std::string(seq.c_str() + i, k_ + 1),
                        DBG_succ::encode
                    );
                }
            }
            #pragma omp critical
            kmers.insert(kmers.end(),
                std::make_move_iterator(kmer_priv.begin()),
                std::make_move_iterator(kmer_priv.end())
            );
        }
    }
    if (add_bridge) {
        std::deque<char> bridge(seq.end() - k_, seq.end());
        bridge.push_back('$');
        if (std::equal(suffix.begin(), suffix.end(),
                       bridge.begin() + k_ - suffix.length(),
                       equal_encodings)) {
            kmers.emplace_back(bridge, DBG_succ::encode);
        }
    }
}

void DBG_succ::construct_succ(unsigned int parallel) {

    for (size_t i = 1; i < W->size(); ++i) {
        kmers.emplace_back(get_node_str(i) + DBG_succ::decode(get_W(i)),
                           DBG_succ::encode);
    }

    // parallel sort of all kmers
    omp_set_num_threads(std::max(static_cast<int>(parallel), 1));
    __gnu_parallel::sort(kmers.begin(), kmers.end());

    auto unique_end = std::unique(kmers.begin(), kmers.end());
    kmers.erase(unique_end, kmers.end()); 

    //DEBUG: output kmers
    // for (const auto &kmer : kmers) {
    //     std::cout << kmer.to_string(alphabet) << std::endl;
    // }

    // the bit array indicating the last outgoing edge of a node (static container for full init)
    std::vector<uint8_t> last_stat_safe { 0 };

    last_stat_safe.resize(last_stat_safe.size() + kmers.size(), 1);

    // the array containing the edge labels
    std::vector<TAlphabet> W_stat { 0 };

    size_t curpos = W_stat.size();
    W_stat.resize(W_stat.size() + kmers.size());

    #pragma omp parallel num_threads(parallel)
    {
        #pragma omp for nowait
        for (size_t i = 0; i < kmers.size(); ++i) {
            //set last
            if (i + 1 < kmers.size()) {
                if (KMer::compare_kmer_suffix(kmers[i], kmers[i + 1])) {
                    last_stat_safe[curpos + i] = 0;
                }
            }
            //set W
            uint8_t curW = kmers[i][0];
            if (curW == 127) {
                std::cerr << "Failure decoding kmer " << i << "\n" << kmers[i] << "\n"
                          << kmers[i].to_string(alphabet) << "\n";
                exit(1);
            }
            if (i) {
                for (size_t j = i - 1; KMer::compare_kmer_suffix(kmers[j], kmers[i], 1); --j) {
                    //TODO: recalculating W is probably faster than doing a pragma for ordered
                    if (kmers[j][0] == curW) {
                        curW += alph_size;
                        break;
                    }
                    if (!j)
                        break;
                }
            }
            W_stat[curpos + i] = curW;
        }
    }
    std::vector<bool> last_stat(last_stat_safe.begin(), last_stat_safe.end());

    size_t i;
    size_t lastlet = 0;
    F[0] = 0;
    for (i = 0; i < kmers.size(); ++i) {
        while (alphabet[lastlet] != alphabet[kmers[i][k_]] && lastlet + 1 < alph_size) {
            F[++lastlet] = curpos + i - 1;
        }
    }
    while (++lastlet < alph_size) {
        F[lastlet] = curpos + i - 1;
    }

    kmers.clear();

    delete W;
    W = new wavelet_tree_dyn(4, W_stat);

    delete last;
    last = new bit_vector_dyn(last_stat);
}

/**
 * This function takes a character c and appends it to the end of the graph
 * sequence given that the corresponding note is not part of the graph yet.
 */
uint64_t DBG_succ::append_pos(uint64_t c, uint64_t source_node, TAlphabet *ckmer) {
    assert(source_node < W->size());

    // get range of identical nodes (without W) pos current end position
    uint64_t begin = pred_last(source_node - 1) + 1;
    uint64_t end = succ_last(source_node) + 1;

    // get position of the first occurence of c or c- in W after p
    uint64_t prev_c_pos = std::max(pred_W(end - 1, c),
                                   pred_W(end - 1, c + alph_size));
    // if the character already exists return its index
    if (prev_c_pos >= begin)
        return fwd(prev_c_pos);

    /**
     * We found that c does not yet exist in the current range and now have to
     * figure out if we need to add c or c- to the range.
     * To do this, we check if there is a preceding position j1 with W[j1] == c
     * whose node shares a k-1 suffix with the current node.
     * If yes, we add c- instead of c.
     */

    bool is_first_incoming = true;
    if (prev_c_pos > 0)
        is_first_incoming = ckmer ? !compare_node_suffix(ckmer, prev_c_pos)
                                  : !compare_node_suffix(begin, prev_c_pos);

    if (!is_first_incoming) {
        // insert the edge
        insert_edge(c + alph_size, begin, end);
        return fwd(prev_c_pos);
    }

    // adding a new node can influence one of the following nodes sharing the k-1 suffix
    // get position of the first occurence of c after p (including p + 1)
    uint64_t first_c = end < W->size()
                       ? succ_W(end, c)
                       : W->size();

    bool the_only_incoming = true;
    if (first_c < W->size()) {
        the_only_incoming = ckmer != NULL ? !compare_node_suffix(ckmer, first_c)
                                          : !compare_node_suffix(begin, first_c);
        if (!the_only_incoming) {
            // the inserted edge will not be the first incoming for the target node
            // need to adapt the respective cc to a cc-
            W->set(first_c, c + alph_size);
        }
    }

    // insert the edge
    bool shift = insert_edge(c, begin, end);

    // Add sentinel if the target node is the new dead-end
    if (!the_only_incoming)
        return fwd(first_c + shift);

    uint64_t sentinel_pos = select_last(rank_last(F[c]) + rank_W(begin - 1, c)) + 1;

    update_F(c, +1);
    W->insert(sentinel_pos, encode('$'));
    last->insertBit(sentinel_pos, true);
    return sentinel_pos;
}


bool DBG_succ::insert_edge(TAlphabet c, uint64_t begin, uint64_t end) {
    if (begin > 1 && get_W(begin) == encode('$')) {
        // the source node is the dead-end with outgoing sentinel
        // replace this sentinel with proper label
        W->set(begin, c);
        return 0;
    } else {
        // the source node already has some outgoing edges
        update_F(get_node_last_char(begin), +1);
        W->insert(begin, c);
        last->insertBit(begin, false);
        sort_W_locally(begin, end);
        return 1;
    }
}


// Given an edge list, remove them from the graph.
void DBG_succ::remove_edges(const std::set<uint64_t> &edges) {
    uint64_t shift = 0;

    for (const auto &edge : edges) {
        assert(edge >= shift);
        uint64_t edge_id = edge - shift;

        uint64_t d = get_W(edge_id);
        if (d < alph_size) {
            //fix W array
            uint64_t next = edge_id + 1;
            uint64_t j = succ_W(next, d);
            for (uint64_t i = next; i < j; ++i) {
                if (get_W(i) == d + alph_size) {
                    W->set(i, d);
                    break;
                }
            }
        }
        W->remove(edge_id);
        update_F(get_node_last_char(edge_id), -1);
        // If the current node has multiple outgoing edges,
        // remove one of the 0s from last instead of 1.
        if (get_last(edge_id) && (edge >= shift + 1)
                              && !get_last(edge_id - 1)) {
            last->deleteBit(edge_id - 1);
        } else {
            last->deleteBit(edge_id);
        }
        shift++;
    }
}

/**
 * This object collects information about branches during graph traversal for the
 * purpose of merging, so we know where to jump back to when we reached a dead end.
 */
struct BranchInfoMerge {
    uint64_t node_id;
    std::deque<TAlphabet> source_kmer;
};

/**
* Heavily borrowing from the graph sequence traversal, this function gets 
* a graph pointer Gm and merges its nodes into the target graph object Gt.
* The edges of Gm are fully traversed and nodes are added to Gt if not existing yet.
* This function is well suited to merge small graphs into large ones.
*/
void DBG_succ::merge(const DBG_succ &Gm) {
    // FYI: can be improved to handle different k_mer sizes
    assert(k_ == Gm.get_k());

    // bool vector that keeps track of visited nodes
    std::vector<bool> marked(Gm.num_nodes() + 1, false);

    // start at the source node
    uint64_t Gt_source_node = 1;
    uint64_t Gm_source_node = 1;
    // keep a running list of the last k characters we have seen
    std::deque<TAlphabet> k_mer(Gm.get_k(), DBG_succ::encode('$'));

    // store all branch nodes on the way
    std::stack<BranchInfoMerge> branchnodes;

    uint64_t added_counter = 0;

    // keep traversing until we reach the sink and have worked off all branches from the stack
    while (true) {
        // verbose output
        if (added_counter > 0 && added_counter % 1000 == 0) {
            std::cout << "." << std::flush;
            if (added_counter % 10000 == 0) {
                std::cout << "merged " << added_counter
                          << " / " << Gm.num_edges()
                          << " - edges " << num_edges()
                          << " / nodes " << num_nodes() << "\n";
            }
        }

        k_mer.pop_front();
        k_mer.push_back(0);

        // loop over outgoing edges
        for (TAlphabet c = 1; c < alph_size; ++c) {
            uint64_t Gm_target_node = Gm.outgoing(Gm_source_node, c);
            if (!Gm_target_node)
                continue;

            uint64_t num_all_nodes_old = rank_last(last->size() - 1);

            uint64_t Gt_target_node = append_pos(c, Gt_source_node);
            added_counter++;

            if (rank_last(last->size() - 1) > num_all_nodes_old
                    && Gt_target_node <= Gt_source_node) {
                Gt_source_node++;
            }

            if (marked.at(Gm.rank_last(Gm_target_node)))
                continue;

            k_mer.back() = c;
            branchnodes.push({ Gm_target_node, k_mer });
            marked.at(Gm.rank_last(Gm_target_node)) = true;
        }
        if (!branchnodes.size())
            break;

        // get new node
        BranchInfoMerge &branch = branchnodes.top();
        Gm_source_node = branch.node_id;
        k_mer = branch.source_kmer;
        branchnodes.pop();

        // find node where to restart insertion
        Gt_source_node = index(k_mer);
    }
}
