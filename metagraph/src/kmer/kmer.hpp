#ifndef __KMER_HPP__
#define __KMER_HPP__

#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <cassert>


template <typename G, int L>
class KMer {
  public:
    typedef G WordType;
    typedef uint64_t CharType;
    static constexpr int kBitsPerChar = L;

    KMer() {}
    template <typename V>
    KMer(const V &arr, size_t k);
    template <typename T>
    KMer(const std::vector<T> &arr) : KMer(arr, arr.size()) {}

    KMer(WordType&& seq) noexcept : seq_(seq) {}
    explicit KMer(const WordType &seq) noexcept : seq_(seq) {}

    // corresponds to the co-lex order of k-mers
    bool operator<(const KMer &other) const { return seq_ < other.seq_; }
    bool operator<=(const KMer &other) const { return seq_ <= other.seq_; }
    bool operator>(const KMer &other) const { return seq_ > other.seq_; }
    bool operator>=(const KMer &other) const { return seq_ >= other.seq_; }
    bool operator==(const KMer &other) const { return seq_ == other.seq_; }
    bool operator!=(const KMer &other) const { return seq_ != other.seq_; }

    inline CharType operator[](size_t i) const;

    std::string to_string(size_t k, const std::string &alphabet) const;

    /**
     * Construct the next k-mer for s[7]s[6]s[5]s[4]s[3]s[2]s[1].
     * next = s[8]s[7]s[6]s[5]s[4]s[3]s[2]
     *      = ( s[8] << k ) + ( kmer >> 1 ).
     */
    inline void to_next(size_t k, WordType edge_label);
    inline void to_prev(size_t k, CharType first_char);

    inline const WordType& data() const { return seq_; }

    template <typename T>
    inline static bool match_suffix(const T *kmer, size_t k, const std::vector<T> &suffix) {
        assert(k > 0);
        assert(k >= suffix.size());
        return suffix.empty()
                || std::equal(suffix.begin(), suffix.end(), kmer + k - suffix.size());
    }

    void print_hex(std::ostream &os) const;

  private:
    static constexpr CharType kFirstCharMask = (1ull << kBitsPerChar) - 1;
    static inline const WordType kAllSetMask = ~(WordType(0ull));
    WordType seq_; // kmer sequence
};


template <typename G, int L>
template <typename V>
KMer<G, L>::KMer(const V &arr, size_t k) : seq_(0) {
    if (k * kBitsPerChar > sizeof(WordType) * 8 || k < 1) {
        std::cerr << "ERROR: Invalid k-mer size "
                  << k << ": must be between 1 and "
                  << sizeof(WordType) * 8 / kBitsPerChar << std::endl;
        exit(1);
    }

    for (int i = k - 1; i > 0; --i) {
        assert(static_cast<CharType>(arr[i]) <= kFirstCharMask
                 && "Too small Digit size for representing the character");

        seq_ |= arr[i];
        seq_ <<= kBitsPerChar;
    }

    assert(static_cast<CharType>(arr[0]) <= kFirstCharMask
            && "Too small Digit size for representing the character");

    seq_ |= arr[0];
}

template <typename G, int L>
void KMer<G, L>::to_next(size_t k, WordType edge_label) {
    assert(edge_label <= static_cast<WordType>(kFirstCharMask));
    assert(k * kBitsPerChar <= sizeof(WordType) * 8);
    seq_ >>= kBitsPerChar;
    seq_ |= edge_label << static_cast<int>(kBitsPerChar * (k - 1));
}

template <typename G, int L>
void KMer<G, L>::to_prev(size_t k, CharType first_char) {
    assert(k * kBitsPerChar <= sizeof(WordType) * 8);
    seq_ <<= kBitsPerChar;
    seq_ &= kAllSetMask >> (sizeof(WordType) * 8 - kBitsPerChar * k);
    seq_ |= first_char;
}

template <typename G, int L>
typename KMer<G, L>::CharType KMer<G, L>::operator[](size_t i) const {
    static_assert(kBitsPerChar <= 64, "Too large digit!");
    assert(kBitsPerChar * (i + 1) <= sizeof(WordType) * 8);
    return static_cast<uint64_t>(seq_ >> static_cast<int>(kBitsPerChar * i))
             & kFirstCharMask;
}


template <typename G, int L>
std::ostream& operator<<(std::ostream &os, const KMer<G, L> &kmer) {
    kmer.print_hex(os);
    return os;
}

#endif // __KMER_HPP__
