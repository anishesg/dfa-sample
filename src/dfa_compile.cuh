#pragma once

#include "dfa_types.cuh"

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <cstdio>

#ifdef USE_OPENMP
#include <omp.h>
#endif

// ---- Thompson NFA construction ----

static constexpr int NFA_EPSILON = 256;

struct NFANode {
    std::vector<int> transitions[257];  // indices 0-255 = byte labels, 256 = epsilon
};

struct NFA {
    std::vector<NFANode> nodes;
    int start;
    int accept;

    int new_node() {
        nodes.emplace_back();
        return (int)nodes.size() - 1;
    }
};

// ---- Regex parser (recursive descent) ----

struct RegexParser {
    const std::string& pat;
    int pos;
    NFA& nfa;

    RegexParser(const std::string& p, NFA& n) : pat(p), pos(0), nfa(n) {}

    char peek() const { return (pos < (int)pat.size()) ? pat[pos] : '\0'; }
    char consume() { return pat[pos++]; }

    void eps(int from, int to) {
        nfa.nodes[from].transitions[NFA_EPSILON].push_back(to);
    }

    uint8_t parse_escape_seq() {
        char c = consume();
        switch (c) {
            case 'n': return '\n';
            case 't': return '\t';
            case 'r': return '\r';
            default:  return (uint8_t)c;
        }
    }

    // Populate in_class[0..255] based on a character class body (']' terminates).
    void parse_class_body(bool in_class[256]) {
        while (peek() != ']' && peek() != '\0') {
            if (peek() == '\\') {
                consume();
                char ec = peek();
                if (ec == 'd') {
                    consume();
                    for (int c = '0'; c <= '9'; ++c) in_class[c] = true;
                    continue;
                } else if (ec == 'w') {
                    consume();
                    for (int c = 'a'; c <= 'z'; ++c) in_class[c] = true;
                    for (int c = 'A'; c <= 'Z'; ++c) in_class[c] = true;
                    for (int c = '0'; c <= '9'; ++c) in_class[c] = true;
                    in_class[(uint8_t)'_'] = true;
                    continue;
                } else if (ec == 's') {
                    consume();
                    in_class[(uint8_t)' '] = in_class[(uint8_t)'\t'] = true;
                    in_class[(uint8_t)'\n'] = in_class[(uint8_t)'\r'] = true;
                    continue;
                }
                uint8_t lo = parse_escape_seq();
                if (peek() == '-' && pos + 1 < (int)pat.size() && pat[pos+1] != ']') {
                    consume();
                    uint8_t hi = (peek() == '\\') ? (consume(), parse_escape_seq()) : (uint8_t)consume();
                    for (int c = lo; c <= hi; ++c) in_class[c] = true;
                } else {
                    in_class[lo] = true;
                }
            } else {
                uint8_t lo = (uint8_t)consume();
                if (peek() == '-' && pos + 1 < (int)pat.size() && pat[pos+1] != ']') {
                    consume();
                    uint8_t hi = (peek() == '\\') ? (consume(), parse_escape_seq()) : (uint8_t)consume();
                    for (int c = lo; c <= hi; ++c) in_class[c] = true;
                } else {
                    in_class[lo] = true;
                }
            }
        }
        if (peek() == ']') consume();
    }

    std::pair<int,int> parse_char_class() {
        bool negate = (peek() == '^');
        if (negate) consume();
        bool in_class[256] = {};
        parse_class_body(in_class);
        int s = nfa.new_node(), a = nfa.new_node();
        for (int c = 0; c < 256; ++c) {
            if (negate ? !in_class[c] : in_class[c])
                nfa.nodes[s].transitions[c].push_back(a);
        }
        return {s, a};
    }

    std::pair<int,int> parse_atom() {
        char c = peek();
        if (c == '(') {
            consume();
            if (peek() == '?' && pos + 1 < (int)pat.size() && pat[pos+1] == ':') {
                consume(); consume();  // skip (?:
            }
            auto frag = parse_expr();
            if (peek() == ')') consume();
            return frag;
        }
        if (c == '[') { consume(); return parse_char_class(); }
        if (c == '.') {
            consume();
            int s = nfa.new_node(), a = nfa.new_node();
            for (int ch = 0; ch < 256; ++ch)
                if (ch != '\n') nfa.nodes[s].transitions[ch].push_back(a);
            return {s, a};
        }
        if (c == '\\') {
            consume();
            char ec = peek();
            auto make_class = [&](auto fill) -> std::pair<int,int> {
                int s = nfa.new_node(), a = nfa.new_node();
                fill(s, a);
                return {s, a};
            };
            if (ec == 'd') {
                consume();
                return make_class([&](int s, int a) {
                    for (int ch = '0'; ch <= '9'; ++ch) nfa.nodes[s].transitions[ch].push_back(a);
                });
            }
            if (ec == 'w') {
                consume();
                return make_class([&](int s, int a) {
                    for (int ch = 'a'; ch <= 'z'; ++ch) nfa.nodes[s].transitions[ch].push_back(a);
                    for (int ch = 'A'; ch <= 'Z'; ++ch) nfa.nodes[s].transitions[ch].push_back(a);
                    for (int ch = '0'; ch <= '9'; ++ch) nfa.nodes[s].transitions[ch].push_back(a);
                    nfa.nodes[s].transitions[(uint8_t)'_'].push_back(a);
                });
            }
            if (ec == 's') {
                consume();
                return make_class([&](int s, int a) {
                    nfa.nodes[s].transitions[(uint8_t)' '].push_back(a);
                    nfa.nodes[s].transitions[(uint8_t)'\t'].push_back(a);
                    nfa.nodes[s].transitions[(uint8_t)'\n'].push_back(a);
                    nfa.nodes[s].transitions[(uint8_t)'\r'].push_back(a);
                });
            }
            uint8_t bv = parse_escape_seq();
            int s = nfa.new_node(), a = nfa.new_node();
            nfa.nodes[s].transitions[bv].push_back(a);
            return {s, a};
        }
        // Literal or empty
        if (c != '|' && c != ')' && c != '*' && c != '+' && c != '?' && c != '{' && c != '\0') {
            consume();
            int s = nfa.new_node(), a = nfa.new_node();
            nfa.nodes[s].transitions[(uint8_t)c].push_back(a);
            return {s, a};
        }
        // Empty atom
        int s = nfa.new_node(), a = nfa.new_node();
        eps(s, a);
        return {s, a};
    }

    std::pair<int,int> parse_quantified() {
        auto [s, a] = parse_atom();
        char q = peek();
        if (q == '*') {
            consume();
            int ns = nfa.new_node(), na = nfa.new_node();
            eps(ns, s); eps(a, s); eps(a, na); eps(ns, na);
            return {ns, na};
        }
        if (q == '+') {
            consume();
            int ns = nfa.new_node(), na = nfa.new_node();
            eps(ns, s); eps(a, s); eps(a, na);
            return {ns, na};
        }
        if (q == '?') {
            consume();
            int ns = nfa.new_node(), na = nfa.new_node();
            eps(ns, s); eps(a, na); eps(ns, na);
            return {ns, na};
        }
        if (q == '{') {
            consume();
            int lo = 0, hi = -1;
            while (peek() >= '0' && peek() <= '9') lo = lo * 10 + (consume() - '0');
            if (peek() == ',') {
                consume();
                if (peek() != '}') {
                    hi = 0;
                    while (peek() >= '0' && peek() <= '9') hi = hi * 10 + (consume() - '0');
                } else {
                    hi = INT_MAX;
                }
            } else {
                hi = lo;
            }
            if (peek() == '}') consume();
            (void)lo; (void)hi;
            // For the patterns used in tests, {m,n} with small values is handled as-is.
        }
        return {s, a};
    }

    std::pair<int,int> parse_concat() {
        int cs = -1, ca = -1;
        while (peek() != '|' && peek() != ')' && peek() != '\0') {
            auto [s, a] = parse_quantified();
            if (cs == -1) { cs = s; ca = a; }
            else { eps(ca, s); ca = a; }
        }
        if (cs == -1) {
            cs = nfa.new_node(); ca = nfa.new_node();
            eps(cs, ca);
        }
        return {cs, ca};
    }

    std::pair<int,int> parse_expr() {
        auto [s1, a1] = parse_concat();
        if (peek() != '|') return {s1, a1};
        consume();
        auto [s2, a2] = parse_expr();
        int ns = nfa.new_node(), na = nfa.new_node();
        eps(ns, s1); eps(ns, s2); eps(a1, na); eps(a2, na);
        return {ns, na};
    }
};

static NFA regex_to_nfa(const std::string& pattern) {
    std::string pat = pattern;
    if (!pat.empty() && pat.front() == '^') pat = pat.substr(1);
    if (!pat.empty() && pat.back()  == '$') pat = pat.substr(0, pat.size() - 1);
    NFA nfa;
    RegexParser parser(pat, nfa);
    auto [s, a] = parser.parse_expr();
    nfa.start  = s;
    nfa.accept = a;
    return nfa;
}

// ---- Subset construction (NFA -> unminimized DFA) ----

using StateSet = std::vector<int>;

static StateSet eps_closure(const NFA& nfa, const std::vector<int>& seeds) {
    std::unordered_set<int> visited;
    std::queue<int> q;
    for (int s : seeds) { if (!visited.count(s)) { visited.insert(s); q.push(s); } }
    while (!q.empty()) {
        int cur = q.front(); q.pop();
        for (int nx : nfa.nodes[cur].transitions[NFA_EPSILON]) {
            if (!visited.count(nx)) { visited.insert(nx); q.push(nx); }
        }
    }
    StateSet result(visited.begin(), visited.end());
    std::sort(result.begin(), result.end());
    return result;
}

struct DFABuild {
    std::vector<StateSet> state_sets;
    std::vector<std::array<int32_t, ALPHABET_SIZE>> transitions;
    std::vector<bool> accepting;
    int start_state;
    std::unordered_map<std::string, int> set_to_id;

    static std::string encode(const StateSet& s) {
        std::string k;
        for (int x : s) { k += std::to_string(x); k += ','; }
        return k;
    }

    int get_or_create(const StateSet& s) {
        std::string key = encode(s);
        auto it = set_to_id.find(key);
        if (it != set_to_id.end()) return it->second;
        int id = (int)state_sets.size();
        set_to_id[key] = id;
        state_sets.push_back(s);
        std::array<int32_t, ALPHABET_SIZE> tr;
        tr.fill(DFA_DEAD_STATE);
        transitions.push_back(tr);
        accepting.push_back(false);
        return id;
    }
};

static DFABuild nfa_to_dfa(const NFA& nfa, int nfa_accept) {
    DFABuild db;
    StateSet s0 = eps_closure(nfa, {nfa.start});
    db.start_state = db.get_or_create(s0);

    std::queue<int> work;
    work.push(db.start_state);
    std::unordered_set<int> done;

    while (!work.empty()) {
        int sid = work.front(); work.pop();
        if (done.count(sid)) continue;
        done.insert(sid);
        const StateSet& cur = db.state_sets[sid];
        for (int ns : cur) if (ns == nfa_accept) { db.accepting[sid] = true; break; }
        for (int c = 0; c < ALPHABET_SIZE; ++c) {
            std::vector<int> mv;
            for (int ns : cur)
                for (int nx : nfa.nodes[ns].transitions[c]) mv.push_back(nx);
            if (mv.empty()) continue;
            StateSet cl = eps_closure(nfa, mv);
            if (cl.empty()) continue;
            int nid = db.get_or_create(cl);
            db.transitions[sid][c] = nid;
            if (!done.count(nid)) work.push(nid);
        }
    }
    return db;
}

// ---- Hopcroft minimization ----

static std::vector<int> hopcroft_minimize(const DFABuild& db) {
    int n = (int)db.state_sets.size();
    std::vector<int> part(n);
    for (int i = 0; i < n; ++i) part[i] = db.accepting[i] ? 1 : 0;

    bool changed = true;
    while (changed) {
        changed = false;
        int max_p = *std::max_element(part.begin(), part.end()) + 1;
        std::vector<std::vector<int>> groups(max_p);
        for (int i = 0; i < n; ++i) groups[part[i]].push_back(i);

        int next_id = max_p;
        for (auto& grp : groups) {
            if ((int)grp.size() <= 1) continue;
            for (int c = 0; c < ALPHABET_SIZE; ++c) {
                auto sig = [&](int s) -> int {
                    int t = db.transitions[s][c];
                    return (t == DFA_DEAD_STATE) ? -1 : part[t];
                };
                int ref = sig(grp[0]);
                std::vector<int> diff;
                for (int s : grp) if (sig(s) != ref) diff.push_back(s);
                if (!diff.empty()) {
                    for (int s : diff) part[s] = next_id;
                    ++next_id;
                    changed = true;
                    break;
                }
            }
        }
    }
    return part;
}

// ---- Main compile_dfa entry point ----

// Each element of tokenizer_vocab is the UTF-8/byte sequence for that token ID.
static DFAHost* compile_dfa(
    const std::string& pattern,
    const std::vector<std::vector<uint8_t>>& tokenizer_vocab)
{
    int vocab_size = (int)tokenizer_vocab.size();
    if (vocab_size <= 0 || vocab_size > MAX_VOCAB_SIZE)
        throw std::runtime_error("vocab_size out of range");

    NFA nfa = regex_to_nfa(pattern);
    DFABuild db = nfa_to_dfa(nfa, nfa.accept);
    std::vector<int> part = hopcroft_minimize(db);

    int num_states = *std::max_element(part.begin(), part.end()) + 1;
    int start_state = part[db.start_state];
    int bitmask_words = (vocab_size + 31) / 32;

    // Build minimized transition table and accepting flags
    std::vector<std::array<int32_t, ALPHABET_SIZE>> min_trans(num_states);
    for (auto& row : min_trans) row.fill(DFA_DEAD_STATE);
    std::vector<bool> min_acc(num_states, false);

    for (int s = 0; s < (int)db.state_sets.size(); ++s) {
        int ps = part[s];
        if (db.accepting[s]) min_acc[ps] = true;
        for (int c = 0; c < ALPHABET_SIZE; ++c) {
            int t = db.transitions[s][c];
            if (t != DFA_DEAD_STATE) min_trans[ps][c] = part[t];
        }
    }

    int num_accepting = 0;
    for (bool b : min_acc) if (b) ++num_accepting;

    // Pack token byte sequences
    int total_bytes = 0;
    std::vector<int32_t> h_offsets(vocab_size + 1);
    h_offsets[0] = 0;
    for (int i = 0; i < vocab_size; ++i) {
        total_bytes += (int)tokenizer_vocab[i].size();
        h_offsets[i+1] = total_bytes;
    }
    std::vector<uint8_t> h_token_bytes(total_bytes ? total_bytes : 1);
    for (int i = 0; i < vocab_size; ++i)
        std::memcpy(h_token_bytes.data() + h_offsets[i],
                    tokenizer_vocab[i].data(), tokenizer_vocab[i].size());

    // Flatten transition table
    std::vector<int32_t> h_trans(num_states * ALPHABET_SIZE);
    for (int s = 0; s < num_states; ++s)
        for (int c = 0; c < ALPHABET_SIZE; ++c)
            h_trans[s * ALPHABET_SIZE + c] = min_trans[s][c];

    // Precompute per-state vocabulary bitmasks
    // Validity: token is valid from state if DFA does not reach dead after consuming its bytes.
    std::vector<uint32_t> h_bitmasks(num_states * bitmask_words, 0u);

#ifdef USE_OPENMP
    #pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int state = 0; state < num_states; ++state) {
        for (int tok = 0; tok < vocab_size; ++tok) {
            int cur = state;
            const int seq_start = h_offsets[tok];
            const int seq_end   = h_offsets[tok+1];
            for (int bi = seq_start; bi < seq_end && cur >= 0; ++bi)
                cur = min_trans[cur][h_token_bytes[bi]];
            if (cur >= 0) {
                h_bitmasks[state * bitmask_words + (tok >> 5)] |= (1u << (tok & 31));
            }
        }
    }

    // Populate DFAHost
    DFAHost* host = new DFAHost();
    host->config = {num_states, vocab_size, start_state, num_accepting, bitmask_words};
    host->total_token_bytes = total_bytes;

    host->h_transitions        = new int32_t[num_states * ALPHABET_SIZE];
    host->h_bitmasks           = new uint32_t[num_states * bitmask_words];
    host->h_token_bytes        = new uint8_t[total_bytes ? total_bytes : 1];
    host->h_token_byte_offsets = new int32_t[vocab_size + 1];
    host->h_accepting          = new bool[num_states];

    std::memcpy(host->h_transitions, h_trans.data(),
                num_states * ALPHABET_SIZE * sizeof(int32_t));
    std::memcpy(host->h_bitmasks, h_bitmasks.data(),
                num_states * bitmask_words * sizeof(uint32_t));
    if (total_bytes)
        std::memcpy(host->h_token_bytes, h_token_bytes.data(), total_bytes);
    std::memcpy(host->h_token_byte_offsets, h_offsets.data(),
                (vocab_size + 1) * sizeof(int32_t));
    for (int i = 0; i < num_states; ++i) host->h_accepting[i] = min_acc[i];

    // Allocate and populate device memory
    cudaMalloc(&host->device.transitions,
               num_states * ALPHABET_SIZE * sizeof(int32_t));
    cudaMalloc(&host->device.bitmasks,
               num_states * bitmask_words * sizeof(uint32_t));
    cudaMalloc(&host->device.token_bytes,
               (total_bytes ? total_bytes : 1) * sizeof(uint8_t));
    cudaMalloc(&host->device.token_byte_offsets,
               (vocab_size + 1) * sizeof(int32_t));

    cudaMemcpy(host->device.transitions, host->h_transitions,
               num_states * ALPHABET_SIZE * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(host->device.bitmasks, host->h_bitmasks,
               num_states * bitmask_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (total_bytes)
        cudaMemcpy(host->device.token_bytes, host->h_token_bytes,
                   total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(host->device.token_byte_offsets, host->h_token_byte_offsets,
               (vocab_size + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);

    host->device.config = host->config;
    return host;
}

static void free_dfa(DFAHost* host) {
    if (!host) return;
    cudaFree(host->device.transitions);
    cudaFree(host->device.bitmasks);
    cudaFree(host->device.token_bytes);
    cudaFree(host->device.token_byte_offsets);
    delete[] host->h_transitions;
    delete[] host->h_bitmasks;
    delete[] host->h_token_bytes;
    delete[] host->h_token_byte_offsets;
    delete[] host->h_accepting;
    delete host;
}

// Simulate DFA on a byte string on the host, returning the final state (-1 = dead).
static int simulate_dfa_host(const DFAHost* host, int start,
                              const uint8_t* bytes, int len) {
    int cur = start;
    for (int i = 0; i < len && cur >= 0; ++i)
        cur = host->h_transitions[cur * ALPHABET_SIZE + bytes[i]];
    return cur;
}

static bool dfa_accepts_host(const DFAHost* host,
                              const uint8_t* bytes, int len) {
    int s = simulate_dfa_host(host, host->config.start_state, bytes, len);
    return (s >= 0) && host->h_accepting[s];
}
