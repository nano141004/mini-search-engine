"""
Finite State Transducer (FST) for dictionary compression.

An FST maps term strings to integer term IDs, sharing both prefixes
and suffixes between terms for compact storage. This is more space-efficient
than a plain Python dictionary (used in IdMap) for large vocabularies with
common morphological patterns.

Built using the algorithm from Mihov & Maurel (2001):
"Direct Construction of Minimal Acyclic Subsequential Transducers".

Key properties:
  - Prefix sharing: terms with common prefixes share initial states/edges
  - Suffix sharing: terms with common suffixes share final states/edges
  - Output: integer values accumulated along edge traversals

Example from the lecture slides:
  FST for {mop->2, moth->1, pop->5, star->3, stop->4, top->6}:
  Query "stop": s/3 -> t -> o/1 -> p -> output = 3+0+1+0 = 4
"""


class FSTState:
    """A state in the Finite State Transducer."""
    __slots__ = ['trans', 'is_final', 'final_out']

    def __init__(self):
        self.trans = {}         # char -> (FSTState, int output)
        self.is_final = False
        self.final_out = 0

    def signature(self):
        """Hashable signature for minimization (state equivalence check)."""
        return (
            self.is_final,
            self.final_out,
            tuple(sorted((c, id(s), o) for c, (s, o) in self.trans.items()))
        )


class FST:
    """
    Minimal Acyclic Finite State Transducer.

    Maps sorted string keys to non-negative integer outputs, sharing
    both prefixes and suffixes for compression.
    """

    def __init__(self, initial_state=None):
        self.initial_state = initial_state

    @classmethod
    def build(cls, sorted_pairs):
        """
        Build a minimal FST from sorted (string, int) pairs.

        Uses on-the-fly minimization: as each new word is processed,
        suffix states from the previous word are frozen and deduplicated
        via a register of canonical states.

        The output redistribution ensures that integer outputs are pushed
        as far left as possible along shared prefix edges, with excess
        pushed down to diverging suffixes.

        Parameters
        ----------
        sorted_pairs : iterable of (str, int)
            Must be sorted lexicographically by the string key.
            Values must be non-negative integers.

        Returns
        -------
        FST
        """
        sorted_pairs = list(sorted_pairs)
        if not sorted_pairs:
            return cls(FSTState())

        register = {}   # signature -> frozen FSTState
        temp = [FSTState()]
        prev_word = ""

        def freeze(state):
            """Find an equivalent minimized state, or register this one."""
            sig = state.signature()
            if sig in register:
                return register[sig]
            register[sig] = state
            return state

        for word, output in sorted_pairs:
            # Grow temp array if needed
            while len(temp) <= len(word):
                temp.append(FSTState())

            # Common prefix length with previous word
            pref = 0
            while pref < len(word) and pref < len(prev_word) \
                    and word[pref] == prev_word[pref]:
                pref += 1

            # Step 1: Freeze suffix states of previous word (end -> pref+1)
            for i in range(len(prev_word), pref, -1):
                frozen = freeze(temp[i])
                char = prev_word[i - 1]
                _, old_out = temp[i - 1].trans[char]
                temp[i - 1].trans[char] = (frozen, old_out)

            # Step 2: Redistribute output along common prefix
            # (must happen BEFORE creating new states, so excess only
            #  propagates to old transitions, not the new suffix)
            remaining = output
            for i in range(pref):
                char = word[i]
                target, edge_out = temp[i].trans[char]
                common = min(edge_out, remaining)
                excess = edge_out - common

                if excess > 0:
                    # Push excess to all outgoing transitions of target
                    for c in target.trans:
                        t2, o2 = target.trans[c]
                        target.trans[c] = (t2, o2 + excess)
                    if target.is_final:
                        target.final_out += excess

                temp[i].trans[char] = (target, common)
                remaining -= common

            # Step 3: Create new temp states for current word's suffix
            for i in range(pref + 1, len(word) + 1):
                temp[i] = FSTState()
                temp[i - 1].trans[word[i - 1]] = (temp[i], 0)

            temp[len(word)].is_final = True
            temp[len(word)].final_out = 0

            # Step 4: Place remaining output on first edge of new suffix
            if pref < len(word):
                char = word[pref]
                target, edge_out = temp[pref].trans[char]
                temp[pref].trans[char] = (target, edge_out + remaining)
            else:
                # Current word equals common prefix (word is a prefix of
                # the previous word — valid when words are sorted)
                temp[pref].final_out += remaining

            prev_word = word

        # Freeze all remaining temp states
        for i in range(len(prev_word), 0, -1):
            frozen = freeze(temp[i])
            char = prev_word[i - 1]
            _, old_out = temp[i - 1].trans[char]
            temp[i - 1].trans[char] = (frozen, old_out)

        initial = freeze(temp[0])
        return cls(initial)

    def lookup(self, key):
        """
        Look up a string key and return its integer output.
        Returns None if the key is not in the FST.
        """
        state = self.initial_state
        if state is None:
            return None
        output = 0
        for char in key:
            if char not in state.trans:
                return None
            state, edge_out = state.trans[char]
            output += edge_out
        if state.is_final:
            return output + state.final_out
        return None

    def __contains__(self, key):
        return self.lookup(key) is not None

    def state_count(self):
        """Count unique states in the FST."""
        visited = set()
        stack = [self.initial_state]
        while stack:
            s = stack.pop()
            sid = id(s)
            if sid in visited:
                continue
            visited.add(sid)
            for _, (t, _) in s.trans.items():
                stack.append(t)
        return len(visited)

    def edge_count(self):
        """Count total transitions in the FST."""
        visited = set()
        edges = 0
        stack = [self.initial_state]
        while stack:
            s = stack.pop()
            sid = id(s)
            if sid in visited:
                continue
            visited.add(sid)
            edges += len(s.trans)
            for _, (t, _) in s.trans.items():
                stack.append(t)
        return edges

    def _assign_indices(self):
        """BFS to assign stable integer indices to all states."""
        state_map = {}
        state_list = []
        state_map[id(self.initial_state)] = 0
        state_list.append(self.initial_state)
        head = 0
        while head < len(state_list):
            s = state_list[head]
            head += 1
            for _, (t, _) in sorted(s.trans.items()):
                if id(t) not in state_map:
                    state_map[id(t)] = len(state_list)
                    state_list.append(t)
        return state_map, state_list

    def to_bytes(self):
        """
        Pack FST into a compact binary format.

        Format:
          [4B num_states] [4B initial_idx]
          For each state:
            [1B flags: bit0=is_final] [4B final_out if is_final]
            [2B num_transitions]
            For each transition (sorted by char):
              [2B char_codepoint] [4B target_idx] [4B output]
        """
        import struct
        state_map, state_list = self._assign_indices()
        parts = [struct.pack('<II', len(state_list), state_map[id(self.initial_state)])]

        for s in state_list:
            flags = 1 if s.is_final else 0
            parts.append(struct.pack('<B', flags))
            if s.is_final:
                parts.append(struct.pack('<i', s.final_out))
            sorted_trans = sorted(s.trans.items())
            parts.append(struct.pack('<H', len(sorted_trans)))
            for char, (target, out) in sorted_trans:
                parts.append(struct.pack('<HIi', ord(char),
                                         state_map[id(target)], out))

        return b''.join(parts)

    @classmethod
    def from_bytes(cls, data):
        """Reconstruct FST from compact binary format."""
        import struct
        offset = 0
        num_states, initial_idx = struct.unpack_from('<II', data, offset)
        offset += 8

        objs = [FSTState() for _ in range(num_states)]

        for i in range(num_states):
            flags = struct.unpack_from('<B', data, offset)[0]
            offset += 1
            objs[i].is_final = bool(flags & 1)
            if objs[i].is_final:
                objs[i].final_out = struct.unpack_from('<i', data, offset)[0]
                offset += 4
            num_trans = struct.unpack_from('<H', data, offset)[0]
            offset += 2
            for _ in range(num_trans):
                cp, tidx, out = struct.unpack_from('<HIi', data, offset)
                offset += 10
                objs[i].trans[chr(cp)] = (objs[tidx], out)

        fst = cls()
        fst.initial_state = objs[initial_idx]
        return fst

    def __getstate__(self):
        """Serialize as zlib-compressed compact bytes."""
        import zlib
        return zlib.compress(self.to_bytes())

    def __setstate__(self, data):
        """Reconstruct from zlib-compressed compact bytes."""
        import zlib
        restored = FST.from_bytes(zlib.decompress(data))
        self.initial_state = restored.initial_state


class FSTIdMap:
    """
    Drop-in replacement for IdMap that uses an FST for string->id lookup.

    The FST compresses the dictionary by sharing prefixes and suffixes.
    Provides the same __getitem__ interface as IdMap for string keys:
      - fst_id_map["term"]  -> term_id  (via FST lookup)

    Does NOT store the reverse mapping (int -> string) since retrieval
    code only needs string -> int for term lookups. This saves space
    by not duplicating all term strings alongside the FST graph.

    Unlike IdMap, FSTIdMap is read-only — it cannot assign new IDs.
    Build it from an existing IdMap after indexing is complete.
    """

    def __init__(self):
        self.fst = None
        self._size = 0

    @classmethod
    def from_id_map(cls, id_map):
        """Build an FSTIdMap from a populated IdMap."""
        obj = cls()
        obj._size = len(id_map)
        # Build sorted (string, id) pairs for FST construction
        pairs = sorted(id_map.str_to_id.items())
        obj.fst = FST.build(pairs)
        return obj

    def __len__(self):
        return self._size

    def __getitem__(self, key):
        if isinstance(key, str):
            result = self.fst.lookup(key)
            if result is not None:
                return result
            # Term not in vocabulary — return out-of-range ID
            # (won't match any entry in postings_dict, so safely skipped)
            return self._size
        raise TypeError(
            "FSTIdMap only supports string->int lookup. "
            "Reverse lookup (int->str) is not stored for compression."
        )


if __name__ == "__main__":

    # ---- Test 1: Lecture example ----
    print("Test 1: Lecture slide example (mop, moth, pop, star, stop, top)")
    pairs = [
        ("mop", 2), ("moth", 1), ("pop", 5),
        ("star", 3), ("stop", 4), ("top", 6),
    ]
    fst = FST.build(pairs)
    all_pass = True
    for word, expected in pairs:
        result = fst.lookup(word)
        ok = result == expected
        all_pass &= ok
        print(f"  {'PASS' if ok else 'FAIL'}: lookup('{word}') = {result} (expected {expected})")

    for word in ["mo", "sto", "tops", "xyz", ""]:
        result = fst.lookup(word)
        ok = result is None
        all_pass &= ok
        print(f"  {'PASS' if ok else 'FAIL'}: lookup('{word}') = {result} (expected None)")

    print(f"  States: {fst.state_count()}, Edges: {fst.edge_count()}")
    print(f"  All passed: {all_pass}\n")

    # ---- Test 2: Prefix-of-prefix case ----
    print("Test 2: Prefix-of-prefix (a, ab, abc)")
    pairs2 = [("a", 10), ("ab", 20), ("abc", 30)]
    fst2 = FST.build(pairs2)
    all_pass2 = True
    for word, expected in pairs2:
        result = fst2.lookup(word)
        ok = result == expected
        all_pass2 &= ok
        print(f"  {'PASS' if ok else 'FAIL'}: lookup('{word}') = {result} (expected {expected})")
    print(f"  All passed: {all_pass2}\n")

    # ---- Test 3: FSTIdMap from IdMap ----
    print("Test 3: FSTIdMap round-trip with IdMap")
    from util import IdMap
    id_map = IdMap()
    terms = ["apple", "application", "apply", "banana", "band", "bandana",
             "information", "informative", "automation", "automotive", "automata"]
    for t in terms:
        id_map[t]

    fst_map = FSTIdMap.from_id_map(id_map)
    all_pass3 = True
    for i in range(len(id_map)):
        term = id_map[i]
        ok_fwd = fst_map[term] == id_map[term]
        all_pass3 &= ok_fwd
        if not ok_fwd:
            print(f"  FAIL: '{term}' fst={fst_map[term]} expected={id_map[term]}")

    # Unknown term should return out-of-range ID
    unknown_id = fst_map["zzzzz"]
    ok_unk = unknown_id == len(id_map)
    all_pass3 &= ok_unk
    print(f"  Unknown term -> {unknown_id} (expected {len(id_map)}): {'PASS' if ok_unk else 'FAIL'}")
    print(f"  All passed: {all_pass3}\n")

    # ---- Test 4: Pickle round-trip ----
    print("Test 4: Pickle round-trip (compact binary)")
    import pickle
    import io

    buf = io.BytesIO()
    pickle.dump(fst_map, buf)
    fst_size = buf.tell()

    buf2 = io.BytesIO()
    pickle.dump(id_map, buf2)
    idmap_size = buf2.tell()

    buf.seek(0)
    loaded = pickle.load(buf)
    all_pass4 = True
    for i in range(len(id_map)):
        term = id_map[i]
        ok = loaded[term] == id_map[term]
        all_pass4 &= ok

    print(f"  FSTIdMap pickle size: {fst_size} B")
    print(f"  IdMap pickle size:    {idmap_size} B")
    print(f"  Ratio: {fst_size / idmap_size:.3f}x")
    print(f"  All passed: {all_pass4}\n")

    # ---- Test 5: FST compression stats ----
    print("Test 5: FST compression stats")
    fst_inner = fst_map.fst
    print(f"  Vocabulary size: {len(id_map)} terms")
    print(f"  FST states: {fst_inner.state_count()}")
    print(f"  FST edges:  {fst_inner.edge_count()}")
    total_chars = sum(len(t) for t in terms)
    print(f"  Total chars in vocabulary: {total_chars}")
    print(f"  Edges / total chars: {fst_inner.edge_count() / total_chars:.2f} "
          "(< 1.0 means sharing)")

    # Binary FST size
    fst_bytes = fst_inner.to_bytes()
    print(f"  FST binary size: {len(fst_bytes)} B")
