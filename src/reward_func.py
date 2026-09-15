from collections import defaultdict, deque, Counter
import re
from multi_lmlm.constants import (
    ANSWER_START_TOKEN, ANSWER_END_TOKEN, THINKING_START_TOKEN,
    DB_START_TOKEN, DB_SEP_TOKEN, DB_RETRIEVE_TOKEN, DB_END_TOKEN,
)
from eval.metrics import exact_match_score

STEPS_START_TOKEN = "<steps>"
STEPS_END_TOKEN = "</steps>"


def extract_answer_from_tags(text: str):
    """Extract answer from between answer tags."""
    try:
        return text.split(ANSWER_START_TOKEN)[1].split(ANSWER_END_TOKEN)[0]
    except Exception:
        return ""

def em_accuracy(completions, solution, **kwargs):
    """Calculate exact match accuracy for completions."""
    results = []
    for c, s in zip(completions, solution):
        extracted = extract_answer_from_tags(c)
        # Return None if answer extraction failed (empty string)
        if extracted == "" and THINKING_START_TOKEN not in c:
            results.append(None)
        else:
            results.append(1 if exact_match_score(extracted, s) else 0)
    return results

# Note: set the threshold be lower
def db_size_threshold(completions, **kwargs):
    """Reward function that checks if triplet to context character ratio is above 0.017."""
    rewards = []
    contexts = kwargs.get("contexts", [])
    for i, comp in enumerate(completions):
        try:
            triplets = comp.split("\n")
            # Sanity check to ensure this is a db generation (Phase 1)
            if "\t" in triplets[0] and "\t" in triplets[1]:
                # Count number of triplets (non-empty lines with tabs)
                num_triplets = sum(1 for t in triplets if "\t" in t)

                # Calculate context character count
                if i < len(contexts):
                    context = contexts[i]
                    # If context is a list of strings, join them
                    if isinstance(context, list):
                        context_str = "\n\n".join(context)
                    else:
                        context_str = str(context)
                    context_chars = len(context_str)

                    # Calculate ratio and assign reward
                    if context_chars > 0:
                        ratio = num_triplets / context_chars
                        # print(f"DEBUG: Example {i} - Num Triplets: {num_triplets}, Context Chars: {context_chars}, Ratio: {ratio:.4f}")
                        reward = 1 if ratio > 0.005 else 0
                        # reward = 0 # DEBUG - disable size reward
                    else:
                        reward = 0  
                else:
                    reward = None  # Return None instead of 0 - no context available
            else:
                reward = None  # Return None instead of 0 - Phase 2 completion or malformed
        except Exception:
            reward = None  # Return None instead of 0 - if parsing fails
        rewards.append(reward)
    return rewards


def f1_reward(completions, solution, **kwargs):
    """Token-level F1 score between the extracted answer and the gold answer.
    Returns a float in [0, 1] for Phase-2 QA completions, None for Phase-1 triplets."""
    results = []
    for c, s in zip(completions, solution):
        extracted = extract_answer_from_tags(c)
        if extracted == "" and THINKING_START_TOKEN not in c:
            results.append(None)  # no answer format — Phase-1 or malformed
        else:
            pred_tokens = normalize_text(extracted).split()
            gold_tokens = normalize_text(s).split()
            results.append(token_f1(pred_tokens, gold_tokens))
    return results


_PHASE1_COMMENTARY_MARKERS = (
    "Relationship:", "Implicit Fact:", "Triplets:", "Note:", "note:",
    "however,", "Therefore,", "In summary",
)
_EOS_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "<pad>")

FORMAT_PENALTY = -1  # tune here: -0.5 is mild, -1.0 is more aggressive


_DB_TOKEN_PAT = re.compile(
    r"<\|db_entity\|>\s*(.+?)\s*<\|db_relationship\|>\s*(.+?)\s*<\|db_return\|>\s*(.+?)\s*<\|db_end\|>"
)
_PAREN_PAT = re.compile(r"^\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*(.+?)\s*\)\s*$")


def _parse_phase1_triplets(lines: list[str], text: str):
    """Parse Phase 1 triplets, auto-detecting format.

    Tries in order: v6 DB-token → v5 pipe → v1 paren.
    Returns (triplets, is_line_based) where is_line_based=True means
    each non-empty line should correspond to exactly one triplet.
    """
    # v6: <|db_entity|> E <|db_relationship|> R <|db_return|> V <|db_end|>
    if DB_START_TOKEN in text:
        triplets = [
            (m.group(1), m.group(2), m.group(3))
            for m in _DB_TOKEN_PAT.finditer(text)
        ]
        # count lines containing a db_start as "triplet lines"
        triplet_lines = sum(1 for l in lines if DB_START_TOKEN in l)
        return triplets, triplet_lines

    # v5: entity | relationship | value
    pipe_triplets = []
    pipe_triplet_lines = 0
    for line in lines:
        if " | " in line:
            parts = line.split(" | ", 2)
            if len(parts) == 3 and all(p.strip() for p in parts):
                pipe_triplets.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
                pipe_triplet_lines += 1
    if pipe_triplets:
        return pipe_triplets, pipe_triplet_lines

    # v1: (entity, relationship, value)
    paren_triplets = []
    paren_triplet_lines = 0
    for line in lines:
        m = _PAREN_PAT.match(line)
        if m:
            paren_triplets.append((m.group(1), m.group(2), m.group(3)))
            paren_triplet_lines += 1
    return paren_triplets, paren_triplet_lines


def _phase1_format_ok(lines: list[str], text: str) -> bool:
    """Return True if Phase 1 triplet output passes all format checks.

    Compatible with all three zero_rl Phase 1 formats:
        - formatted_zero_rl   : (entity, relationship, value)
        - formatted_zero_rl_v5: entity | relationship | value
        - formatted_zero_rl_v6: <|db_entity|> E <|db_relationship|> R <|db_return|> V <|db_end|>
    """
    if not lines:
        return False

    triplets, triplet_line_count = _parse_phase1_triplets(lines, text)

    # has_triplets
    if not triplets:
        return False

    # no_junk_lines + triplet_line_ratio (≥80% of lines are valid triplets)
    if triplet_line_count < len(lines) * 0.8:
        return False

    # no_commentary
    if any(m in text for m in _PHASE1_COMMENTARY_MARKERS):
        return False

    # no_duplicates
    seen: set[tuple[str, str, str]] = set()
    for t in triplets:
        key = (t[0].lower(), t[1].lower(), t[2].lower())
        if key in seen:
            return False
        seen.add(key)

    # no_nested_parens — only meaningful for paren/pipe formats (v6 uses tags, not parens)
    if DB_START_TOKEN not in text:
        if any(line.startswith("((") for line in lines):
            return False

    return True


def _phase2_format_score(c: str) -> float:
    """
    Single penalty:
    - NO_WELL_FORMED_LOOKUP (-0.5): no complete well-formed lookup found in completion

    Everything else (malformed syntax, missing answer tags) is implicitly
    captured — if the model can't produce even one good lookup, it gets penalized.
    """
    well_formed_lookups = _DB_TOKEN_PAT.findall(c)
    return FORMAT_PENALTY if not well_formed_lookups else 0.0


def format_reward_zero_rl(completions, solution, **kwargs):
    """Format reward for zero_rl two-phase completions.

    Phase 1 (triplet extraction): checks triplet quality — has valid pipe-separated
    triplets, no junk lines, no commentary, no duplicates, no nested parens.

    Phase 2 (QA with DB lookups): two independent -0.5 penalties:
    - NO_LOOKUP: no DB lookup found before <answer>
    - OTHER: structural issues (missing/misordered tags, empty answer, EOS leakage)

    Phase detection: Phase 2 is identified by presence of <steps> or <answer> tags;
    everything else is treated as Phase 1.

    Returns 0.0 if all checks pass, up to -1.0 for Phase 2 failures.
    """
    results = []
    for c in completions:
        lines = [l for l in c.strip().splitlines() if l.strip()]

        # Phase detection: Phase 2 has <steps> or <answer> structural tags
        if STEPS_START_TOKEN in c or ANSWER_START_TOKEN in c:
            results.append(_phase2_format_score(c))
        else:
            results.append(0.0 if _phase1_format_ok(lines, c) else FORMAT_PENALTY)

    return results


def db_coverage_reward(completions, solution, **kwargs):
    """Phase-1 reward: 1 if the gold answer is graph-reachable from the question
    via the parsed triplets (reverse BFS), 0 if not, None for Phase-2 completions."""
    questions = kwargs.get("question", [])
    results = []
    for i, comp in enumerate(completions):
        try:
            lines = [l for l in comp.strip().splitlines() if l.strip()]
            # Detect Phase-1 by presence of tab-separated triplets on the first line
            if lines and "\t" in lines[0]:
                triples = parse_triplets(comp)
                question = questions[i] if i < len(questions) else ""
                answer = solution[i] if i < len(solution) else ""
                reachable, _, _, _ = reverse_bfs_with_path(triples, question, answer)
                results.append(0 if reachable else -0.5)
            else:
                results.append(None)  # Phase-2 QA completion — not applicable
        except Exception:
            results.append(None)
    return results


# ---- basic text utils ----
STOP = {
    "the","a","an","in","on","at","of","for","to","and","or","by","with",
    "what","which","who","whom","where","when","why","how",
    "is","are","was","were","be","been","being",
    "played","play","games","game","home","city","record","population"
}

def norm(x: str) -> str:
    return x.lower().replace(",", "").strip()

def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens_wo_stop(s: str):
    return [t for t in normalize_text(s).split() if t not in STOP]

def token_f1(a_tokens, b_tokens) -> float:
    a = Counter(a_tokens)
    b = Counter(b_tokens)
    overlap = sum((a & b).values())
    if overlap == 0:
        return 0.0
    p = overlap / max(1, sum(a.values()))
    r = overlap / max(1, sum(b.values()))
    return 2 * p * r / (p + r)

# ---- graph utils ----
def build_reverse_graph(triples):
    rev = defaultdict(list)
    for h, r, t in triples:
        rev[t].append((h, r))
    return rev

def reconstruct_path(parent, start):
    path = []
    cur = start
    while parent[cur] is not None:
        nxt, rel = parent[cur]
        path.append((cur, rel, nxt))
        cur = nxt
    return path  # from matched node -> ... -> answer (reverse direction)

# ---- revised search: no question_entities needed ----
def reverse_bfs_with_path(
    triples,
    question: str,
    answer: str,
    max_depth: int = 4,
    f1_thresh: float = 0.35,
    require_overlap: bool = True,
):
    """
    Reverse BFS starting from `answer`. Stop when a visited node has high token-F1 overlap with `question`.
    Returns (reachable: bool, path: list[triples] | None, matched_node: str | None, match_f1: float)
    """
    rev = build_reverse_graph(triples)
    q_tokens = tokens_wo_stop(question)
    q_set = set(q_tokens)

    dq = deque([(answer, 0)])
    parent = {answer: None}

    while dq:
        cur, depth = dq.popleft()

        if depth == 0:
            pass
        else:
            # success: current node matches question by token overlap / F1
            if isinstance(cur, str):
                cur_tokens = tokens_wo_stop(cur)
                if cur_tokens:
                    overlap_ok = (len(q_set.intersection(cur_tokens)) > 0) if require_overlap else True
                    f1 = token_f1(cur_tokens, q_tokens) if overlap_ok else 0.0
                    if f1 >= f1_thresh:
                        return True, reconstruct_path(parent, cur), cur, f1

        if depth == max_depth:
            continue

        for prev, rel in rev.get(cur, []):
            if prev not in parent:
                parent[prev] = (cur, rel)
                dq.append((prev, depth + 1))

    return False, None, None, 0.0


if __name__ == "__main__":
    question = (
        "In 2015 the New York Giants played their home games "
        "in a city with a record population of what in 2010?"
    )

    # try either answer:
    # answer = "8,913"
    answer = "New York City"

    triples = [
        ("2015 New York Giants season", "home games in", "East Rutherford, New Jersey"),
        ("MetLife Stadium", "is located in", "East Rutherford, New Jersey"),
        ("East Rutherford, New Jersey", "is a suburb of", "New York City"),
        ("East Rutherford", "population as of 2010 United States Census", "8,913"),
        ("East Rutherford, New Jersey", "alias", "East Rutherford"),  # manual alias bridge
    ]

    ok, path, matched_node, match_f1 = reverse_bfs_with_path(
        triples,
        question=question,
        answer=answer,
        max_depth=4,
        f1_thresh=0.35,
    )

    print("Reachable?", ok)
    if ok:
        print(f"Matched node: {matched_node}  (F1={match_f1:.3f})")
        print("Supporting path (matched -> ... -> answer):")
        for h, r, t in path:
            print(f"  ('{h}', '{r}') -> {t}")