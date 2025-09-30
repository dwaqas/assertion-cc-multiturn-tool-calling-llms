"""
BFCL V4 (multi_turn_base) confusability selector — RELIABLE.

Changes vs. previous:
- Loads ALL *.json func docs from --func-doc-dir (no brittle fixed map).
- Class-name normalizer + aliases (e.g., 'VehicleControlAPI' -> vehicle_control.json).
- Alternatives are STRICTLY same-class as the gold first function and within involved_classes.
- Returns up to 3 scored 'assertion_candidates' for building assertions.
"""

import argparse, json, math, re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from assertionskit.gen_assertions.io_utils import load_json_or_jsonl

# Dataset-specific helpers;
FUNCNAME_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(")

def get_gold_first(ground_truth_steps: List[List[str]]) -> str:
    if not ground_truth_steps: return ""
    step0 = ground_truth_steps[0]
    if not step0: return ""
    m = FUNCNAME_RE.match(step0[0])
    return m.group(1) if m else ""

def first_user_text(item: Dict[str, Any]) -> str:
    # BFCL question is a list of turns (first turn is usually a list of messages);
    q = item.get("question", [])
    if not q: return ""
    turn0 = q[0]
    if isinstance(turn0, list):
        return "\n".join([str(m.get("content","")) for m in turn0 if isinstance(m, dict)])
    return str(turn0)

# Class name normalization/aliasing;

def _norm(s: str) -> str:
    # Lowercase, remove non-alnum;
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

# Canonical classes from BFCL multi-turn docs;
CANONICAL_CLASSES = {
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI":           "math_api",
    "MessageAPI":        "message_api",
    "TwitterAPI":        "posting_api",
    "TicketAPI":         "ticket_api",
    "TradingBot":        "trading_bot",
    "TravelAPI":         "travel_booking",
    "VehicleControlAPI": "vehicle_control",
}

# Build alias → stem map (accept with/without API suffix, camel/snake/variants);
ALIAS_TO_STEM: Dict[str, str] = {}
for canon, stem in CANONICAL_CLASSES.items():
    variants = {
        canon,
        canon.replace("API", ""),
        stem,
        stem.replace("_", ""),
        "vehiclecontrol", "vehiclecontrolapi" if canon.startswith("Vehicle") else "",
        "postingapi" if stem == "posting_api" else "",
        "twitterapi" if canon == "TwitterAPI" else "",
        "gorillafilesystem" if canon == "GorillaFileSystem" else "",
        "travelapi" if canon == "TravelAPI" else "",
        "tradingbot" if canon == "TradingBot" else "",
        "messageapi" if canon == "MessageAPI" else "",
        "ticketapi" if canon == "TicketAPI" else "",
        "mathapi" if canon == "MathAPI" else "",
    }
    for v in variants:
        if not v: continue
        ALIAS_TO_STEM[_norm(v)] = stem

def resolve_stem_for_class(cls: str, available_stems: set) -> Optional[str]:
    """Map an involved_class to a doc filename stem that actually exists on disk."""
    key = _norm(cls)
    # Direct alias;
    stem = ALIAS_TO_STEM.get(key)
    if stem and stem in available_stems:
        return stem
    # Heuristic: Strip 'api' suffix from key and try to match an available stem;
    k = key[:-3] if key.endswith("api") else key
    for s in available_stems:
        if k == _norm(s) or k == _norm(s.replace("_","")):
            return s
    return None

# Func-doc loader;

def load_all_func_docs(func_doc_dir: Path) -> Tuple[
    Dict[str, Dict[str, Dict[str, Any]]], # by_stem: stem -> {func: {desc, params}}
    Dict[str, Dict[str, Any]] # Flat index: func_name -> {"class_stem", "description", "parameters"}
]:
    """
    Loads *all* *.json files in the directory.

    Returns:
      by_stem: { file_stem : { func_name : {"description": str, "parameters": dict} } }
      flat_index: { func_name : {"class_stem": stem, "description": str, "parameters": dict} }
    """
    by_stem: Dict[str, Dict[str, Dict[str, Any]]] = {}
    flat: Dict[str, Dict[str, Any]] = {}
    for p in func_doc_dir.glob("*.json"):
        stem = p.stem
        try:
            docs = load_json_or_jsonl(p)
        except Exception:
            continue
        fm: Dict[str, Dict[str, Any]] = {}
        for d in docs:
            fn = d.get("name")
            if not fn: continue
            rec = {"description": d.get("description",""), "parameters": d.get("parameters",{})}
            fm[fn] = rec
            # Only index first occurrence per function name (function names are unique within a class);
            if fn not in flat:
                flat[fn] = {"class_stem": stem, **rec}
        by_stem[stem] = fm
    return by_stem, flat

# Lightweight TF-IDF;
WORD_RE = re.compile(r"[A-Za-z0-9]+")
def tok(s: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(s or "")]

def build_tfidf(texts: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    docs_tokens = {k: tok(v) for k, v in texts.items()}
    df = Counter()
    for toks in docs_tokens.values():
        df.update(set(toks))
    n = max(1, len(docs_tokens))
    idf = {t: math.log((1+n)/(1+df[t])) + 1.0 for t in df}
    vecs: Dict[str, Dict[str, float]] = {}
    for k, toks in docs_tokens.items():
        tf = Counter(toks)
        L = max(1, len(toks))
        vecs[k] = {t: (tf[t]/L)*idf[t] for t in tf}
    return vecs

def cosine(v1: Dict[str,float], v2: Dict[str,float]) -> float:
    if not v1 or not v2: return 0.0
    keys = set(v1) | set(v2)
    num = sum(v1.get(k,0.0)*v2.get(k,0.0) for k in keys)
    d1 = math.sqrt(sum(x*x for x in v1.values()))
    d2 = math.sqrt(sum(x*x for x in v2.values()))
    return 0.0 if d1==0.0 or d2==0.0 else num/(d1*d2)

# Core logic;
def prompt_number(bfcl_id: str) -> str:
    m = re.findall(r"\d+", bfcl_id or "")
    return m[-1] if m else (bfcl_id or "")

def build_candidate_pool(involved_classes: List[str],
                        by_stem: Dict[str, Dict[str, Dict[str, Any]]]
                        ) -> Dict[str, Dict[str, Any]]:
    """
    STRICT class restriction:
        returns { func_name:{ "class_stem": stem,
                            "description": desc,
                            "blob": desc + parameter names/descriptions } }
    Only functions from the provided `involved_classes` are included.
    """
    available_stems = set(by_stem.keys())
    pool: Dict[str, Dict[str, Any]] = {}

    for cls in involved_classes or []:
        stem = resolve_stem_for_class(cls, available_stems)
        if not stem: # Unknown class in this dir;
            continue
        funcs = by_stem.get(stem, {})
        for fname, meta in funcs.items():
            params = meta.get("parameters", {})
            param_blob = ""
            if isinstance(params, dict):
                props = params.get("properties") or {}
                if isinstance(props, dict):
                    for k, v in props.items():
                        desc = v.get("description","") if isinstance(v, dict) else ""
                        param_blob += f" {k} {desc}"
            blob = f"{fname}\n{meta.get('description','')}\n{param_blob}".strip()
            pool[fname] = {
                "class_stem": stem,
                "description": meta.get("description",""),
                "blob": blob,
            }
    return pool

def rank_confusable(query_text: str,
                    pool: Dict[str, Dict[str, Any]],
                    gold_first: str,
                    restrict_to_stem: Optional[str],
                    top_k: int) -> List[Tuple[str, float]]:
    """
    Score only pool functions (already class-restricted), optionally filtered to a given class stem.
    Exclude the exact gold function name.
    """
    if not pool: return []
    # Restrict to same-class as gold (if known)
    names = [n for n, r in pool.items()
             if n != gold_first and (restrict_to_stem is None or r.get("class_stem") == restrict_to_stem)]
    if not names: return []

    docs = {name: pool[name]["blob"] for name in names}
    tfidf = build_tfidf(docs | {"__QUERY__": query_text})
    qv = tfidf.get("__QUERY__", {})
    scored = []
    for name in names:
        sv = tfidf.get(name, {})
        scored.append((name, cosine(qv, sv)))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:max(0, top_k)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bfcl", required=True)
    ap.add_argument("--possible-answer", required=True)
    ap.add_argument("--func-doc-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=3)  # cap at 3 as requested
    args = ap.parse_args()

    # Load datasets
    bfcl_items = load_json_or_jsonl(Path(args.bfcl))
    pa_items = load_json_or_jsonl(Path(args.possible_answer))
    by_stem, flat_index = load_all_func_docs(Path(args.func_doc_dir))

    # Index for convenience
    bfcl_by_id = {x.get("id"): x for x in bfcl_items if "id" in x}
    gold_by_id: Dict[str,str] = {}
    for obj in pa_items:
        _id = obj.get("id")
        gt = obj.get("ground_truth")
        if not (_id and isinstance(gt, list)): continue
        gold_by_id[_id] = get_gold_first(gt)

    results: Dict[str, Any] = {}

    for _id, item in bfcl_by_id.items():
        inv = item.get("involved_classes", []) or []
        first_user = first_user_text(item)
        gold = gold_by_id.get(_id, "")

        pool = build_candidate_pool(inv, by_stem)

        # Try to figure out the gold's class/desc either from pool or globally;
        gold_rec_pool = pool.get(gold, {})
        gold_desc = gold_rec_pool.get("description","")
        gold_stem = gold_rec_pool.get("class_stem")

        if (not gold_stem) and gold in flat_index:
            # Found globally (record but we will still restrict picks to involved classes);
            gold_stem = flat_index[gold]["class_stem"]
            gold_desc = gold_desc or flat_index[gold].get("description","")

        # Rank confusables strictly within the same (resolved) class;
        ranked = rank_confusable(first_user, pool, gold, restrict_to_stem=gold_stem, top_k=args.top_k)

        # Prepare output entry (keyed by prompt number);
        key = prompt_number(_id)
        entry: Dict[str, Any] = {
            "involved_classes": inv,
            "pool_size": len(pool),
            "restricted_to_involved_classes": True,
            "gold_first": {
                "func": gold,
                "description": gold_desc,
                # Expose a canonical class name if we can map stem back;
                "class": next((canon for canon, stem in CANONICAL_CLASSES.items() if stem == gold_stem), None)
                        if gold_stem else None,
            },
            "assertion_candidates": []
        }

        # Up to 3 candidates (same-class, scored);
        for fname, score in ranked:
            rec = pool.get(fname, {})
            cstem = rec.get("class_stem")
            canon_class = next((canon for canon, stem in CANONICAL_CLASSES.items() if stem == cstem), None)
            entry["assertion_candidates"].append({
                "func": fname,
                "class": canon_class,
                "description": rec.get("description",""),
                "similarity": round(float(score), 6),
            })

        results[key] = entry

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} entries to {args.out}")

if __name__ == "__main__":
    main()
