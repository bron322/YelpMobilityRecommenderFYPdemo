# convert_user_profiles.py
import json
import re
import difflib
from collections import Counter, defaultdict
import pandas as pd


# -----------------------
# Helpers
# -----------------------
VISITED_RE = re.compile(r"^- Visited\s+(.*?)(?:\s*\(|$)", re.MULTILINE)

def parse_visited_names(history_text: str) -> list[str]:
    """
    Extract visited restaurant names from:
      User History:
      - Visited Luckys Steakhouse (Categories...)
    Returns list of names (strings).
    """
    if not history_text:
        return []
    names = [m.strip() for m in VISITED_RE.findall(history_text)]
    # de-dup while preserving order
    seen = set()
    out = []
    for n in names:
        key = n.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(n)
    return out

CITY_STATE_RE = re.compile(r"City:\s*([^,]+),\s*([A-Z]{2})\.")

def extract_city_state_from_rag_text(rag_text: str):
    m = CITY_STATE_RE.search(str(rag_text or ""))
    if not m:
        return None, None
    return m.group(1).strip(), m.group(2).strip()


def norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def build_name_index(df: pd.DataFrame):
    """
    Build:
      - exact_map: norm_name -> list of row dicts (handles duplicate names)
      - all_names: list of norm_name for difflib fallback
    """
    exact_map = defaultdict(list)
    for _, r in df.iterrows():
        n = norm_name(r.get("name", ""))
        if not n:
            continue
        exact_map[n].append({
            "business_id": str(r.get("business_id", "")),
            "name": str(r.get("name", "")),
            "city": r.get("city"),
            "state": r.get("state"),
        })
    all_names = list(exact_map.keys())
    return exact_map, all_names


def pick_best_candidate(cands: list[dict]) -> dict | None:
    """
    If duplicate names exist, prefer the candidate with city/state present.
    Otherwise pick first.
    """
    if not cands:
        return None
    for c in cands:
        if c.get("city") and c.get("state"):
            return c
    return cands[0]


def match_name_to_business(name: str, exact_map, all_names, cutoff=0.88):
    """
    Try exact match on normalized name.
    Else fuzzy match with difflib.
    Returns candidate dict or None.
    """
    key = norm_name(name)
    if key in exact_map:
        return pick_best_candidate(exact_map[key])

    # fuzzy fallback
    close = difflib.get_close_matches(key, all_names, n=1, cutoff=cutoff)
    if close:
        return pick_best_candidate(exact_map[close[0]])

    return None


# -----------------------
# Main conversion
# -----------------------
def convert(
    user_profiles_path: str,
    restaurant_rag_csv_path: str,
    out_path: str,
    fuzzy_cutoff: float = 0.88,
):
    # Load user profiles (old format)
    with open(user_profiles_path, "r", encoding="utf-8") as f:
        raw_profiles = json.load(f)

    # Load restaurant RAG data, parse city/state from rag_text
    df = pd.read_csv(restaurant_rag_csv_path)
    if "city" not in df.columns or "state" not in df.columns:
        df["city"], df["state"] = zip(*df["rag_text"].fillna("").map(extract_city_state_from_rag_text))

    exact_map, all_names = build_name_index(df)

    new_profiles = {}
    stats = {
        "users_total": 0,
        "users_with_visits": 0,
        "total_visit_names": 0,
        "matched_visits": 0,
        "unmatched_visits": 0,
    }

    for user_id, history_text in raw_profiles.items():
        stats["users_total"] += 1

        visited_names = parse_visited_names(history_text)
        if visited_names:
            stats["users_with_visits"] += 1
        stats["total_visit_names"] += len(visited_names)

        visited_business_ids = []
        visited_city_states = []
        unmatched = []

        for n in visited_names:
            cand = match_name_to_business(n, exact_map, all_names, cutoff=fuzzy_cutoff)
            if cand:
                visited_business_ids.append(cand["business_id"])
                if cand.get("city") and cand.get("state"):
                    visited_city_states.append((cand["city"], cand["state"]))
                stats["matched_visits"] += 1
            else:
                unmatched.append(n)
                stats["unmatched_visits"] += 1

        # Infer top city/state from matched visits
        top_city, top_state = None, None
        city_state_counts = {}
        if visited_city_states:
            c = Counter(visited_city_states)
            (top_city, top_state), top_cnt = c.most_common(1)[0]
            city_state_counts = {f"{city}, {state}": cnt for (city, state), cnt in c.most_common()}

        new_profiles[user_id] = {
            "history_text": history_text,
            "visited_names": visited_names,
            "visited_business_ids": visited_business_ids,
            "visited_city_states": [{"city": city, "state": state} for (city, state) in visited_city_states],
            "top_city": top_city,
            "top_state": top_state,
            "city_state_counts": city_state_counts,
            "unmatched_visited_names": unmatched,  # debug so you can improve matching later
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_profiles, f, ensure_ascii=False, indent=2)

    print("Done.")
    print("Output:", out_path)
    print("Stats:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    # Change these to your real paths on Windows
    USER_PROFILES = r"C:\\Users\\lebro\\OneDrive - Nanyang Technological University\\Github\\fyp-demo\\yelp-mobility-dashboard\\public\\data\\user_profiles.json"
    REST_RAG = r"C:\\Users\\lebro\\OneDrive - Nanyang Technological University\\Github\\fyp-demo\\yelp-mobility-dashboard\\public\\data\\restaurant_rag_data.csv"
    OUT = r"C:\\Users\\lebro\\OneDrive - Nanyang Technological University\\Github\\fyp-demo\\yelp-mobility-dashboard\\public\\data\\user_profiles_enriched.json"

    convert(
        user_profiles_path=USER_PROFILES,
        restaurant_rag_csv_path=REST_RAG,
        out_path=OUT,
        fuzzy_cutoff=0.88,
    )