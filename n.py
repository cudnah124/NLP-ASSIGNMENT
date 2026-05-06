import json
from collections import Counter

with open("BTL2\\data\\ner_training_data.json") as f:
    data = json.load(f)

# ── Check duplicate clauses ───────────────────────────────────────────────────
text_counter = Counter(item["text"] for item in data)
duplicates = {text: count for text, count in text_counter.items() if count > 1}

print(f"Duplicate clause texts found: {len(duplicates)}")
for text, count in duplicates.items():
    print(f"  ({count}x) {text[:80]}{'...' if len(text) > 80 else ''}")

# ── Remove duplicates (keep first occurrence) ─────────────────────────────────
seen = set()
deduped = []
for item in data:
    if item["text"] not in seen:
        seen.add(item["text"])
        deduped.append(item)

removed = len(data) - len(deduped)
print(f"\nClauses before dedup : {len(data)}")
print(f"Clauses after  dedup : {len(deduped)}")
print(f"Removed              : {removed}")

data = deduped

# ── Resolve overlapping spans ─────────────────────────────────────────────────
def resolve_overlaps(entities):
    if not entities:
        return entities
    ents = sorted(entities, key=lambda x: (x[0], -(x[1] - x[0])))
    kept = [ents[0]]
    for curr in ents[1:]:
        prev = kept[-1]
        if curr[0] < prev[1]:
            if (curr[1] - curr[0]) > (prev[1] - prev[0]):
                kept[-1] = curr
        else:
            kept.append(curr)
    return kept

fixed = 0
for item in data:
    original = item["entities"]
    cleaned  = resolve_overlaps(sorted(original, key=lambda x: x[0]))
    if len(cleaned) != len(original):
        fixed += len(original) - len(cleaned)
    item["entities"] = cleaned

# ── Stats ─────────────────────────────────────────────────────────────────────
label_counter = Counter()
for item in data:
    for ent in item["entities"]:
        label_counter[ent[2]] += 1

print(f"\nEntity distribution:")
for label, count in sorted(label_counter.items(), key=lambda x: -x[1]):
    print(f"  {label:10s}: {count}")
print(f"\nTotal entities        : {sum(label_counter.values())}")
print(f"Overlap spans removed : {fixed}")

# ── Verify no overlaps remain ─────────────────────────────────────────────────
overlaps = 0
for item in data:
    ents = sorted(item["entities"], key=lambda x: x[0])
    for i in range(len(ents) - 1):
        if ents[i][1] > ents[i+1][0]:
            overlaps += 1
            print(f"OVERLAP STILL: {ents[i]} vs {ents[i+1]} in: {item['text'][:80]}")
print(f"Overlapping spans after fix: {overlaps}")

# ── Save ──────────────────────────────────────────────────────────────────────
with open("BTL2\\data\\ner_training_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)