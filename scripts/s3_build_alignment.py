#!/usr/bin/env python3
"""S3 step 0: build the PlantWild -> canonical label alignment by explicit rules.

The alignment already present in the repository (`E5b_plantwild`) was produced by
sentence-embedding cosine similarity at threshold 0.65, and a later note records
a fuzzy-string variant at threshold 75. Neither is admissible evidence that two
disease labels denote the same pathology, and both were fitted against a
*different* 18-class space, not the 23-class space this project uses. This script
therefore rebuilds the alignment from scratch and does not read either artefact.

Evidence chain, so that no step rests on intuition:

    canonical label  --(labels_clean.json "source" field)-->  PlantVillage
    directory name   --(PlantVillage taxonomy)-->  (crop, disease)

The PlantVillage directory name is the authoritative statement of what a
canonical class *is*: `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` says
the class is grey leaf spot of maize regardless of what the short label
`Corn_Gray_leaf_spot` happens to abbreviate. PlantWild labels are decomposed the
same way, by stripping a known crop prefix, and the two (crop, disease) pairs are
compared under documented normalisation rules only.

Match types follow the taxonomy required by the task. Only `EXACT` and
`NORMALIZED_EXACT` enter the conservative replication set automatically;
everything weaker is written out for a human decision and excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PLANTDIAG = PROJECT.parent / "plantdiag-edge"
sys.path.insert(0, str(PLANTDIAG))

PLANTWILD_ROOT = Path(
    "data/plantwild_v2"
)

MAPPING_VERSION = "s3-alignment-v1"

# Crops recognised in PlantWild label strings. Longest match wins, so
# "bell pepper" is not truncated to "bell". Derived by reading the directory
# listing, not inferred.
CROPS = [
    "bell pepper", "apple", "banana", "basil", "bean", "blueberry", "broccoli",
    "cabbage", "carrot", "cauliflower", "celery", "cherry", "citrus", "coffee",
    "corn", "cucumber", "eggplant", "garlic", "ginger", "grapevine", "grape",
    "lettuce", "maple", "peach", "plum", "potato", "raspberry", "rice",
    "soybean", "squash", "strawberry", "tobacco", "tomato", "wheat", "zucchini",
]

# Crop-name equivalences that are matters of nomenclature, not of judgement.
CROP_SYNONYMS = {
    "corn (maize)": "corn",
    "corn_(maize)": "corn",
    "pepper, bell": "bell pepper",
    "pepper,_bell": "bell pepper",
    "soyabean": "soybean",
    "grapevine": "grape",
}

# Disease-string normalisations. Each entry is a rewriting rule that removes a
# purely descriptive token or resolves a naming convention; none of them equates
# two different pathologies.
DISEASE_NORMALISATION: list[tuple[str, str, str]] = [
    (r"\bleaf\b", "", "organ qualifier: the whole label space is leaf imagery"),
    (r"\bcercospora\b", "", "genus name of the grey leaf spot pathogen, carried in the PlantVillage string"),
    (r"\bnorthern\b", "", "regional qualifier, tracked separately as pathogen-bearing"),
    (r"\bcommon\b", "", "qualifier, tracked separately as pathogen-bearing"),
    (r"\bcedar\b", "", "alternate-host qualifier, tracked separately as pathogen-bearing"),
    (r"\s+", " ", "whitespace"),
]

# Two further rules are applied structurally rather than as regexes, because both
# are properties of how PlantVillage writes its directory names:
#
#  * the crop is often repeated inside the disease field
#    (`Apple___Apple_scab`, `Tomato___Tomato_mosaic_virus`), so the crop token is
#    removed from the disease side once the crop is already known;
#  * two synonymous names of one disease are joined by a space
#    (`Cercospora_leaf_spot Gray_leaf_spot` is a single pathology under two
#    names), which leaves duplicated content words.
#
# Both are handled by comparing the remaining content words as an unordered set.
# That is weaker than string equality and stronger than substring containment:
# "early blight" and "late blight" stay distinct, while "spot gray spot" and
# "gray spot" coincide.
STRUCTURAL_RULES = [
    "crop token removed from the disease field when PlantVillage repeats it",
    "remaining content words compared as an unordered set, which absorbs "
    "PlantVillage's habit of joining two synonymous disease names",
]

# Qualifiers whose removal changes what pathogen is denoted. If a canonical and
# an external label differ only by one of these, the pair is NOT exact — it is
# reported as needing a decision.
SEMANTIC_QUALIFIERS = {"northern", "southern", "common", "cedar"}


def git_hash(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def split_crop(label: str) -> tuple[str, str]:
    """Split a lower-cased label into (crop, remainder)."""
    s = label.lower().replace("_", " ").strip()
    for crop in sorted(CROPS, key=len, reverse=True):
        if s == crop:
            return crop, ""
        if s.startswith(crop + " "):
            return CROP_SYNONYMS.get(crop, crop), s[len(crop) + 1 :]
    return "", s


def parse_plantvillage(source_dir: str) -> tuple[str, str]:
    """`Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` -> (corn, cercospora leaf spot gray leaf spot)."""
    crop_raw, _, disease_raw = source_dir.partition("___")
    crop = crop_raw.lower().replace("_", " ").strip()
    crop = CROP_SYNONYMS.get(crop, crop)
    crop = re.sub(r"\s*\(.*?\)", "", crop).strip()
    crop = CROP_SYNONYMS.get(crop, crop)
    disease = disease_raw.lower().replace("_", " ").strip()
    return crop, disease


def normalise_disease(disease: str, crop: str = "") -> tuple[frozenset[str], set[str]]:
    """Apply documented rewrites; return content-word set and dropped qualifiers."""
    s = " " + disease.lower().strip() + " "
    dropped = {q for q in SEMANTIC_QUALIFIERS if re.search(rf"\b{q}\b", s)}
    if crop:
        for token in crop.split():
            s = re.sub(rf"\b{re.escape(token)}\b", " ", s)
    for pattern, repl, _why in DISEASE_NORMALISATION:
        s = re.sub(pattern, repl, s)
    return frozenset(s.split()), dropped


def classify(
    ext_crop: str, ext_dis: str, can_crop: str, can_dis: str
) -> tuple[str, str, str]:
    """Return (match_type, confidence, evidence)."""
    if ext_crop != can_crop:
        return "NO_MATCH", "high", f"crop differs: '{ext_crop}' vs '{can_crop}'"

    if ext_dis.strip().lower() == can_dis.strip().lower():
        return "EXACT", "high", "crop and disease strings identical before normalisation"

    n_ext, q_ext = normalise_disease(ext_dis, ext_crop)
    n_can, q_can = normalise_disease(can_dis, can_crop)
    s_ext = " ".join(sorted(n_ext)) or "<empty>"
    s_can = " ".join(sorted(n_can)) or "<empty>"

    if n_ext == n_can:
        if q_ext != q_can:
            only = sorted((q_ext | q_can) - (q_ext & q_can))
            return (
                "TAXONOMIC_EQUIVALENT",
                "medium",
                "content words agree but the pair differs by the pathogen-bearing "
                f"qualifier(s) {only}; deciding equivalence needs the pathogen species, "
                "which neither dataset states",
            )
        return (
            "NORMALIZED_EXACT",
            "high",
            f"content words identical after documented normalisation: {{{s_ext}}}",
        )

    if not n_ext or not n_can:
        return "AMBIGUOUS", "low", "one side normalises to an empty disease string"

    if n_ext < n_can or n_can < n_ext:
        return (
            "SYNONYM",
            "low",
            f"one content-word set is a strict subset of the other ({{{s_ext}}} vs "
            f"{{{s_can}}}); containment is not evidence of the same pathology",
        )

    return "NO_MATCH", "high", f"same crop, different disease ({{{s_ext}}} vs {{{s_can}}})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plantwild", type=Path, default=PLANTWILD_ROOT)
    ap.add_argument("--labels", type=Path, default=PLANTDIAG / "configs" / "labels_clean.json")
    ap.add_argument("--out-yaml", type=Path, default=PROJECT / "label_alignment.yaml")
    ap.add_argument("--out-json", type=Path, default=PROJECT / "label_alignment.json")
    args = ap.parse_args()

    spec = json.loads(args.labels.read_text())
    canonical = list(spec["target_labels"])
    src_of = {p["target_label"]: p["source"] for p in spec["pairs"]}

    can_parsed = {}
    for lab in canonical:
        crop, dis = parse_plantvillage(src_of[lab])
        can_parsed[lab] = {"crop": crop, "disease": dis, "pv_source": src_of[lab]}

    pw_classes = sorted(d.name for d in args.plantwild.iterdir() if d.is_dir())
    pw_counts = {c: len(list((args.plantwild / c).glob("*.jpg"))) for c in pw_classes}

    rows = []
    for ext in pw_classes:
        ecrop, edis = split_crop(ext)
        best = None
        for lab, meta in can_parsed.items():
            mt, conf, ev = classify(ecrop, edis, meta["crop"], meta["disease"])
            rank = ["EXACT", "NORMALIZED_EXACT", "TAXONOMIC_EQUIVALENT", "SYNONYM",
                    "AMBIGUOUS", "NO_MATCH"].index(mt)
            if best is None or rank < best[0]:
                best = (rank, lab, mt, conf, ev)
        rank, lab, mt, conf, ev = best
        auto = mt in ("EXACT", "NORMALIZED_EXACT")
        crop_known = ecrop in {m["crop"] for m in can_parsed.values()}
        rows.append({
            "external_label": ext,
            "n_images": pw_counts[ext],
            "external_crop": ecrop,
            "external_disease": edis,
            "canonical_label": lab if mt != "NO_MATCH" else None,
            "canonical_crop": can_parsed[lab]["crop"] if mt != "NO_MATCH" else None,
            "canonical_disease": can_parsed[lab]["disease"] if mt != "NO_MATCH" else None,
            "match_type": mt,
            "confidence": conf,
            "evidence": ev,
            "action": (
                "INCLUDE_CONSERVATIVE" if auto
                else "AUTHOR_DECISION" if mt in ("TAXONOMIC_EQUIVALENT", "SYNONYM", "AMBIGUOUS")
                else "OOD_KNOWN_CROP" if crop_known
                else "OOD_UNSEEN_CROP"
            ),
        })

    # A canonical label must not be claimed by two external classes.
    claimed: dict[str, list[str]] = {}
    for r in rows:
        if r["action"] == "INCLUDE_CONSERVATIVE":
            claimed.setdefault(r["canonical_label"], []).append(r["external_label"])
    for lab, ext_list in claimed.items():
        if len(ext_list) > 1:
            for r in rows:
                if r["external_label"] in ext_list:
                    r["match_type"] = "MANY_TO_ONE"
                    r["action"] = "AUTHOR_DECISION"
                    r["evidence"] += f"; conflict: {ext_list} all map to {lab}"

    conservative = {r["external_label"]: r["canonical_label"]
                    for r in rows if r["action"] == "INCLUDE_CONSERVATIVE"}
    payload = {
        "mapping_version": MAPPING_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git": {"edge-reliability-gate": git_hash(PROJECT), "plantdiag-edge": git_hash(PLANTDIAG)},
        "provenance": {
            "external_dataset": "PlantWild v2 (uqtwei2/PlantWild), 115 classes, 11488 images",
            "external_root": str(args.plantwild),
            "canonical_space": str(args.labels),
            "canonical_n_classes": len(canonical),
            "authority_for_canonical_disease": "PlantVillage source directory in labels_clean.json 'source'",
            "rejected_prior_alignments": [
                "experiments/E5b_plantwild (sentence-embedding cosine, threshold 0.65, 18-class space)",
                "label_mapping_pwv2_fuzzy/sbert/hybrid (rapidfuzz token_sort_ratio, threshold 75)",
            ],
            "rejection_reason": "string or embedding similarity is not evidence of pathological identity, "
                                "and both were fitted against a different label space",
        },
        "rules": {
            "crop_vocabulary": CROPS,
            "crop_synonyms": CROP_SYNONYMS,
            "disease_normalisation": [
                {"pattern": p, "replacement": r, "justification": w}
                for p, r, w in DISEASE_NORMALISATION
            ],
            "structural_rules": STRUCTURAL_RULES,
            "semantic_qualifiers_blocking_exactness": sorted(SEMANTIC_QUALIFIERS),
            "auto_accept_match_types": ["EXACT", "NORMALIZED_EXACT"],
        },
        "counts": {
            "external_classes": len(rows),
            "conservative_included": len(conservative),
            "author_decision_required": sum(1 for r in rows if r["action"] == "AUTHOR_DECISION"),
            "ood_known_crop": sum(1 for r in rows if r["action"] == "OOD_KNOWN_CROP"),
            "ood_unseen_crop": sum(1 for r in rows if r["action"] == "OOD_UNSEEN_CROP"),
            "conservative_images": sum(r["n_images"] for r in rows if r["action"] == "INCLUDE_CONSERVATIVE"),
            "ood_known_crop_images": sum(r["n_images"] for r in rows if r["action"] == "OOD_KNOWN_CROP"),
            "ood_unseen_crop_images": sum(r["n_images"] for r in rows if r["action"] == "OOD_UNSEEN_CROP"),
        },
        "conservative_mapping": conservative,
        "canonical_labels_unmatched": sorted(set(canonical) - set(conservative.values())),
        "rows": rows,
    }

    body = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False)
    args.out_json.write_text(body)
    # Hash covers the mapping content only, not the timestamp, so re-running
    # without a data change reproduces the same identifier.
    stable = json.dumps(
        {"version": MAPPING_VERSION, "rows": rows, "conservative": conservative},
        sort_keys=True, ensure_ascii=False,
    )
    digest = hashlib.sha256(stable.encode()).hexdigest()
    payload["mapping_sha256"] = digest
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    yaml_lines = [
        f"# Generated by scripts/s3_build_alignment.py — do not edit by hand.",
        f"mapping_version: {MAPPING_VERSION}",
        f"mapping_sha256: {digest}",
        f"canonical_space: {args.labels}",
        f"external_dataset: PlantWild v2 (uqtwei2/PlantWild)",
        "",
        "conservative_mapping:",
    ]
    for ext, lab in sorted(conservative.items()):
        yaml_lines.append(f'  "{ext}": {lab}')
    yaml_lines += ["", "author_decision_required:"]
    for r in rows:
        if r["action"] == "AUTHOR_DECISION":
            yaml_lines.append(f'  - external: "{r["external_label"]}"')
            yaml_lines.append(f'    proposed_canonical: {r["canonical_label"]}')
            yaml_lines.append(f'    match_type: {r["match_type"]}')
            yaml_lines.append(f'    n_images: {r["n_images"]}')
            yaml_lines.append(f'    evidence: "{r["evidence"]}"')
    yaml_lines += ["", "ood_known_crop:"]
    for r in rows:
        if r["action"] == "OOD_KNOWN_CROP":
            yaml_lines.append(f'  - "{r["external_label"]}"   # n={r["n_images"]}')
    yaml_lines += ["", "ood_unseen_crop:"]
    for r in rows:
        if r["action"] == "OOD_UNSEEN_CROP":
            yaml_lines.append(f'  - "{r["external_label"]}"   # n={r["n_images"]}')
    yaml_lines += ["", "canonical_labels_with_no_external_counterpart:"]
    for lab in payload["canonical_labels_unmatched"]:
        yaml_lines.append(f"  - {lab}")
    args.out_yaml.write_text("\n".join(yaml_lines) + "\n")

    c = payload["counts"]
    print(f"mapping version {MAPPING_VERSION}  sha256 {digest[:16]}")
    print(f"external classes                {c['external_classes']}")
    print(f"  conservative (auto-accepted)  {c['conservative_included']:>4}  images {c['conservative_images']}")
    print(f"  author decision required      {c['author_decision_required']:>4}")
    print(f"  OOD, known crop               {c['ood_known_crop']:>4}  images {c['ood_known_crop_images']}")
    print(f"  OOD, unseen crop              {c['ood_unseen_crop']:>4}  images {c['ood_unseen_crop_images']}")
    print(f"\ncanonical labels with no counterpart ({len(payload['canonical_labels_unmatched'])}):")
    for lab in payload["canonical_labels_unmatched"]:
        print(f"  {lab}")
    print("\nauthor decision required:")
    for r in rows:
        if r["action"] == "AUTHOR_DECISION":
            print(f"  {r['external_label']:<34} -> {r['canonical_label']:<28} {r['match_type']:<22} n={r['n_images']}")
            print(f"      {r['evidence']}")
    print(f"\nwritten -> {args.out_yaml}\n           {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
