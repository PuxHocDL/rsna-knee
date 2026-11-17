"""Score fixed rank blends on the same Gold-58 studies; no weight fitting."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "kaggle_out" / "gold58-combinations-model1"
SOURCES = {
    "R6": ROOT / "kaggle_out" / "r6-gold58" / "r6_gold58.csv",
    "EffV2": ROOT / "train_kit" / "kaggle_train" / "output-effv2s-latest" / "effv2s_gold.csv",
    "ConvNeXt": ROOT / "model1" / "cnxs_gold.csv",
}


def auc(y, p):
    positive = p[y == 1]
    negative = p[y == 0]
    if not len(positive) or not len(negative):
        raise ValueError("AUC requires both positive and negative labels")
    difference = positive[:, None] - negative[None, :]
    return float(np.mean((difference > 0) + 0.5 * (difference == 0)))


def main():
    train = pd.read_csv(ROOT / "train_kit" / "labels" / "train.csv")
    labels = [c for c in train.columns if c not in ("StudyInstanceUID", "Report")]
    gold = train.loc[train[labels].notna().all(axis=1)].set_index("StudyInstanceUID")[labels]
    gold = gold.sort_index()
    if len(gold) != 58 or not gold.index.is_unique:
        raise ValueError("Expected exactly 58 unique gold studies")
    if not gold.isin([0, 1]).all().all():
        raise ValueError("Gold labels must be binary")
    ranks = {}
    raw_auc = {}
    for name, path in SOURCES.items():
        frame = pd.read_csv(path, index_col=0)
        if not frame.index.is_unique or set(frame.index) != set(gold.index):
            raise ValueError(f"{name}: gold study IDs do not match")
        if list(frame.columns) != labels:
            raise ValueError(f"{name}: label columns do not match")
        frame = frame.loc[gold.index]
        if not np.isfinite(frame.to_numpy()).all() or not frame.ge(0).all().all() or not frame.le(1).all().all():
            raise ValueError(f"{name}: invalid probabilities")
        ranks[name] = frame.rank(method="average", pct=True)
        raw_auc[name] = [auc(gold[c].to_numpy(), frame[c].to_numpy()) for c in labels]

    recipes = {name: {name: 1.0} for name in SOURCES}
    for a, b in combinations(SOURCES, 2):
        recipes[f"{a}50_{b}50"] = {a: 0.5, b: 0.5}
    recipes.update({
        "R6_80_EffV2_20": {"R6": 0.8, "EffV2": 0.2},
        "R6_80_ConvNeXt_20": {"R6": 0.8, "ConvNeXt": 0.2},
        "R6_80_EffV2_10_ConvNeXt_10": {"R6": 0.8, "EffV2": 0.1, "ConvNeXt": 0.1},
        "R6_70_EffV2_15_ConvNeXt_15": {"R6": 0.7, "EffV2": 0.15, "ConvNeXt": 0.15},
        "R6_ratio85_EffV2_12p5_ConvNeXt_12p5": {"R6": 85 / 110, "EffV2": 12.5 / 110, "ConvNeXt": 12.5 / 110},
        "Equal_three": {name: 1 / 3 for name in SOURCES},
    })
    results = {}
    predictions = {}
    for name, weights in recipes.items():
        # Preserve mathematical rank-blend ties despite floating-point summation.
        prediction = sum(ranks[k] * w for k, w in weights.items()).round(12)
        predictions[name] = prediction
        results[name] = [auc(gold[c].to_numpy(), prediction[c].to_numpy()) for c in labels]
        if name in raw_auc:
            np.testing.assert_allclose(results[name], raw_auc[name], atol=1e-12)

    reference = json.loads((ROOT / "kaggle_out" / "r6-gold58" / "r6_gold58_auc.json").read_text())
    np.testing.assert_allclose(results["R6"], [reference["auc"][c] for c in labels], atol=1e-12)
    logs = {}
    for name in ("EffV2", "ConvNeXt"):
        log = json.loads(SOURCES[name].with_name(
            "effv2s_log.json" if name == "EffV2" else "cnxs_log.json"
        ).read_text())
        logs[name] = {k: log[k] for k in ("status", "epoch", "next_batch", "gold_auc")}
        np.testing.assert_allclose(np.mean(results[name]), log["gold_auc"], atol=1e-12)

    table = pd.DataFrame(results, index=labels)
    table.loc["Macro"] = table.mean()
    correlations = pd.DataFrame({
        f"{a}__{b}": ranks[a].corrwith(ranks[b])
        for a, b in combinations(SOURCES, 2)
    })
    correlations.loc["Mean"] = correlations.mean()
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "auc_by_finding.csv", index_label="Finding")
    correlations.to_csv(OUT / "spearman_by_finding.csv", index_label="Finding")
    for name, prediction in predictions.items():
        prediction.to_csv(OUT / f"{name}_rank_predictions.csv", index_label="StudyInstanceUID")
    (OUT / "evaluation.json").write_text(json.dumps({
        "studies": 58,
        "method": "Per-label average percentile rank, fixed weights, macro ROC AUC",
        "sources": {k: str(v.relative_to(ROOT)) for k, v in SOURCES.items()},
        "versions": {"R6": "E_lrprior 4 folds + flip TTA",
                     "EffV2": "16 epochs, final SWA",
                     "ConvNeXt": logs["ConvNeXt"]},
        "training_logs": logs,
        "latest_remote_outputs": "ConvNeXt supplied locally in model1; EffV2 continuation not retrieved",
        "recipes": recipes,
        "macro_auc": table.loc["Macro"].to_dict(),
        "warning": "Exploratory Gold-58 comparison; not an independent leaderboard estimate.",
    }, indent=2), encoding="utf-8")
    print(table.round(4).to_string())
    print("\nSpearman correlations\n" + correlations.round(4).to_string())
    print("\nValidation passed: exact ID/column alignment, binary labels, finite probabilities,")
    print("single-model rank invariance, R6 published AUCs and training logs reproduced.")
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
