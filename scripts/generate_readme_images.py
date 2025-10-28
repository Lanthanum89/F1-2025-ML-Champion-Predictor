import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plotting style
sns.set_theme(style="whitegrid")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
ASSETS_DIR = os.path.join(ROOT, "assets")
RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(ROOT, "data")

os.makedirs(ASSETS_DIR, exist_ok=True)

TOP10_IMG = os.path.join(ASSETS_DIR, "top10_probabilities.png")
STANDINGS_IMG = os.path.join(ASSETS_DIR, "current_standings_top12.png")
CALIB_IMG = os.path.join(ASSETS_DIR, "calibrated_vs_model_top10.png")
H2H_IMG = os.path.join(ASSETS_DIR, "h2h_swing_heatmap_race1.png")
MC_DELTA_IMG = os.path.join(ASSETS_DIR, "model_minus_mc_top10.png")

PREDICTIONS_CSV = os.path.join(RESULTS_DIR, "f1_2025_championship_predictions.csv")
CALIBRATED_CSV = os.path.join(RESULTS_DIR, "f1_2025_championship_predictions_calibrated.csv")
DATA_CSV = os.path.join(DATA_DIR, "f1_championship_data.csv")


def _save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_top10_probabilities():
    if not os.path.exists(PREDICTIONS_CSV):
        print(f"[WARN] Missing {PREDICTIONS_CSV} — skipping top10 image")
        return
    df = pd.read_csv(PREDICTIONS_CSV)
    # Normalize in case not summing to 1
    if "championship_probability" not in df.columns:
        print("[WARN] 'championship_probability' not in predictions — skipping top10 image")
        return
    df = df.copy()
    # Ensure numeric
    df["championship_probability"] = pd.to_numeric(df["championship_probability"], errors="coerce").fillna(0)
    top10 = df.sort_values("championship_probability", ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top10,
        x="championship_probability",
        y="driver_name",
        hue="constructor_name",
        dodge=False,
    )
    plt.xlabel("Championship Probability")
    plt.ylabel("Driver")
    plt.title("Top 10 Predicted Contenders for 2025 Drivers' Title")
    xmax = float(top10["championship_probability"].max())
    plt.xlim(0, max(0.4, xmax * 1.1))
    plt.legend(title="Constructor", bbox_to_anchor=(1.02, 1), loc="upper left")

    _save_fig(TOP10_IMG)


def plot_current_standings_top12():
    if not os.path.exists(DATA_CSV):
        print(f"[WARN] Missing {DATA_CSV} — skipping standings image")
        return
    df = pd.read_csv(DATA_CSV)
    if "year" not in df.columns or "final_points" not in df.columns:
        print("[WARN] Expected columns missing in data — skipping standings image")
        return
    cur = (
        df[df["year"] == 2025]
        .loc[:, ["driver_name", "constructor_name", "final_points", "wins", "races_started"]]
        .rename(columns={"final_points": "points"})
        .sort_values("points", ascending=False)
        .reset_index(drop=True)
    )
    if cur.empty:
        print("[WARN] No rows for year==2025 — skipping standings image")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(data=cur.head(12), x="points", y="driver_name", hue="constructor_name", dodge=False)
    plt.title("2025 Current Standings (Top 12)")
    plt.xlabel("Points so far")
    plt.ylabel("Driver")
    plt.legend(title="Constructor", bbox_to_anchor=(1.02, 1), loc="upper left")

    _save_fig(STANDINGS_IMG)


def plot_calibrated_vs_model_top10():
    if not os.path.exists(PREDICTIONS_CSV) or not os.path.exists(CALIBRATED_CSV):
        print(f"[WARN] Missing predictions or calibrated CSV — skipping calibrated image")
        return
    base = pd.read_csv(PREDICTIONS_CSV)
    calib = pd.read_csv(CALIBRATED_CSV)
    if "championship_probability" not in base.columns or "calibrated_probability" not in calib.columns:
        print("[WARN] Expected columns missing — skipping calibrated image")
        return

    merged = (
        base[["driver_name", "constructor_name", "championship_probability"]]
        .merge(calib[["driver_name", "calibrated_probability"]], on="driver_name", how="left")
        .sort_values("calibrated_probability", ascending=False)
        .head(10)
    )

    plot_df = merged.melt(
        id_vars=["driver_name"],
        value_vars=["championship_probability", "calibrated_probability"],
        var_name="source",
        value_name="probability",
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="probability", y="driver_name", hue="source")
    plt.title("Model vs Calibrated Probabilities (Top 10 by calibrated)")
    plt.xlabel("Probability")
    plt.ylabel("Driver")
    plt.legend(title="Source", bbox_to_anchor=(1.02, 1), loc="upper left")

    _save_fig(CALIB_IMG)


def plot_h2h_heatmap_race1():
    """Generate head-to-head swing heatmap for Race 1 (others neutral) using top two drivers."""
    if not os.path.exists(DATA_CSV):
        print(f"[WARN] Missing {DATA_CSV} — skipping H2H heatmap")
        return
    df = pd.read_csv(DATA_CSV)
    required = {"year", "driver_name", "final_points"}
    if not required.issubset(df.columns):
        print("[WARN] Expected columns missing in data — skipping H2H heatmap")
        return

    cur = (
        df[df["year"] == 2025]
        .loc[:, ["driver_name", "final_points"]]
        .rename(columns={"final_points": "points"})
        .sort_values("points", ascending=False)
        .reset_index(drop=True)
    )
    if len(cur) < 2:
        print("[WARN] Need at least two drivers for H2H — skipping H2H heatmap")
        return

    driver_A = cur.loc[0, "driver_name"]
    driver_B = cur.loc[1, "driver_name"]

    POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

    def pts(pos):
        if pos is None or pos == 0:
            return 0
        return POINTS.get(int(pos), 0)

    base_points = dict(cur.set_index("driver_name")["points"])

    def simulate_h2h(scenario):
        pA = base_points[driver_A]
        pB = base_points[driver_B]
        for (pa, pb) in scenario:
            pA += pts(pa)
            pB += pts(pb)
        return pA - pB

    neutral = [(3, 3), (3, 3), (3, 3), (3, 3)]
    rows = []
    pos_list = [1, 2, 3, 4, 5, 0]
    for a_pos in pos_list:
        for b_pos in pos_list:
            scen = list(neutral)
            scen[0] = (a_pos, b_pos)  # Race 1 only
            diff = simulate_h2h(scen)
            rows.append({"A_pos": a_pos, "B_pos": b_pos, "A_minus_B": diff})
    swing_df = pd.DataFrame(rows)

    heat = swing_df.pivot(index="A_pos", columns="B_pos", values="A_minus_B").sort_index(ascending=False)
    plt.figure(figsize=(8, 6))
    sns.heatmap(heat, annot=True, fmt=".0f", cmap="coolwarm", center=0)
    plt.title(f"Points swing for Race 1 (others neutral) — {driver_A} minus {driver_B}")
    plt.xlabel(f"{driver_B} finishing position (0=DNF)")
    plt.ylabel(f"{driver_A} finishing position (0=DNF)")

    _save_fig(H2H_IMG)


def plot_mc_delta_top10():
    """Run a Monte Carlo of remaining races and plot model - MC delta for top 10 by MC."""
    if not os.path.exists(DATA_CSV) or not os.path.exists(PREDICTIONS_CSV):
        print(f"[WARN] Missing data or predictions — skipping MC delta image")
        return
    df = pd.read_csv(DATA_CSV)
    preds = pd.read_csv(PREDICTIONS_CSV)
    need_cols = {"year", "driver_name", "constructor_name", "final_points", "wins", "races_started"}
    if not need_cols.issubset(df.columns) or "championship_probability" not in preds.columns:
        print("[WARN] Expected columns missing — skipping MC delta image")
        return

    race_points = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
    remaining_races = 4

    cur = (
        df[df["year"] == 2025]
        .loc[:, ["driver_name", "constructor_name", "final_points", "wins", "races_started"]]
        .rename(columns={"final_points": "points"})
        .sort_values("points", ascending=False)
        .reset_index(drop=True)
    )
    if cur.empty:
        print("[WARN] No 2025 rows — skipping MC delta image")
        return

    beta = 0.40
    cur["points_per_race"] = cur["points"] / cur["races_started"].replace(0, np.nan)
    skill = np.asarray(cur["points_per_race"].fillna(cur["points_per_race"].median()).values, dtype=float)
    mu = float(np.mean(skill))
    sigma = float(np.std(skill)) + 1e-9
    skill = (skill - mu) / sigma
    weights = np.exp(beta * skill)
    weights = weights / weights.sum()

    names = cur["driver_name"].tolist()

    S = 3000  # sims
    champ_counts = np.zeros(len(names), dtype=np.int64)

    rng = np.random.default_rng(42)
    for _ in range(S):
        sim_points = cur["points"].values.astype(float).copy()
        sim_wins = cur["wins"].values.astype(int).copy()

        for _r in range(remaining_races):
            available = list(range(len(names)))
            order = []
            w = weights.copy()
            for _pos in range(len(names)):
                w_norm = w[available] / w[available].sum()
                choice = int(rng.choice(available, p=w_norm))
                order.append(choice)
                available.remove(choice)
            for pos, driver_idx in enumerate(order[: len(race_points)]):
                sim_points[driver_idx] += race_points[pos]
            sim_wins[order[0]] += 1

        top_points = sim_points.max()
        contenders = np.where(sim_points == top_points)[0]
        if len(contenders) > 1:
            wins_subset = sim_wins[contenders]
            max_wins = wins_subset.max()
            contenders = contenders[wins_subset == max_wins]
            winner = int(rng.choice(contenders)) if len(contenders) > 1 else int(contenders[0])
        else:
            winner = int(contenders[0])
        champ_counts[winner] += 1

    mc_probs = champ_counts / S
    mc_df = (
        pd.DataFrame({"driver_name": names, "mc_championship_probability": mc_probs})
        .merge(
            preds[["driver_name", "championship_probability"]],
            on="driver_name",
            how="left",
        )
        .sort_values("mc_championship_probability", ascending=False)
    )
    comp = mc_df.copy()
    comp["delta_model_minus_mc"] = comp["championship_probability"] - comp["mc_championship_probability"]

    plt.figure(figsize=(10, 6))
    sns.barplot(data=comp.head(10), x="delta_model_minus_mc", y="driver_name", orient="h")
    plt.axvline(0, color="k", linewidth=1)
    plt.title("Model probability minus Monte Carlo (Top 10 by MC)")
    plt.xlabel("Model - MC probability")
    plt.ylabel("Driver")

    _save_fig(MC_DELTA_IMG)


if __name__ == "__main__":
    print("Generating README images...")
    plot_top10_probabilities()
    plot_current_standings_top12()
    plot_calibrated_vs_model_top10()
    plot_h2h_heatmap_race1()
    plot_mc_delta_top10()
    print("Done.")
