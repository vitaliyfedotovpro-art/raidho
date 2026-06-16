"""Generate the benchmark & test-suite visuals for the README / docs.

Every number traces back to:
  - evidence/2026-06-11_opus_vs_raidho/README.md   (Benchmark 1)
  - evidence/2026-06-12_autodistill_curve/README.md (Benchmark 2)
  - the pytest suite (9 modules, 83 tests)

Nothing here is invented or extrapolated. Run:
    python scripts/generate_visuals.py
    # or, isolated:  uv run --with matplotlib python scripts/generate_visuals.py
Outputs SVG (README) + PNG @300dpi to docs/visuals/.
"""
import os
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "visuals")
os.makedirs(OUT_DIR, exist_ok=True)

plt.style.use("default")
# Colorblind-safe, consistent across figures.
COLOR_A = "#2ca02c"   # procedure (green)
COLOR_C = "#ff7f0e"   # hybrid (orange)
COLOR_B = "#1f77b4"   # pure loop (blue)
COLOR_LIGHT = "#9467bd"
COLOR_HEAVY = "#d62728"

# NOTE: the rune ᚱ is intentionally kept out of matplotlib titles — the default
# font has no Runic glyph and renders it as a tofu box. It lives in the README
# branding instead.


def save_fig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f"{name}.svg"), format="svg",
                transparent=True, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), format="png", dpi=300,
                transparent=True, bbox_inches="tight")
    plt.close(fig)


def add_footnote(fig, text):
    fig.text(0.5, -0.04, text, ha="center", va="top", fontsize=9, color="gray")


# 1. Cost & tokens vs pure loop ------------------------------------------------
def fig1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    labels = ["A: Procedure", "C: Hybrid", "B: Pure Loop"]
    colors = [COLOR_A, COLOR_C, COLOR_B]
    cost = [0.0501, 0.1156, 0.3010]
    tokens = [2744, 14484, 45288]
    x = np.arange(len(labels))

    ax1.bar(x, cost, color=colors)
    ax1.set_title("Cost vs pure loop")
    ax1.set_ylabel("Cost (USD)")
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.text(0, cost[0] + 0.012, "×6 cheaper", ha="center", fontweight="bold")
    ax1.text(1, cost[1] + 0.012, "×2.6 cheaper", ha="center", fontweight="bold")

    ax2.bar(x, tokens, color=colors)
    ax2.set_title("Tokens vs pure loop")
    ax2.set_ylabel("Tokens (in+out)")
    ax2.set_xticks(x); ax2.set_xticklabels(labels)
    ax2.text(0, tokens[0] + 1800, "×16.5 fewer", ha="center", fontweight="bold")
    ax2.text(1, tokens[1] + 1800, "×3.1 fewer", ha="center", fontweight="bold")

    fig.suptitle("Raidho vs pure Opus 4.8 on an audit task", fontsize=14, fontweight="bold")
    add_footnote(fig, 'Data: Benchmark 1 "Raidho procedure vs pure Opus 4.8" (2026-06-11)')
    plt.tight_layout()
    save_fig(fig, "01_cost_tokens")


# 2. Four metrics at a glance --------------------------------------------------
def fig2():
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    labels = ["A", "C", "B"]
    series = [
        ([0.0501, 0.1156, 0.3010], "Cost (USD)", axs[0, 0]),
        ([2744, 14484, 45288], "Tokens (count)", axs[0, 1]),
        ([22.7, 32.6, 55.9], "Wall time (s)", axs[1, 0]),
        ([1, 1, 8], "LLM calls (count)", axs[1, 1]),
    ]
    for data, ylabel, ax in series:
        ax.bar(labels, data, color=[COLOR_A, COLOR_C, COLOR_B])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(data) * 1.2)
        for i, val in enumerate(data):
            pct = (val / data[2]) * 100
            ax.text(i, val, f"{pct:.0f}%", ha="center", va="bottom")
    fig.suptitle("Four metrics at a glance  (B = 100% baseline)", fontsize=14, fontweight="bold")
    add_footnote(fig, 'Data: Benchmark 1 "Raidho procedure vs pure Opus 4.8" (2026-06-11)')
    plt.tight_layout()
    save_fig(fig, "02_metrics_glance")


# 4. Levers against the loop ---------------------------------------------------
def fig4():
    fig, ax = plt.subplots(figsize=(10, 6))
    x_light = [0, 1]
    y_light = [0.00034, 0.00004]
    x_heavy = [3, 4, 5, 6]
    y_heavy = [0.00100, 0.00046, 0.00059, 0.00013]
    labels = ["Light: Base", "Light: Distill",
              "Heavy: Base", "Heavy: Distill\n(lottery)",
              "Heavy: Ctx-First", "Heavy: Combined"]

    ax.bar(x_light, y_light, color=COLOR_LIGHT)
    ax.bar(x_heavy[0], y_heavy[0], color=COLOR_HEAVY)
    ax.bar(x_heavy[1], y_heavy[1], color=COLOR_HEAVY, hatch="//")
    # distill-on-heavy is a high-variance "lottery" (×1–×5) — show the range.
    ax.errorbar(x_heavy[1], y_heavy[1],
                yerr=[[y_heavy[1] - 0.00020], [0.00100 - y_heavy[1]]],
                fmt="none", ecolor="black", capsize=5, lw=1.5)
    ax.bar(x_heavy[2], y_heavy[2], color=COLOR_HEAVY)
    ax.bar(x_heavy[3], y_heavy[3], color=COLOR_HEAVY)

    ax.set_xticks(x_light + x_heavy)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Cost (USD)")
    ax.text(x_light[1], y_light[1] + 0.00002, "×9.7", ha="center", va="bottom", fontweight="bold")
    ax.text(x_heavy[1], 0.00104, "×1–×5", ha="center", va="bottom")
    ax.text(x_heavy[2], y_heavy[2] + 0.00002, "×1.7", ha="center", va="bottom", fontweight="bold")
    ax.text(x_heavy[3], y_heavy[3] + 0.00002, "×7.7", ha="center", va="bottom", fontweight="bold")
    ax.set_title("Levers against the loop", fontsize=14, fontweight="bold")
    add_footnote(fig, 'Data: Benchmark 2 "Auto-distillation curve" (2026-06-12)')
    plt.tight_layout()
    save_fig(fig, "04_levers")


# 5. Saving scales with iteration overhead, not task size ----------------------
#    Honest reframe: compare DISTILL-ALONE speedup on light vs heavy. That is the
#    literal finding — distill's win collapses when cost is the data, not the loop.
def fig5():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = ["Light\n(overhead-dominated)", "Heavy\n(data-dominated)"]
    mult = [9.7, 2.2]   # distill-alone multiplier, from the evidence tables
    bars = ax.bar(labels, mult, color=[COLOR_LIGHT, COLOR_HEAVY])
    bars[1].set_hatch("//")
    # heavy distill alone is a lottery (×1–×5)
    ax.errorbar(1, 2.2, yerr=[[1.2], [2.8]], fmt="none", ecolor="black", capsize=6, lw=1.5)

    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.text(0.5, 1.15, "×1 = no saving", color="gray", fontsize=9, ha="center")
    ax.set_ylabel("Distill-alone speedup  (× cheaper per repeat)")
    ax.set_ylim(0, 11)
    ax.text(0, 9.9, "×9.7", ha="center", fontweight="bold")
    ax.text(1, 5.2, "×2.2 (×1–×5)", ha="center", fontweight="bold")

    ax.annotate("Many cheap iterations over little data:\nthe loop IS the cost → big, stable win",
                xy=(0, 9.7), xytext=(0.15, 7.3), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.annotate("Cost is the data in context, few\niterations to cut → small & variable",
                xy=(1, 2.2), xytext=(0.45, 3.6), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=1.2))

    ax.set_title("Distillation's saving scales with iteration overhead, not task size",
                 fontsize=12, fontweight="bold")
    add_footnote(fig, 'Data: Benchmark 2 "Auto-distillation curve" (2026-06-12). '
                      'Heavy is a single stochastic sample.')
    plt.tight_layout()
    save_fig(fig, "05_savings_scale")


# 8. Test coverage by subsystem ------------------------------------------------
def fig8():
    fig, ax = plt.subplots(figsize=(11, 5))
    systems = ["Distillation & safety", "Memory / VSA core",
               "External-review regressions", "Context-first", "Council → memory"]
    counts = [42, 17, 11, 7, 6]
    purposes = ["safety limits & homeostasis", "algebraic retrieval & bit-packing",
                "issue prevention", "workspace state collection", "consensus recording"]
    y = np.arange(len(systems))
    bars = ax.barh(y, counts, color="#7f7f7f")
    ax.set_yticks(y); ax.set_yticklabels(systems)
    ax.set_xlabel("Number of tests")
    ax.set_xlim(0, 62)   # room for the annotation text
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2,
                f"{counts[i]}  ({purposes[i]})", va="center", fontsize=10)
    ax.invert_yaxis()
    ax.set_title("Test coverage by subsystem  (83 tests, 9 modules)", fontsize=14, fontweight="bold")
    add_footnote(fig, "Data: pytest suite (def test_* counts)")
    plt.tight_layout()
    save_fig(fig, "08_test_coverage")


if __name__ == "__main__":
    fig1(); fig2(); fig4(); fig5(); fig8()
    print(f"Wrote figures to {os.path.normpath(OUT_DIR)}")
