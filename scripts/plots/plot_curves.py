#!/usr/bin/env python
"""Training curves of the chopper / answerer runs (2026-10-30, user: "log ce tok etc, some accuracy measures for these
trainings? can you put them on some chart? … in the readme page, too; i will see if training curve flattens").
Parses the train.log files, puts a continuation run after its parent on one x axis (training rows seen), writes
  release/figures/curves.json            (data for the HTML chart page)
  release/figures/training_curves{,_dark}.png   (README; light + dark)
usage: MM_DIR=<mm runs> KEEPER_DIR=<keeper runs> MLA_DIR=<mla runs> python plot_curves.py"""
import json, os, re
import numpy as np
MM, KP = os.environ.get("MM_DIR", "mm"), os.environ.get("KEEPER_DIR", "keeper")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures")
MLA = os.environ.get("MLA_DIR", "mla")
# entity → [(log, row offset)]; a continuation starts where its parent ended
CHOPPERS = {
    "mmBERT-base (0.31B), 54k mixed": [(f"{MM}/chopper_basemix54k/train.log", 0)],
    "10% Qwen3-1.7B (0.14B)": [(f"{KP}/w10_q17/train.log", 0), (f"{KP}/w10_q17_27k/train.log", 9000),
                               (f"{KP}/w10_q17_54k/train.log", 27000)],
    "mmBERT-small (0.14B), 54k mixed": [(f"{MM}/chopper_smallmix54k/train.log", 0)],
    "10% MiniCPM3-4B (MLA), 54k mixed": [(f"{MLA}/mla10_mix54k/train.log", 0)],   # 2026-11-01, stopped at 4.5k
}
# 2026-10-31: only the mmBERTs trained once on all 54k rows mixed are kept (the sequential ones were deleted). The dashed
# line is the Qwen slice trained the same way, from scratch, against the solid slice that saw the benchmark rows first
# 2026-11-02: the first 54k mixed Qwen slice (lr 3e-4) diverged at ~16k; its retrain from scratch at lr 1.5e-4 replaces it
REFS = {"10% Qwen3-1.7B, 54k mixed": (1, [(f"{KP}/w10_q17_mix54k_lr15/train.log", 0)]),   # name: (colour index, logs)
        "20% MiniCPM3-4B (MLA), 54k mixed": (3, [(f"{MLA}/mla20_mix54k/train.log", 0),   # 2nd epoch (lr 5e-5, seed 1) continues at 54k
                                                     (f"{MLA}/mla20_mix54k_ep2/train.log", 54000)])}
ANSWERER = ("Qwen3-1.7B answerer", [(f"{MM}/qans17/train.log", 0), (f"{MM}/qans17_27k/train.log", 9000)])
DS = ("musique", "hotpot", "triviaqa")
num = lambda pat, s: float(m.group(1)) if (m := re.search(pat, s)) else None


def chopper(parts):
    tr, va = [], {}
    for path, off in parts:
        if not os.path.exists(path):
            continue
        for l in open(path):
            if re.search(r"\| step \d+/\d+ \|", " | " + l[9:]) and "keepL" in l:
                st = int(re.search(r"step (\d+)/", l).group(1)) + off
                tr.append({"rows": st, "loss": num(r"keepL ([\d.]+)", l), "f1": num(r"train KEEP@0\.5 bal-acc [\d.]+ F1 ([\d.]+)", l),
                           "bal": num(r"bal-acc ([\d.]+)", l)})
            elif l[9:].startswith("EVAL step") and "answer in kept" in l:
                # mmBERT logs its in-training eval at p>0.5 with no shaping; the keeper logs "raw p>0.5" — the comparable line
                if "BASELINE" in l or ("CHAR" in l and "raw p>0.5" not in l):
                    continue
                st = re.search(r"EVAL step (\d+)", l)
                if not st:
                    continue
                ds = next((d for d in DS if f"| {d}" in l), None)
                e = va.setdefault(int(st.group(1)) + off, {})
                e[ds] = {"ans": num(r"answer in kept ([\d.]+)", l), "kept": num(r"\| kept ([\d.]+)", l)}
    val = [{"rows": r, **{k: float(np.mean([v[d][k] for d in DS])) for k in ("ans", "kept")},
            "per": {d: v[d]["ans"] for d in DS}} for r, v in sorted(va.items()) if all(d in v for d in DS)]
    return tr, val


def answerer(parts):
    tr, va = [], {}
    for path, off in parts:
        if not os.path.exists(path):
            continue
        for l in open(path):
            if re.search(r" step \d+/\d+ \| CE", l):
                st = int(re.search(r"step (\d+)/", l).group(1)) + off
                tr.append({"rows": st, "ce": num(r"CE ([\d.]+)", l), "tok": num(r"tok ([\d.]+)", l)})
            elif "EVAL step" in l and ("notes=reader" in l or "notes=chopper" in l) and "base model" not in l:
                st = int(re.search(r"EVAL step (\d+)", l).group(1)) + off
                ds = next((d for d in DS if f"| {d}" in l), None)
                va.setdefault(st, {})[ds] = num(r"exact ([\d.]+)", l)
    val = [{"rows": r, "exact": float(np.mean([v[d] for d in DS])), "per": v} for r, v in sorted(va.items()) if all(d in v for d in DS)]
    return tr, val


D = {"choppers": {k: dict(zip(("train", "val"), chopper(v))) for k, v in CHOPPERS.items()},
     "reference": {k: {**dict(zip(("train", "val"), chopper(v))), "ci": ci} for k, (ci, v) in REFS.items()},
     "answerer": dict(zip(("train", "val"), answerer(ANSWERER[1])))}
os.makedirs(OUT, exist_ok=True)
json.dump(D, open(f"{OUT}/curves.json", "w"), indent=0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
THEMES = {"": dict(bg="#fcfcfb", fg="#0b0b0b", fg2="#52514e", grid="#e4e3df",
                   c=["#2a78d6", "#eb6834", "#1baf7a", "#c2449f"], a="#6250d6"),
          "_dark": dict(bg="#1a1a19", fg="#f0efec", fg2="#c3c2b7", grid="#34332f",
                        c=["#3987e5", "#d95926", "#199e70", "#d55fb3"], a="#9085e9")}


def smooth(y, k=5):
    y = np.array([np.nan if v is None else v for v in y], float)
    return np.array([np.nanmean(y[max(0, i - k + 1):i + 1]) for i in range(len(y))])


for suf, T in THEMES.items():
    plt.rcParams.update({"font.size": 13, "axes.edgecolor": T["grid"], "axes.labelcolor": T["fg2"], "xtick.color": T["fg2"],
                         "ytick.color": T["fg2"], "text.color": T["fg"], "axes.titlecolor": T["fg"], "axes.titleweight": "bold",
                         "axes.titlesize": 15, "axes.titlelocation": "left"})
    fig, axs = plt.subplots(4, 2, figsize=(16, 19), facecolor=T["bg"])   # 2 wide so each panel stays large on GitHub
    ax = axs.flatten().reshape(2, 4)                                     # keep the ax[r, c] panel names
    for a in ax.flat:
        a.set_facecolor(T["bg"]); a.grid(True, color=T["grid"], lw=0.6); a.spines[["top", "right"]].set_visible(False)
        a.axvline(9000, color=T["fg2"], lw=0.8, ls=":"); a.axvline(27000, color=T["fg2"], lw=0.8, ls=":")
        a.set_xlabel("training rows seen"); a.set_xlim(0, 55000)
    P = [("loss", "Chopper · train KEEP loss", ax[0, 0], "train"),
         ("f1", "Chopper · train KEEP F1 (p>0.5)", ax[0, 1], "train"),
         ("ans", "Chopper · val answer-in-kept (p>0.5)", ax[0, 2], "val"),
         ("kept", "Chopper · val share of text kept (p>0.5)", ax[1, 0], "val"),
         ("bal", "Chopper · train balanced accuracy (p>0.5)", ax[0, 3], "train")]
    for key, title, a, part in P:
        a.set_title(title)
        for (name, run), col in zip(D["choppers"].items(), T["c"]):
            pts = [p for p in run[part] if p.get(key) is not None]
            if not pts:
                continue
            x = [p["rows"] for p in pts]; y = [p[key] for p in pts]
            y = smooth(y) if part == "train" else y
            a.plot(x, y, color=col, lw=2, marker=None if part == "train" else "o", ms=4, label=name)
            a.annotate(name.split(" (")[0], (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", color=T["fg2"], fontsize=11, va="center")
        for name, run in D["reference"].items():
            col = T["c"][run["ci"]]
            pts = [p for p in run[part] if p.get(key) is not None]
            if pts:
                y = [p[key] for p in pts]
                a.plot([p["rows"] for p in pts], smooth(y) if part == "train" else y, color=col, lw=1.3, ls="--", label=name)
        a.legend(frameon=False, fontsize=11, labelcolor=T["fg2"])
    for key, title, a, part in (("ce", "Answerer · train CE", ax[1, 1], "train"),
                                ("tok", "Answerer · train token accuracy", ax[1, 2], "train"),
                                ("exact", "Answerer · val exact (chopper notes)", ax[1, 3], "val")):
        a.set_title(title); pts = D["answerer"][part]
        if pts:
            a.plot([p["rows"] for p in pts], smooth([p[key] for p in pts]) if part == "train" else [p[key] for p in pts],
                   color=T["a"], lw=2, marker=None if part == "train" else "o", ms=4)
    ax[1, 3].set_ylim(0, 1)
    ax[0, 2].set_ylim(0.5, 1); ax[1, 0].set_ylim(0, 0.25); ax[0, 1].set_ylim(0, 1); ax[0, 3].set_ylim(0.5, 1)
    for a in (ax[0, 1], ax[0, 3]):
        if not any(p.get("f1") for r in D["choppers"].values() for p in r["train"]):
            a.text(0.5, 0.5, "logged from the next runs on", transform=a.transAxes, ha="center", color=T["fg2"])
    fig.suptitle("Training curves — train lines are 500-row rolling means; val points = mean of MuSiQue, HotpotQA, TriviaQA\n"
                 "mmBERTs and MiniCPM3 MLA slice: one epoch over all 54k rows mixed; solid Qwen slice: 9k, then 18k more, then Wikipedia (dotted lines); dashed: Qwen slice 54k mixed (stopped), MiniCPM3 20% slice",
                 x=0.01, ha="left", color=T["fg2"], fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(f"{OUT}/training_curves{suf}.png", dpi=110, facecolor=T["bg"]); plt.close(fig)
print("CURVES", {k: (len(v["train"]), len(v["val"])) for k, v in D["choppers"].items()}, "answerer", len(D["answerer"]["train"]))
import subprocess, sys as _sys                       # the HTML artifact page from the same curves.json
subprocess.run([_sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "curves_page.py"), f"{OUT}/curves.html"], check=False)
