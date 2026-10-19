"""拣货到位缺货分析：定位断点、量化损耗、评估拦截策略。

用法（仓库根目录）：
    python scripts/analyze.py [--csv data/picks.csv]

输出：stdout 报告 + output/*.png
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MINUTES_LOST_PER_STOCKOUT = 90
MINUTES_LOST_PER_EARLY_FLAG = 20
BREAKPOINT_MIN = 20


def load(path):
    raw = pd.read_csv(path)
    n_raw = len(raw)

    df = raw.drop_duplicates()
    n_dupes = n_raw - len(df)

    # pick_ts 带 +08:00，stock_synced_at 带 Z：必须按绝对时刻相减。
    # 若直接去掉时区做朴素相减，会得到恒定 ~8h 的假滞后，断点信号完全消失。
    df["pick_ts"] = pd.to_datetime(df["pick_ts"], utc=True)
    df["sync_ts"] = pd.to_datetime(df["stock_synced_at"], utc=True)
    df["lag_min"] = (df["pick_ts"] - df["sync_ts"]).dt.total_seconds() / 60

    df["hour"] = df["pick_ts"].dt.tz_convert("Asia/Shanghai").dt.hour
    df["date"] = df["pick_ts"].dt.tz_convert("Asia/Shanghai").dt.date.astype(str)
    df["stale"] = df["lag_min"] > BREAKPOINT_MIN

    scored = df[df["pick_success"].notna()].copy()
    scored["fail"] = 1 - scored["pick_success"].astype(int)
    unscored = df[df["pick_success"].isna()]

    return df, scored, unscored, {"rows_raw": n_raw, "rows_dup": n_dupes, "rows": len(df), "unscored": len(unscored)}


def rate_table(scored, by, bins=None):
    if bins is not None:
        key = pd.cut(scored["lag_min"], bins, right=False)
    else:
        key = scored[by]
    g = scored.groupby(key, observed=True)["fail"].agg(n="size", fails="sum", rate="mean")
    g["rate_pct"] = (g["rate"] * 100).round(1)
    return g.drop(columns="rate")


def find_breakpoint(scored):
    """在 1..65 分钟里找能把缺货率劈得最开的阈值。"""
    best = None
    for t in range(1, 66):
        hi = scored.loc[scored["lag_min"] > t, "fail"]
        lo = scored.loc[scored["lag_min"] <= t, "fail"]
        if len(hi) < 50 or len(lo) < 50:
            continue
        gap = hi.mean() - lo.mean()
        if best is None or gap > best["gap"]:
            best = {"gap": gap, "threshold_min": t, "gap_pp": round(gap * 100, 1), "n_stale": int(len(hi)),
                    "rate_stale_pct": round(hi.mean() * 100, 1), "n_fresh": int(len(lo)),
                    "rate_fresh_pct": round(lo.mean() * 100, 1)}
    best.pop("gap")
    best["lift"] = round(best["rate_stale_pct"] / best["rate_fresh_pct"], 1)
    return best


def evaluate(scored, flag):
    flag = flag.astype(bool)
    tp = int((flag & (scored["fail"] == 1)).sum())
    fp = int((flag & (scored["fail"] == 0)).sum())
    missed = int((~flag & (scored["fail"] == 1)).sum())
    baseline_min = int(scored["fail"].sum()) * MINUTES_LOST_PER_STOCKOUT
    policy_min = tp * MINUTES_LOST_PER_EARLY_FLAG + fp * MINUTES_LOST_PER_EARLY_FLAG + missed * MINUTES_LOST_PER_STOCKOUT
    return {
        "flagged": int(flag.sum()),
        "caught": tp,
        "false_flags": fp,
        "missed": missed,
        "recall_pct": round(tp / max(tp + missed, 1) * 100, 1),
        "precision_pct": round(tp / max(tp + fp, 1) * 100, 1),
        "baseline_min": baseline_min,
        "policy_min": policy_min,
        "saved_min": baseline_min - policy_min,
        "saved_hours": round((baseline_min - policy_min) / 60, 1),
        "saved_pct": round((baseline_min - policy_min) / baseline_min * 100, 1),
    }


def charts(scored, out_dir):
    bins = np.arange(0, 70, 5)
    b = rate_table(scored, None, bins=bins)
    centers = [i.left + 2.5 for i in b.index]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(centers, b["rate_pct"], marker="o", lw=2)
    ax.axvline(BREAKPOINT_MIN, color="crimson", ls="--", label=f"{BREAKPOINT_MIN} 分钟断点")
    ax.set_xlabel("账面库存同步滞后（分钟）")
    ax.set_ylabel("到位缺货率 (%)")
    ax.set_title("同步滞后 vs 到位缺货率")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "lag_vs_stockout.png"))
    plt.close(fig)

    z = scored.groupby("zone_code")["fail"].agg(n="size", rate="mean")
    zs = scored[scored["stale"]].groupby("zone_code")["fail"].agg(n_stale="size", rate_stale="mean")
    zf = scored[~scored["stale"]].groupby("zone_code")["fail"].agg(n_fresh="size", rate_fresh="mean")
    tab = z.join([zs, zf])
    for c in ("rate", "rate_stale", "rate_fresh"):
        tab[c] = (tab[c] * 100).round(1)
    tab = tab.sort_values("rate", ascending=False)
    x = np.arange(len(tab))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - 0.2, tab["rate_fresh"], 0.4, label="同步 ≤20 分钟")
    ax.bar(x + 0.2, tab["rate_stale"], 0.4, label="同步 >20 分钟")
    ax.set_xticks(x, tab.index)
    ax.set_ylabel("到位缺货率 (%)")
    ax.set_title("各库区缺货率：账面新鲜度是主要分水岭")
    ax.legend()
    fig.savefig(os.path.join(out_dir, "zone_stockout.png"))
    plt.close(fig)
    return tab.round(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/picks.csv")
    args = ap.parse_args()

    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    df, scored, unscored, meta = load(args.csv)

    print("=" * 72)
    print("0. 数据体检")
    print("=" * 72)
    print(f"原始 {meta['rows_raw']} 行 → 去重后 {meta['rows']} 行（{meta['rows_dup']} 行为逐字段完全重复，已剔除）")
    print(f"结果未回传 {meta['unscored']} 条（{meta['unscored']/meta['rows']*100:.1f}%），不进入缺货率分母")
    print(f"可判定 {len(scored)} 条，到位缺货 {int(scored.fail.sum())} 次，缺货率 {scored.fail.mean()*100:.2f}%")
    print(f"reported_stock=0 仍派单 {int((df.reported_stock == 0).sum())} 条，"
          f"其缺货率 {scored.loc[scored.reported_stock == 0, 'fail'].mean()*100:.1f}%（与整体持平 → 账面数量本身无预警力）")
    print(f"滞后相关性 r={scored['fail'].corr(scored['lag_min']):.3f}，"
          f"账面剩余量相关性 r={scored['fail'].corr(scored['reported_stock']):.3f}")

    print("\n" + "=" * 72)
    print(f"1. 断点定位")
    print("=" * 72)
    bp = find_breakpoint(scored)
    print(json.dumps(bp, ensure_ascii=False, indent=2))
    print("\n按 5 分钟分箱：")
    print(rate_table(scored, None, bins=np.arange(0, 70, 5)).to_string())

    print("\n" + "=" * 72)
    print("2. 断点两侧")
    print("=" * 72)
    print(scored.groupby("stale").agg(n=("fail", "size"), fails=("fail", "sum"), rate=("fail", "mean"))
          .assign(rate=lambda t: (t.rate * 100).round(1)).to_string())
    print(f"\n>20 分钟的派单占全部派单 {scored.stale.mean()*100:.1f}%")
    zone_tab = charts(scored, out_dir)
    print("\n各库区（缺货率 %）：")
    print(zone_tab.to_string())
    print("\n滞后最严重的库区 TOP：")
    print(scored.groupby("zone_code").agg(中位滞后min=("lag_min", "median"), 超20min占比=("stale", "mean"))
          .assign(超20min占比=lambda t: (t.超20min占比 * 100).round(1)).sort_values("超20min占比", ascending=False).to_string())

    print("\n" + "=" * 72)
    print("3. 拦截策略成本模型（省 90 分钟/次真缺货，花 20 分钟/次误标）")
    print("=" * 72)
    policies = {
        "A 仅按滞后>20min 拦截": scored["stale"],
        "B 滞后>20min 且 高架E区豁免": scored["stale"] & (scored.zone_code != "高架E区"),
        "C 滞后>20min 或 (滞后>10min 且当日已拣>80)": scored["stale"] | ((scored.lag_min > 10) & (scored.daily_picks > 80)),
    }
    rows = []
    for name, flag in policies.items():
        r = evaluate(scored, flag)
        r["policy"] = name
        rows.append(r)
        print(f"\n{name}")
        print("  " + json.dumps(r, ensure_ascii=False))
    table = pd.DataFrame(rows).set_index("policy")
    print("\n汇总：")
    print(table[["flagged", "caught", "false_flags", "missed", "recall_pct", "precision_pct", "saved_min", "saved_hours", "saved_pct"]].to_string())
    print(f"\n基线（不拦截）：{int(scored.fail.sum())} 次到位缺货 × 90 分钟 = {int(scored.fail.sum())*90} 分钟 = {int(scored.fail.sum())*90/60:.1f} 小时/周")

    stale_g = scored[scored["stale"]]
    fresh_rate = scored.loc[~scored["stale"], "fail"].mean()
    avoided = stale_g["fail"].sum() - len(stale_g) * fresh_rate
    d_min = round(avoided * MINUTES_LOST_PER_STOCKOUT)
    print(f"\nD 治本（反事实估算，不是拦截规则）：派单前把这 {len(stale_g)} 条 >{BREAKPOINT_MIN}min 的账面刷新到 {BREAKPOINT_MIN}min 以内，")
    print(f"  假设其缺货率降到新鲜组水平 {fresh_rate*100:.1f}% → 可避免 {avoided:.1f} 次缺货，"
          f"省 {d_min:.0f} 分钟 = {d_min/60:.1f} 小时/周，且零误标成本")
    d_row = {"policy": "D 治本：刷新 >20min 账面（反事实）", "flagged": 0, "caught": round(avoided, 1),
             "false_flags": 0, "missed": int(scored.fail.sum() - avoided),
             "recall_pct": round(avoided / scored.fail.sum() * 100, 1), "precision_pct": 100.0,
             "baseline_min": int(scored.fail.sum()) * 90, "policy_min": int(scored.fail.sum() * 90 - d_min),
             "saved_min": d_min, "saved_hours": round(d_min / 60, 1),
             "saved_pct": round(d_min / (scored.fail.sum() * 90) * 100, 1)}
    print("  " + json.dumps(d_row, ensure_ascii=False))
    all_rows = pd.DataFrame(rows + [d_row]).set_index("policy")
    print("\n含治本项的汇总：")
    print(all_rows[["flagged", "caught", "false_flags", "recall_pct", "saved_min", "saved_hours", "saved_pct"]].to_string())

    print("\n" + "=" * 72)
    print("4. 根因排查：滞后从哪来")
    print("=" * 72)
    print("按小时的超20min占比（%）：")
    hh = scored.groupby("hour").agg(n=("stale", "size"), 超20min占比=("stale", "mean"), 缺货率=("fail", "mean"))
    print(hh.assign(超20min占比=(hh.超20min占比 * 100).round(1), 缺货率=(hh.缺货率 * 100).round(1)).to_string())
    print("\n按日期：")
    dd = scored.groupby("date").agg(n=("stale", "size"), 超20min占比=("stale", "mean"), 缺货率=("fail", "mean"))
    print(dd.assign(超20min占比=(dd.超20min占比 * 100).round(1), 缺货率=(dd.缺货率 * 100).round(1)).to_string())

    with open(os.path.join(out_dir, "key_numbers.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "breakpoint": bp,
                   "overall_fail_pct": round(scored.fail.mean() * 100, 2),
                   "policies": {**{k: evaluate(scored, v) for k, v in policies.items()}, "D": d_row}}, fh, ensure_ascii=False, indent=2)
    print(f"\n图表与关键数字已写入 {out_dir}/")


if __name__ == "__main__":
    main()
