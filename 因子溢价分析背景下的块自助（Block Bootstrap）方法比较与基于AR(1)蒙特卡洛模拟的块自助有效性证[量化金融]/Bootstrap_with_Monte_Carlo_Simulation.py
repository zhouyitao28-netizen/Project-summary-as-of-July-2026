"""
bootstrap_new.py
================
对 118 个因子进行三种 Bootstrap 方法的比较分析：
  - Stationary Bootstrap (SB) — 随机几何块长度，保持平稳性
  - Circular Block Bootstrap (CBB) — 固定块长度，循环首尾
  - Moving Block Bootstrap (MBB) — 固定块长度，不循环

对每个因子的风险溢价均值、夏普比率、t 统计量构建 95% 置信区间，
比较三种方法在自相关保持、区间宽度、覆盖概率方面的表现。

最后选取表现突出的因子，做三种方法下的可视化。

值得注意的点：
1. 所有因子都是月度因子，不用担心Bootstrap或者蒙特卡洛模拟时，不同因子单次时间跨度不一致的问题
2. 分割样本覆盖概率是分割样本法得到的最终结果，分割样本法是把数据样本分为训练期 (2004-2019) 和验证期 (2020-2024)
    用训练期的置信区间覆盖验证期，这里结果72-85%（偏低）原因被推测为市场环境不同，因子溢价真实均值发生漂移。
3. 蒙特卡洛+Bootstrap全流程：
- 定下一个因子与其带来的一连串月份收益率数据，用AR(1)在这个时间序列中做模拟（用AR(1)的原因：1.假定平稳则均值方差可知，无需分两个样本分别训练 2.AR(1)是一个典型的自相关模型，其应该被研究自相关性的3钟bootstrape捕捉。）
- 创建一个rng(某seed)，这个rng下创建一系列不同的初始点（但是可以保证每次创造的一系列不同的初始点完全相同）。对于创造的一系列（100个）初始点，分别用3钟Bootstrape各做200次，观察覆盖范围和95%做对比，越接近越好。

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from scipy import stats as sp_stats
from numpy.random import default_rng
import warnings
import os
import time

warnings.filterwarnings("ignore")

# ============================================================
# 0. 全局配置
# ============================================================
BLOCK_SIZE = 12          # 块长度（月度数据建议 6~24）
N_REPLICATIONS = 1000    # Bootstrap 重复次数（主分析）
N_AR1_REPS = 200         # 自相关比较用较少重复
N_COVERAGE_REPS = 500    # 覆盖概率检验用重复
ALPHA = 0.05             # 显著性水平 → 95% CI
SEED = 42                # 随机种子，保证可复现
N_STANDOUT = 6           # 用于详细可视化的突出因子数量

# 数据路径
DATA_PATH_PRIMARY = (
    r"E:\Study materials\因子投资小组学习\Bootstrap方法及应用 "
    r"可回想同为distrbution free的"
    r"（1.百分位数在某两个样本数据之间的概率估计 "
    r"2.Wilcoxon Test）\factor_returns.csv"
)
DATA_PATH_FALLBACK = "factor_returns.csv"

# 输出目录：确保有一个固定的文件夹来存放程序运行产生的所有结果文件（如图片 .png 和数据 .csv）
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "bootstrap_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei",
                                    "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

print("=" * 70)
print("  因子 Bootstrap 置信区间 — 三种方法比较")
print("=" * 70)
print(f"  块长度: {BLOCK_SIZE}  |  重复次数: {N_REPLICATIONS}  |  "
      f"CI: {(1-ALPHA)*100:.0f}%")
print(f"  随机种子: {SEED}")
print("=" * 70)


# ============================================================
# 1. 数据加载
# ============================================================
def load_data():
    """加载因子收益率数据，返回 (DataFrame, array, 因子名列表)."""
    data_path = (DATA_PATH_PRIMARY
                 if os.path.exists(DATA_PATH_PRIMARY)
                 else DATA_PATH_FALLBACK)
    print(f"\n[1/7] 加载数据: {os.path.basename(data_path)}")

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    # 删除全为 NaN 的行 / 列
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    # 对剩余 NaN 做前向填充（少量缺失时）
    df = df.ffill().bfill()

    factor_names = df.columns.tolist()
    data = df.values  # (T, K) float64
    T, K = data.shape
    print(f"  样本期: {df.index[0].strftime('%Y-%m')} → "
          f"{df.index[-1].strftime('%Y-%m')}  |  "
          f"{T} 个月 × {K} 个因子")
    return df, data, factor_names


# ============================================================
# 2. Bootstrap 指标生成 (纯 NumPy 从零实现)
# 不直接取具体的数据，而取对应的行号，方便套用给不同因子下的数据
# ============================================================
def stationary_bootstrap_indices(n_obs, block_size, n_reps, rng):
    """
    Stationary Bootstrap (Politis & Romano, 1994).
    块长度服从几何分布，均值 = block_size。
    每个观测以概率 p = 1/block_size 开始新块，
    以概率 1-p 延续当前块。序列首尾循环。
    """
    indices = np.zeros((n_reps, n_obs), dtype=np.int32)
    p_new_block = 1.0 / block_size
    for b in range(n_reps):
        indices[b, 0] = rng.integers(0, n_obs)
        for t in range(1, n_obs):
            if rng.random() < p_new_block:
                indices[b, t] = rng.integers(0, n_obs)
            else:
                indices[b, t] = (indices[b, t - 1] + 1) % n_obs
    return indices


def circular_block_bootstrap_indices(n_obs, block_size, n_reps, rng):
    """
    Circular Block Bootstrap.
    与 MBB 类似，但从首尾循环的圆环上抽取定长块。
    n 个可能的块起始位置（任意位置均可），块可以跨越序列末端。
    """
    n_blocks = int(np.ceil(n_obs / block_size))
    indices = np.zeros((n_reps, n_obs), dtype=np.int32)
    for b in range(n_reps):
        starts = rng.integers(0, n_obs, size=n_blocks)
        pos = 0
        for start in starts:
            length = min(block_size, n_obs - pos)
            for j in range(length):
                indices[b, pos + j] = (start + j) % n_obs
            pos += length
    return indices


def moving_block_bootstrap_indices(n_obs, block_size, n_reps, rng):
    """
    Moving Block Bootstrap (Künsch, 1989).
    从 n - block_size + 1 个可能块中等概率抽取，不循环。
    这是最经典的块 Bootstrap。
    """
    n_blocks = int(np.ceil(n_obs / block_size))
    n_possible = n_obs - block_size + 1
    if n_possible < 1:
        raise ValueError(f"block_size ({block_size}) > n_obs ({n_obs})")
    indices = np.zeros((n_reps, n_obs), dtype=np.int32)
    for b in range(n_reps):
        starts = rng.integers(0, n_possible, size=n_blocks)
        pos = 0
        for start in starts:
            length = min(block_size, n_obs - pos)
            indices[b, pos:pos + length] = np.arange(start, start + length)
            pos += length
    return indices


# ---- 索引缓存辅助 ----
_bootstrap_index_cache = {}


def get_bootstrap_indices(method, n_obs, block_size, n_reps, seed):
    """带缓存的索引生成，避免重复计算。"""
    key = (method, n_obs, block_size, n_reps, seed)
    if key not in _bootstrap_index_cache:
        rng = default_rng(seed + hash(method) % 2**31)
        if method == "SB":
            idx = stationary_bootstrap_indices(n_obs, block_size, n_reps, rng)
        elif method == "CBB":
            idx = circular_block_bootstrap_indices(n_obs, block_size, n_reps, rng)
        elif method == "MBB":
            idx = moving_block_bootstrap_indices(n_obs, block_size, n_reps, rng)
        else:
            raise ValueError(f"未知方法: {method}")
        _bootstrap_index_cache[key] = idx
    return _bootstrap_index_cache[key]


# ============================================================
# 3. 统计量计算 (全部向量化)
# ============================================================
def compute_statistics(sample):
    """
    对 bootstrap 样本计算三个统计量。
    sample: (n_reps, n_obs, n_factors) 或 (n_obs, n_factors)

    返回:
        means, sharpes, tstats — 形状与输入的前导维度一致
    """
    if sample.ndim == 3:
        # (n_reps, n_obs, n_factors)
        n_reps, n_obs, n_factors = sample.shape
        means = sample.mean(axis=1)          # (n_reps, n_factors)
        stds = sample.std(axis=1, ddof=1)    # (n_reps, n_factors)
    else:
        # (n_obs, n_factors) — 原始数据
        n_obs, n_factors = sample.shape
        means = sample.mean(axis=0)          # (n_factors,)
        stds = sample.std(axis=0, ddof=1)    # (n_factors,)

    # 避免除零
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(stds > 1e-12, means / stds, 0.0)
        tstats = np.where(stds > 1e-12,
                          means / (stds / np.sqrt(n_obs)), 0.0)

    return means, sharpes, tstats


def ar1_coefficient(x):
    """
    AR(1) 系数 (向量化).
    x: (..., n_obs, n_factors) — 沿 n_obs 轴计算 AR(1).

    返回: (..., n_factors)
    """
    x_dm = x - x.mean(axis=-2, keepdims=True)  # demean
    # 滞后 1
    num = (x_dm[..., :-1, :] * x_dm[..., 1:, :]).sum(axis=-2)
    den = (x_dm[..., :-1, :] ** 2).sum(axis=-2)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(den > 1e-12, num / den, 0.0)
    return rho


# ============================================================
# 4. 主 Bootstrap 分析
# ============================================================
def run_bootstrap_analysis(data, factor_names, block_size, n_reps,
                           n_ar1_reps, seed):
    """
    对全部因子执行三种方法的 Bootstrap，计算三个统计量的 CI。

    返回:
        results: dict，包含所有 CI、分布统计量
    """
    T, K = data.shape
    methods = ["SB", "CBB", "MBB"]
    stats_names = ["mean", "sharpe", "tstat"]

    print(f"\n[2/7] 运行 Bootstrap 分析 ({K} 个因子 × {len(methods)} 种方法)...")

    # ---- 原始统计量 ----
    orig_mean, orig_sharpe, orig_tstat = compute_statistics(data)
    originals = {"mean": orig_mean, "sharpe": orig_sharpe, "tstat": orig_tstat}

    results = {
        "originals": originals,
        "ci": {},          # ci[method][stat] = (lower, upper) — (K,) each
        "distributions": {},  # distributions[method][stat] = (n_reps, K)
    }

    for method in methods:
        t0 = time.time()
        print(f"    {method}...", end=" ", flush=True)

        # 生成索引
        idx = get_bootstrap_indices(method, T, block_size, n_reps, seed)

        # 向量化: 一次取所有 bootstrap 样本
        boot_data = data[idx]  # (n_reps, T, K)

        # 计算统计量
        means, sharpes, tstats = compute_statistics(boot_data)

        results["distributions"][method] = {
            "mean": means, "sharpe": sharpes, "tstat": tstats}

        # 百分位 CI
        ci = {}
        for sn, dist in zip(stats_names, [means, sharpes, tstats]):
            lo = np.percentile(dist, 100 * ALPHA / 2, axis=0)   # (K,)
            hi = np.percentile(dist, 100 * (1 - ALPHA / 2), axis=0)
            ci[sn] = (lo, hi)
        results["ci"][method] = ci

        elapsed = time.time() - t0
        print(f"完成 ({elapsed:.1f}s)")

    # ---- 自相关保持检验 ----
    print(f"\n[3/7] 自相关保持检验...")
    orig_ar1 = ar1_coefficient(data)  # (K,)
    results["orig_ar1"] = orig_ar1
    results["ar1_boot"] = {}

    for method in methods:
        idx_ar1 = get_bootstrap_indices(method, T, block_size,
                                        n_ar1_reps, seed + 1)
        boot_data_ar1 = data[idx_ar1]  # (n_ar1_reps, T, K)
        boot_ar1 = ar1_coefficient(boot_data_ar1)  # (n_ar1_reps, K)
        results["ar1_boot"][method] = boot_ar1
        bias = boot_ar1.mean(axis=0) - orig_ar1
        print(f"    {method}: AR(1) 平均偏差 = {bias.mean():+.4f}")

    return results


# ============================================================
# 5. 三种方法比较
# ============================================================
def compare_methods(results, data):
    """
    比较三种方法在三个维度上的表现:
      (a) 区间宽度
      (b) 自相关保持
      (c) 覆盖概率（分割样本 + 真实数据检验）
    """
    T, K = data.shape
    methods = ["SB", "CBB", "MBB"]
    stats_names = ["mean", "sharpe", "tstat"]
    comparison = {}

    print(f"\n[4/7] 三种方法比较...")

    # ---- (a) 区间宽度 ----
    print(f"\n  ── (a) 区间宽度 ──")
    comparison["width"] = {}
    for method in methods:
        widths = {}
        for sn in stats_names:
            lo, hi = results["ci"][method][sn]
            w = hi - lo
            widths[sn] = w
        comparison["width"][method] = widths

    for sn in stats_names:
        ws = [comparison["width"][m][sn].mean() for m in methods]
        best = methods[np.argmin(ws)]
        vals = "  ".join(f"{m}={ws[i]:.4f}" for i, m in enumerate(methods))
        print(f"    {sn:>6s}: {vals}  → 最窄: {best}")

    # ---- (b) 自相关保持 ----
    print(f"\n  ── (b) 自相关保持 ──")
    orig_ar1 = results["orig_ar1"]
    comparison["ar1"] = {}
    for method in methods:
        boot_ar1 = results["ar1_boot"][method]  # (n_reps, K)
        bias = boot_ar1.mean(axis=0) - orig_ar1  # (K,)
        rmse = np.sqrt(((boot_ar1 - orig_ar1) ** 2).mean(axis=0))  # (K,)
        # 原始 AR(1) 是否落在 bootstrap AR(1) 的 95% CI 内
        lo = np.percentile(boot_ar1, 2.5, axis=0)
        hi = np.percentile(boot_ar1, 97.5, axis=0)
        covered = (orig_ar1 >= lo) & (orig_ar1 <= hi)
        comparison["ar1"][method] = {
            "bias_mean": bias.mean(),
            "bias_median": np.median(bias),
            "rmse_mean": rmse.mean(),
            "coverage_rate": covered.mean(),
        }
        c = comparison["ar1"][method]
        print(f"    {method}: bias={c['bias_mean']:+.4f}  "
              f"RMSE={c['rmse_mean']:.4f}  "
              f"AR(1)覆盖率={c['coverage_rate']:.2%}")

    # ---- (c) 覆盖概率 (分割样本) ----
    print(f"\n  ── (c) 覆盖概率 (前 180 月估计, 后 57 月验证) ──")
    split = 180
    data_train = data[:split]     # (180, K)
    data_test = data[split:]      # (57, K)

    test_mean, test_sharpe, test_tstat = compute_statistics(data_test)
    test_stats = {"mean": test_mean, "sharpe": test_sharpe, "tstat": test_tstat}

    comparison["coverage_split"] = {}

    for method in methods:
        t0 = time.time()
        idx_cov = get_bootstrap_indices(method, split, BLOCK_SIZE,
                                        N_COVERAGE_REPS, SEED + 2)
        boot_train = data_train[idx_cov]  # (N_COVERAGE_REPS, 180, K)
        tr_means, tr_sharpes, tr_tstats = compute_statistics(boot_train)
        tr_dists = {"mean": tr_means, "sharpe": tr_sharpes, "tstat": tr_tstats}

        cov_rates = {}
        for sn in stats_names:
            lo = np.percentile(tr_dists[sn], 100 * ALPHA / 2, axis=0)
            hi = np.percentile(tr_dists[sn], 100 * (1 - ALPHA / 2), axis=0)
            covered = (test_stats[sn] >= lo) & (test_stats[sn] <= hi)
            cov_rates[sn] = covered.mean()

        comparison["coverage_split"][method] = cov_rates
        vals = "  ".join(f"{sn}={cov_rates[sn]:.2%}" for sn in stats_names)
        t_elapsed = time.time() - t0
        print(f"    {method}: {vals}  ({t_elapsed:.1f}s)")

    return comparison


# ============================================================
# 6. 汇总报告
# ============================================================
def print_summary(data, factor_names, results, comparison):
    """打印完整的分析汇总。"""
    K = len(factor_names)
    methods = ["SB", "CBB", "MBB"]
    stats_names = ["mean", "sharpe", "tstat"]

    print(f"\n\n{'='*70}")
    print(f"  汇总报告")
    print(f"{'='*70}")

    # --- 原始统计量概览 ---
    print(f"\n  ── 原始统计量 (全部 {K} 个因子) ──")
    orig = results["originals"]
    for sn, label in zip(stats_names,
                         ["风险溢价均值", "夏普比率", "t 统计量"]):
        vals = orig[sn]
        print(f"    {label:12s}: 均值={vals.mean():+.4f}  "
              f"中位数={np.median(vals):+.4f}  "
              f"标准差={vals.std():.4f}  "
              f"显著比例(5%)= {(sp_stats.t.sf(np.abs(vals), data.shape[0]-1) * 2 < 0.05).mean():.1%}")

    # --- CI 宽度汇总 ---
    print(f"\n  ── 平均 CI 宽度 ──")
    header = f"    {'统计量':<10s}"
    for m in methods:
        header += f"  {m:>10s}"
    print(header)
    print(f"    {'-'*40}")
    for sn in stats_names:
        row = f"    {sn:<10s}"
        for m in methods:
            w = comparison["width"][m][sn].mean()
            row += f"  {w:>10.4f}"
        print(row)

    # --- 自相关保持 ---
    print(f"\n  ── 自相关保持 ──")
    print(f"    {'指标':<18s}", end="")
    for m in methods:
        print(f"  {m:>10s}", end="")
    print()
    print(f"    {'-'*46}")
    for metric in ["bias_mean", "rmse_mean", "coverage_rate"]:
        label = {"bias_mean": "AR(1)偏差(均值)",
                 "rmse_mean": "AR(1) RMSE",
                 "coverage_rate": "AR(1)覆盖率"}[metric]
        print(f"    {label:<18s}", end="")
        for m in methods:
            val = comparison["ar1"][m][metric]
            fmt = ".4f" if metric != "coverage_rate" else ".2%"
            print(f"  {val:>10{fmt}}", end="")
        print()

    # --- 覆盖概率 ---
    print(f"\n  ── 分割样本覆盖概率 ──")
    header = f"    {'统计量':<10s}"
    for m in methods:
        header += f"  {m:>10s}"
    print(header)
    print(f"    {'-'*40}")
    for sn in stats_names:
        row = f"    {sn:<10s}"
        for m in methods:
            cr = comparison["coverage_split"][m][sn]
            row += f"  {cr:>9.1%}"
        print(row)

    # --- 综合评分 ---
    print(f"\n  ── 综合排名 (越小越好) ──")
    scores = {m: 0 for m in methods}
    # 区间宽度: 排名 (1=最窄)
    for sn in stats_names:
        ws = [comparison["width"][m][sn].mean() for m in methods]
        ranked = np.argsort(ws)  # 0=最小
        for rank, mi in enumerate(ranked):
            scores[methods[mi]] += rank
    # AR(1) RMSE: 排名
    rmses = [comparison["ar1"][m]["rmse_mean"] for m in methods]
    ranked = np.argsort(rmses)
    for rank, mi in enumerate(ranked):
        scores[methods[mi]] += rank
    # 覆盖概率 (均值接近 95% 最好): 排名
    for sn in stats_names:
        crs = [comparison["coverage_split"][m][sn] for m in methods]
        # 距离 95% 的绝对值
        dists = [abs(cr - (1 - ALPHA)) for cr in crs]
        ranked = np.argsort(dists)
        for rank, mi in enumerate(ranked):
            scores[methods[mi]] += rank

    for m in methods:
        print(f"    {m}: 综合得分={scores[m]} (越低越好)")
    best = min(scores, key=scores.get)
    print(f"    → 推荐方法: {best}")


# ============================================================
# 7. 突出因子选择
# ============================================================
def select_standout_factors(results, factor_names, n_standout=6):
    """选择在均值、夏普、t 统计量方面表现突出的因子。"""
    orig = results["originals"]
    K = len(factor_names)

    # 综合评分: 各指标的标准化排名
    rankings = np.zeros((K, 3))
    for i, sn in enumerate(["mean", "sharpe", "tstat"]):
        # 按绝对值排名
        abs_vals = np.abs(orig[sn])
        # rank (1 = 最大绝对值)
        rankings[:, i] = K - np.argsort(np.argsort(abs_vals))

    # 平均排名
    avg_rank = rankings.mean(axis=1)
    top_idx = np.argsort(avg_rank)[:n_standout]

    print(f"\n[5/7] 突出因子选择 (Top {n_standout}):")
    print(f"    {'因子':<35s}  {'均值':>8s}  {'夏普':>8s}  "
          f"{'t值':>8s}  {'排名':>6s}")
    print(f"    {'-'*70}")
    for i in top_idx:
        print(f"    {factor_names[i]:<35s}  {orig['mean'][i]:>+8.4f}  "
              f"{orig['sharpe'][i]:>8.4f}  {orig['tstat'][i]:>8.4f}  "
              f"{avg_rank[i]:>6.1f}")

    return top_idx, avg_rank


# ============================================================
# 8. 可视化
# ============================================================
def plot_summary_comparison(comparison, methods, stats_names):
    """汇总比较图: 三方法在三指标上的表现。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("三种 Bootstrap 方法比较", fontsize=15, fontweight="bold",
                 y=0.98)

    colors = {"SB": "#2E86AB", "CBB": "#A23B72", "MBB": "#F18F01"}
    x = np.arange(len(stats_names))
    w = 0.25

    # (a) 平均 CI 宽度
    ax = axes[0, 0]
    for i, m in enumerate(methods):
        ws = [comparison["width"][m][sn].mean() for sn in stats_names]
        ax.bar(x + i * w, ws, w, label=m, color=colors[m], alpha=0.85,
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x + w)
    ax.set_xticklabels(["均值", "夏普比率", "t 统计量"])
    ax.set_ylabel("平均 CI 宽度")
    ax.set_title("(a) 置信区间宽度 (越小越好)", fontweight="bold")
    ax.legend(fontsize=9)

    # (b) AR(1) 偏差
    ax = axes[0, 1]
    for i, m in enumerate(methods):
        boot_ar1 = comparison.get("_boot_ar1_ref", {}).get(m)
        # 用 stored results
        bias_vals = comparison["ar1"][m]["bias_mean"]
        rmse_vals = comparison["ar1"][m]["rmse_mean"]
        ax.bar(x + i * w - w/2, [bias_vals], w/2,
               label=f"{m} bias" if i == 0 else "",
               color=colors[m], alpha=0.6)
        ax.bar(x + i * w, [rmse_vals], w/2,
               label=f"{m} RMSE" if i == 0 else "",
               color=colors[m], alpha=1.0)
    # 简化: 直接对比 bias 和 RMSE
    ax.clear()
    metrics_ar1 = ["|偏差|", "RMSE"]
    x2 = np.arange(len(metrics_ar1))
    for i, m in enumerate(methods):
        vals = [abs(comparison["ar1"][m]["bias_mean"]),
                comparison["ar1"][m]["rmse_mean"]]
        ax.bar(x2 + i * w, vals, w, label=m, color=colors[m], alpha=0.85,
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x2 + w)
    ax.set_xticklabels(metrics_ar1)
    ax.set_ylabel("数值")
    ax.set_title("(b) 自相关保持 (越小越好)", fontweight="bold")
    ax.legend(fontsize=9)

    # (c) 覆盖概率
    ax = axes[1, 0]
    for i, m in enumerate(methods):
        crs = [comparison["coverage_split"][m][sn] for sn in stats_names]
        ax.bar(x + i * w, crs, w, label=m, color=colors[m], alpha=0.85,
               edgecolor="white", linewidth=0.5)
    ax.axhline(y=0.95, color="black", linestyle="--", linewidth=1,
               label="名义水平 95%")
    ax.set_xticks(x + w)
    ax.set_xticklabels(["均值", "夏普比率", "t 统计量"])
    ax.set_ylabel("覆盖概率")
    ax.set_title("(c) 分割样本覆盖概率 (越接近 95% 越好)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    # (d) AR(1) 覆盖率
    ax = axes[1, 1]
    for i, m in enumerate(methods):
        cr = comparison["ar1"][m]["coverage_rate"]
        ax.bar(i, cr, 0.5, color=colors[m], alpha=0.85,
               edgecolor="white", linewidth=0.5)
    ax.axhline(y=0.95, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_ylabel("AR(1) 覆盖率")
    ax.set_title("(d) AR(1) 系数落入 Bootstrap CI 的比例", fontweight="bold")
    ax.set_ylim(0, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "summary_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  汇总图已保存: {path}")
    plt.close(fig)


def plot_standout_factor(results, factor_names, top_idx, data, comparison):
    """为每个突出因子绘制 Bootstrap 分布图。"""
    methods = ["SB", "CBB", "MBB"]
    stats_info = [
        ("mean", "风险溢价均值", "mean"),
        ("sharpe", "夏普比率", "sharpe"),
        ("tstat", "t 统计量", "tstat"),
    ]
    colors = {"SB": "#2E86AB", "CBB": "#A23B72", "MBB": "#F18F01"}
    orig = results["originals"]

    for fi, idx in enumerate(top_idx):
        fname = factor_names[idx]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"{fname}  — 三种 Bootstrap 方法 95% CI 比较",
                     fontsize=14, fontweight="bold")

        for col, (sn, sn_label, sn_key) in enumerate(stats_info):
            ax = axes[col]
            orig_val = orig[sn_key][idx]

            for method in methods:
                dist = results["distributions"][method][sn_key][:, idx]
                # KDE + 直方图
                ax.hist(dist, bins=40, density=True, alpha=0.3,
                        color=colors[method], edgecolor="white",
                        linewidth=0.3)
                # KDE 曲线
                kde = sp_stats.gaussian_kde(dist)
                x_kde = np.linspace(dist.min(), dist.max(), 200)
                ax.plot(x_kde, kde(x_kde), color=colors[method],
                        linewidth=2, label=method)

                # CI 竖线
                lo_arr, hi_arr = results["ci"][method][sn_key]
                lo = lo_arr[idx]
                hi = hi_arr[idx]
                ax.axvline(lo, color=colors[method], linestyle="--",
                           linewidth=1, alpha=0.6)
                ax.axvline(hi, color=colors[method], linestyle="--",
                           linewidth=1, alpha=0.6)

            # 原始值
            ax.axvline(orig_val, color="black", linestyle="-",
                       linewidth=2, label="原始值")
            ax.set_xlabel(sn_label)
            ax.set_ylabel("密度")
            if col == 0:
                ax.legend(fontsize=8, loc="upper right")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        safe_name = fname.replace("/", "_").replace("\\", "_")
        path = os.path.join(OUTPUT_DIR, f"standout_{fi+1:02d}_{safe_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  因子图已保存: {path}")
        plt.close(fig)


def plot_ci_width_heatmap(results, factor_names, top_idx, comparison):
    """CI 宽度热力图: 突出因子 × 方法 × 统计量。"""
    methods = ["SB", "CBB", "MBB"]
    stats_names = ["mean", "sharpe", "tstat"]

    n_factors = len(top_idx)
    data_matrix = np.zeros((n_factors, len(methods) * len(stats_names)))

    col_labels = []
    for m in methods:
        for sn in stats_names:
            col_labels.append(f"{m}\n{sn}")

    for i, idx in enumerate(top_idx):
        col = 0
        for m in methods:
            for sn in stats_names:
                lo_arr, hi_arr = results["ci"][m][sn]
                data_matrix[i, col] = hi_arr[idx] - lo_arr[idx]
                col += 1

    fig, ax = plt.subplots(figsize=(12, max(5, n_factors * 0.6)))
    im = ax.imshow(data_matrix, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(n_factors))
    ax.set_yticklabels([factor_names[i] for i in top_idx], fontsize=9)

    # 在每个格子标注宽度
    for i in range(n_factors):
        for j in range(data_matrix.shape[1]):
            ax.text(j, i, f"{data_matrix[i, j]:.3f}",
                    ha="center", va="center", fontsize=7)

    ax.set_title("突出因子 CI 宽度热力图", fontweight="bold", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8, label="CI 宽度")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ci_width_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  热力图已保存: {path}")
    plt.close(fig)


def plot_coverage_summary(comparison, methods, stats_names):
    """覆盖概率汇总柱状图 (分割样本)."""
    colors = {"SB": "#2E86AB", "CBB": "#A23B72", "MBB": "#F18F01"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("分割样本覆盖概率 — 三种方法对比",
                 fontweight="bold", fontsize=14)

    for col, sn in enumerate(stats_names):
        ax = axes[col]
        sn_labels = {"mean": "风险溢价均值",
                     "sharpe": "夏普比率",
                     "tstat": "t 统计量"}
        crs = [comparison["coverage_split"][m][sn] for m in methods]
        bars = ax.bar(methods, crs, color=[colors[m] for m in methods],
                      alpha=0.85, edgecolor="white")
        ax.axhline(y=0.95, color="black", linestyle="--", linewidth=1.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("覆盖概率")
        ax.set_title(sn_labels.get(sn, sn), fontweight="bold")

        # 标注数值
        for bar, cr in zip(bars, crs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{cr:.1%}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(OUTPUT_DIR, "coverage_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  覆盖概率图已保存: {path}")
    plt.close(fig)


# ============================================================
# 9. 额外: 蒙特卡洛覆盖概率 (针对突出因子, AR(1) 模拟)
# ============================================================
def mc_coverage_simulation(data, factor_names, top_idx, results,
                           n_sim=100, n_boot=200):
    """
    蒙特卡洛覆盖概率检验:
    1. 对每个突出因子拟合 AR(1)
    2. 模拟 n_sim 条路径 (已知真实均值)
    3. 对每条路径做 Bootstrap, 检查 CI 是否覆盖真实均值
    """
    methods = ["SB", "CBB", "MBB"]
    T, K = data.shape

    print(f"\n[6/7] 蒙特卡洛覆盖概率 (Top {len(top_idx)} 因子, "
          f"{n_sim} 条模拟路径)...")

    mc_results = {m: {"mean": [], "sharpe": [], "tstat": []}
                  for m in methods}

    rng = default_rng(SEED + 100)

    for fi, idx in enumerate(top_idx):
        fname = factor_names[idx]
        x = data[:, idx]  # (T,)

        # ---- 拟合 AR(1): x_t = c + ρ x_{t-1} + ε_t ----
        x_lag = x[:-1]
        x_cur = x[1:]
        X_dm = x_lag - x_lag.mean()
        Y_dm = x_cur - x_cur.mean()
        rho_hat = (X_dm * Y_dm).sum() / (X_dm ** 2).sum()
        # 即/beta_LS
        c_hat = x_cur.mean() - rho_hat * x_lag.mean()
        # 即/alpha_LS
        resid = Y_dm - rho_hat * X_dm
        sigma_eps = resid.std(ddof=1)

        # 真实均值 (平稳 AR(1) 的无条件均值)
        if abs(rho_hat) < 1.0:
            true_mean = c_hat / (1 - rho_hat)
        else:
            true_mean = x.mean()  # 非平稳时回退

        # 真实波动率 (近似)
        true_std = np.sqrt(sigma_eps ** 2 / (1 - rho_hat ** 2)) if abs(rho_hat) < 1 else x.std()
        true_sharpe = true_mean / true_std if true_std > 1e-12 else 0.0
        true_tstat = true_mean / (true_std / np.sqrt(T)) if true_std > 1e-12 else 0.0

        truths = {"mean": true_mean, "sharpe": true_sharpe, "tstat": true_tstat}
# 注：参数n_sim=100, n_boot=200
        for si in range(n_sim):
            # 模拟 AR(1) 路径
            sim = np.zeros(T)
            sim[0] = rng.normal(true_mean, true_std)
            for t in range(1, T):
                sim[t] = c_hat + rho_hat * sim[t - 1] + rng.normal(0, sigma_eps)

            # 对模拟路径做 Bootstrap
            for method in methods:
                idx_boot = get_bootstrap_indices(method, T, BLOCK_SIZE,
                                                 n_boot, SEED + 200 + si)
                boot_sim = sim[idx_boot]  # (n_boot, T)
                b_mean = boot_sim.mean(axis=1)
                b_std = boot_sim.std(axis=1, ddof=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    b_sharpe = np.where(b_std > 1e-12, b_mean / b_std, 0.0)
                    b_tstat = np.where(b_std > 1e-12,
                                       b_mean / (b_std / np.sqrt(T)), 0.0)

                # 覆盖检查
                for sn, b_dist, tv in [("mean", b_mean, true_mean),
                                       ("sharpe", b_sharpe, true_sharpe),
                                       ("tstat", b_tstat, true_tstat)]:
                    lo = np.percentile(b_dist, 100 * ALPHA / 2)
                    hi = np.percentile(b_dist, 100 * (1 - ALPHA / 2))
                    mc_results[method][sn].append(1.0 if lo <= tv <= hi else 0.0)

    # 汇总
    print(f"\n  ── MC 覆盖概率 (真实 DGP: AR(1)) ──")
    header = f"    {'方法':<6s}"
    for sn in ["mean", "sharpe", "tstat"]:
        header += f"  {sn:>10s}"
    print(header)
    print(f"    {'-'*40}")
    for method in methods:
        row = f"    {method:<6s}"
        for sn in ["mean", "sharpe", "tstat"]:
            cr = np.mean(mc_results[method][sn])
            row += f"  {cr:>9.1%}"
        print(row)

    return mc_results


# ============================================================
# 10. 综合可视化: 全因子排名
# ============================================================
def plot_all_factors_ranking(results, factor_names, comparison):
    """绘制全部 118 个因子的统计量排名 + CI 宽度。"""
    methods = ["SB", "CBB", "MBB"]
    orig = results["originals"]
    K = len(factor_names)

    # 排序: 按绝对 t 统计量
    sort_idx = np.argsort(np.abs(orig["tstat"]))[::-1]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"全部 {K} 个因子 Bootstrap 分析总览",
                 fontweight="bold", fontsize=15)

    colors = {"SB": "#2E86AB", "CBB": "#A23B72", "MBB": "#F18F01"}

    # (a) t 统计量排名 (带 95% CI) — 取前 30
    ax = axes[0, 0]
    n_show = min(30, K)
    top_sort = sort_idx[:n_show]
    x_pos = np.arange(n_show)
    # 用 SB 的 CI
    lo = results["ci"]["SB"]["tstat"][0][top_sort]
    hi = results["ci"]["SB"]["tstat"][1][top_sort]
    tvals = orig["tstat"][top_sort]
    ax.errorbar(x_pos, tvals,
                yerr=[tvals - lo, hi - tvals],
                fmt="o", capsize=3, markersize=5,
                color=colors["SB"], alpha=0.7, linewidth=1)
    ax.axhline(y=1.96, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(y=-1.96, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_xticks(x_pos[::3])
    ax.set_xticklabels([factor_names[i][:12] for i in top_sort[::3]],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("t 统计量")
    ax.set_title(f"(a) Top {n_show} |t| 因子 + 95% CI (SB)", fontweight="bold")

    # (b) 三种方法 CI 宽度分布 (所有因子)
    ax = axes[0, 1]
    width_data = []
    labels = []
    for m in methods:
        for sn, sn_label in [("mean", "均值"), ("sharpe", "夏普"), ("tstat", "t值")]:
            w = comparison["width"][m][sn]
            width_data.append(w)
            labels.append(f"{m}-{sn_label}")
    bp = ax.boxplot(width_data, labels=labels, patch_artist=True,
                    showfliers=False)
    for patch, m in zip(bp["boxes"],
                        methods * 3):
        patch.set_facecolor(colors[m])
        patch.set_alpha(0.6)
    ax.set_ylabel("CI 宽度")
    ax.set_title("(b) CI 宽度分布 (所有因子)", fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=8)

    # (c) AR(1) 偏差分布
    ax = axes[1, 0]
    ar1_bias_data = []
    ar1_labels = []
    for m in methods:
        boot_ar1 = results["ar1_boot"][m]  # (n_reps, K)
        orig_ar1 = results["orig_ar1"]      # (K,)
        bias = boot_ar1.mean(axis=0) - orig_ar1
        ar1_bias_data.append(bias)
        ar1_labels.append(m)
    bp2 = ax.boxplot(ar1_bias_data, labels=ar1_labels, patch_artist=True)
    for patch, m in zip(bp2["boxes"], methods):
        patch.set_facecolor(colors[m])
        patch.set_alpha(0.6)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("AR(1) 偏差 (Bootstrap - 原始)")
    ax.set_title("(c) 自相关保持: AR(1) 偏差分布", fontweight="bold")

    # (d) 原始均值 vs 夏普散点图 (所有因子)
    ax = axes[1, 1]
    scatter = ax.scatter(orig["mean"], orig["sharpe"],
                         c=np.abs(orig["tstat"]), cmap="viridis",
                         alpha=0.6, edgecolors="white", linewidth=0.3,
                         s=40)
    # 标注突出因子
    standout_mask = np.zeros(K, dtype=bool)
    # top by abs mean
    standout_mask[np.argsort(np.abs(orig["mean"]))[-5:]] = True
    standout_mask[np.argsort(np.abs(orig["sharpe"]))[-5:]] = True
    standout_mask[np.argsort(np.abs(orig["tstat"]))[-5:]] = True
    for i in np.where(standout_mask)[0]:
        ax.annotate(factor_names[i][:15],
                    (orig["mean"][i], orig["sharpe"][i]),
                    fontsize=6, alpha=0.8,
                    arrowprops=dict(arrowstyle="->", color="gray",
                                    alpha=0.5))
    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.set_xlabel("风险溢价均值")
    ax.set_ylabel("夏普比率")
    ax.set_title("(d) 均值-夏普散点图 (颜色=|t值|)", fontweight="bold")
    plt.colorbar(scatter, ax=ax, shrink=0.8, label="|t 统计量|")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUTPUT_DIR, "all_factors_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  全因子总览图已保存: {path}")
    plt.close(fig)


# ============================================================
# 主函数
# ============================================================
def main():
    t_start = time.time()

    # 1. 加载数据
    df, data, factor_names = load_data()
    T, K = data.shape

    # 2. Bootstrap 分析
    results = run_bootstrap_analysis(data, factor_names, BLOCK_SIZE,
                                     N_REPLICATIONS, N_AR1_REPS, SEED)

    # 3. 三种方法比较
    comparison = compare_methods(results, data)

    # 4. 汇总报告
    print_summary(data, factor_names, results, comparison)

    # 5. 突出因子选择
    top_idx, avg_rank = select_standout_factors(results, factor_names,
                                                N_STANDOUT)

    # 6. 蒙特卡洛覆盖概率 (AR(1) 模拟)
    mc_results = mc_coverage_simulation(data, factor_names, top_idx, results,
                                        n_sim=100, n_boot=200)

    # 7. 可视化
    print(f"\n[7/7] 生成可视化...")
    plot_summary_comparison(comparison, ["SB", "CBB", "MBB"],
                            ["mean", "sharpe", "tstat"])
    plot_standout_factor(results, factor_names, top_idx, data, comparison)
    plot_ci_width_heatmap(results, factor_names, top_idx, comparison)
    plot_coverage_summary(comparison, ["SB", "CBB", "MBB"],
                          ["mean", "sharpe", "tstat"])
    plot_all_factors_ranking(results, factor_names, comparison)

    # 8. 导出结果 CSV
    export_results(results, factor_names, comparison, top_idx)

    t_total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  全部完成! 总耗时: {t_total:.1f}s  "
          f"({t_total/60:.1f} min)")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"{'='*70}")


def export_results(results, factor_names, comparison, top_idx):
    """导出关键结果到 CSV 文件。"""
    methods = ["SB", "CBB", "MBB"]
    orig = results["originals"]
    K = len(factor_names)

    # 各因子各方法的 CI
    rows = []
    for i in range(K):
        row = {
            "factor": factor_names[i],
            "orig_mean": orig["mean"][i],
            "orig_sharpe": orig["sharpe"][i],
            "orig_tstat": orig["tstat"][i],
            "orig_ar1": results["orig_ar1"][i],
        }
        for m in methods:
            for sn in ["mean", "sharpe", "tstat"]:
                lo, hi = results["ci"][m][sn]
                row[f"{m}_{sn}_ci_lower"] = lo[i]
                row[f"{m}_{sn}_ci_upper"] = hi[i]
                row[f"{m}_{sn}_ci_width"] = hi[i] - lo[i]
        rows.append(row)

    df_out = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "all_factors_ci.csv")
    df_out.to_csv(path, index=False)
    print(f"\n  结果 CSV 已导出: {path}")

    # 突出因子单独导出
    df_top = df_out.iloc[top_idx]
    path_top = os.path.join(OUTPUT_DIR, "standout_factors_ci.csv")
    df_top.to_csv(path_top, index=False)
    print(f"  突出因子 CSV 已导出: {path_top}")


# ============================================================
if __name__ == "__main__":
    main()
