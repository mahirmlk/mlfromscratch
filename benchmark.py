"""
Benchmark: Our implementations vs sklearn — runtime and accuracy.

Produces log-log Big-O curves showing actual measured scaling behavior.
Saves plots to benchmark_results/.

Run: python benchmark.py
"""

import numpy as np
import time
import os
import json
from collections import defaultdict

# sklearn imports
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.linear_model import Ridge as SkRidge
from sklearn.neighbors import KNeighborsClassifier as SkKNN
from sklearn.naive_bayes import GaussianNB as SkGaussianNB
from sklearn.tree import DecisionTreeClassifier as SkDecisionTree
from sklearn.ensemble import GradientBoostingClassifier as SkGradientBoosting
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split

# our implementations
import sys
_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)

from linear_regression.linear_regression import LinearRegression
from ridge_regression.ridge_regression import RidgeRegression
from knn.knn import KNN
from naive_bayes.naive_bayes import GaussianNB
from decision_tree.decision_tree import DecisionTree
from gradient_boosting.gradient_boosting import GradientBoosting
from knn.knn_naive import KNNNaive
from decision_tree.decision_tree_naive import DecisionTreeNaive
from linear_regression.linear_regression_naive import LinearRegressionNaiveGD

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

np.random.seed(42)

OUT_DIR = os.path.join(_root, "benchmark_results")
os.makedirs(OUT_DIR, exist_ok=True)


def timer(fn, *args, **kwargs):
    """Run fn and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


# ─────────────────────────────────────────────────────────────────────
# 1. LINEAR REGRESSION: Ours vs sklearn, varying n_samples
# ─────────────────────────────────────────────────────────────────────
def benchmark_linear_regression():
    print("\n" + "=" * 60)
    print("LINEAR REGRESSION: Ours vs sklearn")
    print("=" * 60)

    d = 10
    sizes = [100, 500, 1000, 5000, 10000, 50000]
    results = {"n": [], "ours_time": [], "sklearn_time": [], "ours_mse": [], "sklearn_mse": []}

    for n in sizes:
        X, y = make_regression(n_samples=n, n_features=d, noise=10, random_state=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        _, t_ours = timer(lambda: LinearRegression(solver='closed_form').fit(X_tr, y_tr))
        m = LinearRegression(solver='closed_form')
        m.fit(X_tr, y_tr)
        mse_ours = mean_squared_error(y_te, m.predict(X_te))

        _, t_sk = timer(lambda: SkLinearRegression().fit(X_tr, y_tr))
        m_sk = SkLinearRegression()
        m_sk.fit(X_tr, y_tr)
        mse_sk = mean_squared_error(y_te, m_sk.predict(X_te))

        results["n"].append(n)
        results["ours_time"].append(t_ours)
        results["sklearn_time"].append(t_sk)
        results["ours_mse"].append(mse_ours)
        results["sklearn_mse"].append(mse_sk)

        print(f"  n={n:>6d}  ours={t_ours:.4f}s  sklearn={t_sk:.4f}s  "
              f"mse_ratio={mse_ours/mse_sk:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────
# 2. KNN: Ours vs sklearn, varying n_samples
# ─────────────────────────────────────────────────────────────────────
def benchmark_knn():
    print("\n" + "=" * 60)
    print("KNN: Ours vs sklearn")
    print("=" * 60)

    d = 10
    sizes = [100, 500, 1000, 2000, 5000]
    results = {"n": [], "ours_time": [], "sklearn_time": [], "ours_acc": [], "sklearn_acc": []}

    for n in sizes:
        X, y = make_classification(n_samples=n, n_features=d, n_informative=5,
                                   n_redundant=2, random_state=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        _, t_ours = timer(lambda: (KNN(k=5).fit(X_tr, y_tr), None))
        m = KNN(k=5)
        m.fit(X_tr, y_tr)
        _, t_pred_ours = timer(lambda: m.predict(X_te))
        acc_ours = accuracy_score(y_te, m.predict(X_te))

        _, t_sk = timer(lambda: SkKNN(n_neighbors=5).fit(X_tr, y_tr))
        m_sk = SkKNN(n_neighbors=5)
        m_sk.fit(X_tr, y_tr)
        _, t_pred_sk = timer(lambda: m_sk.predict(X_te))
        acc_sk = accuracy_score(y_te, m_sk.predict(X_te))

        total_ours = t_ours + t_pred_ours
        total_sk = t_sk + t_pred_sk

        results["n"].append(n)
        results["ours_time"].append(total_ours)
        results["sklearn_time"].append(total_sk)
        results["ours_acc"].append(acc_ours)
        results["sklearn_acc"].append(acc_sk)

        print(f"  n={n:>6d}  ours={total_ours:.4f}s  sklearn={total_sk:.4f}s  "
              f"acc_ours={acc_ours:.3f}  acc_sk={acc_sk:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────
# 3. NAIVE BAYES: Ours vs sklearn
# ─────────────────────────────────────────────────────────────────────
def benchmark_naive_bayes():
    print("\n" + "=" * 60)
    print("NAIVE BAYES: Ours vs sklearn")
    print("=" * 60)

    d = 10
    sizes = [100, 500, 1000, 5000, 10000, 50000]
    results = {"n": [], "ours_time": [], "sklearn_time": [], "ours_acc": [], "sklearn_acc": []}

    for n in sizes:
        X, y = make_classification(n_samples=n, n_features=d, n_informative=5,
                                   n_redundant=2, random_state=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        m = GaussianNB()
        _, t_ours = timer(lambda: m.fit(X_tr, y_tr))
        acc_ours = accuracy_score(y_te, m.predict(X_te))

        m_sk = SkGaussianNB()
        _, t_sk = timer(lambda: m_sk.fit(X_tr, y_tr))
        acc_sk = accuracy_score(y_te, m_sk.predict(X_te))

        results["n"].append(n)
        results["ours_time"].append(t_ours)
        results["sklearn_time"].append(t_sk)
        results["ours_acc"].append(acc_ours)
        results["sklearn_acc"].append(acc_sk)

        print(f"  n={n:>6d}  ours={t_ours:.4f}s  sklearn={t_sk:.4f}s  "
              f"acc_ours={acc_ours:.3f}  acc_sk={acc_sk:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────
# 4. DECISION TREE: Ours vs sklearn
# ─────────────────────────────────────────────────────────────────────
def benchmark_decision_tree():
    print("\n" + "=" * 60)
    print("DECISION TREE: Ours vs sklearn")
    print("=" * 60)

    d = 10
    sizes = [100, 500, 1000, 2000, 5000]
    results = {"n": [], "ours_time": [], "sklearn_time": [], "ours_acc": [], "sklearn_acc": []}

    for n in sizes:
        X, y = make_classification(n_samples=n, n_features=d, n_informative=5,
                                   n_redundant=2, random_state=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        m = DecisionTree(max_depth=10)
        _, t_ours = timer(lambda: m.fit(X_tr, y_tr))
        acc_ours = accuracy_score(y_te, m.predict(X_te))

        m_sk = SkDecisionTree(max_depth=10, random_state=42)
        _, t_sk = timer(lambda: m_sk.fit(X_tr, y_tr))
        acc_sk = accuracy_score(y_te, m_sk.predict(X_te))

        results["n"].append(n)
        results["ours_time"].append(t_ours)
        results["sklearn_time"].append(t_sk)
        results["ours_acc"].append(acc_ours)
        results["sklearn_acc"].append(acc_sk)

        print(f"  n={n:>6d}  ours={t_ours:.4f}s  sklearn={t_sk:.4f}s  "
              f"acc_ours={acc_ours:.3f}  acc_sk={acc_sk:.3f}")

    return results


# ─────────────────────────────────────────────────────────────────────
# 5. NAIVE vs OPTIMIZED: KNN
# ─────────────────────────────────────────────────────────────────────
def benchmark_naive_vs_optimized_knn():
    print("\n" + "=" * 60)
    print("NAIVE vs OPTIMIZED: KNN")
    print("=" * 60)

    d = 5
    sizes = [50, 100, 200, 500, 1000]
    results = {"n": [], "naive_time": [], "optimized_time": [], "speedup": []}

    for n in sizes:
        X, y = make_classification(n_samples=n, n_features=d, n_informative=3,
                                   random_state=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        # naive
        m_naive = KNNNaive(k=5)
        m_naive.fit(X_tr, y_tr)
        _, t_naive = timer(lambda: m_naive.predict(X_te))

        # optimized
        m_opt = KNN(k=5)
        m_opt.fit(X_tr, y_tr)
        _, t_opt = timer(lambda: m_opt.predict(X_te))

        speedup = t_naive / t_opt if t_opt > 0 else float('inf')
        results["n"].append(n)
        results["naive_time"].append(t_naive)
        results["optimized_time"].append(t_opt)
        results["speedup"].append(speedup)

        print(f"  n={n:>5d}  naive={t_naive:.4f}s  optimized={t_opt:.4f}s  "
              f"speedup={speedup:.1f}x")

    return results


# ─────────────────────────────────────────────────────────────────────
# 6. NAIVE vs OPTIMIZED: Decision Tree
# ─────────────────────────────────────────────────────────────────────
def benchmark_naive_vs_optimized_dt():
    print("\n" + "=" * 60)
    print("NAIVE vs OPTIMIZED: Decision Tree")
    print("=" * 60)

    d = 5
    sizes = [50, 100, 200, 500, 1000]
    results = {"n": [], "naive_time": [], "optimized_time": [], "speedup": []}

    for n in sizes:
        X, y = make_classification(n_samples=n, n_features=d, n_informative=3,
                                   random_state=42)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        # naive
        m_naive = DecisionTreeNaive(max_depth=5)
        _, t_naive = timer(lambda: m_naive.fit(X_tr, y_tr))

        # optimized
        m_opt = DecisionTree(max_depth=5)
        _, t_opt = timer(lambda: m_opt.fit(X_tr, y_tr))

        speedup = t_naive / t_opt if t_opt > 0 else float('inf')
        results["n"].append(n)
        results["naive_time"].append(t_naive)
        results["optimized_time"].append(t_opt)
        results["speedup"].append(speedup)

        print(f"  n={n:>5d}  naive={t_naive:.4f}s  optimized={t_opt:.4f}s  "
              f"speedup={speedup:.1f}x")

    return results


# ─────────────────────────────────────────────────────────────────────
# 7. NAIVE vs OPTIMIZED: Gradient Descent
# ─────────────────────────────────────────────────────────────────────
def benchmark_naive_vs_optimized_gd():
    print("\n" + "=" * 60)
    print("NAIVE vs OPTIMIZED: Gradient Descent (Linear Regression)")
    print("=" * 60)

    d = 5
    epochs = 500
    sizes = [50, 100, 200, 500, 1000]
    results = {"n": [], "naive_time": [], "optimized_time": [], "speedup": []}

    for n in sizes:
        X, y = make_regression(n_samples=n, n_features=d, noise=10, random_state=42)

        # naive
        m_naive = LinearRegressionNaiveGD(lr=0.001, epochs=epochs)
        _, t_naive = timer(lambda: m_naive.fit(X, y))

        # optimized
        m_opt = LinearRegression(solver='gradient_descent', lr=0.001, epochs=epochs)
        _, t_opt = timer(lambda: m_opt.fit(X, y))

        speedup = t_naive / t_opt if t_opt > 0 else float('inf')
        results["n"].append(n)
        results["naive_time"].append(t_naive)
        results["optimized_time"].append(t_opt)
        results["speedup"].append(speedup)

        print(f"  n={n:>5d}  naive={t_naive:.4f}s  optimized={t_opt:.4f}s  "
              f"speedup={speedup:.1f}x")

    return results


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────
def plot_all(all_results):
    if not HAS_PLT:
        print("\nmatplotlib not available, skipping plots.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Benchmark: Ours vs Sklearn (log-log)", fontsize=14, fontweight='bold')

    # 1. Linear Regression
    r = all_results["linear_regression"]
    ax = axes[0, 0]
    ax.loglog(r["n"], r["ours_time"], 'o-', label='Ours', linewidth=2)
    ax.loglog(r["n"], r["sklearn_time"], 's--', label='sklearn', linewidth=2)
    ax.set_xlabel('n samples')
    ax.set_ylabel('Time (s)')
    ax.set_title('Linear Regression')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. KNN
    r = all_results["knn"]
    ax = axes[0, 1]
    ax.loglog(r["n"], r["ours_time"], 'o-', label='Ours', linewidth=2)
    ax.loglog(r["n"], r["sklearn_time"], 's--', label='sklearn', linewidth=2)
    ax.set_xlabel('n samples')
    ax.set_ylabel('Time (s)')
    ax.set_title('KNN (k=5)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Naive Bayes
    r = all_results["naive_bayes"]
    ax = axes[0, 2]
    ax.loglog(r["n"], r["ours_time"], 'o-', label='Ours', linewidth=2)
    ax.loglog(r["n"], r["sklearn_time"], 's--', label='sklearn', linewidth=2)
    ax.set_xlabel('n samples')
    ax.set_ylabel('Time (s)')
    ax.set_title('Naive Bayes')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Decision Tree
    r = all_results["decision_tree"]
    ax = axes[1, 0]
    ax.loglog(r["n"], r["ours_time"], 'o-', label='Ours', linewidth=2)
    ax.loglog(r["n"], r["sklearn_time"], 's--', label='sklearn', linewidth=2)
    ax.set_xlabel('n samples')
    ax.set_ylabel('Time (s)')
    ax.set_title('Decision Tree')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Naive vs Optimized KNN
    r = all_results["naive_vs_optimized_knn"]
    ax = axes[1, 1]
    ax.loglog(r["n"], r["naive_time"], 'o-', label='Naive (loops)', linewidth=2)
    ax.loglog(r["n"], r["optimized_time"], 's-', label='Optimized (numpy)', linewidth=2)
    ax.set_xlabel('n samples')
    ax.set_ylabel('Time (s)')
    ax.set_title(f'KNN: Naive vs Optimized')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # add speedup annotations
    for i, n in enumerate(r["n"]):
        ax.annotate(f'{r["speedup"][i]:.0f}x', (n, r["optimized_time"][i]),
                    textcoords="offset points", xytext=(0, 10), fontsize=8, ha='center')

    # 6. Naive vs Optimized GD
    r = all_results["naive_vs_optimized_gd"]
    ax = axes[1, 2]
    ax.loglog(r["n"], r["naive_time"], 'o-', label='Naive (loops)', linewidth=2)
    ax.loglog(r["n"], r["optimized_time"], 's-', label='Optimized (numpy)', linewidth=2)
    ax.set_xlabel('n samples')
    ax.set_ylabel('Time (s)')
    ax.set_title('Gradient Descent: Naive vs Optimized')
    ax.legend()
    ax.grid(True, alpha=0.3)
    for i, n in enumerate(r["n"]):
        ax.annotate(f'{r["speedup"][i]:.0f}x', (n, r["optimized_time"][i]),
                    textcoords="offset points", xytext=(0, 10), fontsize=8, ha='center')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "benchmark_curves.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("MLFROMSCRATCH BENCHMARK SUITE")
    print("Testing runtime scaling and accuracy parity with sklearn")

    all_results = {}
    all_results["linear_regression"] = benchmark_linear_regression()
    all_results["knn"] = benchmark_knn()
    all_results["naive_bayes"] = benchmark_naive_bayes()
    all_results["decision_tree"] = benchmark_decision_tree()
    all_results["naive_vs_optimized_knn"] = benchmark_naive_vs_optimized_knn()
    all_results["naive_vs_optimized_dt"] = benchmark_naive_vs_optimized_dt()
    all_results["naive_vs_optimized_gd"] = benchmark_naive_vs_optimized_gd()

    plot_all(all_results)

    # save raw results
    results_path = os.path.join(OUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw results saved to {results_path}")
