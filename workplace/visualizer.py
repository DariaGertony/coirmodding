
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.use('Agg')

# === Настройки стиля ===
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'legend.framealpha': 0.92,
    'grid.alpha': 0.35,
    'axes.grid': True,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def clear_folder(folder_path):
    """Очищает папку (удаляет всё содержимое)"""
    if not os.path.exists(folder_path):
        return
    try:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                import shutil
                shutil.rmtree(item_path)
        print(f"✓ Очистка: {folder_path}")
    except Exception as e:
        print(f"⚠ Ошибка при очистке {folder_path}: {e}")


def load_tasks_from_folder(root_folder):
    
    root = Path(root_folder)
    tasks = {}

    for json_file in root.rglob("*.json"):
        parts = json_file.parts
        if len(parts) < 3:
            continue

        try:
            task_idx = parts.index(root.name) + 1 if root.name in parts else 1
            task_name = parts[task_idx]
        except ValueError:
            if len(parts) >= 4:
                task_name = parts[-4]
            else:
                continue

        method_name = json_file.stem

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"⚠ Пропущен файл {json_file}: {e}")
            continue

        raw_metrics = None
        if "metrics " in data:
            raw_metrics = data["metrics "]
        elif "metrics" in data:
            raw_metrics = data["metrics"]
        else:
            print(f"⚠ Нет 'metrics' в {json_file}")
            continue

        metrics_flat = {}
        for metric_type_key, values in raw_metrics.items():
            clean_metric_type = metric_type_key.strip()
            for k, v in values.items():
                clean_k = k.strip()
                metrics_flat[clean_k] = v

        if task_name not in tasks:
            tasks[task_name] = {}
        tasks[task_name][method_name] = metrics_flat

    return tasks

def plot_scatter_for_task(task_name, df, output_dir):
    if df.empty:
        return False

    pairs = [
        ("NDCG@10", "MAP@100"),
        ("Recall@100", "NDCG@100"),
        ("P@10", "Recall@10")
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, min(len(df), 10)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for x_metric, y_metric in pairs:
        if x_metric not in df.columns or y_metric not in df.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))
        for idx, method in enumerate(df.index):
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            ax.scatter(
                df.loc[method, x_metric],
                df.loc[method, y_metric],
                c=[color], marker=marker, s=100,
                edgecolors='black', linewidth=0.6, alpha=0.85,
                label=method
            )

        corr = df[x_metric].corr(df[y_metric])
        ax.set_xlabel(x_metric, fontweight='bold')
        ax.set_ylabel(y_metric, fontweight='bold')
        ax.set_title(f'{task_name}\n{x_metric} vs {y_metric} (ρ = {corr:.3f})', fontweight='bold')

        if len(df) <= 8:
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
            fig.subplots_adjust(right=0.75)
        else:
            ax.legend(loc='lower right', frameon=True, ncol=1)

        safe_x = x_metric.replace("@", "_").replace(" ", "")
        safe_y = y_metric.replace("@", "_").replace(" ", "")
        plt.savefig(os.path.join(output_dir, f'scatter_{safe_x}_vs_{safe_y}.png'))
        plt.close()
        print(f"  ✓ scatter: {x_metric} vs {y_metric}")
    return True


def extract_topk_series(metrics_dict, prefix):
    k_vals = [1, 3, 5, 10, 100, 1000]
    series = {}
    for method, metrics in metrics_dict.items():
        vals = []
        for k in k_vals:
            key = f"{prefix}@{k}"
            if key in metrics:
                vals.append(metrics[key])
        if len(vals) == len(k_vals):
            series[method] = vals
    return series, k_vals


def plot_topk_for_task(task_name, metrics_dict, output_dir):
    prefixes = {"NDCG": "NDCG", "MAP": "MAP", "Recall": "Recall", "Precision": "P"}
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(metrics_dict), 10)))
    linestyles = ['-', '--', '-.', ':']

    for name, prefix in prefixes.items():
        series, k_vals = extract_topk_series(metrics_dict, prefix)
        if not series:
            continue

        fig, ax = plt.subplots(figsize=(11, 6.5))
        for idx, (method, values) in enumerate(series.items()):
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            ax.plot(
                k_vals, values,
                marker='o', markersize=5, linewidth=2.2,
                color=color, linestyle=linestyle,
                label=method
            )

        ax.set_xscale('log')
        ax.set_xticks(k_vals)
        ax.set_xticklabels([f'@{k}' for k in k_vals])
        ax.set_xlabel('Top-k', fontweight='bold')
        ax.set_ylabel(f'{name} score', fontweight='bold')
        ax.set_title(f'{task_name}: {name} vs top-k', fontweight='bold')

        if len(series) <= 6:
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
            fig.subplots_adjust(right=0.78)
        else:
            ax.legend(loc='lower right', frameon=True, ncol=2)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'topk_{name.lower()}.png'))
        plt.close()
        print(f"  ✓ top-k: {name}")
    return True


def plot_barchart_for_task(task_name, df, output_dir):
    if df.empty:
        return False

    key_metrics = [
        "NDCG@1", "NDCG@10", "NDCG@100",
        "MAP@10", "MAP@100",
        "Recall@10", "Recall@100",
        "P@1", "P@10"
    ]
    available = [m for m in key_metrics if m in df.columns]
    if not available:
        return False

    colors = plt.cm.tab10(np.linspace(0, 1, min(len(df), 10)))

    for metric in available:
        models = df.index.tolist()
        values = df[metric].values
        sorted_idx = np.argsort(-values)
        models_sorted, values_sorted, colors_sorted = (
            [models[i] for i in sorted_idx],
            values[sorted_idx],
            [colors[i % len(colors)] for i in sorted_idx]
        )

        fig, ax = plt.subplots(figsize=(max(8, len(models)*0.9), 6))
        bars = ax.bar(models_sorted, values_sorted, color=colors_sorted,
                      edgecolor='black', linewidth=0.5, alpha=0.85)

        if len(models) <= 10:
            for bar, val in zip(bars, values_sorted):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(values_sorted)*0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xlabel('Метод', fontweight='bold')
        ax.set_ylabel(metric, fontweight='bold')
        ax.set_title(f'{task_name}: {metric}', fontweight='bold')
        ax.set_ylim(0, max(values_sorted)*1.12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        safe_metric = metric.replace("@", "_").replace(" ", "")
        plt.savefig(os.path.join(output_dir, f'barchart_{safe_metric}.png'))
        plt.close()
        print(f"  ✓ barchart: {metric}")
    return True


def visualize_tasks(input_root="./piplines_results", output_base="./visualizations"):
    print("=" * 60)
    print("ВИЗУАЛИЗАЦИЯ ПО ЗАДАЧАМ")
    print("=" * 60)

    os.makedirs(output_base, exist_ok=True)
    clear_folder(output_base)

    tasks = load_tasks_from_folder(input_root)
    if not tasks:
        print("❌ Не найдено задач (JSON-файлов в подпапках)")
        return

    print(f"Найдено задач: {len(tasks)}")
    for task_name, methods_data in tasks.items():
        print(f"\nОбработка задачи: {task_name} ({len(methods_data)} методов)")

        df = pd.DataFrame.from_dict(methods_data, orient='index')
        if df.empty:
            continue

        task_output = os.path.join(output_base, task_name)
        os.makedirs(task_output, exist_ok=True)

        plot_scatter_for_task(task_name, df, task_output)
        plot_topk_for_task(task_name, methods_data, task_output)
        plot_barchart_for_task(task_name, df, task_output)

        df.to_csv(os.path.join(task_output, 'summary.csv'), encoding='utf-8-sig')
        print(f"  → Сохранено в: {task_output}")

    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print(f"Графики: {output_base}/")
    print("=" * 60)


if __name__ == "__main__":
    visualize_tasks()