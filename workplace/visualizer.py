
"""
Визуализатор результатов по задачам.
— Подписи моделей вынесены в легенду
— Улучшен масштаб и читаемость
— Каждая задача — отдельная папка
"""
import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Настройки стиля
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


def load_json_files(folder_path):
    """Загружает JSON-файлы: каждый файл = одна задача"""
    json_files = glob.glob(os.path.join(folder_path, '*.json'))
    data = {}
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                task_name = os.path.splitext(os.path.basename(file_path))[0]
                data[task_name] = content
        except Exception as e:
            print(f"⚠ Пропущен файл {file_path}: {e}")
    return data


def extract_task_metrics(task_data):
    """Извлекает метрики из одного task-файла (формат evaluation.py)"""
    metrics_dict = {}
    if "clear" not in task_data:
        return metrics_dict
    
    for model_name, model_data in task_data["clear"].items():
        if "metrics" not in model_data:
            continue
        clean_model_name = model_name.strip()
        metrics_dict[clean_model_name] = {}
        for metric_type, values in model_data["metrics"].items():
            for k, v in values.items():
                clean_k = k.strip()
                metrics_dict[clean_model_name][clean_k] = v
    return metrics_dict


def plot_scatter_for_task(task_name, df, output_dir):
    """Создаёт scatter plot с легендой вместо подписей"""
    if df.empty:
        return False
    
    pairs = [
        ("NDCG@10", "MAP@10"),
        ("Recall@10", "NDCG@10"),
        ("P@10", "Recall@10")
    ]
    
    # Цвета и маркеры
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(df), 10)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for x_metric, y_metric in pairs:
        if x_metric not in df.columns or y_metric not in df.columns:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for idx, model_name in enumerate(df.index):
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            ax.scatter(
                df.loc[model_name, x_metric],
                df.loc[model_name, y_metric],
                c=[color], marker=marker, s=100,
                edgecolors='black', linewidth=0.6, alpha=0.85,
                label=model_name  # ← легенда!
            )
        
        corr = df[x_metric].corr(df[y_metric])
        ax.set_xlabel(x_metric, fontweight='bold')
        ax.set_ylabel(y_metric, fontweight='bold')
        ax.set_title(f'{task_name}\n{x_metric} vs {y_metric} (ρ = {corr:.3f})', fontweight='bold')
        
        # Легенда вне графика, если моделей ≤ 8, иначе внутри
        if len(df) <= 8:
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
            fig.subplots_adjust(right=0.75)
        else:
            ax.legend(loc='lower right', frameon=True, ncol=1)
        
        plt.tight_layout()
        safe_x = x_metric.replace("@", "_")
        safe_y = y_metric.replace("@", "_")
        plt.savefig(os.path.join(output_dir, f'scatter_{safe_x}_vs_{safe_y}.png'))
        plt.close()
        print(f"  ✓ scatter: {x_metric} vs {y_metric}")
    return True


def extract_topk_series(metrics_dict, prefix):
    """Извлекает значения @1, @3, ..., @1000"""
    k_vals = [1, 3, 5, 10]
    series = {}
    for model, metrics in metrics_dict.items():
        vals = []
        for k in k_vals:
            key = f"{prefix}@{k}"
            if key in metrics:
                vals.append(metrics[key])
        if len(vals) == len(k_vals):
            series[model] = vals
    return series, k_vals


def plot_topk_for_task(task_name, metrics_dict, output_dir):
    """Создаёт top-k графики с легендой"""
    prefixes = {"NDCG": "NDCG", "MAP": "MAP", "Recall": "Recall", "Precision": "P"}
    
    # Цвета и стили линий
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(metrics_dict), 10)))
    linestyles = ['-', '--', '-.', ':']
    
    for name, prefix in prefixes.items():
        series, k_vals = extract_topk_series(metrics_dict, prefix)
        if not series:
            continue
        
        fig, ax = plt.subplots(figsize=(11, 6.5))
        
        for idx, (model_name, values) in enumerate(series.items()):
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            ax.plot(
                k_vals, values,
                marker='o', markersize=5, linewidth=2.2,
                color=color, linestyle=linestyle,
                label=model_name  # ← легенда!
            )
        
        ax.set_xscale('log')
        ax.set_xticks(k_vals)
        ax.set_xticklabels([f'@{k}' for k in k_vals])
        ax.set_xlabel('Top-k', fontweight='bold')
        ax.set_ylabel(f'{name} score', fontweight='bold')
        ax.set_title(f'{task_name}: {name} per top-k', fontweight='bold')
        
        # Легенда
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
    """Создаёт барчарты: один график = одна метрика, один столбец = одна модель"""
    if df.empty:
        return False

    # Выбираем ключевые метрики для визуализации
    key_metrics = [
        "NDCG@1", "NDCG@10", "NDCG@100",
        "MAP@10", "MAP@100",
        "Recall@10", "Recall@100",
        "P@1", "P@10"
    ]

    # Фильтруем только доступные метрики
    available_metrics = [m for m in key_metrics if m in df.columns]
    if not available_metrics:
        return False

    # Цвета для столбцов
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(df), 10)))

    for metric in available_metrics:
        fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.9), 6))
        
        models = df.index.tolist()
        values = df[metric].values
        
        # Сортируем по убыванию для лучшей читаемости
        sorted_idx = np.argsort(-values)
        models_sorted = [models[i] for i in sorted_idx]
        values_sorted = values[sorted_idx]
        colors_sorted = [colors[i % len(colors)] for i in sorted_idx]

        bars = ax.bar(
            models_sorted, values_sorted,
            color=colors_sorted, edgecolor='black', linewidth=0.5, alpha=0.85
        )

        # Подписи значений над столбцами (если не слишком много моделей)
        if len(models) <= 10:
            for bar, val in zip(bars, values_sorted):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values_sorted) * 0.01,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold'
                )

        ax.set_xlabel('Модель / Метод', fontweight='bold')
        ax.set_ylabel(metric, fontweight='bold')
        ax.set_title(f'{task_name}: {metric}', fontweight='bold')
        ax.set_ylim(0, max(values_sorted) * 1.12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        safe_metric = metric.replace("@", "_")
        plt.savefig(os.path.join(output_dir, f'barchart_{safe_metric}.png'))
        plt.close()
        print(f"  ✓ barchart: {metric}")
    
    return True


def visualize_tasks(input_folder="./results", output_base="./visualizations"):
    """Основная функция: обрабатывает каждый task отдельно"""
    print("="*60)
    print("ВИЗУАЛИЗАЦИЯ ПО ЗАДАЧАМ")
    print("="*60)
    
    tasks = load_json_files(input_folder)
    if not tasks:
        print("❌ Не найдено JSON-файлов")
        return
    
    print(f"Найдено задач: {len(tasks)}")
    
    for task_name, task_data in tasks.items():
        print(f"\nОбработка задачи: {task_name}")
        metrics_dict = extract_task_metrics(task_data)
        if not metrics_dict:
            print(f"  ⚠ Пропущено: нет данных")
            continue
        
        task_output = os.path.join(output_base, task_name)
        os.makedirs(task_output, exist_ok=True)
        
        df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        plot_scatter_for_task(task_name, df, task_output)
        plot_topk_for_task(task_name, metrics_dict, task_output)
        plot_barchart_for_task(task_name, df, task_output)
        
        df.to_csv(os.path.join(task_output, 'summary.csv'), encoding='utf-8-sig')
        print(f"  → Сохранено в: {task_output}")
    
    print("\n" + "="*60)
    print("ГОТОВО!")
    print(f"Все графики: {output_base}/")
    print("="*60)


if __name__ == "__main__":
    visualize_tasks()