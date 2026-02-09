import json
import os
import glob
from collections import defaultdict
import pandas as pd
import sys
import numpy as np

def load_json_files(folder_path):
    """Рекурсивно загружает все JSON файлы из указанной папки"""
    json_files = glob.glob(os.path.join(folder_path, '**', '*.json'), recursive=True)
    all_data = {}
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data[file_path] = data
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Ошибка при загрузке файла {file_path}: {e}")
    
    return all_data

def extract_metrics(data_dict):
    """Извлекает метрики из структуры JSON файла"""
    metrics_dict = {}
    file_info = {}  # Дополнительная информация о файлах
    
    for file_path, data in data_dict.items():
        if "clear" in data:
            for model_name, model_data in data["clear"].items():
                if "metrics" in model_data:
                    metrics = model_data["metrics"]
                    # Создаем уникальный ключ
                    key = f"{os.path.basename(file_path)} ({model_name})"
                    metrics_dict[key] = {}
                    
                    # Сохраняем информацию о файле
                    file_info[key] = {
                        'file_path': file_path,
                        'file_name': os.path.basename(file_path),
                        'model_name': model_name,
                        'folder': os.path.dirname(file_path)
                    }
                    
                    # Извлекаем все метрики
                    for metric_type, metric_values in metrics.items():
                        for metric_name, metric_value in metric_values.items():
                            metrics_dict[key][metric_name] = metric_value
    
    return metrics_dict, file_info

def calculate_overall_score(metrics_row, weights=None):
    """Вычисляет общий балл для файла по всем метрикам"""
    if weights is None:
        # Веса по умолчанию (можно настроить)
        weights = {
            'NDCG@10': 1.5,
            'NDCG@100': 1.2,
            'MAP@100': 1.2,
            'Recall@100': 1.0,
            'NDCG@5': 1.0,
            'MAP@10': 1.0,
            'Recall@10': 0.8,
            'Recall@5': 0.7,
            'MAP@5': 0.7,
            'NDCG@3': 0.6,
            'NDCG@1': 0.5,
            'MAP@1': 0.5,
            'Recall@1': 0.5,
            'P@1': 0.4,
            'P@3': 0.3,
            'P@5': 0.3,
            'P@10': 0.2,
            'NDCG@1000': 0.1,
            'MAP@1000': 0.1,
            'Recall@1000': 0.05,
            'P@100': 0.1,
            'P@1000': 0.05
        }
    
    total_score = 0
    total_weight = 0
    
    for metric, value in metrics_row.items():
        weight = weights.get(metric, 1.0)
        total_score += value * weight
        total_weight += weight
    
    return total_score / total_weight if total_weight > 0 else 0

def create_comprehensive_ranking(df, file_info, top_n=20):
    """Создает комплексный рейтинг по всем метрикам"""
    print("\n" + "=" * 100)
    print("КОМПЛЕКСНЫЙ ТОП ПО ВСЕМ МЕТРИКАМ В СОВОКУПНОСТИ")
    print("=" * 100)
    
    # Вычисляем общий балл для каждого файла
    overall_scores = {}
    details = {}
    
    for idx, row in df.iterrows():
        # Общий взвешенный балл
        weighted_score = calculate_overall_score(row)
        
        # Среднее по всем метрикам
        avg_score = row.mean()
        
        # Среднее по основным метрикам (без @1000)
        main_metrics = [col for col in df.columns if '@1000' not in col]
        main_avg = row[main_metrics].mean() if main_metrics else 0
        
        # Сумма рангов по всем метрикам (чем меньше, тем лучше)
        ranks = row.rank(ascending=False)
        rank_sum = ranks.sum()
        
        overall_scores[idx] = {
            'weighted_score': weighted_score,
            'average_score': avg_score,
            'main_metrics_avg': main_avg,
            'rank_sum': rank_sum,
            'file_info': file_info.get(idx, {})
        }
    
    # Топ по взвешенному баллу
    print("\n" + "═" * 100)
    print("ТОП ПО ВЗВЕШЕННОМУ БАЛЛУ (основные метрики имеют больший вес)")
    print("═" * 100)
    sorted_weighted = sorted(overall_scores.items(), 
                             key=lambda x: x[1]['weighted_score'], 
                             reverse=True)
    
    for i, (model, scores) in enumerate(sorted_weighted[:top_n], 1):
        file_name = scores['file_info'].get('file_name', model)
        folder = scores['file_info'].get('folder', '')
        print(f"{i:2}. {file_name:40} | "
              f"Взвешенный: {scores['weighted_score']:.5f} | "
              f"Средний: {scores['average_score']:.5f} | "
              f"Папка: {folder}")
    
    # Топ по среднему баллу всех метрик
    print("\n" + "═" * 100)
    print("ТОП ПО СРЕДНЕМУ БАЛЛУ ВСЕХ МЕТРИК")
    print("═" * 100)
    sorted_avg = sorted(overall_scores.items(), 
                        key=lambda x: x[1]['average_score'], 
                        reverse=True)
    
    for i, (model, scores) in enumerate(sorted_avg[:top_n], 1):
        file_name = scores['file_info'].get('file_name', model)
        print(f"{i:2}. {file_name:40} | "
              f"Средний: {scores['average_score']:.5f} | "
              f"Взвешенный: {scores['weighted_score']:.5f} | "
              f"Основные метрики: {scores['main_metrics_avg']:.5f}")
    
    # Топ по сумме рангов (чем меньше сумма, тем выше место)
    print("\n" + "═" * 100)
    print("ТОП ПО СУММЕ РАНГОВ (чем меньше, тем лучше)")
    print("═" * 100)
    sorted_ranks = sorted(overall_scores.items(), 
                          key=lambda x: x[1]['rank_sum'])
    
    for i, (model, scores) in enumerate(sorted_ranks[:top_n], 1):
        file_name = scores['file_info'].get('file_name', model)
        max_possible_rank = len(df.columns) * len(df)  # Максимально возможная сумма рангов
        rank_percentage = (scores['rank_sum'] / max_possible_rank) * 100
        print(f"{i:2}. {file_name:40} | "
              f"Сумма рангов: {scores['rank_sum']:.1f} | "
              f"Процентиль: {100 - rank_percentage:.1f}% | "
              f"Взвешенный: {scores['weighted_score']:.5f}")
    
    # Лучший файл по каждой группе метрик
    print("\n" + "═" * 100)
    print("ЛУЧШИЕ ФАЙЛЫ ПО КАТЕГОРИЯМ МЕТРИК")
    print("═" * 100)
    
    metric_groups = {
        'NDCG': [col for col in df.columns if 'NDCG' in col],
        'MAP': [col for col in df.columns if 'MAP' in col],
        'Recall': [col for col in df.columns if 'Recall' in col],
        'Precision': [col for col in df.columns if 'P@' in col],
        'Основные (@1-@100)': [col for col in df.columns if '@1000' not in col],
        'Точность (@1-@10)': [col for col in df.columns if '@1' in col or '@3' in col or '@5' in col or '@10' in col]
    }
    
    for group_name, metrics in metric_groups.items():
        if metrics:
            group_scores = {}
            for idx, row in df.iterrows():
                group_scores[idx] = row[metrics].mean()
            
            best_file = max(group_scores.items(), key=lambda x: x[1])
            file_name = overall_scores[best_file[0]]['file_info'].get('file_name', best_file[0])
            print(f"{group_name:20} → {file_name:40} | Средний балл: {best_file[1]:.5f}")
    
    return overall_scores

def create_ranking_tables(metrics_dict, file_info):
    """Создает таблицы рейтинга для каждой метрики"""
    if not metrics_dict:
        print("Не найдено метрик для анализа")
        return None, None
    
    # Создаем DataFrame из словаря метрик
    df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    
    print("=" * 100)
    print("ОБЩИЙ ОБЗОР ВСЕХ МЕТРИК")
    print("=" * 100)
    print(f"Всего файлов: {len(df)}")
    print(f"Всего метрик: {len(df.columns)}")
    print("-" * 100)
    
    # Выводим первые несколько строк для ознакомления
    print(df.head().round(4))
    
    # Топ по каждой метрике
    print("\n" + "=" * 100)
    print("ТОП-3 ПО КАЖДОЙ МЕТРИКЕ")
    print("=" * 100)
    
    for column in df.columns:
        print(f"\n{column}:")
        print("-" * 60)
        
        # Сортируем по убыванию
        sorted_df = df[column].sort_values(ascending=False)
        
        for i, (model, value) in enumerate(sorted_df.head(3).items(), 1):
            file_name = file_info.get(model, {}).get('file_name', model)
            print(f"{i}. {file_name:45} : {value:.5f}")
    
    return df, file_info

def export_comprehensive_results(overall_scores, output_file="comprehensive_ranking.csv"):
    """Экспортирует комплексные результаты в CSV"""
    data = []
    for model, scores in overall_scores.items():
        row = {
            'file': scores['file_info'].get('file_name', model),
            'model': scores['file_info'].get('model_name', ''),
            'folder': scores['file_info'].get('folder', ''),
            'weighted_score': scores['weighted_score'],
            'average_score': scores['average_score'],
            'main_metrics_avg': scores['main_metrics_avg'],
            'rank_sum': scores['rank_sum']
        }
        data.append(row)
    
    df_export = pd.DataFrame(data)
    df_export = df_export.sort_values('weighted_score', ascending=False)
    df_export.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\nКомплексный рейтинг экспортирован в: {output_file}")
    
    return df_export

def main():
    if len(sys.argv) != 2:
        print("Использование: python script.py <путь_к_папке>")
        print("Пример: python script.py ./experiments")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    if not os.path.exists(folder_path):
        print(f"Ошибка: Папка '{folder_path}' не существует!")
        sys.exit(1)
    
    print(f"Поиск JSON файлов в папке: {folder_path}")
    print("=" * 100)
    
    # Загружаем все JSON файлы
    json_data = load_json_files(folder_path)
    
    if not json_data:
        print("Не найдено JSON файлов для анализа")
        return
    
    print(f"Найдено JSON файлов: {len(json_data)}")
    
    # Извлекаем метрики
    metrics_dict, file_info = extract_metrics(json_data)
    
    if not metrics_dict:
        print("Не удалось извлечь метрики из файлов")
        return
    
    print(f"Извлечено моделей с метриками: {len(metrics_dict)}")
    
    # Создаем рейтинги по отдельным метрикам
    df, file_info = create_ranking_tables(metrics_dict, file_info)
    
    if df is not None:
        # Создаем комплексный рейтинг по всем метрикам
        overall_scores = create_comprehensive_ranking(df, file_info)
        
        # Экспортируем результаты
        export_comprehensive_results(overall_scores)
        
        # Дополнительная статистика
        print("\n" + "=" * 100)
        print("СТАТИСТИКА ПО ВСЕМ ФАЙЛАМ")
        print("=" * 100)
        
        weighted_scores = [s['weighted_score'] for s in overall_scores.values()]
        avg_scores = [s['average_score'] for s in overall_scores.values()]
        
        print(f"Средний взвешенный балл: {np.mean(weighted_scores):.5f}")
        print(f"Медианный взвешенный балл: {np.median(weighted_scores):.5f}")
        print(f"Максимальный взвешенный балл: {np.max(weighted_scores):.5f}")
        print(f"Минимальный взвешенный балл: {np.min(weighted_scores):.5f}")
        print(f"Стандартное отклонение: {np.std(weighted_scores):.5f}")
        
        # Лучшие 5 файлов
        print("\nТОП-5 ЛУЧШИХ ФАЙЛОВ:")
        print("-" * 80)
        sorted_files = sorted(overall_scores.items(), 
                              key=lambda x: x[1]['weighted_score'], 
                              reverse=True)
        
        for i, (model, scores) in enumerate(sorted_files[:5], 1):
            file_name = scores['file_info'].get('file_name', model)
            folder = scores['file_info'].get('folder', '')
            print(f"{i}. {file_name}")
            print(f"   Папка: {folder}")
            print(f"   Взвешенный балл: {scores['weighted_score']:.5f}")
            print(f"   Средний балл: {scores['average_score']:.5f}")
            print(f"   Сумма рангов: {scores['rank_sum']:.1f}")
            print()
    
    print("\n" + "=" * 100)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print("=" * 100)

if __name__ == "__main__":
    main()