import pandas as pd
import argparse
import os
import streamlit as st
import matplotlib.pyplot as plt

def count_frames(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Папка {path} не существует")

    stats = {
        'total_sum' : 0,
        'total_num' : 0,
        'min_frames' : float('inf'),
        'max_frames' : 0,
        'average' : 0,
        'min_name' : None,
        'max_name' : None
    }

    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        try:
            df = pd.read_csv(filepath)
            file_len = len(df)
            stats['total_sum'] += file_len
            if file_len > stats['max_frames']:
                stats['max_frames'] = file_len
                stats['max_name'] = filepath
            elif file_len < stats['min_frames']:
                stats['min_frames'] = file_len
                stats['min_name'] = filepath
            stats['total_num'] += 1
        except Exception as e:
            print("error error")
    stats['average'] = stats['total_sum'] / stats['total_num']
    print("total num: ", stats['total_num'])
    print("average: ", stats['average'])
    print("min: ", stats['min_frames'])
    print("min name: ", stats['min_name'])
    print('max: ', stats['max_frames'])
    print('max name: ', stats['max_name'])

    return stats['average']

def build_parameter_chart(categories, features):
    
    category_names = {
        'fon': 'fon',
        'other_face': 'other face',
        'own_face': 'own face'
    }
    
    # Создаем графики для каждой комбинации признака и категории
    for feature_idx, feature in enumerate(features):
        for category_name, category_data in categories.items():
            print(category_name)
            #print(category_data)
            # Получаем данные для конкретного признака и категории
            feature_data = category_data[feature_idx]
            print(feature_data)
            # Сортируем видео по названию и извлекаем значения
            sorted_values = [v for k, v in sorted(feature_data.items())]
            
            # Создаем новый график
            plt.figure(figsize=(10, 4))
            
            # Рисуем линейный график
            plt.plot(
                range(1, len(sorted_values)+1),  # Порядковые номера видео
                sorted_values,                    # Средние значения признака
                marker='o',
                linestyle='-',
                linewidth=2,
                markersize=5,
                color='#1f77b4'  # Синий цвет для всех графиков
            )
            
            # Настройки графика
            plt.title(f"{category_names[category_name]} - {feature}")
            plt.xlabel('Порядковый номер видео')
            plt.ylabel('Среднее значение')
            plt.grid(True, linestyle='--', alpha=0.5)
            
            # Отображаем график в Streamlit
            st.pyplot(plt.gcf())
            plt.close()

def calc_average_correlation(average_frame_num, path, features):
    categories = {
        'fon': [{} for _ in range(len(features))],
        'other_face': [{} for _ in range(len(features))],
        'own_face': [{} for _ in range(len(features))]
    }  # словарь списков со словарями

    for file in os.listdir(path):   # проходимся по файлам 
        if not file.endswith('.csv'):
            continue
    
        filepath = os.path.join(path, file)
        full_df = pd.read_csv(filepath)

        try:
            full_df = pd.read_csv(filepath)
            shorten_df = full_df.iloc[:int(average_frame_num)] if len(full_df) > int(average_frame_num) else full_df
            
            if 'fon' in file.lower():
                category = 'fon'
            elif 'other' in file.lower():
                category = 'other_face'
            elif 'own' in file.lower():
                category = 'own_face'
            else:
                continue
                
            for i, feature in features:
                if feature in shorten_df.columns:
                    categories[category][i][file] = shorten_df[feature].mean()
                    
        except Exception as e:
            print(f"Ошибка при обработке файла {file}: {str(e)}")
            continue
            
    return categories

def check_min_max_for_feature(source_path, features):
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Папка {source_path} не существует")
    
    min_dict = {}
    max_dict = {}
    for feature in features:
        min_dict[feature] = 1
        max_dict[feature] = 0

    for file in os.listdir(source_path):
        if not file.endswith('.csv'):
            continue
        filepath = os.path.join(source_path, file)
        df = pd.read_csv(filepath)
        #print('-' * 50)
        #print(file)
        for column in df.columns:
            min_val = df[column].min()
            max_val = df[column].max()
            if min_val < min_dict[column]:
                min_dict[column] = min_val
            if max_val > max_dict[column]:
                max_dict[column] = max_val
            #print(f"Name: {column},      min: {min_val},        max: {max_val}")
    print("FEATURE             MIN                  MAX")
    for i in range(len(features)):
        feature = list(min_dict.keys())[i]
        print(f"{feature}       {min_dict[feature]}          {max_dict[feature]}")

    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_path", type=str, help="path to directory with videos")
    args = parser.parse_args()

    features = [
        "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10",
        "AU11", "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24",
        "AU25", "AU26", "AU28", "AU43", "anger", "disgust", "fear",
        "happiness", "sadness", "surprise", "neutral"
    ]


    #average_frame_num = count_frames(args.dir_path)
    #categories = calc_average_correlation(average_frame_num, args.dir_path, features)
    #build_parameter_chart(categories, features)    

    check_min_max_for_feature(args.dir_path, features)


if __name__ == '__main__':
    main()


