import argparse
import os
import pandas as pd

def filter_data(source_path, target_path):
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Папка {source_path} не существует")
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    
    features = [ "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11", "AU12", "AU14", 
               "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26", "AU28", "AU43", 
               "anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

    for file in os.listdir(source_path):
        if not file.endswith('.csv'):
            continue
        filepath = os.path.join(source_path, file)
        old_df = pd.read_csv(filepath)
        new_df = old_df[features].copy()
        new_df.to_csv(os.path.join(target_path, file), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_path", type=str, help="path to directory with original csv")
    parser.add_argument("target_path", type=str, help="path to directory to save csv")
    args = parser.parse_args()

    filter_data(args.source_path, args.target_path)

   

if __name__ == '__main__':
    main()


