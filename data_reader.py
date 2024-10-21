import streamlit as st
import pandas as pd
def read_iqdat_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        # Читаем заголовок файла
        lines = content.split('\n')

        df = pd.DataFrame([line.split() for line in lines])

    return df

def main():
    df1 = read_iqdat_file('Som_T_001_v1.iqdat')
    st.dataframe(df1)
    df2 = pd.read_csv("Som_T_001_v1_Face.csv", sep='\t')
    st.dataframe(df2)

if __name__ == "__main__":
    main()
