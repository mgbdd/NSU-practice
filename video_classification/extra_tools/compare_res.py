import streamlit as st
import csv
import pandas as pd
import matplotlib.pyplot as plt

def build_chart(column_name, df1, df2, df3):

    plt.figure(figsize=(20, 6))
    plt.plot(range(len(df1)), df1[column_name], color='red')
    plt.plot(range(len(df2)), df2[column_name], color='green')
    plt.plot(range(len(df3)), df3[column_name], color='blue')
    plt.xlabel('red - fon, green - other face, blue - own face')
    plt.ylabel('frame')
    plt.title(column_name)
    plt.grid(True)
    st.pyplot(plt)


def main():
    df_fon = pd.read_csv("./Co_y6_004_Fon1.csv")
    df_other = pd.read_csv("./Co_Y6_004_Other_face.csv")
    df_own = pd.read_csv("./Co_Y6_004_Own_face.csv")
    st.text("Results of empty screen video")
    st.dataframe(df_fon)
    st.text("Results of other face video")
    st.dataframe(df_other)
    st.text("Results of own face video")
    st.dataframe(df_own)

    emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
    aus = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09', 'AU10', 'AU11', 'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU24', 'AU25', 'AU26', 'AU28', 'AU43']

    for emotion in emotions:
        build_chart(emotion, df_fon, df_other, df_own)

    for au in aus:
        build_chart(au, df_fon, df_other, df_own)
    return


if __name__ == '__main__':
    main()