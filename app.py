import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re




df = pd.read_csv('clean.csv')
df_judgment_text = df[['judgment_text']]
# tf_idf = pd.read_csv('dt_tfidf.csv')
cv_tfidf = TfidfVectorizer(min_df=2, max_df=0.8)


def show_predict_page():
    st.title("Ministry of Justice")

    st.write("""### Plese write your judgement text""")

  
    title = st.text_input('Judgemant Text')


    ok = st.button("Show  Judgement Similarities")

    if ok:
        df_judgment_text_new = df_judgment_text.append({'judgment_text':title},ignore_index=True)
        x = cv_tfidf.fit_transform(df_judgment_text_new['judgment_text']).toarray()
        dt_tfidf = pd.DataFrame(x,columns=cv_tfidf.get_feature_names())
        similars = cosine_similarity(dt_tfidf.iloc[-1:,:],dt_tfidf).argsort()[0][-6:]
        similars = similars[-2::-1][:dt_tfidf.shape[0]]
        
        # st.write(similars)
        for i in similars:
            percents = round(cosine_similarity(dt_tfidf.iloc[-1:,:], dt_tfidf.loc[i:i,:])[0][0],2)
            st.write("---")
            st.write("قضية رقم : ",i.astype('str'))
            st.write("رابط القضية : ", df.link.iloc[i])
            st.write(percents.astype('str'), " %")


show_predict_page()