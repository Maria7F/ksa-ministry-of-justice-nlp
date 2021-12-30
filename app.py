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
        st.write(title)
        last = df_judgment_text.append([title])
        x = cv_tfidf.fit_transform(df_judgment_text['judgment_text']).toarray()
        dt_tfidf = pd.DataFrame(x,columns=cv_tfidf.get_feature_names())
        similars = cosine_similarity(dt_tfidf.iloc[-1:,:],dt_tfidf).argsort()[0][-6:]
        
        st.write(similars)
        for i in similars:
            st.write(i, df.link.iloc[i])


show_predict_page()