#!/usr/bin/env python
# coding: utf-8

# Tugas Akhir Pariwisata 
# 
# Nama : JAN PHILIP FAITH
# 
# Nim  : 190411100112

# In[1]:


#Mengoupload File ke dalam google colabs
# from google.colab import files


# uploaded = files.upload()


# In[10]:


import numpy as np
import pandas as pd


# In[16]:


# get_ipython().system('pip install Sastrawi')
# get_ipython().system('pip install tensorflow')


# In[17]:

import streamlit as st
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px 

from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.metrics.pairwise import cosine_similarity

from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

def main():
    st.title("Sistem Rekomendasi Content Based Filtering")
    tv = TfidfVectorizer(max_features=5000)
    stem = StemmerFactory().create_stemmer()
    stopword = StopWordRemoverFactory().create_stop_word_remover()


    # In[18]:


    read_file = pd.read_csv('hotel.csv')
    # read_file.to_csv ('hotel.csv', index = None, header=True)


    # In[20]:


    df = pd.read_csv('./hotel.csv')
    # df


    # In[21]:


    data_tourism_rating = df


    # In[22]:


    data_tourism_rating.isna().sum()


    # In[9]:


    def preprocessing(data):
        data = str(data)
        data = data.lower()
        data = stem.stem(data)
        data = stopword.remove(data)
        return data

    data_rekomendasi = df


    # In[10]:


    data_content_based_filtering = data_rekomendasi.copy()
    data_content_based_filtering['Tags'] = data_content_based_filtering['ulasan']
    data_content_based_filtering.drop(['ulasan'],axis=1,inplace=True)
    # data_content_based_filtering


    # In[11]:


    data_content_based_filtering.Tags = data_content_based_filtering.Tags.apply(preprocessing)
    # data_content_based_filtering


    # In[ ]:


    vectors = tv.fit_transform(data_content_based_filtering.Tags).toarray()
    # vectors


    # In[ ]:


    similarity = cosine_similarity(vectors)
    similarity[0][1:10]


    # In[ ]:


    def recommend_by_content_based_filtering(nama_tempat):
        nama_tempat_index = data_content_based_filtering[data_content_based_filtering['nama_hotel']==nama_tempat].index[0]
        distancess = similarity[nama_tempat_index]
        nama_tempat_list = sorted(list(enumerate(distancess)),key=lambda x: x[1],reverse=True)[1:20]
        
        recommended_nama_tempats = []
        for i in nama_tempat_list:
            recommended_nama_tempats.append(([data_content_based_filtering.iloc[i[0]].nama_hotel]+[i[1]]))
            
        return recommended_nama_tempats
    
    def print_recommendations(recommendations):
        st.write("------ Hasil Rekomendasi Tempat Berdasarkan Kota ------")
        col1, col2 = st.columns(2)  # Mengatur jumlah kolom menjadi 2

        for index, recommendation in enumerate(recommendations, start=1):
            place_name, similarity_score = recommendation
            if index % 2 == 1:
                with col1:
                    st.write(f"{index}. {place_name}, Similarity Score: {similarity_score}")
            else:
                 with col2:
                    st.write(f"{index}. {place_name}, Similarity Score: {similarity_score}")
            



    # In[ ]:


    recommendations = recommend_by_content_based_filtering('abian harmony hotel')
    print_recommendations(recommendations)

    # In[ ]:


    from sklearn.cluster import KMeans
    import numpy as np  


    # In[ ]:

    st.write("---------------")
    hasil_rekomendasi= recommend_by_content_based_filtering('abian harmony hotel')
    


    # In[ ]:


    hasil_rekomendasi = np.array([
    ['anvaya hotel', 0.28977642414067206],
    ['anvaya hotel', 0.28977642414067206],
    ['kuta paradiso hotel', 0.25129041590527484],
    ['eden kuta hotel bali', 0.24453207262410825],
    ['potato head suites', 0.24383333420372777],
    ['primebiz hotel kuta', 0.23310975018328106],
    ['primebiz hotel kuta', 0.23140170237199423],
    ['anvaya hotel', 0.2136112590943877],
    ['anvaya hotel', 0.2136112590943877],
    ['kuta paradiso hotel', 0.19197232640899622],
    ['eden kuta hotel bali', 0.18965354036408813],
    ['anvaya hotel', 0.18333433765681129],
    ['anvaya hotel', 0.18333433765681129],
    ['infinity8 bali', 0.1825394643533978],
    ['primebiz hotel kuta', 0.18103713443568376],
    ['kuta paradiso hotel', 0.17899119079533488],
    ['eden kuta hotel bali', 0.17795665888724074],
    ['eden kuta hotel bali', 0.17758591690338613],
    ['anvaya hotel', 0.17628719130614134]]
    )


    # In[ ]:

    def perform_clustering(hasil_rekomendasi):
        features = hasil_rekomendasi[:, 1].astype(float)


        # In[ ]:


        features = features.reshape(-1, 1)


        # In[ ]:


        num_clusters = 4


        # In[ ]:


        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(features)


        # In[ ]:


        cluster_labels = kmeans.labels_


        # In[ ]:


        clustered_hasil_rekomendasi = []
        for i, item in enumerate(hasil_rekomendasi):
            item_cluster = cluster_labels[i]
            clustered_hasil_rekomendasi.append([item[0], item_cluster])


        # In[ ]:


        temp = []
        for item in clustered_hasil_rekomendasi:
            temp.append(item)


        # In[ ]:


        import matplotlib.pyplot as plt


        # In[ ]:


        hasil_cluster = np.array([
        ['anvaya hotel', 0],
        ['anvaya hotel', 0],
        ['kuta paradiso hotel', 0],
        ['eden kuta hotel bali', 0],
        ['potato head suites', 0],
        ['primebiz hotel kuta', 0],
        ['primebiz hotel kuta', 0],
        ['anvaya hotel', 1],
        ['anvaya hotel', 1],
        ['kuta paradiso hotel', 1],
        ['eden kuta hotel bali', 1],
        ['anvaya hotel', 1],
        ['anvaya hotel', 1],
        ['infinity8 bali', 1],
        ['primebiz hotel kuta', 1],
        ['kuta paradiso hotel', 1],
        ['eden kuta hotel bali', 1],
        ['eden kuta hotel bali', 1],
        ['anvaya hotel', 1]]
        )


        # In[ ]:


        features = hasil_cluster[:, 1].astype(float)

        features = features.reshape(-1, 1)

        k_values = range(1, len(hasil_cluster) + 1)
        inertia_values = []


    # In[ ]:


        for k in k_values:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(features)
            inertia_values.append(kmeans.inertia_)

    # Plot elbow curve
        st.write("Grafik Elbow Berdasarkan Hasil Clustering")
        fig, ax = plt.subplots()
        ax.plot(list(k_values), inertia_values, 'bx-')
        ax.set(xlabel='Number of Clusters (k)', ylabel='Inertia', title='Elbow Curve')
        st.pyplot(fig)

        return hasil_cluster
    
    st.write(" Hasil Clustering ")
    clustered_hasil_rekomendasi = perform_clustering(hasil_rekomendasi)
    st.write(clustered_hasil_rekomendasi)


#RUN PROGRAM
if __name__=='__main__':
    main()


# In[ ]:




