import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import plotly.express as px

# Modelleri ve veriyi yükle
@st.cache_data
def load_data():
    try:
        # Kitap verilerini yükle
        data = pd.read_csv("books_data.csv", encoding='latin-1', on_bad_lines='skip')
        
        # Popüler kitapları yükle
        popular_books = pd.read_csv("popular_books.csv")
        
        # Benzerlik modelini yükle
        with open("model.pkl", "rb") as file:
            cosine_sim = pickle.load(file)
        
        # Book content sütunu oluştur
        data['book_content'] = data['title'] + ' ' + data['authors']
        
        return data, popular_books, cosine_sim
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {e}")
        return None, None, None

def recommend_books(book_title, data, cosine_sim):
    try:
        idx = data[data['title'] == book_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        book_indices = [i[0] for i in sim_scores]
        return data.iloc[book_indices][['title', 'authors', 'average_rating']]
    except Exception as e:
        st.error(f"Öneri oluşturulurken hata: {e}")
        return None

def main():
    st.set_page_config(page_title="📚 Kitap Öneri Sistemi", page_icon="📖")
    
    # Veriyi ve modelleri yükle
    data, popular_books, cosine_sim = load_data()
    if data is None:
        return
    
    # Sayfa başlığı ve açıklama
    st.title("📚 Kitap Öneri Sistemi")
    st.write("Kişiselleştirilmiş kitap önerileri için aşağıdaki seçenekleri kullanabilirsiniz.")
    
    # Yan menü
    menu = st.sidebar.radio("Öneri Türünü Seçin", 
        ["Popüler Kitaplar", "İçerik Tabanlı Öneri"])
    
    if menu == "Popüler Kitaplar":
        st.subheader("🔥 En Popüler Kitaplar")
        st.dataframe(popular_books)
    
    else:
        st.subheader("📖 Benzer Kitap Önerisi")
        
        # Kullanıcıdan kitap seçimi
        book_titles = sorted(data['title'].unique())
        selected_book = st.selectbox("Bir kitap seçin:", book_titles)
        
        if st.button("Kitap Önerilerini Göster"):
            recommendations = recommend_books(selected_book, data, cosine_sim)
            if recommendations is not None:
                st.write(f"'{selected_book}' kitabı için öneriler:")
                st.dataframe(recommendations)
    
    # Görselleştirmeler
    st.sidebar.header("📊 Veri Görselleştirmeleri")
    
    if st.sidebar.checkbox("Ortalama Değerlendirme Dağılımı"):
        fig = px.histogram(data, x='average_rating', 
                           nbins=30, 
                           title='Ortalama Değerlendirme Dağılımı')
        fig.update_xaxes(title_text='Ortalama Değerlendirme')
        fig.update_yaxes(title_text='Sıklık')
        st.plotly_chart(fig)
    
    if st.sidebar.checkbox("En Çok Kitap Yazan Yazarlar"):
        top_authors = data['authors'].value_counts().head(10)
        fig = px.bar(top_authors, x=top_authors.values, y=top_authors.index, 
                     orientation='h',
                     labels={'x': 'Kitap Sayısı', 'y': 'Yazar'},
                     title='Yazara Göre Kitap Sayısı')
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()
    