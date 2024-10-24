import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import Sastrawi
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from translate import Translator
import preprocessor as p
from textblob import TextBlob
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import preprocessor as p
from textblob.classifiers import NaiveBayesClassifier
import random

norm = {" nggak ": " tidak ", " gak ": " tidak ", " tdk ": " tidak ", " yg ": " yang "}
more_stop_words = ["ada", "adalah", "adanya", "akan", "amat", "an", "anda", "andalah", "antara", "apa", "apaan", "apakah", "apalagi", "bagi", "bahkan", "bagaimana", "bahwa", "bahwasanya", "baik", "beberapa", "bagian", "banyak", "baru", "bawah", "berikut", "berbagai", "bersama", "bersama-sama", "bisa", "boleh", "bukan", "kepada", "kalian", "kami", "kamu", "karena", "dari", "daripada", "dalam", "dengan", "di", "dia", "dirimu", "juga", "jika", "lagi", "lain", "lalu", "mana", "maka", "atau", "telah", "kemudian", "kalau", "sedang", "dan", "tapi", "dapat", "itu", "saja", "hanya", "lebih", "setiap", "sangat", "sudah", "ini", "pada", "lebih", "saja", "lagi", "maka", "sangat", "atau", "kami", "atau", "sangat", "hanya", "lebih", "dan", "saja", "maka", "telah", "tetapi", "baru"]
stop_words = StopWordRemoverFactory().get_stop_words()
stop_words.extend(more_stop_words)
new_array = ArrayDictionary(stop_words)
stop_words_remover_new = StopWordRemover(new_array)

def cleaning(text):
    # Menghapus mention (contoh: @username)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)

    # Menghapus hashtag (contoh: #hashtag)
    text = re.sub(r'#\w+', '', text)

    # Menghapus retweet marker (contoh: RT)
    text = re.sub(r'RT[\s]+', '', text)

    # Menghapus URL
    text = re.sub(r'https?://\S+', '', text)

    # Menghapus karakter selain huruf, angka, dan spasi
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)

    # Menghapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def normalisasi(str_text):
    for i in norm:
        str_text = str_text.replace(i, norm[i])
    return str_text

def stopword(str_text):
    str_text = stop_words_remover_new.remove(str_text)
    return str_text

def stemming(text_cleaning):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    do = []
    for w in text_cleaning:
        dt = stemmer.stem(w)
        do.append(dt)
    d_clean = []
    d_clean = " ".join(do)
    return d_clean

def convert(text):
    translator = Translator(to_lang="en", from_lang="id")
    translation = translator.translate(text)
    return translation

# Create a sidebar with menu options
menu = ["Input Text", "Upload File"]
choice = st.sidebar.selectbox("Select Menu", menu)

# Display the appropriate input widget based on the menu choice
# Saya membenci                           pres??>?<>:iden yg suka tidur
# Saya suka dengan pelayanan hotel bintang 5 yang bagus
# Saya tidur jam 2
if choice == "Input Text":
    text_input = st.text_input("Enter some text")
    if st.button("Proses"):
        st.subheader("Teks yang dimasukkan:")
        st.write(text_input)
        text_input = cleaning(text_input)
        st.subheader("Cleaning")
        st.write(text_input)
        st.subheader("Lowercase")
        text_input = text_input.lower()
        st.write(text_input)
        st.subheader("Normalisasi")
        text_input = normalisasi(text_input)
        st.write(text_input)
        st.subheader("Stopword")
        text_input = stopword(text_input)
        st.write(text_input)
        st.subheader("Tokenizing")
        text_input = (lambda x: x.split())(text_input)
        st.write(text_input)
        st.subheader("Stemming")
        text_input = stemming(text_input)
        st.write(text_input)
        st.subheader("Translate")
        text_input = convert(text_input)
        st.write(text_input)
        st.subheader("Hasil Analisis Sentimen")
        analysis = TextBlob(text_input)
        if analysis.sentiment.polarity > 0:
            st.write("Positif")
        elif analysis.sentiment.polarity < 0:
            st.write("Negatif")
        else:
            st.write("Netral")
        
        

    
elif choice == "Upload File":
    uploaded_file = st.file_uploader("Upload an XLSX or CSV file", type=['xlsx', 'csv'])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        if st.button("Proses"):
            st.subheader("Data Awal")
            st.dataframe(df)
            df = df[['full_text', 'username', 'created_at']]
            st.dataframe(df)
            df = df.drop_duplicates(subset=['full_text'])
            df.duplicated().sum()
            df = df.dropna()
            df.isnull().sum()
            st.subheader("Cleaning")
            df['full_text'] = df['full_text'].apply(cleaning)
            df['full_text']=df['full_text'].str.lower()
            st.dataframe(df)
            st.subheader("Normalisasi")
            df["full_text"] = df["full_text"].apply(normalisasi)
            st.dataframe(df)
            st.subheader("Stopword")
            df["full_text"] = df["full_text"].apply(stopword)
            st.dataframe(df)
            st.subheader("Tokenizing")
            df["full_text"] = df['full_text'].apply(lambda x:x.split())
            st.dataframe(df)
            st.subheader("Stemming")
            df["full_text"] = df["full_text"].apply(stemming)
            st.dataframe(df)
            st.subheader("Translate")
            df["full_text"] = df["full_text"].apply(convert)
            st.dataframe(df)

            data_tweet  = list(df['full_text'])
            pol = 0
            status = []
            positif = negatif = netral = total = 0
            for i, tweet in enumerate(data_tweet):
                analysis = TextBlob(tweet)
                pol += analysis.sentiment.polarity
                
                if analysis.sentiment.polarity == 0:
                    netral += 1
                    status.append('netral')
                elif analysis.sentiment.polarity > 0:
                    positif += 1
                    status.append('positif')
                else:
                    negatif += 1
                    status.append('negatif')
            st.write(f'Positif = {positif}\nNetral = {netral}\nNegatif = {negatif}\n\nTotal = {positif+negatif+netral}')
            df['klasifikasi'] = status
            st.dataframe(df)
            labels = ['Positif', 'Netral', 'Negatif']
            values = [positif, netral, negatif]

            fig, ax = plt.subplots()
            ax.bar(labels, values, color=['green', 'blue', 'red'])
            ax.set_xlabel('Kategori')
            ax.set_ylabel('Jumlah')
            ax.set_title('Distribusi Sentimen')

            # Menampilkan diagram batang di Streamlit
            st.pyplot(fig)
            dataset = df.drop(columns=['username', 'created_at'])
            st.dataframe(df)
            dataset = [tuple(x) for x in dataset.to_records(index=False)]
            set_positif = []
            set_negatif = []
            set_netral = []

            for i in dataset:
                if i[1] == 'positif':
                    set_positif.append(i)
                elif i[1] == 'negatif':
                    set_negatif.append(i)
                else:
                    set_netral.append(i)

            set_positif = random.sample(set_positif, k=int(len(set_positif)/2))
            set_negatif = random.sample(set_negatif, k=int(len(set_negatif)/2))
            set_netral = random.sample(set_netral, k=int(len(set_netral)/2))

            train = set_positif + set_negatif + set_netral

            train_set = []

            for n in train:
                train_set.append(n)
            cl = NaiveBayesClassifier(train_set)
            st.write(cl.accuracy(dataset))
            data_tweet  = list(df['full_text'])
            pol = 0

            status = []
            positif = negatif = netral = total = 0

            for i, tweet in enumerate(data_tweet):
                analysis = TextBlob(tweet, classifier=cl)
                
                if analysis.classify() == 'netral':
                    netral += 1
                elif analysis.classify() == 'positif':
                    positif += 1
                else:
                    negatif += 1

                status.append(analysis.classify())


            st.write(f'Positif = {positif}\nNetral = {netral}\nNegatif = {negatif}\n\nTotal = {positif+negatif+netral}')
            status = pd.DataFrame(status)

            df['klasifikasi_bayes'] = status
            st.dataframe(df)
            labels = ['Positif', 'Netral', 'Negatif']
            values = [positif, netral, negatif]

            fig, ax = plt.subplots()
            ax.bar(labels, values, color=['green', 'blue', 'red'])
            ax.set_xlabel('Kategori')
            ax.set_ylabel('Jumlah')
            ax.set_title('Distribusi Sentimen')

            # Menampilkan diagram batang di Streamlit
            st.pyplot(fig)
        