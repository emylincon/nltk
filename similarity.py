from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

ps = PorterStemmer()

stop_words = set(stopwords.words("english"))


def remove_stopwords():
    enter_words = input('enter sentence: ').strip()
    words = word_tokenize(enter_words)
    stemmed = [ps.stem(w) for w in words]
    filt_sen = ' '.join([w for w in stemmed if not w in stop_words])
    return filt_sen


def train(sentence):
    documents = (
        sentence,
        "weather forecast today",
        "weather forecast city",
        "what weather london",
        "light off",
        "what weather forecast",
    )
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    # print(tfidf_matrix.shape)
    array = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
    return array[0][1:]


def similarity(arr):
    dic_comp = {1:"weather forecast today",
        2:"weather forecast city",
        3:"what weather london",
        4:"light off",
        5:"what weather forecast"}
    dic_sim = {1:arr[0], 2:arr[1], 3:arr[2], 4:arr[3], 5:arr[4]}
    max_dic_sim = max(dic_sim, key=dic_sim.get)
    if dic_sim[max_dic_sim] > 0.50:
        return dic_comp[max_dic_sim]
    else:
        return 0


def main():
    y = similarity(train(remove_stopwords()))
    if y == 0:
        print('No Match')
    else:
        print('sentence similar to: ', y)


if __name__ == '__main__':
    main()
