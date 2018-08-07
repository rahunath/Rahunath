import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams



stop = stopwords.words('english')
data = pd.read_csv("data.tsv", sep="\t", header=None)
data.columns = ["label", "body_text"]
data.head()


# Removing Punctuation

data['punct_Removed'] = data['body_text'].str.replace(r'[^\w\s]+', '')



# Setting to Lower Case

data['to_Lower'] = data['punct_Removed'].str.lower()


# Removing Stop Words

data['Stopwords_Removed'] = data['to_Lower'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))



# Word Tokenizing
data['Tokenized_Words'] = data['Stopwords_Removed'].apply(nltk.word_tokenize)



# Stemming using the PorterStemmer
stemmer = PorterStemmer()
data['Stemmed_Words'] = data['Tokenized_Words'].apply(lambda x: [stemmer.stem(y) for y in x])



# Lemmatizing using the WordnetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
data['Lemmatized_Words'] = data['Tokenized_Words'].apply(lambda x: [wordnet_lemmatizer.lemmatize(y) for y in x])


# create Bigrams

data['Bigrams'] = data['Lemmatized_Words'].apply(lambda x: list(ngrams(x,2)))


# Groupby label as Spam, hanm and aggregate to df,df1 dataframes.

df = data.groupby('label').agg({'Bigrams': 'sum'})
df1 =data.groupby('label').agg({'Lemmatized_Words': 'sum'})
df['count']=df['Bigrams'].str.len()


# Method to create Bigram dictionary having its bigrams and frequencies

def bigram_dic(bigram_corpus):

    bigram_dic = {}
    for bigram in bigram_corpus:
        if bigram not in bigram_dic:
            bigram_dic[bigram] = 1
        else:
            bigram_dic[bigram] += 1
    return bigram_dic



# Method to create Unigram dictionary having its Unigrams and frequencies

def unigram_dic(unigram_corpus):

    unigram_dic = {}
    for unigram in unigram_corpus:
        if unigram not in unigram_dic:
            unigram_dic[unigram] = 1
        else:
            unigram_dic[unigram] += 1

    return unigram_dic



# Method to calculate Ham probablity

def cal_ham_prob(word1, word2):
    try:
        no_word1 = Unigram_ham_dic[word1]
    except KeyError:
        no_word1 = 0

    try:
        biword = Bigram_ham_dic[word1, word2]
    except KeyError:
        biword = 0

    V = len(Unigram_ham_dic)
    prob = (biword + 1) / (no_word1 + V)
    return prob


# Method to calculate spam probablity
def cal_spam_prob(word1, word2):
    try:
        no_word1 = Unigram_spam_dic[word1]
    except KeyError:
        no_word1 = 0

    try:
        biword = Bigram_spam_dic[word1, word2]
    except KeyError:
        biword = 0

    V = len(Unigram_spam_dic)
    prob = (biword + 1) / (no_word1 + V)
    return prob


# Calling Methods and creating Unigram ham,spam dictonaries and bigram ham,spam dictonaries

Bigram_ham_corpus=df.iat[0,0]
Bigram_spam_corpus=df.iat[1,0]
Unigram_ham_corpus=df1.iat[0,0]
Unigram_spam_corpus=df1.iat[1,0]

Unigram_ham_dic = unigram_dic(Unigram_ham_corpus)
Unigram_spam_dic = unigram_dic(Unigram_spam_corpus)
Bigram_ham_dic = bigram_dic(Bigram_ham_corpus)
Bigram_spam_dic = bigram_dic(Bigram_spam_corpus)




# Method of preprocess of Given Sentence.

def preprocess(Sentence):
    Sentence = Sentence.lower()
    Sentence = re.sub(r'[^\w\s]','',Sentence)
    stop_words = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(Sentence)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    Sentence = " ".join(filtered_sentence)
    word_tokens = nltk.word_tokenize(Sentence)

    LemmatizedWords = []
    for ms in word_tokens:
        LemmatizedWords.append(WordNetLemmatizer().lemmatize(ms))

    bigrams = list(ngrams(LemmatizedWords, 2))
    bigram_of_given_word = bigram_dic(bigrams)

    return bigram_of_given_word



# Method to return total spam and ham probablity

def tot_pro(Sentence):
    bigram_of_given_word = preprocess(Sentence)
    total_spam_prob = 1
    total_ham_prob = 1
    for i in bigram_of_given_word:

        total_ham_prob = total_ham_prob*cal_ham_prob(i[0],i[1])
        total_spam_prob = total_spam_prob * cal_spam_prob(i[0],i[1])

    return total_ham_prob,total_spam_prob




def Final_results(Sentence):

    total_ham_probablity,total_spam_probablity = tot_pro(Sentence)

    if total_spam_probablity > total_ham_probablity :
        results="Spam"
    else:
        results="Ham"

    print("Sentence is :"+Sentence)
    print("Ham probablity :",total_ham_probablity)
    print("Spam probablity :", total_spam_probablity)
    print("Calculated type :"+results+'\n')



#input test cases
Sentence1="Sorry,  ..use your brain dear"
Sentence2="Six chances to win cash."

Final_results(Sentence1)
Final_results(Sentence2)