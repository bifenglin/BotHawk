from nltk.corpus import stopwords
import string
import pandas as pd

BASE_DIR = '/Users/bifenglin/Code/maxcompute/scripts/bothawk/'
issue_logs_df = pd.read_csv(BASE_DIR+'data/bothawk_issue_logs.txt')
issue_logs_df


# turn a doc into clean tokens
def clean_doc(doc):
    # split into tokens by white space
    if isinstance(doc, str):  # check if the input is a string
        doc = doc.lower()  # convert the string to lowercase
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 2]
    return tokens

def clean_doc_test():
    string_list = []
    for index in df['issue_comment_body'].iterrows():
        string_list.append(index['issue_comment_body'])
    clean_list = []
    for i in string_list:
        clean_list.append(clean_doc(i))
    clean_list

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def jaccard_similarity(s1, s2):
    def add_space(s):
        return ' '.join(s)

    s1, s2 = add_space(s1), add_space(s2)
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    numerator = np.sum(np.min(vectors, axis=0))
    denominator = np.sum(np.max(vectors, axis=0))
    return 1.0 * numerator / denominator

def get_jaccard_similarity(clean_list):
    total = 0.0
    num = 0.0
    for i in clean_list:
        for j in clean_list:
            if i != j:
                num += 1
                total += jaccard_similarity(i, j)
    if num == 0:
        return 0
    return total/num

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.linalg import norm

def tfidf_similarity(s1, s2):
    def add_space(s):
        return ' '.join(s)

    s1, s2 = add_space(s1), add_space(s2)
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))

def get_tfidf_similarity(clean_list):
    total = 0.0
    num = 0.0
    for i in clean_list:
        for j in clean_list:
            if i != j:
                num += 1
                total += tfidf_similarity(i, j)
    if num == 0:
        return 0
    return total/num

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def cosine_similarity(x, y):
    """Calculate cosine similarity"""
    numerator = np.dot(x, y)  # 分子
    denominator = np.linalg.norm(x) * np.linalg.norm(y)  # 分母
    similarity = numerator / denominator
    return similarity

def get_cosine_similarity(clean_list):
    total = 0.0
    num = 0.0
    # Convert text to word vector matrix
    cv = CountVectorizer(tokenizer=lambda s: s.split(), lowercase=False, min_df=1, max_df=0.95)
    corpus = [' '.join(text) for text in clean_list]
    vectors = cv.fit_transform(corpus).toarray()
    for i in range(len(clean_list)):
        for j in range(i+1, len(clean_list)):
            # Calculate cosine similarity
            similarity = cosine_similarity(vectors[i], vectors[j])
            num += 1
            total += similarity
    if num == 0:
        return 0
    return total / num

from tqdm import tqdm
import types
df_test = issue_logs_df[['actor_id', 'actor_login', 'issue_comment_body']]
# df_test
result = []
err_list = []
grouped = df_test.groupby('actor_id')
i = 0
print(len(grouped))
for actor_id,group in grouped:
    # print('actor_id:', actor_id)
    # 清洗数据
    i = i + 1
    print(i)
    # if i < 5399:
    #     continue
    string_list = []
    for index in group['issue_comment_body']:
        if isinstance(index, str):
            string_list.append(index)
    clean_list = []
    for index in string_list:
        clean_list.append(clean_doc(index))

    if len(clean_list) < 2:
        result.append({'actor_id': actor_id, 'cosin_similarity': 0})
        # result.append({'jaccard_similarity': 0, 'tfidf_similarity':0})
        continue
    # jaccard = get_jaccard_similarity(clean_list)
    # tfidf = get_tfidf_similarity(clean_list)
    try:
        cosine = get_cosine_similarity(clean_list)
    except:
        cosine = 0
        print('error:{}', actor_id)
        err_list.append(actor_id)
    res_dic = {'actor_id': actor_id,  'cosin_similarity': cosine}
    print(res_dic)
    result.append(res_dic)

    if len(result) == 5:
        result_df = pd.DataFrame(result)
        result_df.to_csv(BASE_DIR+'data/sim5.csv')
    if len(result) == 8000:
        result_df = pd.DataFrame(result)
        result_df.to_csv(BASE_DIR+'data/sim_cos_8000.csv')
    if len(result) == 16000:
        result_df = pd.DataFrame(result)
        result_df.to_csv(BASE_DIR+'data/sim_cos_16000.csv')

result_df = pd.DataFrame(result)
result_df.to_csv(BASE_DIR+'data/similar_cos.csv')
err_df = pd.DataFrame(err_list)
err_df.to_csv(BASE_DIR+'data/err_list.csv')
