import spacy
import pandas as pd
from string import punctuation

from nltk import ngrams
from nltk.corpus import wordnet, stopwords
from collections import Counter
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_lg")

context_threshold = 0.7

def words_relatedness(word1, word2):
    # Process the words with spaCy
    token1 = nlp(word1)
    token2 = nlp(word2)

    max_similarity = 0
    
    # Iterate through all tokens of each word
    for t1 in token1:
        for t2 in token2:
            similarity = t1.similarity(t2)
            if similarity > max_similarity:
                max_similarity = similarity
    
    return max_similarity

def get_category_row(current_categories, df_row):
    #print(df_row['trigger_words'])
    cat_word_scores = {}
    scores = {word:score for word, score in df_row['context_score'].items() if score > context_threshold}
    for category in current_categories:
        for word in scores.keys():
            if len(word.split("_")) > 1:
                subwords = word.split("_")
                for subword in subwords:
                    cat_word_scores[(category, subword)] = words_relatedness(subword, category)
            else:
                cat_word_scores[(category, word)] = words_relatedness(word, category)
    
    max_score = max(cat_word_scores.values())
    new_category = [k[0] for k, v in cat_word_scores.items() if v == max_score][0]
    return current_categories, new_category


def get_category_df(df, current_categories):
    df['category'] = None
    for idx, row in df.iterrows():
        try:
            current_categories, category = get_category_row(current_categories, row)
            df.at[idx, 'category'] = category
        except Exception as e:
            print("Error at %d: %s" % (idx, e))
            print(current_categories)
            break
    
    return df, current_categories

# Function to generate n-grams for the title
def generate_ngrams(text, n):
    # Tokenize the text into words
    tokens = word_tokenize(text)
    # Generate n-grams
    return list(ngrams(tokens, n))

def get_contextual_features(title):
    doc = nlp(title)
    lemma = []
    pos = []
    tag = []
    dep = []
    label = []
    
    for token in doc:
        if token.text in punctuation:
            continue
        lemma.append(token.lemma_)
        pos.append(token.pos_)
        tag.append(token.tag_)
        dep.append(token.dep_)
        label.append(token.ent_type_)
        
    return lemma, pos, tag, dep, label

def get_contextual_features_df(df):
    lemma = []
    pos = []
    tag = []
    dep = []
    label = []

    for idx, row in df.iterrows():
        title = row['title']
        l, p, t, d, la = get_contextual_features(title)
        lemma.append(l)
        pos.append(p)
        tag.append(t)
        dep.append(d)
        label.append(la)

    df['lemma'] = lemma
    df['pos'] = pos
    df['tag'] = tag
    df['dep'] = dep
    df['label'] = label
    
    return df

def contains_digit(word):
    for char in word:
        if char.isdigit():
            return True
    return False

def extract_trigger_words(title):
    result = []
    pos_tag = ['ADJ', 'NOUN', 'VERB', 'ADV', 'NNP', 'PROPN'] 
    label_type = ['TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
    doc = nlp(title.lower())

    prev_label = None
    trigger_combi = None
    for token in doc:
        if token.text in nlp.Defaults.stop_words or token.text in punctuation or contains_digit(token.text):
            continue
        if (token.pos_ in pos_tag) and (token.ent_type_ not in label_type):
            if prev_label:
                if prev_label == token.ent_type_:
                    trigger_combi += '_' + token.text
                    continue
                else:
                    if trigger_combi:
                        result.append(trigger_combi.lower())
                        trigger_combi = None
                        prev_label = None
                    result.append(token.text.lower())
                    continue
            
            if len(token.ent_type_) > 0:
                prev_label = token.ent_type_
                trigger_combi = token.text
            else:
                if trigger_combi:
                    result.append(trigger_combi.lower())
                    trigger_combi = None
                    prev_label = None
                result.append(token.text.lower())
        
    if trigger_combi:
        result.append(trigger_combi.lower())
        trigger_combi = None
        prev_label = None

    return result

def get_context_score(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['title'])
    words = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    word_scores = dict(zip(words, tfidf_scores))
    
    df['context_score'] = None
    for index, row in df.iterrows():
        score = {}
        for word in row['trigger_words']:
            sum_score = 0
            word_ls = word.split('_')
            for w in word_ls:
                if w in word_scores:
                    sum_score += word_scores[w]
                else:
                    sum_score += 0
                    
            if sum_score > 0:
                score[word] = sum_score / len(word_ls)
            else:
                score[word] = 0
                
            
        if len(score) > 1:
            max_score = max(score.values())
            if max_score > 0:
                for key in score:
                    score[key] = score[key] / max_score
        
        elif len(score) == 1:
            for key in score:
                score[key] = 1.0
            
        score = {k: v for k, v in score.items()}
        
        df.at[index, 'context_score'] = score
        
    return df

def get_features(df):
    df['word_count'] = df['title'].str.split().str.len()
    df['character_count'] = df['title'].str.len()
    df['bigrams'] = df['title'].apply(lambda title: generate_ngrams(title, 2))
    df = get_contextual_features_df(df)
    
    df['trigger_words'] = df['title'].apply(extract_trigger_words)
    df = df[df['trigger_words'].map(len) > 0]
    df = get_context_score(df)
    df = df[df['context_score'].map(len) > 0]
    
    return df


input_file_location = "data/full_maven.csv"
output_file_location = "data/full_maven_with_category.csv"

categories = ["business", "politics", "technology", "entertainment", "sports", "lifestyle", "health", "science", "education", "editorial", "international", "environment", "crime", "travel", "social"]
columns = ['title', 'word_count', 'character_count', 'bigrams', 'lemma', 'pos', 'tag', 'dep', 'label', 'context_score', 'trigger_words', 'category']

if __name__ == "__main__":
    df = pd.read_csv(input_file_location) # Original dataset file location
    df = get_features(df)
    
    df, current_categories = get_category_df(df, categories)
    df = df[columns]
    
    df.to_csv(output_file_location, index=False) # Output dataset file location
    