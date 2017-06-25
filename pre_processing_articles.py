'''
Created on 16 Jun 2017

@author: H155936
'''

import re
import json
import pandas as pd
import numpy as np
import string
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

ARTICLES_FILE = 'articles.json'

# TODO: "bags of words"
# TODO: word bigrams

def cleanup_text(text):
    replacement_specification = [
        # fix sentence that end before quotation mark, 'sent_tokenize' doesn't recognise these
        (r'.”', '.”.'),
        # remove 'about the author' sections
        (r'ABOUT BRIAN AND AHA!(.*\s*)*', ''),
        (r'ABOUT THE AUTHOR(.*\s*)*', ''),
        (r'Ian Bremmer is president of(.*\s*)*', ''),
        (r'If you liked this article and want more content to help you transform into a better leader, join the Radiate community by clicking here.', ''),
        (r'For more, read my book, Driver in the Driverless Car, follow me on Twitter: @wadhwa, and visit my website: www.wadhwa.com', ''),
        (r'“Better Off” is sponsored by Betterment.(.*\s*)*', ''),
        (r'For more, go to JillonMoney.com', ''),
        (r'For more, go to Jill on Money', ''),
        (r'Image by Flickr User(.*\s*)*', ''),
        (r'Dustin McKissen is the founder(.*\s*)*', '')]
    for pattern in replacement_specification:
        text = re.sub(pattern[0], pattern[1], text)
    return text

def read_articles_from_file(file_name):
    with open(file_name, 'r') as f:
        article_lines = f.readlines()
    return article_lines

def get_article_metrics(article_lines):
    articles_list = pd.DataFrame()
    num_articles = article_lines.__len__()
    for idx, article in enumerate(article_lines):
        print('processing article {0} of {1}'.format(idx, num_articles))
        article_json = json.loads(article)
        article_json['body'] = cleanup_text(article_json['body'])
        paragraphs = split_paragraphs(article_json['body'])
        sentences = tokenize_sentences(paragraphs)
        article_words, sentences_words = tokenize_words(sentences)
        data = pd.DataFrame(data = {
            'author': [article_json['author']],
            'body': [article_json['body']], # raw text of the article 
            'sentences_words': [sentences_words], # list of words nested in list of sentences
            'headline': [article_json['headline']],
            'url': [article_json['url']],
            'total_number_of_sentences': [article_words.shape[0]],
            'total_number_of_words': [article_words.sum(axis=0)],
            'number_of_words_each_sentence': [article_words],
            'number_of_words_sentence_mean': [article_words.mean(axis='index')],
            'number_of_words_sentence_median': [article_words.median(axis='index')],
            'number_of_words_sentence_min': [article_words.min(axis='index')],
            'number_of_words_sentence_max': [article_words.max(axis='index')]})
        articles_list = articles_list.append(data, ignore_index = True)
    return articles_list

def split_paragraphs(text):
    paragraphs = [p for p in text.split('\n')]
    return paragraphs

def tokenize_sentences(paragraphs):
    sentences = []
    for p in paragraphs:
        sentences.extend(sent_tokenize(p))
    return sentences

def tokenize_words(sentences):
    words_per_sentence = []
    sentences_words = []
    for sent in sentences:
        words = word_tokenize(sent)
        # remove punctuation from word list
        words = [w for w in words if not re.fullmatch('[' + string.punctuation + '’“‘”–…' ']', w)]
        words_per_sentence.append(words.__len__())
        sentences_words.append(words)
    article_words = pd.Series(data=words_per_sentence)
    return article_words, sentences_words

def save_data_frame_to_disk(articles_list):
    articles_list.to_pickle('articles.pickle', compression='gzip')

def load_data_frame_from_disk():
    return pd.read_pickle('articles.pickle', compression='gzip')

def get_articles_with_longest_sentences(articles_list, num):
    longest_sentence_articles = articles_list.sort_values('number_of_words_sentence_max',
                                                          ascending = False).head(num)
    for index, row in longest_sentence_articles.iterrows():
        print('author: {0}, url: {1}'.format(row.author, row.url))
        print('max sentence length: {0}'.format(int(row.number_of_words_sentence_max)))
        max_value_idx = row.number_of_words_each_sentence.idxmax()
        longest_sentence = ' '.join(row.sentences_words[max_value_idx])
        print('longest sentence: {0}'.format(longest_sentence))
        print('---------------------------------------------------------------')

def plot_words_per_sentence_histogram(articles_list):
    authors = iter(articles_list.author.unique())
    rows, cols = get_dims_of_subplots(articles_list.author.unique().__len__())
    fig, ax_lst = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)
    for c in range(cols):
        for r in range(rows):
            axSubplt = ax_lst[r, c]
            axSubplt.set_axisbelow(True)
            axSubplt.grid(True, color='grey', linestyle='dashed')
            try:
                author = next(authors)
            except StopIteration:
                break
            axSubplt.set_title(author)
            author_articles = articles_list[articles_list.author == author]
            print('{0}: {1} articles'.format(author, author_articles.__len__()))
            words_each_sentence = [element for s in author_articles.number_of_words_each_sentence for element in s]
            n, bins, patches = axSubplt.hist(words_each_sentence, 50, facecolor='green')
    plt.show(block=False)

def plot_feature_scatters(articles_list):
    # scatter plots of the following features:
    #    total_number_of_sentences
    #    total_number_of_words
    #    number_of_words_sentence_mean
    #    number_of_words_sentence_median
    #    number_of_words_sentence_min
    #    number_of_words_sentence_max
    author1 = 'Vivek Wadhwa'
    author2 = 'Jill Schlesinger'
    rows = 1
    cols = 2
    fig, ax_lst = plt.subplots(nrows=rows, ncols=cols)
    # total_number_of_sentences vs. total_number_of_words
    x1 = articles_list[articles_list.author == author1].total_number_of_sentences
    y1 = articles_list[articles_list.author == author1].total_number_of_words
    x2 = articles_list[articles_list.author == author2].total_number_of_sentences
    y2 = articles_list[articles_list.author == author2].total_number_of_words
    
    axSubplt = ax_lst[0]
    axSubplt.scatter(x1, y1, s=0.5, c='green')
    axSubplt.scatter(x2, y2, s=0.5, c='blue')
    axSubplt.set_xlabel('number of sentences')
    axSubplt.set_ylabel('number of words')
    plt.show(block=False)

def get_dims_of_subplots(num_of_authors):
    (frac, intpart) = np.modf(np.sqrt(num_of_authors))
    if frac < 0.5:
        rows = int(intpart)
        cols = int(intpart + 1)
    else:
        rows = int(intpart + 1)
        cols = int(intpart + 1)
    return rows, cols

if __name__ == '__main__':
    if False:
        # run this to load articles from JSON file and pre-process the articles
        article_lines = read_articles_from_file(ARTICLES_FILE)
        articles_list = get_article_metrics(article_lines)
        save_data_frame_to_disk(articles_list)
    if True:
        # run this to load a pandas data frame with the already pre-processed
        # articles. This is mainly to save time during development.
        articles_list = load_data_frame_from_disk()
    get_articles_with_longest_sentences(articles_list, 25)
    plot_words_per_sentence_histogram(articles_list)
    plot_feature_scatters(articles_list)
    plt.show()





