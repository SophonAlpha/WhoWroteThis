'''
Created on 16 Jun 2017

@author: H155936
'''

import re
import json
import pandas as pd
import numpy as np
import string
import time
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import bigrams
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # @UnusedImport
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from cycler import cycler
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import tree
from collections import Counter
from sklearn.naive_bayes import GaussianNB

ARTICLES_FILE = 'articles.json'

# TODO: "bags of words"
# TODO: word bigrams

def cleanup_text(text_corpus):
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
        text_corpus = re.sub(pattern[0], pattern[1], text_corpus)
    return text_corpus

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
            'body': [article_json['body']], # raw text_corpus of the article 
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
        if not data.isnull().values.any(): # skip articles with NaNs
            articles_list = articles_list.append(data, ignore_index = True)
    return articles_list

def split_paragraphs(text_corpus):
    paragraphs = [p for p in text_corpus.split('\n')]
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

def print_articles_with_longest_sentences(articles_list, num_articles):
    longest_sentence_articles = articles_list.sort_values('number_of_words_sentence_max',
                                                          ascending = False).head(num_articles)
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
    fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)
    
    for c in range(cols):
        for r in range(rows):
            axSubplt = axes[r, c]
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

    return fig

def plot_feature_scatters(articles_list, authors):
    plot_features = ['total_number_of_sentences',
                     'total_number_of_words',
                     'number_of_words_sentence_mean',
                     'number_of_words_sentence_median',
                     'number_of_words_sentence_min',
                     'number_of_words_sentence_max']
    plot_labels = ['total number\nof sentences',
                   'total number\nof words',
                   'words per\nsentence mean',
                   'words per\nsentence median',
                   'words per\nsentence min',
                   'words per\nsentence max']
    num_features = plot_features.__len__()
    
    # set up the figure 
    fig, axes = plt.subplots(nrows=num_features, ncols=num_features)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.suptitle("Feature Relationships", fontsize=14)
    
    # set up the axes
    for ax in axes.flat:
        ax.tick_params(axis='both',
                       bottom='off', top='off', left='off', right='off',
                       labelbottom='off', labeltop='off',
                       labelleft='off', labelright='off')
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
            ax.tick_params(axis='y', left='on', labelleft='on')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
            ax.tick_params(axis='y', right='on', labelright='on')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
            ax.tick_params(axis='x', top='on', labeltop='on')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')
            ax.tick_params(axis='x', bottom='on', labelbottom='on')
        # set colour cycler, each author a different colour 
        ax.set_prop_cycle(cycler('color', ['blue', 'green', 'red', 'cyan',
                                           'magenta', 'yellow', 'black', 'white']))
        
    # plot the scatter plots
    for c in range(num_features):
        for r in range(num_features):
            if c == r:
                # leave diagonal plots empty and remove ticks and labels
                axes[r, c].xaxis.set_visible(False)
                axes[r, c].yaxis.set_visible(False)
            elif r > c:
                axes[r, c].axis('off')
            else:
                axes[r, c].set_prop_cycle(None) # reset colour cycler
                for author in authors:
                    x = articles_list[articles_list.author == author][plot_features[c]]
                    y = articles_list[articles_list.author == author][plot_features[r]]
                    axes[r, c].scatter(x, y, s=2, alpha=0.5, label=author)

    # place a legend above top left plot
    axes[0, 1].legend(bbox_to_anchor=(-1, 1), loc=3)

    # add axis labels in diagonal boxes
    for i, label in enumerate(plot_labels):
        axes[i,i].annotate(label, (0.5, 0.5), 
                           xycoords='axes fraction', ha='center', va='center',
                           size=8)
    
    return fig

def get_dims_of_subplots(num_of_authors):
    (frac, intpart) = np.modf(np.sqrt(num_of_authors))
    if frac < 0.5:
        rows = int(intpart)
        cols = int(intpart + 1)
    else:
        rows = int(intpart + 1)
        cols = int(intpart + 1)
    return rows, cols

def plot_3D_GridSearch_results(df):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # transform data from dataframe
    
    startTime = time.time()
    X, Y, Z = transform_dataframe(df)
    elapsedTime = time.time() - startTime
    print('transform data via loops: {0}s'.format(elapsedTime))
    startTime = time.time()
    X, Y, Z = transform_dataframe_pivot(df)
    elapsedTime = time.time() - startTime
    print('transform data via df.pivot: {0}s'.format(elapsedTime))
    # results from the timing above:
    #     transform data via loops: 4.733000040054321s
    #     transform data via df.pivot: 0.005000114440917969s
    # Holy Sun!!!  Using pandas functions is 946.59 times faster!
    
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,  # @UndefinedVariable
                           linewidth=0, antialiased=False)
    
    # customise the z axis.
    ax.set_zlim(0, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()

def transform_dataframe(df):
    x_param = 'param_max_depth'
    y_param = 'param_min_samples_split'
    z_param = 'mean_test_score'
    X = df[x_param].drop_duplicates()
    Y = df[y_param].drop_duplicates()
    Z = pd.DataFrame(np.zeros((Y.__len__(), X.__len__())))
    for i_x, x in enumerate(X):
        for i_y, y in enumerate(Y):
            value = df[df[x_param] == x][df[y_param] == y][z_param].iloc[0]
            Z.iloc[i_y, i_x] = value
    X, Y = np.meshgrid(X, Y)
    return X, Y, Z

def transform_dataframe_pivot(df):
    Z = df.pivot(index='param_min_samples_split',
                 columns='param_max_depth',
                 values='mean_test_score')
    X = Z.columns
    Y = Z.index
    X, Y = np.meshgrid(X, Y)
    return X, Y, Z

def my_tokenizer(s):
    return s

if __name__ == '__main__':
    # ---------- data load ----------
    if False:
        # run this to load articles from JSON file and pre-process the articles
        article_lines = read_articles_from_file(ARTICLES_FILE)
        articles_list = get_article_metrics(article_lines)
        save_data_frame_to_disk(articles_list)
    if True:
        # run this to load a pandas data frame with the already pre-processed
        # articles. This is mainly to save time during development.
        articles_list = load_data_frame_from_disk()

    if False:
        # ---------- longest sentences ----------
        print_articles_with_longest_sentences(articles_list, 25)
    
        # ---------- charts ----------
        figure1 = plot_words_per_sentence_histogram(articles_list)
        plt.show(block=False)
        figure2 = plot_feature_scatters(articles_list, ['Vivek Wadhwa', 'Jill Schlesinger', 'Betty Liu'])
        figure2 = plot_feature_scatters(articles_list, ['Brian de Haaff', 'Anurag Harsh'])
        plt.show()
    
    # ---------- predict author using machine learning ----------
    # load data into separate pandas data frame
    data = articles_list[['author',
                          'total_number_of_sentences',
                          'total_number_of_words',
                          'number_of_words_sentence_mean',
                          'number_of_words_sentence_median',
                          'number_of_words_sentence_min',
                          'number_of_words_sentence_max']]
    # get the author with the smallest number of articles
    authors = data.author.unique()
    authors_num_articles = [[author, data[data.author == author].__len__()] for author in authors]
    articles_sum = pd.DataFrame(authors_num_articles, columns=['author', 'num_of_articles'])
    min_articles = articles_sum['num_of_articles'].min()
    # shuffle data set
    data = data.sample(frac=1)
    # delete data sets so that each author has the same number as the author with
    # smallest number of articles
    for author in authors:
        indexes_to_drop = data[data.author == author].index.tolist()[min_articles:]
        data.drop(indexes_to_drop, inplace=True)
        print('author: {0}, articles: {1}'.format(author, data[data.author == author].__len__()))
    data = data.reset_index(drop=True)
    # split labels into separate data frame
    labels = data[['author']]
    data = data.drop('author', 1)
    # shuffle training test split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=23)
    # decision tree prediction model
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    score_train = accuracy_score(y_train, clf.predict(X_train))
    score_test = accuracy_score(y_test, clf.predict(X_test))
    print('prediction accuracy score on training data set: {0}'.format(score_train))
    print('prediction accuracy score on test data set: {0}'.format(score_test))
    # decision tree prediction model with GridSearch
    if False:
        cv_sets = ShuffleSplit(n_splits=10, test_size = 0.25, random_state = 23)
        learning_model = tree.DecisionTreeClassifier()
        params = {'max_depth': list(range(1,30)),   
                  'min_samples_split': list(range(2,50))}
        scoring_fnc = make_scorer(accuracy_score, greater_is_better=True)
        grid = GridSearchCV(estimator=learning_model,
                            param_grid=params,
                            scoring=scoring_fnc,
                            cv=cv_sets)
        grid = grid.fit(X_train, y_train)
        best_clf = grid.best_estimator_
        score_test = accuracy_score(y_test, best_clf.predict(X_test))
        print('GridSearch prediction accuracy score on test data set: {0}'.format(score_test))
        print('best parameters:')
        print('    max_depth:{0}'.format(best_clf.get_params()['max_depth']))
        print('    min_samples_split:{0}'.format(best_clf.get_params()['min_samples_split']))
        cv_results_df = pd.DataFrame(grid.cv_results_)
        cv_results_df.to_pickle('cv_results_df.pickle', compression='gzip')
    cv_results_df = pd.read_pickle('cv_results_df.pickle', compression='gzip')
    plot_3D_GridSearch_results(cv_results_df[['param_max_depth', 'param_min_samples_split', 'mean_test_score']])
    # counting terms
    article_words = articles_list[articles_list.author == 'Jill Schlesinger'].iloc[0].sentences_words
    article_words = [w for s in article_words for w in s] # "unpack" list of words from list of sentences
    print(Counter(article_words).most_common(10))
    # exclude stop words
    stop = stopwords.words('english')
    article_words = [w for w in article_words if w not in stop]
    print('top word frequencies:\n {0}'.format(Counter(article_words).most_common(10)))
    # count bigrams
    terms_bigram = bigrams(article_words)
    fdist = nltk.FreqDist(terms_bigram)
    print('top bigram frequencies:\n{0}'.format(fdist.most_common(10)))

    text_corpus = [' '.join([' '.join(sentence) \
                    for sentence in article]) \
                        for article in articles_list.sentences_words]
    labels = articles_list.author
    vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words='english')
    X = vectorizer.fit_transform(text_corpus)
    print('no. of features with stop_words: {0}'.format(vectorizer.get_feature_names().__len__()))
    word_freq_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'occurrences':np.asarray(X.sum(axis=0)).ravel().tolist()})
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(X.toarray())
    # split training and test data
    X_train, X_test, y_train, y_test = train_test_split(tfidf, labels, test_size=0.25, random_state=23)

    # Gaussian Naive Bayes classifier
    clf = GaussianNB()
    clf = clf.fit(X_train.toarray(), y_train)
    score_train = accuracy_score(y_train, clf.predict(X_train.toarray()))
    score_test = accuracy_score(y_test, clf.predict(X_test.toarray()))
    print('Gaussian Naive Bayes training accuracy: {0}'.format(score_train))
    print('Gaussian Naive Bayes test accuracy: {0}'.format(score_test))

    # Decision Tree classifier
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train.toarray(), y_train)
    score_train1 = accuracy_score(y_train, clf.predict(X_train.toarray()))
    score_test1 = accuracy_score(y_test, clf.predict(X_test.toarray()))
    print('Decision Tree training accuracy: {0}'.format(score_train1))
    print('Decision Tree test accuracy: {0}'.format(score_test1))

    # Decision Tree classifier optimisation with GridSearch
    cv_sets = ShuffleSplit(n_splits=10, test_size = 0.25, random_state = 23)
    learning_model = tree.DecisionTreeClassifier()
    params = {'max_depth': list(range(1,30)),
              'min_samples_split': list(range(2,50))}
    scoring_fnc = make_scorer(accuracy_score, greater_is_better=True)
    grid = GridSearchCV(estimator=learning_model, 
                        param_grid=params, 
                        scoring=scoring_fnc, 
                        cv=cv_sets,
                        n_jobs=8,
                        verbose=5)
#     grid = grid.fit(X_train, y_train)
#     best_clf = grid.best_estimator_
#     score_test = accuracy_score(y_test, best_clf.predict(X_test))
#     cv_results_df = pd.DataFrame(grid.cv_results_)
#     cv_results_df.to_pickle('text_frequency_cv_results_df.pickle', compression='gzip')
#     print('GridSearch prediction accuracy score on test data set: {0}'.format(score_test))
#     print('best parameters:')
#     print('    max_depth:{0}'.format(best_clf.get_params()['max_depth']))
#     print('    min_samples_split:{0}'.format(best_clf.get_params()['min_samples_split']))



    # train decision tree model with k-fold and cross validation
    # plot R2 performance metric over size of training set
    # predict
    # report precision & recall
