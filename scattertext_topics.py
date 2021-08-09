import scattertext as st
from scattertext import LogOddsRatioInformativeDirichletPrior
import pandas as pd
import os
import glob
import os.path
import codecs
from nltk.tokenize import WordPunctTokenizer
import numpy as np
import spacy

def read_articles_from_file_list(folder_name, file_pattern="*.txt"):
    """
    Read articles from files matching patterns <file_pattern> from
    the directory <folder_name>.
    The content of the article is saved in the dictionary whose key
    is the id of the article (extracted from the file name).
    Each element of <sentence_list> is one line of the article.
    """
    file_list = glob.glob(os.path.join(folder_name, file_pattern))
    articles = {}
    for filename in sorted(file_list):
        article_id = os.path.basename(filename).split(".")[0][7:]
        with codecs.open(filename, "r", encoding="utf8") as f:
            articles[article_id] = f.read()
    return articles


def return_annotated_articles(spansfile, articles, train=True):
    token_labels = dict()
    # we first read all the propaganda spans from spans file and we label the corresponding tokens
    if train:
        with open(spansfile, "r") as f:
            for row in f.readlines():
                if len(row.split('\t')) == 3:
                    continue
                article_id, methods, span_start, span_end = row.rstrip().split("\t")
                tokenized_article = list(WordPunctTokenizer().span_tokenize(articles[article_id]))
                token_start = 0
                token_end = 0
                for poz in range(0, len(tokenized_article)):
                    t_poz = tokenized_article[poz]
                    if t_poz[0] <= int(span_start) < t_poz[1]:
                        token_start = poz
                    if int(span_end) <= t_poz[1]:
                        token_end = poz
                        break
                if article_id not in token_labels:
                    token_labels[article_id] = [['O']] * len(tokenized_article)
                # token_labels[article_id][token_strt]='B-prop'
                for i in range(token_start, token_end):
                    token_labels[article_id][i] = token_labels[article_id][i] + [methods]

    tokens = []
    labels = []
    sentences = []
    spans = []
    multilabel = 0
    # we now create the output matrices
    to_skip = set()
    for id in articles:
        article = articles[id].strip().split("\n")
        index = 0
        if id in token_labels:
            list_labels = token_labels[id]
        else:
            list_labels = ['O'] * len(WordPunctTokenizer().tokenize(articles[id]))
        spans_article = list(WordPunctTokenizer().span_tokenize(articles[id]))
        for sentence in article:
            if len(sentence) > 0:
                sentence_tokens = list(WordPunctTokenizer().tokenize(sentence))
                if len(sentence_tokens) == 0:
                    continue
                label_list = list_labels[index: index + len(sentence_tokens)]
                label = [y for x in label_list for y in x]
                tokens.append(sentence_tokens)
                spans.append((id, spans_article[index: index + len(sentence_tokens)]))
                index = index + len(sentence_tokens)
                unique = np.unique(label)
                list_u = []
                for u in unique:
                    if u != 'O' and u not in to_skip:
                        list_u.append(u)
                if len(list_u):
                    if len(list_u) > 1:
                        for u in list_u:
                            sentences.append(sentence)
                            labels.append(u)
                    else:
                        sentences.append(sentence)
                        labels.append(list_u[0])
                else:
                    sentences.append(sentence)
                    labels.append("Not_Propaganda")
    return sentences, labels


if __name__ == '__main__':

    articles = read_articles_from_file_list("../data/train")

    tokens, labels = return_annotated_articles("../data/articles_train.labels", articles)
    d = {'sentences':tokens, 'labels':labels}
    df = pd.DataFrame(d, columns=['sentences', 'labels'])
    fn = 'demo_log_odds_ratio_prior.html'
    nlp = spacy.load('en')
    corpus = (st.CorpusFromPandas(df,
                                  category_col='labels',
                                  text_col='sentences',
                                  nlp=nlp)
              .build().get_stoplisted_unigram_corpus())
    categories = ["Appeal_to_Authority",
    "Appeal_to_fear-prejudice",
    "Bandwagon",
    "Black-and-White_Fallacy",
    "Causal_Oversimplification",
    "Doubt",
    "Exaggeration,Minimisation",
    "Flag-Waving",
    "Loaded_Language",
    "Name_Calling,Labeling",
    "Obfuscation,Intentional_Vagueness,Confusion",
    "Red_Herring",
    "Reductio_ad_hitlerum",
    "Repetition",
    "Slogans",
    "Straw_Men",
    "Thought-terminating_Cliches",
    "Whataboutism",
    "Not_Propaganda"]
    for category in categories:
        not_categories = [c for c in corpus.get_categories() if c != category]
        tdf = corpus.apply_ranker(st.termranking.AbsoluteFrequencyRanker, False)

        cat_freqs = tdf[category + ' freq']
        if not_categories:
            not_cat_freqs = tdf[[c + ' freq' for c in not_categories]].sum(axis=1)
        else:
            not_cat_freqs = tdf.sum(axis=1) - tdf[category + ' freq']
        term_scorer = st.LogOddsRatioUninformativeDirichletPrior()

        scores = term_scorer.get_scores(cat_freqs, not_cat_freqs)
        sorted_series = scores.sort_values(ascending=False)
        count = 0
        output = category + ": "
        for index, value in sorted_series.items():
            output = output + index + ', '
            count +=1
            if count == 20:
                break
        print(output)
