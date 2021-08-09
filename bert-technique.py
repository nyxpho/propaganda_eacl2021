"""
This file for used for the task
Propaganda Technique Identification
"""

from flair.datasets import ClassificationCorpus
from flair.data import Sentence, Token
from flair.trainers import ModelTrainer
from flair.embeddings import DocumentRNNEmbeddings, TransformerDocumentEmbeddings
from flair.models import TextClassifier
import os
import glob
import os.path
import codecs
from nltk.tokenize import WordPunctTokenizer
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


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


def return_annotated_articles(spansfile, articles,train=True):
    token_labels = dict()
    # we first read all the propaganda spans from spans file and we label the corresponding tokens
    if train:
        with open(spansfile, "r") as f:
            for row in f.readlines():
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
                    token_labels[article_id] =  [['O']] * len(tokenized_article)
                #token_labels[article_id][token_strt]='B-prop'
                for i in range(token_start, token_end):
                    token_labels[article_id][i] = token_labels[article_id][i] + [methods]
                

    tokens = []
    labels = []
    sentences = []
    spans = []
    multilabel = 0
    # we now create the output matrices
    to_skip = set()
    to_skip = set(["Bandwagon", "Obfuscation,Intentional_Vagueness,Confusion","Red_Herring", "Straw_Men", "Thought-terminating_Cliches"])
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
                    if u !='O' and u not in to_skip:
                            list_u.append(u)
                if len(list_u):
                    if len(list_u)>1:
                        multilabel +=1
                    sentences.append(sentence)
                    labels.append(list_u)
    print("multi label: " + str(multilabel))
    return sentences, labels


def print_predictions(model, sentences, outfile):
    wb = open(outfile, 'w')
    for sentence in sentences:
        sen = Sentence(sentence)
        model.predict(sen)
        print(sentence)
        print(sentence.labels)
    wb.close()


def write_to_file_flair_corpus(fileout, sentences, tags):
    wb = open(fileout, 'w', encoding="utf-8")
    for sentence, labels in zip(sentences, tags):
        for l in labels:
            wb.write('__label__' + l + '__label__ ' + sentence + '\n')
    wb.close()


if __name__ == '__main__':
    params = {'train_folder': "../data/train",
              'dev_folder': "../data/dev",
              'test_folder': "../data/test",
              'train_labels_file': "../data/articles_train.labels",
              'test_labels_file': "../data/articles_test.labels",
              'dev_labels_file': "../data/articles_dev.labels",
              'results': "bert-technique-nofilter.txt",
              'max_seq_length': 210,
              'model': "bert",
              'version_model': "bert-base-cased",
              'model_dir': "technique-bert-model-filter-20epochs",
              'batch_size': 16,
              'epochs': 20,
              'train': True,
              'learning_rate': 0.01,
              'anneal_factor': 0.5,
              'patience': 2,
              'data_bert_format_dir': '../data/',
              'embeddings_storage_mode': 'gpu'
              }

    articles_train = read_articles_from_file_list(params["train_folder"])
    articles_valid = read_articles_from_file_list(params["dev_folder"])
    articles_test = read_articles_from_file_list(params["test_folder"])

    tokens, labels = return_annotated_articles(params["train_labels_file"], articles_train, train=True)
    tokens_valid, labels_valid = return_annotated_articles(params["dev_labels_file"], articles_valid,
                                                           train=True)
    tokens_test, labels_test = return_annotated_articles(params["test_labels_file"], articles_test,
                                                         train=True)

    write_to_file_flair_corpus(params['data_bert_format_dir']+'train.txt', tokens, labels)
    write_to_file_flair_corpus(params['data_bert_format_dir']+'dev.txt', tokens_valid, labels_valid)
    write_to_file_flair_corpus(params['data_bert_format_dir']+'test.txt', tokens_test, labels_test)
    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus = ClassificationCorpus(params['data_bert_format_dir'],
                                  train_file='train.txt',
                                  test_file='test.txt',
                                  dev_file='dev.txt')

    corpus.filter_empty_sentences()
    print(corpus)

    label_dictionary = corpus.make_label_dictionary()

    print(label_dictionary)

    flat_labels = [item for sublist in labels for item in sublist]
    class_weights = compute_class_weight('balanced', np.unique(flat_labels), flat_labels)
    unique_labels = np.unique(flat_labels)
    weights = {}
    for i in range(len(unique_labels)):
        weights[unique_labels[i]] = class_weights[i]

    document_embeddings = TransformerDocumentEmbeddings(params['version_model'], fine_tune=True)

    classifier = TextClassifier(document_embeddings, label_dictionary=label_dictionary, loss_weights=weights, multi_label=False)

    trainer = ModelTrainer(classifier, corpus)

    trainer.train(params['model_dir'],
                  learning_rate=params['learning_rate'],
                  mini_batch_size=params['batch_size'],
                  anneal_factor=params['anneal_factor'],
                  patience=params['patience'],
                  max_epochs = params['epochs'],
                  embeddings_storage_mode= params['embeddings_storage_mode'])

    # print_predictions(trainer, tokens_test, params['results']+'gloveSentence')
