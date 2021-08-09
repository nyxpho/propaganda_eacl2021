# EACL 2021: From the Stage to the Audience: Propaganda on Reddit

The data required for running this code can be found here:
https://propaganda.qcri.org/

This repository contains the following files:
1. bert-propaganda.py will train a model for classifying sentences in propaganda or non propaganda
2. bert-technique.py will train a model classifying a propaganda sentences according to the type of propaganda
3. mediafact.txt contains the labels of newspapers crawled from the site https://mediabiasfactcheck.com/ with their corresponding label
4. stattertext_topics.py will give the confounding tokens 

In order to obtain the Roberta and XLNet models from the paper, you need just to modify the parameter 'version_model' in the python files above. In order to get the MGN ReLU model, please use the same link as for the data.
