# HAN

Replication of some of the experiments reported in:

[Yang, Zichao, et al. "Hierarchical attention networks for document classification." Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies. 2016.](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)

The main goal of these experiments is to compare the performances of a Hierarchical Attention Network with a Flat Attention and a classic BoW approach.

## Datasets

I wasn't able to find the exactly datasets of the paper, so I have created three datasets from three cited sources: Yelp, Yahoo and Amazon. The datasets are perfectly balanced in all the classes.

|  **Dataset**  | **Classes** | **Documents** |
|---------------|-------------|---------------|
|     Yelp      |      5      |   700,000     |
| Yahoo Answer  |      10     |   1,450,000   |
| Amazon review |      5      |   3,650,000   |

As reported in the paper, 80% of the data is used for training, 10% for validation and the remaining 10% for test. All the dataset used in this experiments, together with the embeddings, are available [here](https://drive.google.com/open?id=1U2f7PfDYUrvfkIkNQPYR2OufAkuldNu6).

### Dataset statistics

In this section some statistics about the number of sentences and words in each document are reported. This kind of information are useful with the padding of the features. These data are obtained on the training and the validation set.

## Bag-Of-Words

To train and test a BoW model (both with and without TFIDF), run

    $ python bow.py {yelp,yelp-sample,yahoo,amazon}

A Stochastic Gradient Descent classifier is used together with a logistic regression loss. The 50,000 most frequent words from the training set are selected and the count of each word is used as features. A grid search cross-validation is used to find the best `alpha` parameter, that is the constant that multiplies the regularization term.

## Word embedding

The first layer of the attention networks is an embedding layer, which transform a word into a fix sized vector that is a numerical representation of the meaning of that word.

As mentioned in the paper, this layer is pretrained on the training and validation sets. To train a word embedding matrix run

    $ python word2vec.py {yelp,yahoo,amazon}

This script will produce a file in the directory `embedding` containing the matrix of the learned word embedding.

The words that appear less then five times in the whole dataset are substituted with a special `UNK` token.

## Attention models

In addition to the model proposed in the paper (Hierarchical Attention Network or HAN), I have implemented a Flat Attention Network (FAN), that is the same model but with only one attention layer, without taking into account the different sentences of the document.

With the script `train.py` is possible to train a WAN or HAN model:

    $ python train.py {yelp,yelp-sample,yahoo,amazon} {fan,han}

The trained model will be saved into a directory called `models` and the logging of the training process will be stored into a directory called `runs`. The plots of the training phase can be viewed with Tensorboard:

    $ tensorboard --logdir=runs

To test a model, run

    $ python test.py {yelp,yelp-sample,yahoo,amazon} {fan,han} <model_file>

## Results

Document classification, in percentage

|                           | **Yelp** | **Yahoo** | **Amazon** |
|---------------------------|----------|-----------|------------|
| **BoW**                   |   61.3   |   66.9    |    52.2    |
| **BoW-TFIDF**             |   61.3   |   66.9    |    52.2    |
| **Flat Attention**        |   67.0   |   73.8    |    60.4    |
| **Hierachical Attention** |   67.3   |   73.9    |    60.9    |

## Notes

The Yahoo Answer categories are mapped as the following:

|        **Domain**      | **Label** |
|------------------------|-----------|
| Business & Finance     | 0         |
| Computers & Internet   | 1         |
| Education & Reference  | 2         |
| Entertainment & Music  | 3         |
| Family & Relationships | 4         |
| Health                 | 5         |
| Politics & Government  | 6         |
| Science & Mathematics  | 7         |
| Society & Culture      | 8         |
| Sports                 | 9         |
