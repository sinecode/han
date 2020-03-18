# HAN

Replication of some of the experiments reported in [1].

The main goal of these experiments is to compare the performances of a Hierarchical Attention Network with a classic BoW approach.

## Datasets

To create the datasets used in the experiments, first download all the dataset accordingly to the instructions and then run

    $ python setup_data.py

This script creates `csv` files containing the same number of documents reported in the paper. **Each dataset is balanced**: there are the same number of elements for each class.

To have more info about where to find the datasets and how to name the files read the following sections.

### Yelp

I wasn't able to find the Yelp 2015 dataset used in the paper. I downloaded a newer version of that dataset [here](https://www.yelp.com/dataset/download). I've kept only the `json` with the reviews and I named it `yelp.json`.

### Yahoo

I downloaded the dataset [here](https://webscope.sandbox.yahoo.com/). I created the `.xml` dataset following the instruction then I renamed the final file `yahoo.xml`.

### Amazon

I downloaded the "small" subset of Amazon review data about books [here](https://nijianmo.github.io/amazon/index.html), then I renamed it `amazon_books.json.gz`.

## Experiments

### BoW / BoW-TFIDF

These experiments are described in [2].

## Results

Document classification, in percentage

|               | **Yelp** | **Amazon** | **Yahoo** |
|---------------|----------|------------|-----------|
| **BoW**       |   54.8   |            |           |
| **BoW-TFIDF** |   54.8   |            |           |


## References

[1] *Yang, Zichao, et al. "Hierarchical attention networks for document classification." Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies. 2016.*

[2] *Zhang, Xiang, Junbo Zhao, and Yann LeCun. "Character-level convolutional networks for text classification." Advances in neural information processing systems. 2015.*
