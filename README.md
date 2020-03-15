# HAN

Replication of some of the experiments reported in [1].

The main goal of these experiments is to compare the performances of a Hierarchical Attention Network with a classic BoW approach.

## Datasets

### Yelp

The Yelp dataset was downloaded from [this page](https://www.yelp.com/dataset/download). It is not the dataset version reported in the paper, this one is a newer version that is a lot bigger then the Yelp dataset 2015. From all the dataset, I've kept the file `reviews.json`, which contains the reviews. Since in the paper, the experiment is done with a Yelp dataset with 1,569,264 reviews, I have resized my dataset to that size with the command:

    $ sort -R reviews.json | tail -1569264 > yelp_full.json

which extracts randomly 1,569,264 reviews from the entire dataset.

To create the dataset used in the experiments, run:

    $ python yelp_setup.py

This script will create a file called `yelp.csv`, which is a simplified version of `yelp_full.json`, where the columns with the reviews texts and the reviews stars are kept.

#### BoW / BoW-TFIDF

The experiment with a BoW approach is described in [2].

The classifier used is a multinomial logistic regression. The train set has size 130,000 and the test set has 10,000 elements for each star (there are in total five stars). 50,000 most frequent words are selected from the training sample. The features are normalized by dividing the largest feature value.

**why not to use the entire dataset?**

**It's not specified how the error rate is obtained. Is it a simple train and test? Or is it used a cross-validation approach?**

**How about the hyper paramenters of the multinomial Logistic Regression?**


## Results

Document classification, in percentage

|               | **Yelp** | **Amazon** | **??** |
|---------------|----------|------------|--------|
| **BoW**       |   54.7   |            |        |
| **BoW-TFIDF** |   54.7   |            |        |


## References

[1] *Yang, Zichao, et al. "Hierarchical attention networks for document classification." Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies. 2016.*

[2] *Zhang, Xiang, Junbo Zhao, and Yann LeCun. "Character-level convolutional networks for text classification." Advances in neural information processing systems. 2015.*
