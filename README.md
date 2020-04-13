# HAN

Replication of some of the experiments reported in [1].

The main goal of these experiments is to compare the performances of a Hierarchical Attention Network with a classic BoW approach.

## Datasets

I wasn't able to find the exactly datasets of the paper, so I have created three datasets from three cited sources: Yelp, Yahoo and Amazon. The datasets are perfectly balanced in all the classes.

|  **Dataset**  | **Classes** | **Documents** |
|---------------|-------------|---------------|
|     Yelp      |      5      |   1,569,000   |
| Yahoo Answer  |      10     |   1,450,000   |
| Amazon review |      5      |   3,650,000   |

As reported in the paper, 80% of the data is used for training, 10% for validation and the remaining 10% for test.

Here are more information about the datasets. #s denotes the number of sentences for each document and #w denotes the number of words for each sentence. This data are useful with the padding of the features.

|  **Dataset**  | **Average #s** | **80% #s** | **90% #s** | **Max #s** | **Average #w** | **80% #w** | **90% #w** | **Max #w** |
|---------------|----------------|------------------------|------------------------|------------|----------------|------------------------|------------------------|------------|
|     Yelp      |      8.6       |           12           |           17           |    151     |      16.5      |          23            |        30              |     846    |
| Yahoo Answer  |      5.5       |            8           |           11           |    154     |      19.7      |          28            |        37              |    3977    |
| Amazon review |      7.0       |           10           |           16           |    660     |      19.3      |          28            |        35              |    1981    |

## Results

Document classification, in percentage

|               | **Yelp** | **Yahoo** | **Amazon** |
|---------------|----------|-----------|------------|
| **BoW**       |   61.3   |   66.9    |    52.2    |
| **BoW-TFIDF** |   61.3   |   66.9    |    52.2    |
| **HAN**       |   ....   |   ....    |    ....    |

## References

[1] *Yang, Zichao, et al. "Hierarchical attention networks for document classification." Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies. 2016.*
