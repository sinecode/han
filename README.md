# HAN

Replication of some of the experiments reported in [1].

The main goal of these experiments is to compare the performances of a Hierarchical Attention Network with a classic BoW approach.

## Datasets

I wasn't able to find the exactly datasets of the paper, so I have created three datasets from three cited sources: Yelp, Yahoo and Amazon. The datasets are perfectly balanced with all the classes.

|  **Dataset**  | **Classes** | **Documents** |
|---------------|-------------|---------------|
|     Yelp      |      5      |   1,569,000   |
| Yahoo Answer  |      10     |   1,450,000   |
| Amazon review |      5      |   3,650,000   |

## Results

Document classification, in percentage

|               | **Yelp** | **Yahoo** | **Amazon** |
|---------------|----------|-----------|------------|
| **BoW**       |   61.3   |   66.9    |    52.2    |
| **BoW-TFIDF** |   61.3   |   66.9    |    52.2    |

## Notes

The Yahoo answer categories are mapped as the following:

* 0: *Business & Finance*
* 1: *Computers & Internet*
* 2: *Education & Reference*
* 3: *Entertainment & Music*
* 4: *Family & Relationships*
* 5: *Health*
* 6: *Politics & Government*
* 7: *Science & Mathematics*
* 8: *Society & Culture*
* 9: *Sports*

## References

[1] *Yang, Zichao, et al. "Hierarchical attention networks for document classification." Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies. 2016.*
