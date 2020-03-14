# han

Hierachical Attention Networks for Document Classification

## Notes

Yelp dataset was dowloaded from [this page](https://www.yelp.com/dataset/download). It is not the dataset version reported in the paper, this one is a newer version that is a lot bigger then the Yelp dataset 2015. From all the dataset, I've kept the file `reviews.json`, which contains the reviews. Since in the paper, the experiment is done with a Yelp dataset with 1,569,264 reviews, I have resized my dataset to that size with the command:

    $ sort -R reviews.json | tail -1569264 > yelp_reviews.json

which extracts randomly 1,569,264 reviews from the downloaded dataset.

