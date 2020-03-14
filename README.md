# han

Hierachical Attention Networks for Document Classification

## Notes

Yelp dataset was dowloaded from [this page](https://www.yelp.com/dataset/download). It is not the dataset version reported in the paper, this one is a newer version. From all the dataset, I've kept the file `reviews.json`, which contains only the reviews. That file was renamed `yelp_reviews.json` and it was moved in the directory `datasets`. Since the entire dataset is quite large (about 5 GB), I extracted a sample composed by the first 100000 elements with the shell command `$ head -100000 yelp_reviews.json > yelp_reviews_sample.json`
