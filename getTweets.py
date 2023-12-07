import nltk
from nltk.corpus import twitter_samples
nltk.download('twitter_samples')
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
all_tweets = all_positive_tweets + all_negative_tweets
index = 0
for tweet in all_tweets:
  with open(f"data/sampleTweets/{index}.txt", "w") as f:
    f.write(tweet)
    index += 1
