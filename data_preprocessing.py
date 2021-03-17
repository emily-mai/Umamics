import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub


# load the data
def load_data(start=0, end=10000, path="data/yelp_academic_dataset_review.json"):
    datafile = open(path, encoding='utf-8')
    data = []
    for i, line in tqdm(enumerate(datafile)):
        if i in list(range(start, end)):
            data.append(json.loads(line))
        if i > end:
            break
    datafile.close()
    df = pd.DataFrame(data)
    return df


# embed reviews for clustering
def load_embeddings(dataframe, column):
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    messages = np.asarray(dataframe[column])
    tensor_list = tf.convert_to_tensor(messages)
    embeddings = np.array(embed(tensor_list)).tolist()
    return messages, embeddings


# show embedded reviews
def show_embeddings(n, reviews, embeddings):
    # print first n review embeddings
    for i, message_embedding in enumerate(embeddings[:n]):
        print("Message: {}".format(reviews[i]))
        print("Embedding size: {}".format(len(message_embedding)))
        message_embedding_snippet = ", ".join(
            (str(x) for x in message_embedding[:3]))
        print("Embedding: [{}, ...]\n".format(message_embedding_snippet))


# process batches of data
def batch_process(step=10000, stop=50000):
    review_df = pd.DataFrame()
    # generate ranges (i.e. [(0,1000), (1000,2000), ...]
    batches = [(n, min(n + step, stop)) for n in range(0, stop, step)]
    for s, e in batches:
        print("Processing {}:{}...".format(s, e))
        # load data and embeddings
        df = load_data(s, e)
        text, embeddings = load_embeddings(df, "text")
        # store embeddings in dataframe
        df["embedding"] = embeddings
        # add new embeddings to main dataframe
        review_df = pd.concat([review_df, df], ignore_index=True, sort=False)

    # save to pickle file to avoid reprocessing of embeddings
    review_df.to_pickle("data/data.pkl")
    print(review_df)


if __name__ == "__main__":
    # run this is you have dataset
    # batch_process()

    # run this if you have pkl file
    data = pd.read_pickle("data/data.pkl")
    print(data.head())
