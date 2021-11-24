# Umamics

## Purpose
Yelp is a commonly used platform for both customers and businesses. Customers
are able to leave ratings and reviews for different restaurants, which can be viewed by
other users as well as restaurant owners. Our goal is to provide restaurants with useful
information to manage their customer relationships.

## Overview
We will first preprocess and if necessary, clean, the dataset. After the
data preprocessing process, we will perform exploratory data analysis by analyzing the
dataset and creating different data visualizations so that we can gain a better
understanding of our data - perhaps even spot patterns. From here, we will apply different
unsupervised learning techniques to cluster the reviews for each restaurant and generate a
text summary for each cluster. From performing all the steps we mentioned, we hope that
businesses can leverage our findings to improve their services and gain a better
understanding of their customers.

## Data Collection and Preprocessing
*Data Collection:* Our data was obtained from Yelp Inc. for academic use. The dataset is a
subset of Yelp’s business, review, and user data. We will focus specifically on the dataset of
the business review. The original dataset gathers the data of eight million restaurants
located in the United States and Canada and contains nine different attributes. We mainly
focused on attributes we believe are the most useful: business_id, stars, and reviews.
Business_id is the unique identification tag for the restaurant. Stars is the rating system
provided to Yelp users ranging from zero to five stars where the highest/best possible
rating that could be given is five stars. Lastly, review is the review given by a user to each
restaurant.

*Data Preprocessing:* Fortunately, our data came in the form of a json file and contained only
a few attributes with no empty values. Therefore, we did not have to do much cleaning of
our data. In order to prepare our data for analysis, we had to first embed the reviews since
it was given in the form of text. To perform the embedding, we used the Universal Sentence
Encoder from TensorFlow and encoded the text into a high-dimensional vector that can be
used for text classification, semantic similarity, and clustering. The embedding process
tended to take a long time and given that we had such a large amount of data (~ 8 million
reviews), we decided to process the embeddings in batches. Therefore, we processed 5
batches of reviews each containing 10,000 reviews giving us a total of 50,000 data points.
Once we obtained the review embeddings, we created a new column in our dataframe to
store it for ease of accessing. Lastly, we saved the dataframe containing the embedded
reviews into a pickle file to avoid reprocessing of the embeddings. Below is an example of
our data with the embedded reviews using the “text” column.

## Exploratory Data Analysis
Initial analysis shows the distribution of ratings, total number of reviews, and the past trend of a particular
restaurant's ratings. These graphs can be found in the Jupyter notebook.

## Methods/Algorithms
In designing our project, we decided to go with an unsupervised
learning approach. For that reason, we used methods for clustering such as DBSCAN and
Gaussian Mixture Model. For the generating the summaries for each cluster, we use text
summarization, specifically extractive summarization, along with cosine similarity. Finally,
we tried to use DBSCAN, Local Outlier Factor (LOF), and Gaussian Mixture Models for the
Anomaly Detection. However, due to the time and a problem we were faced with in which
we could not solve, we were not able to perform anomaly detection in the end.

*Reviews Clustering:* To cluster the reviews, we experimented with different methods such as
DBSCAN and Gaussian Mixture Model to find what worked the best in generating the
clusters.
- **DBSCAN:** Density Based Spatial Clustering of Applications with Noise or
DBSCAN is a density-based clustering method. The idea behind it is that
instead of clustering based on some notion of similarity, we try to define
some density notion to identify the clusters. DBSCAN groups “densely
grouped” data points together into a single cluster while also identifying
outliers (usually points in low-density regions). Unlike k-means clustering, it
is robust to outliers, it does not require that we specify k, it gives us outliers
for free and it can create clusters of arbitrary shapes which would be an
advantage in the case that we have some outliers and the shapes of our
cluster may be in some arbitrary form.
- **Gaussian Mixture Models (GMM):** GMM is a probabilistic model that assumes
all points are generated from a mixture of a finite number of Gaussian
distributions and can be thought of as a generalization to k-means clustering.
GMM essentially performs “soft classification” as opposed to k-mean which
performs “hard classification” meaning that GMM provides the probability of
a given data point belonging to each of the clusters whereas k-means tells
whether a point belongs or doesn’t belong to a cluster. GMM can give us a
better idea of the clustering as it does not assume them to be circular which
is an advantage to us in the case that our clusters are in some arbitrary shape.

From examining and comparing the two model selections for the number of clusters, we can see
that DBSCAN, in this case, is pretty much useless to us as it gave us 1 cluster
with 149 outliers. The GMM, however, chose 4 as the number of clusters
based on a BIC (Bayesian information criterion) score being the lowest as it shifted from 3 to 4.
After comparing the 2 methods, we went with GMM. When using GMM to
cluster the business with the most number of reviews. From doing so, we
found that GMM chose a diagonal model and 15 clusters. The choice of the
cluster can be reinforced by the graph below where we are graphing the
number of clusters with respect to the BIC score which measures how well
our model will perform at predicting our data. Therefore, in general, we
would like the BIC score to be as low as possible given a certain value of
n_components.

*Text Summarization:* In order to generate summaries for each cluster, we used a type of text
summarization called extractive method. Extractive methods try to summarize articles of
text by selecting a subset of words part of sentences that retain the most important points
and that is how the summary is formed. This is useful for us because Text Summarization
cna produce concise and fluent summaries while preserving key information and overall
meaning which is what we would like to do for the clusters that we generated so that we
can have a good understanding of what each cluster of review is about.

Note: in order to generate the summary, we need to clean the sentences and remove stop
words. Stopwords are the words in any language which does not add much meaning to a
sentence. They can safely be ignored without sacrificing the meaning of the sentence. We
find similarity between sentences and the stopword using cosine similarity and to perform
ranking, we would need to use graph-based ranking algorithm called TextRank.

## Conclusion
Through our various methods of data analysis, we have been able to identify
many attributes about the most reviewed restaurant. Our first demonstration of our
findings is through EDA where we analyzed 366 ratings for a particular restaurant and
showed the distribution of its star ratings as a whole, and as a trend over time. By doing so,
we were able to showcase the overwhelming distribution of positive ratings both as a
whole in a bar graph and over time in a scatter plot. Next we decided to apply
unsupervised learning using DBSCAN and Gaussian Mixture Model to generate the clusters.
After creating the clusters, we used extractive summarization and cosine similarity to
summarize each cluster. By doing so, we were able to showcase our findings of each cluster
that was associated with a set of ratings. From the cluster with the highest overall average
rating, we found that customers were very satisfied with their time at the restaurant and
left long reviews highlighting their experience. However, from the cluster with the lowest
overall average rating, it is apparent that customers were very unpleasant about their time
spent at the restaurant, shown in both the reviews and the ratings.



