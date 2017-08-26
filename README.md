# data_science_challenge
This repository consists of similar item clustering implemenation. A simple approach to cluster the listings data into similar items clusters such that if o user view one item from a cluster, it is reasonable to recommend the user another item from the same cluster.

<h3>Approach:</h3>

- The dataset consists of sample item listing from the classified ads and sample of user queries.
- Since the classified ads are already organized into few broad categories, generating similar item clusters within these categories is a reasonable start. This approach helps in getting relevant clusters which are computationally faster to generate and is also good for the use case where user lands to a page via search query.
- User queries are used to get the feature set for the items. 
- User queries are tokenized and ordered by thier frequency of usage. Only those tokens are used that meets a minimum threshold value(based on tailed frequency).
- These tokens are matched with the item title listing and their description.
- Matched tokens are mapped to the feature set which are further reduced in dimenstionality.
- The extracted feature set along with other features like sub-category details and item pricing are used for clustering.
- Minibatch kmeans clustering is applied where optimal k is determined using silhouette score.

<h3>Shortcomings:</h3>

- This approach is highly dependent on the broad category classification of the items. Items in different categories would always be in different clsuters. This would disable cross category recommendation≥
- Does not include much of the quality factors like seller details, closeness to the user location.
- Based highly on user queries. Any important text that was not the part of user queries is discarded.

<h3>Enhancements:</h3>

- Make it scalable and generic across all categories for cross category recommendation.
- Make clustering independent of search queries
- Dimensions tunning for reduction
