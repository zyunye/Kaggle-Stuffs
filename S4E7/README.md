# S4E7 - Binary Classification of Insurance Cross Selling 

Prediction for whether a user responded positively or negatively to being cross sold insurance.

Because of the size of the dataset (around 11 mil rows), I used this opportunity to check out some of the libraries designed for this sort of scale of data, such as Polars for a more optimized Dataframe library.
Additionally, I spun up a VM to experiment with PySpark and Hadoop as I was having trouble with both RAM and compute trying to do everything in memory. 
This path was eventually shelved as I found a way to preprocess the data to make it more amenably fit into the memory I had available and, following the discussions in the competition, there seemed to be simpler paths I hadn't explored yet.