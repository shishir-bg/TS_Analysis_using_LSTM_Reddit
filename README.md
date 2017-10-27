# TS_Analysis_using_LSTM_Reddit
This is an Implmentation of a Recurrent Neural Network using LSTM's. Here I have used the model to train it on Stock Price History data as well as World News data from Reddit. We use daily world news headlines from Reddit to predict the opening value of the Dow Jones Industrial Average. The data for this project comes from a dataset on Kaggle, and covers nearly eight years (2008–08–08 to 2016–07–01).
I have used GloVe’s larger common crawl vectors to create our word embeddings and Keras to build our model. This model was inspired by the work described in the mentioned paper. Similar to the paper, we will use CNNs followed by RNN with LSTM's ( not GRU's), and also various articles and blogs mentioned below.
To help construct a better model, we will use a grid search to alter our hyperparameters’ values and the architecture of our model.

Requirements:
1. Python 3 or above
2. TF 1.3 or Above
3. Keras
4. Pandas
5. Numpy
6. NLTK
7. SKLearn
8. MatPlotLib
9. Glove v1.2 Word Embeddings

Data Sources: 
1. Glove : https://nlp.stanford.edu/projects/glove/
2. Stock and News Data : https://www.kaggle.com/aaron7sun/stocknews

Steps:
1. Import Dependencies
2. Read and Preprocess data
3. Load contractions, clean and aggregate news text
4. Create word count library
5. Load GloVe's Embeddings
6. Match our vocab to gloVes vectors
7. Split Testing and Training data
8. Define placeholders and build Sequential RNN model with conditionals for grid search extension
9. Train the model using model.fit and and use grid search to find the best model and save it.
10. Make Predictions with the best weights

All steps have been described and implemented in the Jupyter Notebook

Reference Links:
1. https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
2. https://medium.com/@TalPerry/deep-learning-the-stock-market-df853d139e02
3. https://medium.com/machine-learning-world/neural-networks-for-algorithmic-trading-2-1-multivariate-time-series-ab016ce70f57
4. https://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/
5. https://www.youtube.com/watch?v=ftMq5ps503w
6. https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
7. https://medium.com/@Currie32/predicting-the-stock-market-with-the-news-and-deep-learning-7fc8f5f639bc
8. https://www.kaggle.com/marklam/a-neutral-network-to-read-btc-price-action

Research Papers:
0. https://www.aclweb.org/anthology/C/C16/C16-1229.pdf - Combination of Convolutional and Recurrent Neural Network for
Sentiment Analysis of Short Texts
1. http://www.aclweb.org/anthology/D14-1162 - GloVe: Global Vectors for Word Representation
2. https://people.cs.pitt.edu/~hashemi/papers/CISIM2010_HBHashemi.pdf - Stock Market Value Prediction Using Neural Networks 
3. https://www.hindawi.com/journals/cin/2016/4742515/ - Financial Time Series Prediction Using Elman Recurrent Random Neural Networks
4. ftp://ftp.idsia.ch/pub/juergen/icann2001predict.pdf - Applyling LSTM to TS Predictable through Time Window Approachs
5. ftp://ftp.idsia.ch/pub/juergen/L-IEEE.pdf - LSTM and RNN Simple Context Free and Sensitive Language

