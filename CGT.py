import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Add, Dense, Concatenate, Dropout, ActivityRegularization
from tensorflow.keras.models import Model


def get_baseline(df, train_index, test_index, user_col, item_col):
    
    '''
    Calculate baseline features from an explicit ratings dataset. Receives a dataframe
    and returns train and test splits with added bias column and mean rating value.
    User and item biases are calculated as average difference from global mean rating.
    Baseline factors are only calculated from training observations, with users or
    items that do not appear in train receiving the global average as default.
    
    Args:
        df          : explicit ratings dataframe with columns userId, movieId and rating
        train_index : train index splits taken from KFold.splits()
        test_index  : test index splits taken from KFold.splits()
        
    Returns:
        train, test : train/test splits of df, with added bias column
        global_mean : average rating of all training observations
    '''
    
    train = df.iloc[train_index]
    test = df.iloc[test_index]
    
    # compute global mean
    global_mean = train.rating.mean()

    # compute average item ratings
    item_averages = train.groupby(
        item_col
    ).agg(
        {'rating':'mean'}
    ).rename(
        {'rating': 'item_avg'}, axis=1
    ).reset_index()
    
    # add as column to train and test
    train = pd.merge(train, item_averages, how='left', on=item_col)
    test = pd.merge(test, item_averages, how='left', on=item_col).fillna(global_mean)
    
    # compute average user bias
    train['user_bias'] = train['rating'] - train['item_avg']
    
    user_biases = train.groupby(
        user_col
    ).agg(
        {'user_bias':'mean'}
    ).rename(
        {'user_bias': 'user_avg'}, axis=1
    ).reset_index()
    
    # add as column to train and test
    train = pd.merge(train, user_biases, how='left', on=user_col)
    test = pd.merge(test, user_biases, how='left', on=user_col).fillna(0.0)
    
    # interaction bias
    train['bias'] = (train['user_avg'] + train['item_avg'] - global_mean)/2
    test['bias'] = (test['user_avg'] + test['item_avg'] - global_mean)/2
    
    return train, test, global_mean


def compile_genre_model(n_items, n_users, min_rating, max_rating, mean_rating, 
                        n_latent=1000, n_hidden_2=[200], activation='relu', l1_reg=0, l2_reg=0, dropout_2=.2, random_seed=42):
    
    # for reproducibility
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    # item latent factors
    item_in = Input(shape=[1])  # name='item'
    item_em = Embedding(n_items, n_latent)(item_in)
    item_vec = Flatten()(item_em)
    # user latent factors
    user_in = Input(shape=[1])
    user_em = Embedding(n_users, n_latent)(user_in)
    user_vec = Flatten()(user_em)
    # user x item bias
    bias = Input(shape=[1])
    # dot product
    x1 = Dot(axes=1)([item_vec, user_vec])
    x1 = ActivityRegularization(l1=l1_reg, l2=l2_reg)(x1)
        
    # add interaction bias to get adjusted rating
    x1 = tf.math.add(Add()([x1, bias]), mean_rating)
    # clip rating to be between min and max
    rating = tf.clip_by_value(x1, min_rating, max_rating)
    # create model and compile it
    model = Model([user_in, item_in, bias], rating)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # model 2
    # hidden layer with ReLU and dropout
    x2 = Dense(n_hidden_2[0], activation=activation)(item_vec)
    x2 = Dropout(dropout_2)(x2)
    # add subsequent hidden layers
    for i in range(len(n_hidden_2)-1):
        x2 = Dense(n_hidden_2[i+1], activation=activation)(x2)
        x2 = Dropout(dropout_2)(x2)
    # add sigmoid activation function
    genre = Dense(1, activation='sigmoid')(x2)
    # create model and compile it
    model2 = Model(item_in, genre)
    # freeze the embedding layer
    model2.layers[1].trainable = False
    model2.compile(optimizer='adam', loss='binary_crossentropy' , metrics=['accuracy', 'AUC'])
    
    return model, model2


def compile_multigenre_model(n_items, n_users, min_rating, max_rating, mean_rating, n_genres,
                        n_latent=1000, n_hidden_2=[200], activation='relu', l1_reg=0, l2_reg=0, dropout_2=.2, random_seed=42):
    
    # for reproducibility
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    # item latent factors
    item_in = Input(shape=[1])  # name='item'
    item_em = Embedding(n_items, n_latent)(item_in)
    item_vec = Flatten()(item_em)
    # user latent factors
    user_in = Input(shape=[1])
    user_em = Embedding(n_users, n_latent)(user_in)
    user_vec = Flatten()(user_em)
    # user x item bias
    bias = Input(shape=[1])
    # user x item bias
    bias = Input(shape=[1])
    # dot product
    x1 = Dot(axes=1)([item_vec, user_vec])
    x1 = ActivityRegularization(l1=l1_reg, l2=l2_reg)(x1)
        
    # add interaction bias to get adjusted rating
    x1 = tf.math.add(Add()([x1, bias]), mean_rating)
    # clip rating to be between min and max
    rating = tf.clip_by_value(x1, min_rating, max_rating)
    # create model and compile it
    model = Model([user_in, item_in, bias], rating)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # model 2
    x2 = Dense(n_hidden_2[0], activation=activation)(item_vec)
    x2 = Dropout(dropout_2)(x2)
    # add subsequent hidden layers
    for i in range(len(n_hidden_2)-1):
        x2 = Dense(n_hidden_2[i+1], activation=activation)(x2)
        x2 = Dropout(dropout_2)(x2)
    # add sigmoid activation function
    genre = Dense(n_genres, activation='sigmoid')(x2)
    # create model and compile it
    model2 = Model(item_in, genre)
    # freeze the embedding layer
    model2.layers[1].trainable = False
    model2.compile(optimizer='adam', loss='binary_crossentropy' , metrics=['accuracy', 'AUC'])
    
    return model, model2