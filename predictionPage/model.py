from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import concat
import pandas as pd
from matplotlib import pyplot

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from pandas import concat
from numpy import concatenate

import io
import base64

global prediction_model
global file

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def choose_model(modelname):
    if modelname == 'model1':

        prediction_model = "model/" + "my_model.h5"
        return prediction_model

def choose_file(csvfile):
    if csvfile == 'csv1':
        file = "csv/" + "ADBL_SCORE" + ".csv"
    elif csvfile == 'csv2':
        file = "csv/" + "cit_SCORE" + ".csv"
    else:
        file = "csv/" + "mnbbl_SCORE.csv"

    return file

def predict(csvfile, modelname):
    prediction_model = choose_model(modelname)
    file = choose_file(csvfile)
    dataset, df, n_features = loadDataset(file)  
    n_lag = 12
    model = keras.models.load_model(prediction_model)
    df_price = dataset[["LTP","Open", "High", "Low"]]
    df_qty = dataset[["Quantity"]]
    df_sentiment = dataset[["positive", "negative", "neutral"]]
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    qty_scaler = MinMaxScaler(feature_range=(0, 1))
    sentiment_scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_price_ltp = price_scaler.fit_transform(df_price[["LTP"]].values)
    scaled_price_high = price_scaler.transform(df_price[["High"]].values)
    scaled_price_low = price_scaler.transform(df_price[["Low"]].values)
    scaled_price_open = price_scaler.transform(df_price[["Open"]].values)

    scaled_qty = qty_scaler.fit_transform(df_qty)
    scaled_sentiment = sentiment_scaler.fit_transform(df_sentiment)
    scaled = np.concatenate((scaled_price_ltp, scaled_price_open), axis=1)
    scaled = np.concatenate((scaled, scaled_price_high), axis=1)
    scaled = np.concatenate((scaled, scaled_price_low), axis=1)
    scaled = np.concatenate((scaled, scaled_qty), axis=1)
    scaled = np.concatenate((scaled, scaled_sentiment), axis=1)
    print(scaled.shape)

    # # frame as supervised learning
    # reframed = series_to_supervised(values, n_lag, 1)
    reframed = series_to_supervised(scaled, n_lag, 1)

    # split into train, validation and test sets
    values = reframed.values
    n_train = int(365 * 4)
    n_val=int(n_train*0.8)
    train = values[:n_val, :]
    val=values[n_val:n_train,:]
    test = values[n_train:, :]
    x_test_panda = dataset.iloc[n_train:,:]

    print(train)
    print(train.shape)
    print(test)
    print(test.shape)

    #store the test data set index for comparing it with result
    df_x_test= df.iloc[n_train:,:]

    # split into input and outputs
    n_obs = n_lag * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    val_X, val_y=val[:,:n_obs],val[:,-n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    print(train_X.shape, len(train_X), train_y.shape)

    # reshape input to be 3D [samples, timesteps, features]
    train_X = np.reshape(train_X, (train_X.shape[0], n_lag, n_features))
    val_X = np.reshape(val_X, (val_X.shape[0], n_lag, n_features))
    test_X = np.reshape(test_X, (test_X.shape[0], n_lag, n_features))

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    yhat = model.predict(test_X)
    test_X_history = test_X.reshape((test_X.shape[0], n_lag*n_features))

    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X_history[:, -(n_features-1):]), axis=1)
    inv_yhat = price_scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X_history[:, -(n_features-1):]), axis=1)
    inv_y = price_scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    a = model.predict(np.expand_dims(test_X[0], axis=0))
    print(
    price_scaler.inverse_transform(a)
    )

    print(df.shape)
    print(x_test_panda.shape)

    y=x_test_panda.iloc[12:,]
    true=y['LTP']

    # true_x=np.array(true)
    # pred_x=np.array(df['Actual'])

    # count=0
    # for i in range(true_x.size):
    #     if(true_x[i]==int(pred_x[i])):
    #         count=count+1
    # print(count)

    X, y = reframe_data(dataset.iloc[-14:-1], price_scaler, qty_scaler, sentiment_scaler, n_lag, n_features)
    # X, y = reframe_data(dataset.iloc[:13])

    price_scaler.inverse_transform(X[0][0][0].reshape(-1,1))


    true_y = np.expand_dims(y, axis=0)
    pred_y = model.predict(X)
    
    return show_plot(
    [
    price_scaler.inverse_transform(X[0,:,-8].reshape(-1,1)),

    price_scaler.inverse_transform(true_y.reshape(-1,1)),
    price_scaler.inverse_transform(pred_y.reshape(-1,1))
    ],
    1,
    "Single Step Prediction",
    ), price_scaler.inverse_transform(pred_y)


def loadDataset(file):
    df = pd.read_csv(file)
    dataset = df[['LTP','Open','High','Low','Quantity','positive','negative','neutral']]
    n_features = 8
    return dataset, df, n_features
       

def da(y_true,y_pred):
    sum=0
    for i in range(y_true.size-1):
        dt=(y_true[i+1]-y_true[i])*(y_pred[i+1]-y_true[i])
        if(dt>=0):
            sum=sum+1
    return sum/(y_true.size-1)



def reframe_data(dataset, price_scaler, qty_scaler, sentiment_scaler, n_lag, n_features):
    df_price = dataset[["LTP","Open", "High", "Low"]]
    df_qty = dataset[["Quantity"]]
    df_sentiment = dataset[["positive", "negative", "neutral"]]

    # price_scaler = MinMaxScaler(feature_range=(0, 1))
    # qty_scaler = MinMaxScaler(feature_range=(0, 1))
    # sentiment_scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_price_ltp = price_scaler.transform(df_price[["LTP"]].values)
    scaled_price_high = price_scaler.transform(df_price[["High"]].values)
    scaled_price_low = price_scaler.transform(df_price[["Low"]].values)
    scaled_price_open = price_scaler.transform(df_price[["Open"]].values)

    scaled_qty = qty_scaler.transform(df_qty)
    scaled_sentiment = sentiment_scaler.transform(df_sentiment)

    scaled = np.concatenate((scaled_price_ltp, scaled_price_open), axis=1)
    scaled = np.concatenate((scaled, scaled_price_high), axis=1)
    scaled = np.concatenate((scaled, scaled_price_low), axis=1)
    scaled = np.concatenate((scaled, scaled_qty), axis=1)
    scaled = np.concatenate((scaled, scaled_sentiment), axis=1)
    # print(scaled.shape)

    
    
    # #prepare for the time-series generation
    # data=series_to_supervised(dataset, n_lag,1)
    # values = dataset.values
    # # print(values)
    # # print(values.shape)
    # values = values.astype('float32')

    # #normalize the data using min-max normalizations
    # scaled = price_scaler.transform(values)
    # # print(scaled)
    # # print(scaled.shape)
    # # # frame as supervised learning
    reframed = series_to_supervised(scaled, n_lag, 1)
    # print(reframed.shape)
    # split into train, validation and test sets
    values = reframed.values
    n_obs = n_lag * n_features
    # print(values.shape)
    # print(values)
    X, y = values[:, :n_obs], values[:, -n_features]
    # print(values)
    # print(y)
    # print(values.shape)
    # print( values[:, -n_features].shape)
    # print( values[:, -n_features])
    
    X = np.reshape(X, (X.shape[0], n_lag, n_features))
    return X, y




# The trained model above is now able to make predictions for 5 sets of values from validation set.

def show_plot(plot_data, delta, title):
    labels = ["History", "True Future","Model Prediction"]
    marker = [".-","rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    flike = io.BytesIO()
    # print(time_steps)
    if delta:
        future = delta
    else:
        future = 0

    plt.figure()
    plt.title(title)
    for i, val in enumerate(plot_data):
      # print(val)
      if i:
        plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
      else:
        # plt.plot(i, val, marker[0], markersize=10, label=labels[0])
        plt.plot(time_steps, plot_data[i], marker[i], label=labels[i])
        # plt.plot(time_steps, plot_data[i], marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    # plt.show()
    plt.savefig(flike)
    b64 =  base64.b64encode(flike.getvalue()).decode()
    return b64


# show_plot(
#     [fut_inp,  inv_yhat],
#     1,
#     "Single Step Prediction",
# )

