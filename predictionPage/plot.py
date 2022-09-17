import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import io
import base64


global file



def choose_file():
    file = "csv/ADBL_SCORE.csv"
    return file

def plotImage():
    images = []
    file = choose_file()
    df = pd.read_csv(file)
    df.Date = pd.to_datetime(df.Date, utc=True)
    mask = (df['Date'] > '2019-7-29') & (df['Date'] <= '2022-3-6')
    df1 = df.loc[mask]

    #Rolling means
    rolling_mean_10=df1.LTP.rolling(10).mean()
    rolling_mean_50=df1.LTP.rolling(50).mean()

    plt.figure()
    flike = io.BytesIO()
    df1['LTP'].plot(figsize=(10,5),title='Simple Moving Average')
    plt.plot(rolling_mean_10,label='10 Day SMA',color='red')
    plt.plot(rolling_mean_50,label='50 Day SMA',color='orange')
    plt.legend()
    plt.savefig(flike)
    first_image =  base64.b64encode(flike.getvalue()).decode()
    images.append(first_image)

    plt.figure()
    blike = io.BytesIO()
        #EMA
    EMA_10=df1.LTP.ewm(span=10, adjust=False).mean()
    EMA_50=df1.LTP.ewm(span=50, adjust=False).mean()

    df1['LTP'].plot(figsize=(15,8),title='Exponential Moving Average')
    plt.plot(EMA_10,label='10 Day EMA',color='red')
    plt.plot(EMA_50,label='50 Day EMA',color='orange')
    plt.legend()
    plt.savefig(blike)
    second_image =  base64.b64encode(blike.getvalue()).decode()
    images.append(second_image)

    plt.figure()
    clike = io.BytesIO()
    #Bollinger Bands
    rolling_mean_20=df1.LTP.rolling(20).mean()
    upper_band=rolling_mean_20+2*df1.LTP.rolling(20).std()
    lower_band=rolling_mean_20-2*df1.LTP.rolling(20).std()
    df1['LTP'].plot(figsize=(18,9),title='20 Day Rolling Bollinger Bands').fill_between(df1.index,lower_band,upper_band,alpha=0.1)
    plt.plot(upper_band,label='Upper Band',color='red')
    plt.plot(lower_band,label='Lower Band',color='orange')
    plt.legend()
    plt.savefig(clike)
    third_image =  base64.b64encode(clike.getvalue()).decode()
    images.append(third_image)

    plt.figure()
    dlike = io.BytesIO()
    #MACD
    EMA_12= df1.LTP.ewm(span=12, adjust=False).mean()
    EMA_26= df1.LTP.ewm(span=26, adjust=False).mean()
    MACD=EMA_12-EMA_26
    SignalLine=MACD.ewm(span=9, adjust=False).mean()


    df1['LTP'].plot(subplots=True,figsize=(18,9),title='Moving Average Convergence Divergence')
    plt.plot(EMA_12,label='12 Day EMA',color='red')
    plt.plot(EMA_26,label='26 Day EMA',color='orange')
    plt.legend()
    # plt.show()
    MACD.plot(figsize=(18,9),label='MACD',color='green',title='MACD vs Signal')
    plt.plot(SignalLine,label='Signal Line',color='Purple')
    plt.legend()
    plt.savefig(dlike)
    fourth_image =  base64.b64encode(dlike.getvalue()).decode()
    images.append(fourth_image)

    return images
