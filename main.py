#import required libraries
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.offline as py

def load_file(fileName):
    excelFile = pd.read_excel(fileName)
    #Rename columns to fit into prophet instance
    excelFile.rename(columns={'Date':'ds', 'Close':'y'},inplace = True)
    return excelFile

def prophet_predict(dataset):
    #make prophet instance
    prophetInstance = Prophet()
    prophetInstance.fit(dataset)
    #predict the future
    future = prophetInstance.make_future_dataframe(periods=1)
    return (prophetInstance,future)

def plot_data():
    returnData = prophet_predict(load_file('usdmxndataset.xlsm'))
    #create a forecast
    forecast = returnData[0].predict(returnData[1])    
    #plot the results
    return returnData[0].plot(forecast)
    
plot_data()