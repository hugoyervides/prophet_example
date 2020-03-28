#import required libraries
import pandas as pd
from fbprophet import Prophet

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
    future = prophetInstance.make_future_dataframe(periods=365)
    return (prophetInstance,future)

def main():
    returnData = prophet_predict(load_file('usdmxndataset.xlsm'))
    #create a forecast
    forecast = returnData[0].predict(returnData[1])    
    print(forecast[['ds','yhat']].tail())

main()