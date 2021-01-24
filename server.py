from flask import Flask,render_template,request
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

app=Flask(__name__,template_folder='templates')
app = Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def result():
    print("start results")
    if request.method == 'POST':
        try:
            etc=request.form['etp']
            #we gonna get tesl/apple/iam
            print("etc= "+etc)
            if(etc=='tesla'):
                modelname='models\tesla.h5'
                df= pd.read_csv("datasets\TSLA.csv")

            elif(etc=="apple"):
                modelname='models\apple.h5'
                df= pd.read_csv("datasets\AAPL.csv")

            elif(etc=="bank"):
                modelname='models\bank.h5'
                df=pd.read_csv("datasets\BAC.csv")

            else:
                modelname='models\iam.h5'
                df= pd.read_csv("datasets\IAM.csv")
            #prediction

            with graph.as_default():
                model1 = keras.models.load_model(modelname)
                df = df.dropna(how='any',axis=0) #remove null rows
                train, test = train_test_split(df, test_size=0.2,shuffle=False)
                dataset_train=train
                dataset_test =test
                real_stock_price = dataset_test.iloc[:, 1:2].values
                print(df.head())
                sc = MinMaxScaler(feature_range=(0,1))
                training_set_scaled = sc.fit_transform(real_stock_price)
                dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
                inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
                inputs = inputs.reshape(-1,1)
                inputs = sc.transform(inputs)
                X_test = []
                for i in range(60, len(inputs)):
                    X_test.append(inputs[i-60:i, 0])
                X_test = np.array(X_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                dates=test.Date.tolist()
                #print(dates)
                origin=real_stock_price
                print("origin")
                print("real stock")
                #origin,X_test,dates=prepar_data(df)
                #predicting
                predicted_stock_price = model1.predict(X_test)
            sc = MinMaxScaler(feature_range=(0,1))
            #predicted_stock_price = sc.inverse_transform(predicted_stock_price)
            #transform to list
            origin=sc.fit_transform(origin)
            #origin=sc.inverse_transform(origin)
            #predicted_stock_price=sc.inverse_transform(predicted_stock_price)
            origin=origin.tolist()
            predicted_stock_price=predicted_stock_price.tolist()
            print(len(origin))
            print(len(predicted_stock_price))
            print(len(dates))
            return(render_template('index.html',origin=origin,predict=predicted_stock_price,getdates=dates))
        except Exception as e:
            return "something is wrong "+str(e)
    else:
        return "shit"

def prepar_data(df):
    df = df.dropna(how='any',axis=0) #remove null rows
    train, test = train_test_split(df, test_size=0.2,shuffle=False)
    dataset_train=train
    dataset_test =test
    real_stock_price = dataset_test.iloc[:, 1:2].values
    
    sc = MinMaxScaler(feature_range=(0,1))
    
    training_set_scaled = sc.fit_transform(real_stock_price)
    
    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, len(inputs)):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    dates=getDate(test)
    return real_stock_price,X_test,dates

def getDate(df):
    return df.Date.tolist()



  

if __name__ == "__main__":
    global graph
    graph = tf.compat.v1.get_default_graph()
    app.run()