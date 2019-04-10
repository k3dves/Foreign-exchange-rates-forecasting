
from IMF_LSTM import *

def test_series(imf):
        testdata=pd.read_csv('USD_EURTest.csv')
        # print(testdata)
        testseries=testdata.iloc[:,2:3]
        temp=imf[len(imf)-15:].reshape(-1,1)
        # np.append(temp,imf[len(imf)-15:],axis=0)
        temp=np.append(temp,testseries,axis=0)
        # temp=temp+testseries
        # testseries=imf[len(imf)-15:]+testseries
        testseries=temp
        # testseries=testseries.astype('float64')
        #testseries=scaler.transform(testseries)
        testseries=testseries.reshape(-1,1);
        return testseries


# scaler = MinMaxScaler(feature_range = (0, 1))
data=pd.read_csv('USD_EUR')
series=data.iloc[:,2:3].values
series=series.astype('float64')
series=np.reshape(series,(series.shape[0]))
#scalar did not use
# print(series[len(series)-15:])
emd=EMD()
emd.emd(series)
imfs, res = emd.get_imfs_and_residue()
# print(imfs[4][1000:1200])
testseries=test_series(series)
predictions=[]
for imf in imfs:
    predictions.append(imf_trainer(imf,testseries))
predictions.append(imf_trainer(res,testseries))
predictions1=sum(predictions)
print(predictions[1])
np.shape(predictions)
print(predictions1)
#predictions = scaler.inverse_transform(predictions)
# print(predictions)
plt.figure(figsize=(10,6))
plt.plot(testseries[15:], color='blue', label='Actual Stock Price')
plt.plot(predictions1 , color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
