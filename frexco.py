
import pandas as pd  ### Pandas para subir os dados
import matplotlib.pyplot as plt  ### Plots
from statsmodels.graphics.tsaplots import plot_acf  ### Covariâncias entre valores
from statsmodels.graphics.tsaplots import plot_pacf ### Covariâncias entre valores
from statsmodels.tsa.arima_model import ARIMA  ### Modelo ARIMA
from statsmodels.tsa.stattools import adfuller ### Teste de estacionariedade
import warnings  ### Evitar warnings desnecessários
warnings.filterwarnings('ignore')


### Subindo inicialmente um dataframe para ser modificado e outro para se manter estável
data = pd.read_excel('C:/Users/caio/Downloads/Dados.xlsx')
data1 = pd.read_excel('C:/Users/caio/Downloads/Dados.xlsx')



### Plot da série para vermos sazonalidade e tendência
df = pd.DataFrame(data, columns = ['Data', 'Vendas'])
df['Data'] = pd.to_datetime(df['Data'])
df.index = df['Data']
del df['Data']
df.plot(figsize=(15, 8))
plt.show()


### Checagem da acf e pacf da série com  uma diferença sazonal de período 7 (semanal) para ver como iremos modelar
data['Vendas'] = data['Vendas'].diff(7)
data = data.drop([0,1, 2, 3, 4, 5, 6], axis=0).reset_index(drop=True)
plot_acf(data.Vendas,lags=12)
plot_pacf(data.Vendas,lags=12)


### Modelagem como um modelo ARMA(1,0) com uma diferença sazonal e comparação deste modelo com a série real
arima = ARIMA(data.Vendas,order=(1,0,0))
model = arima.fit()
print(model.summary())
model.plot_predict(dynamic=False)
plt.show()


### Checagem se os erros satisfazem as pressuposições do modelo (ruído branco)
derro = data['Vendas'] - model.fittedvalues
plot_acf(derro,lags=12)
plot_pacf(derro,lags=12)
teste = adfuller(derro)
print(teste[1]) ### p-value (abaixo de 0.05 concluímos que os erros são estacionários)


### Realização da previsão retornando os dados sem a diferença sazonal
prevdif = [model.predict(start=1,end=5)[1],model.predict(start=1,end=5)[2],model.predict(start=1,end=5)[3],model.predict(start=1,end=5)[4],model.predict(start=1,end=5)[5]]
previsão =  data1['Vendas'][39:44] + prevdif
print(previsão) ### Resultado Final