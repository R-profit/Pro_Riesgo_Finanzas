## seleccion de serie de tiempo
install.packages("installr")
library(installr)
version
Upda



install.packages("fpp2")
library(fpp3)
library(Performanceanalytics)
library(xts)
library(quantmod)
library(fUnitRoots)
library(forecast)
library(ggplot2)
library(tseries)
library(lmtest)
library(TSA)
install.packages("Metrics")
library(Metrics)
library(FitAR)
install.packages('forecast',dependencies = TRUE)
options(digits = 3)
optim(warn = -1)


##obtener precios de Amazon
AMZN <- getSymbols("AMZN", from="2020-08-01", to="2021-03-31", auto.assign = FALSE)

##eliminar faltantes:
AMZN<-na.omit(AMZN) 

##mantener columnas con precio cierre columna 4
AMZN <- AMZN[,4]

##podemos graficar
plot(AMZN, ylab = "precios")
length(AMZN)

##partimos la serie tomamos el 7% para la prueba
h<- round(length(AMZN)=0.07,digits = 0)
h
train <-AMZN[1:(nrow(AMZN) - h),]
test <- AMZN[nrow(AMZN) - h + 1):nrow(AMZN),]


####### Modelos ARIMA #######

#### veamos si la serie es estacionaria ####


adftest(train)


