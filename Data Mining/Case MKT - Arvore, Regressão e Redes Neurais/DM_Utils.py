#-*- coding: utf-8 -*-
##------------------------------------------------------------------------
## Objective: Funções gerais
## Autor: Prof. Roberto Ãngelo
##------------------------------------------------------------------------

# Bibliotecas padrão
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from scipy.stats import ks_2samp
import pickle


#----------------------------------------------
#-- Função para cálculo de KS2
#----------------------------------------------
def KS2(y, y_pred):
    df_ks2 = pd.DataFrame([x for x in y_pred], columns=['REGRESSION_RLog'])
    df_ks2['ALVO'] = [x for x in y]
    return ks_2samp(df_ks2.loc[df_ks2.ALVO==0,"REGRESSION_RLog"], df_ks2.loc[df_ks2.ALVO==1,"REGRESSION_RLog"])[0]
#----------------------------------------------

#--------------------------------------------------------
#-- Funções mostrar fórmula das regressões em formato SQL Server
#--------------------------------------------------------
def LogisticFormulaSQL(model, X):
    print('Regressão Logística')
    print('SCORE = ROUND(1/(1 + exp(-(       ' + str(model.intercept_[0]))
    for i in range(0, len(model.coef_[0])):
        print('             + {:30}     *     {:.6}'.format(X.columns[i], str(model.coef_[0,i])))
    print('             ))) * 100,2) 	')

def LinearFormulaSQL(model, X):
    print('Regressão Linear')
    print('SCORE = ' + str(model.intercept_))
    for i in range(0, len(model.coef_)):
        print('             + {:30}     *     {:.6}'.format(X.columns[i], str(model.coef_[i])))
#--------------------------------------------------------
        
        
#--------------------------------------------------------
#-- Função para salvar modelo
#--------------------------------------------------------
def SaveModel(model,name):
    File_Model = open(str(name) + '.model', 'wb')
    pickle.dump(model, File_Model)
    File_Model.close()
#--------------------------------------------------------
    
#--------------------------------------------------------
#-- Função para carregar modelo
#--------------------------------------------------------
def OpenModel(name):
    File_Model = open(str(name) + '.model', 'rb')
    model = pickle.load(File_Model)        
    File_Model.close()
    return model
#--------------------------------------------------------
        