#-*- coding: utf-8 -*-
##------------------------------------------------------------------------
## Case: Marketing
## Autor: Prof. Roberto Ãngelo
##------------------------------------------------------------------------

# Bibliotecas padrão
import DM_Utils as du
import numpy as np
import pandas as pd


print()
print('----------------     INÍCIO DE CARGA DE DADOS   ------------------------------------')
## Carregando os dados
dataset = pd.read_csv('BASE_MARKETING.TXT',sep='\t') # Separador TAB
print('-----------------     FIM DA CARGA DE DADOS   --------------------------------------')
#print(dataset.describe())

#------------------------------------------------------------------------------------------
# Pré-processamento das variáveis de entrada
#------------------------------------------------------------------------------------------
# ATRIBUTO VL_TOTAL_INVESTIMENTO_T0
dataset['pre_tem_investimento']   = [1 if np.isnan(x) or x > 0 else 0 for x in dataset['VL_TOTAL_INVESTIMENTO_T0']]
# ATRIBUTO RESTRICAO_INTERNA
dataset['pre_restricao_interna']   = [x for x in dataset['RESTRICAO_INTERNA']]
# ATRIBUTO RISCO
dataset['pre_alto'] = [1 if x=='ALTO' else 0 for x in dataset['RISCO']]
dataset['pre_medio'] = [1 if x=='MEDIO' else 0 for x in dataset['RISCO']]
dataset['pre_baixo'] = [1 if x=='BAIXO' else 0 for x in dataset['RISCO']]

# ATRIBUTO SEG_UF
dataset['pre_UF'] = [1 if x=='RJ' or x=='SP' else 0 for x in dataset['SG_UF']]
dataset['pre_UF'] = [1 if x=='SP' else 0 for x in dataset['SG_UF']]
dataset['pre_UF'] = [1 if x=='MG' else 0 for x in dataset['SG_UF']]
# ATRIBUTO CD_SEXO
dataset['pre_sexo_h'] = [1 if x=='H' else 0 for x in dataset['CD_SEXO']]
dataset['pre_sexo_m'] = [1 if x=='M' else 0 for x in dataset['CD_SEXO']]
# ATRIBUTO VL_TOTL_REND
dataset['pre_rend'] = [1 if x<=2000 else 0 for x in dataset['VL_TOTL_REND']]
# ATRIBUTO PERFIL
dataset['pre_perfil_investidor'] = [1 if x=='INVESTIDOR' else 0 for x in dataset['PERFIL']]
dataset['pre_perfil_tomador'] = [1 if x=='TOMADOR' else 0 for x in dataset['PERFIL']]


# ATRIBUTO DS_ESTADO_CIVIL
dataset['pre_viuvo'] = [1 if x =='VIUVO' else 0 for x in dataset['DS_ESTADO_CIVIL']]
dataset['pre_solteiro'] = [1 if x =='SOLTEIRO' else 0 for x in dataset['DS_ESTADO_CIVIL']]
dataset['pre_divorciado_nao_informado'] = [1 if x=='DIVORCIADO' or x=='NÃO INFORMADO' else 0 for x in dataset['DS_ESTADO_CIVIL']]
# ATRIBUTO TEM_PRE_APROV_CDC
dataset['pre_tem_pre_aprov_cdc']   = [x for x in dataset['TEM_PRE_APROV_CDC']]
#  ATRIBUTO TIPO_CLIENTE
dataset['pre_tipo_cliente_vip'] = [1 if x =='VIP' else 0 for x in dataset['TIPO_CLIENTE']]
dataset['pre_tipo_cliente_classico'] = [1 if x =='CLÁSSICO' else 0 for x in dataset['TIPO_CLIENTE']]
# ATRIBUTO IDADE
dataset['pre_idade'] = [18 if np.isnan(x) or x < 18 else x for x in dataset['IDADE']] 
dataset['pre_idade'] = [80 if x > 80 else x for x in dataset['pre_idade']] 
dataset['pre_idade'] = [(x-18)/(80-18) for x in dataset['pre_idade']] 
# ATRIBUTO ESCOLARIDADE
dataset['pre_ensino_fundamental'] = [1 if x=='ENSINO FUNDAMENTAL' else 0 for x in dataset['ESCOLARIDADE']] 
dataset['pre_superior'] = [1 if x=='SUPERIOR' else 0 for x in dataset['ESCOLARIDADE']] 
dataset['pre_escolaridade_demais'] = [1 if x!='ENSINO FUNDAMENTAL' and x!='SUPERIOR' else 0 for x in dataset['ESCOLARIDADE']] 
# ATRIBUTO QTDE_PRODUTOS_PF_12
dataset['pre_qtde_produtos_pf_12m'] = [0 if np.isnan(x) or x < 0 else x for x in dataset['QTDE_PRODUTOS_12M']] 
dataset['pre_qtde_produtos_pf_12m'] = [10 if x > 10 else x for x in dataset['pre_qtde_produtos_pf_12m']] 
dataset['pre_qtde_produtos_pf_12m'] = [x/10 for x in dataset['pre_qtde_produtos_pf_12m']] 

# ---------------------------------------------------------------------------
# Selecionando as colunasjá pré-processadas
# ---------------------------------------------------------------------------
cols_in =  [   'pre_tem_investimento'
              ,'pre_restricao_interna'
              ,'pre_alto'
              ,'pre_medio'
              ,'pre_baixo'
              ,'pre_viuvo'
              ,'pre_solteiro'
              ,'pre_divorciado_nao_informado'
              ,'pre_tem_pre_aprov_cdc'
              ,'pre_tipo_cliente_vip'
              ,'pre_tipo_cliente_classico'
              ,'pre_idade'
              ,'pre_ensino_fundamental'
              ,'pre_superior'
              ,'pre_escolaridade_demais'
              ,'pre_qtde_produtos_pf_12m'
              ,'pre_UF'
              ,'pre_sexo_h'
              ,'pre_sexo_m'
              ,'pre_rend'
              ,'pre_perfil_investidor'
              ,'pre_perfil_tomador'
              ,'ALVO'] # Com ALVO temporariamente

##------------------------------------------------------------
## Separando em dados de treinamento e teste com Oversampling
##------------------------------------------------------------
y = dataset['ALVO']
X = dataset[cols_in] # Com ALVO temporariamente
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 99)

# Replica o número de bons - Oversampling no treino
X_train_1 = X_train.query("ALVO == 1")
X_train = pd.concat([X_train, X_train_1,X_train_1,X_train_1] , sort = False)

# Refaz o y_train
y_train = X_train['ALVO']
#retirar alvo das variáveis de entrada
del X_train['ALVO'] 


#---------------------------------------------------------------------------
## Selecionando Atributos com RFE - Recursive Feature Elimination
#---------------------------------------------------------------------------
# feature extraction
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression(solver='newton-cg')
selected = RFE(model,step=1,n_features_to_select=10).fit(X_train, y_train)

print()
print('----------------     SELEÇÃO DE VARIÁVEIS--------------------------------------')
print("Num Features: %d" % selected.n_features_)
used_cols = []
for i in range(0, len(selected.support_)):
    if selected.support_[i]: 
        used_cols.append(X_train.columns[i]) 
        print('             -> {:30}     '.format(X_train.columns[i]))
print('-------------------------------------------------------------------------------')

X_train = X_train[used_cols]     # Carrega colunas de entrada selecionadas por RFE
X_test = X_test[used_cols]       # Carrega colunas de entrada selecionadas por RFE
#---------------------------------------------------------------------------
## Ajustando modelos - Aprendizado supervisionado  
#---------------------------------------------------------------------------

# Regressão linear com dados de treinamento
from sklearn.linear_model import LinearRegression
LinearReg = LinearRegression(fit_intercept=True)
LinearReg.fit(X_train, y_train) #Treinamento do x que são as variaveis de entrada e o y como alvo


# Árvore de decisão com dados de treinamento
from sklearn.tree import DecisionTreeClassifier
#dtree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dtree = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=30, min_samples_split=30,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=0, splitter='best')
dtree.fit(X_train, y_train) #Treinamento



# Regressão logística com dados de treinamento
from sklearn.linear_model import LogisticRegression
LogisticReg = LogisticRegression(solver='newton-cg')
LogisticReg.fit(X_train, y_train) #Treinamento

#Rede Neural com dados de treinamento
from sklearn.neural_network import MLPClassifier 
RNA = MLPClassifier(activation='logistic', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=True,
       epsilon=1e-08, hidden_layer_sizes=(25), learning_rate='constant',
       learning_rate_init=0.01, max_iter=2000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='adam', tol=0.0001, validation_fraction=0.25, verbose=False,
       warm_start=False)
RNA.fit(X_train, y_train) #Treinamento


#---------------------------------------------------------------------------
## Salvando os Modelos para Implnatar com outro código Python
#---------------------------------------------------------------------------
du.SaveModel(dtree,'dtree')
du.SaveModel(LinearReg,'LinearReg')
du.SaveModel(LogisticReg,'LogisticReg')
du.SaveModel(RNA,'RNA')

#---------------------------------------------------------------------------
## Previsão treinamento e teste - CLASSIFICAÇÃO
#---------------------------------------------------------------------------
# Árvore de Decisão
y_pred_train_DT = dtree.predict(X_train)
y_pred_test_DT  = dtree.predict(X_test)
# Regressão Linear
y_pred_train_RL = np.array([1 if x > 0.5 else 0 for x in LinearReg.predict(X_train)] )
y_pred_test_RL  = np.array([1 if x > 0.5 else 0 for x in LinearReg.predict(X_test)])
# Regressão Logística
y_pred_train_RLog = LogisticReg.predict(X_train)
y_pred_test_RLog  = LogisticReg.predict(X_test)
# Redes Neurais
y_pred_train_RNA = RNA.predict(X_train)
y_pred_test_RNA = RNA.predict(X_test)


#---------------------------------------------------------------------------
## Cálcula e mostra a Acurácia dos modelos
#---------------------------------------------------------------------------
from sklearn import metrics
import pandas as pd
import numpy as np

print()
print('----------------------------------------------------------------------------')
print('----------     VALIDAÇÃO     ------------------------------------------------')
print('----------------------------------------------------------------------------')
print('--  Árvore de Decisão  --')
print('----------------------------------------------------------------------------')
print(pd.crosstab(y_test, y_pred_test_DT, rownames=['Real'], colnames=['Predito'], margins=True))
print('')
print('Acurácia:   ',metrics.accuracy_score(y_test, y_pred_test_DT))
print('Precisão:   ',metrics.precision_score(y_test, y_pred_test_DT))
print('Recall:   ',metrics.recall_score(y_test, y_pred_test_DT))
print('----------------------------------------------------------------------------')
print('')
print('')
print('----------------------------------------------------------------------------')
print('--  Regressão Linear  --')
print('----------------------------------------------------------------------------')
print(pd.crosstab(y_test, y_pred_test_RL, rownames=['Real'], colnames=['Predito'], margins=True))
print('')
print('Acurácia:   ',metrics.accuracy_score(y_test, y_pred_test_RL))
print('Precisão:   ',metrics.precision_score(y_test, y_pred_test_RL))
print('Recall:   ',metrics.recall_score(y_test, y_pred_test_RL))
print('----------------------------------------------------------------------------')
print('')
print('')
print('----------------------------------------------------------------------------')
print('--  Regressão Logística  --')
print('----------------------------------------------------------------------------')
print(pd.crosstab(y_test, y_pred_test_RLog, rownames=['Real'], colnames=['Predito'], margins=True))
print('')
print('Acurácia:   ',metrics.accuracy_score(y_test, y_pred_test_RLog))
print('Precisão:   ',metrics.precision_score(y_test, y_pred_test_RLog))
print('Recall:   ',metrics.recall_score(y_test, y_pred_test_RLog))
print('----------------------------------------------------------------------------')
print('')
print('')
print('----------------------------------------------------------------------------')
print('--  Redes Neurais  --')
print('----------------------------------------------------------------------------')
print(pd.crosstab(y_test, y_pred_test_RNA, rownames=['Real'], colnames=['Predito'], margins=True))
print('')
print('Acurácia:   ',metrics.accuracy_score(y_test, y_pred_test_RNA))
print('Precisão:   ',metrics.precision_score(y_test, y_pred_test_RNA))
print('Recall:   ',metrics.recall_score(y_test, y_pred_test_RNA))
print('----------------------------------------------------------------------------')
print('')
print('')



#---------------------------------------------------------------------------
## Previsão treinamento e teste - REGRESSÂO
#---------------------------------------------------------------------------
# Árvore de Decisão
y_pred_train_DT_R  = dtree.predict_proba(X_train)[:,1]
y_pred_test_DT_R  = dtree.predict_proba(X_test)[:,1]

# Regressão Linear
y_pred_train_RL_R = LinearReg.predict(X_train)
y_pred_test_RL_R  = LinearReg.predict(X_test)
# Regressão Logística
y_pred_train_RLog_R = LogisticReg.predict_proba(X_train)[:,1]
y_pred_test_RLog_R  = LogisticReg.predict_proba(X_test)[:,1]
# Redes Neurais
y_pred_train_RNA_R = RNA.predict_proba(X_train)[:,1]
y_pred_test_RNA_R  = RNA.predict_proba(X_test)[:,1]

#---------------------------------------------------------------------------
## Cálcula e mostra RMSE dos modelos
#---------------------------------------------------------------------------

from math import sqrt
print()
print('----------------------------------------------------------------------------')
print('----------     RMSE ERROR    -----------------------------------------------')
print('----------------------------------------------------------------------------')
print('Árvore de Decisão:  ',  sqrt(np.mean((y_test - y_pred_test_DT_R) **2) ))
print('Regressão Linear:   ',  sqrt(np.mean((y_pred_test_RL_R -  y_test) ** 2) ))
print('Regressão Logística:',  np.mean((y_pred_test_RLog_R - y_test) ** 2) ** 0.5)
print('Redes Neurais:      ',  np.mean((y_pred_test_RNA_R - y_test) ** 2) ** 0.5)
print('----------------------------------------------------------------------------')
print()

#---------------------------------------------------------------------------
## Cálcula o KS2
#---------------------------------------------------------------------------
print()
print('----------------------------------------------------------------------------')
print('----------------     KS2    ------------------------------------------------')
print('----------------------------------------------------------------------------')
print('Árvore de Decisão:   ',du.KS2(y_test,y_pred_test_DT_R))
print('Regressão Linear:    ',du.KS2(y_test,y_pred_test_RL_R))
print('Regressão Logística: ',du.KS2(y_test,y_pred_test_RLog_R))
print('Redes Neurais:       ',du.KS2(y_test,y_pred_test_RNA_R))
print('----------------------------------------------------------------------------')
print()

#---------------------------------------------------------------------------
## Validação Resultados: Matriz Confusão, RMSE ERROR e KS2
#---------------------------------------------------------------------------

print('----------------------------------------------------------------------------')
print('----------     Comparativo     ------------------------------------------------')
print('----------------------------------------------------------------------------')

#Árvore Decisão
dt_acu = round(metrics.accuracy_score(y_test, y_pred_test_DT),2)
dt_pre = round(metrics.precision_score(y_test, y_pred_test_DT),2)
dt_rec = round(metrics.recall_score(y_test, y_pred_test_DT),2)
dt_rmse = round(sqrt(np.mean((y_test - y_pred_test_DT_R) **2)),2)
dt_ks2 = round(du.KS2(y_test,y_pred_test_DT_R),2)

#Regressão Linear
RL_acu = round(metrics.accuracy_score(y_test, y_pred_test_RL),2)
RL_pre = round(metrics.precision_score(y_test, y_pred_test_RL),2)
RL_rec = round(metrics.recall_score(y_test, y_pred_test_RL),2)
RL_rmse = round(sqrt(np.mean((y_pred_test_RL_R -  y_test) ** 2)),2)
RL_ks2 = round(du.KS2(y_test,y_pred_test_RL_R),2)

#Regressão Logistica
RLog_acu = round(metrics.accuracy_score(y_test, y_pred_test_RLog),2)
RLog_pre = round(metrics.precision_score(y_test, y_pred_test_RLog),2)
RLog_rec = round(metrics.recall_score(y_test, y_pred_test_RLog),2)
RLogrmse = round(np.mean((y_pred_test_RLog_R - y_test) ** 2) ** 0.5,2)
RLogks2 = round(du.KS2(y_test,y_pred_test_RLog_R),2)

#Redes Neurais
rna_acu = round(metrics.accuracy_score(y_test, y_pred_test_RNA),2)
rna_pre = round(metrics.precision_score(y_test, y_pred_test_RNA),2)
rna_ec = round(metrics.recall_score(y_test, y_pred_test_RNA),2)
rna_rmse = round(np.mean((y_pred_test_RNA_R - y_test) ** 2) ** 0.5,2)
rna_RNAks2 = round(du.KS2(y_test,y_pred_test_RNA_R),2)





data = np.array([['Árvore Decisão',dt_acu,dt_pre,dt_rec,dt_rmse,dt_ks2],
                 ["Regressçao Linear",RL_acu,RL_pre,RL_rec,RL_rmse,RL_ks2],
                 ['Regressão Logistica',RLog_acu,RLog_pre,RLog_rec,RLogrmse,RLogks2],
                 ['Redes Neurais',rna_acu,rna_pre,rna_ec,rna_rmse,rna_RNAks2]])

df_valid = pd.DataFrame(data, columns=['Treino', 'MC-Acurácia','MC-Precisão','MC-Recall', 'RMSE Error', 'KS2'])

df_valid




#---------------------------------------------------------------------------
## Mostra as fórmulas das regressões para implantação em SQL Server, se precisar
#---------------------------------------------------------------------------

print()
print('----------------   REPRESENTAÇÃO DOS MODELOS EM SQL SERVER  ----------------')
du.LogisticFormulaSQL(LogisticReg, X_train)

print()
du.LinearFormulaSQL(LinearReg, X_train)
print('----------------------------------------------------------------------------')



#----------------------------------------------------------------------
## Montando um Data Frame (Matriz) com os resultados
#----------------------------------------------------------------------
# Conjunto de treinamento
df_train = pd.DataFrame(y_pred_train_DT_R, columns=['REGRESSION_DT'])
df_train['CLASSIF_DT'] = y_pred_train_DT
df_train['REGRESSION_RL'] = y_pred_train_RL_R
df_train['CLASSIF_RL'] =  [1 if x > 0.5 else 0 for x in y_pred_train_RL]
df_train['REGRESSION_RLog'] = y_pred_train_RLog_R
df_train['CLASSIF_RLog'] = y_pred_train_RLog
df_train['REGRESSION_RNA'] = y_pred_train_RNA_R
df_train['CLASSIF_RNA'] = y_pred_train_RNA
df_train['ALVO'] = [x for x in y_train]
df_train['TRN_TST'] = 'TRAIN'

# Conjunto de test
df_test = pd.DataFrame(y_pred_test_DT_R, columns=['REGRESSION_DT'])
df_test['CLASSIF_DT'] = y_pred_test_DT
df_test['REGRESSION_RL'] = y_pred_test_RL_R
df_test['CLASSIF_RL'] =  [1 if x > 0.5 else 0 for x in y_pred_test_RL]
df_test['REGRESSION_RLog'] = y_pred_test_RLog_R
df_test['CLASSIF_RLog'] = y_pred_test_RLog
df_test['REGRESSION_RNA'] = y_pred_test_RNA_R
df_test['CLASSIF_RNA'] = y_pred_test_RNA
df_test['ALVO'] = [x for x in y_test]
df_test['TRN_TST'] = 'TEST' 

print()
print('----------------    INÍCIO DA EXPORTAÇÃO RESULTADOS   ----------------------------------')
# Juntando Conjunto de Teste e Treinamento
df_total = pd.concat([df_test, df_train], sort = False)

## Exportando os dados para avaliação dos resultados em outra ferramenta
df_total.to_csv('resultado_comparacao.csv')
print('----------------     FIM DA EXPORTAÇÃO RESULTADOS   ------------------------------------')












