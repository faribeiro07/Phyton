#-*- coding: utf-8 -*-
##------------------------------------------------------------------------
## Case: Deployment Marketing
## Autor: Prof. Roberto Ãngelo - 10/02/2021
##------------------------------------------------------------------------

# Bibliotecas padrão
import DM_Utils as du
import numpy as np
import pandas as pd

dataset = pd.read_csv('NOVA_BASE_MARKETING.TXT',sep='\t') # Separador TAB

#------------------------------------------------------------------------------------------
# Pré-processamento das variáveis de entrada que estão no modelo
# ATRIBUTO RESTRICAO_INTERNA
dataset['pre_restricao_interna']   = [x for x in dataset['RESTRICAO_INTERNA']]
# ATRIBUTO RISCO
dataset['pre_medio'] = [1 if x=='MEDIO' else 0 for x in dataset['RISCO']]
dataset['pre_baixo'] = [1 if x=='BAIXO' else 0 for x in dataset['RISCO']]
# ATRIBUTO DS_ESTADO_CIVIL
dataset['pre_viuvo'] = [1 if x =='VIUVO' else 0 for x in dataset['DS_ESTADO_CIVIL']]
dataset['pre_solteiro'] = [1 if x =='SOLTEIRO' else 0 for x in dataset['DS_ESTADO_CIVIL']]
dataset['pre_divorciado_nao_informado'] = [1 if x=='DIVORCIADO' or x=='NÃO INFORMADO' else 0 for x in dataset['DS_ESTADO_CIVIL']]
#  ATRIBUTO TIPO_CLIENTE
dataset['pre_tipo_cliente_vip'] = [1 if x =='VIP' else 0 for x in dataset['TIPO_CLIENTE']]
dataset['pre_tipo_cliente_classico'] = [1 if x =='CLÁSSICO' else 0 for x in dataset['TIPO_CLIENTE']]
# ATRIBUTO IDADE
dataset['pre_idade'] = [18 if np.isnan(x) or x < 18 else x for x in dataset['IDADE']] 
dataset['pre_idade'] = [80 if x > 80 else x for x in dataset['pre_idade']] 
dataset['pre_idade'] = [(x-18)/(80-18) for x in dataset['pre_idade']] 
# ATRIBUTO QTDE_PRODUTOS_PF_12
dataset['pre_qtde_produtos_pf_12m'] = [0 if np.isnan(x) or x < 0 else x for x in dataset['QTDE_PRODUTOS_12M']] 
dataset['pre_qtde_produtos_pf_12m'] = [10 if x > 10 else x for x in dataset['pre_qtde_produtos_pf_12m']] 
dataset['pre_qtde_produtos_pf_12m'] = [x/10 for x in dataset['pre_qtde_produtos_pf_12m']] 

# ---------------------------------------------------------------------------
# Selecionando as colunas já pré-processadas
cols_in =  [  'pre_restricao_interna'
              ,'pre_medio'
              ,'pre_baixo'
              ,'pre_viuvo'
              ,'pre_solteiro'
              ,'pre_divorciado_nao_informado'
              ,'pre_tipo_cliente_vip'
              ,'pre_tipo_cliente_classico'
              ,'pre_idade'
              ,'pre_qtde_produtos_pf_12m' ] 
X = dataset[cols_in] 

#---------------------------------------------------------------------------
## Abrindo o Modelo escolhido - Redes Neurais
RNA = du.OpenModel('RNA')
## Predição da nova base
y_pred_RNA = RNA.predict(X)
y_pred_RNA_R = RNA.predict_proba(X)[:,1]
#----------------------------------------------------------------------
## Montagem de Data Frame (Matriz) com os resultados
df = pd.DataFrame(y_pred_RNA_R, columns=['REGRESSION_RN'])
df['CLASSIF_RNA'] = y_pred_RNA
df.to_csv('resultado_saida.csv')
print()
print('-->   REGISTROS PROCESSADOS:', df.count()[0])
print()
print(df)












