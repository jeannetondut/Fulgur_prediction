#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 14:07:00 2022

@author: tj103659


"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import datetime
from datetime import datetime
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, confusion_matrix, f1_score
from sklearn.metrics import mean_absolute_error, roc_auc_score, roc_curve, auc
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import fonctions_definition as fulgur
import allMetrics

"""
CODE
"""

#%% DATA 
nom_fichier = list(['/Users/tj103659/Documents/Python/Data/fulgur-ffbob-16Nov22.csv',
                    '/Users/tj103659/Documents/Python/Data/fulgur-ffr-11Janv23.csv',
                    '/Users/tj103659/Documents/Python/Data/fulgur -ffa-11Janv23.csv'])



X_ffr, Tab_ffr,lst_code_ffr,sortie_ffr = fulgur.MAIN(nom_fichier[1])
X_ffa, Tab_ffa,lst_code_ffa,sortie_ffa = fulgur.MAIN(nom_fichier[2])
X_bob, Tab_bob,lst_code_bob,sortie_bob = fulgur.MAIN(nom_fichier[0]) 

X,Y = fulgur.concat_X(X_ffr,X_ffa,X_bob,sortie_ffr,sortie_ffa,sortie_bob)
X = X.drop(columns=['id','CODE'])
Y = Y.replace(0.33,0)
Y = Y.replace(0.66,1)
Y = Y.astype('int')

lst_code_total = pd.concat([lst_code_ffa,lst_code_ffr,lst_code_bob]) 
lst_code_total =lst_code_total['Code FULGUR'].drop_duplicates()

#%% SMOTE 

#%% Modèle de prédiction 
best_param, metrics, facteur = fulgur.DT_gridsearch(X, Y)
best_param, metrics = fulgur.RF_gridsearch(X, Y)


lst_metrics = ['rocProba','tnPred','fpPred','fnpred','tpPred','accPred','recPred','spePred','prePred','fprPred']

Metrics_moy =pd.DataFrame(index=lst_metrics,columns=['Moyenne','STD','MIN','MAX'])
for i in range(np.size(lst_metrics)):
    Metrics_moy.loc[lst_metrics[i]]['Moyenne'] = np.mean(metrics[lst_metrics[i]])
    Metrics_moy.loc[lst_metrics[i]]['STD'] = np.std(metrics[lst_metrics[i]])
    Metrics_moy.loc[lst_metrics[i]]['MIN'] = np.min(metrics[lst_metrics[i]])
    Metrics_moy.loc[lst_metrics[i]]['MAX'] = np.max(metrics[lst_metrics[i]])
 
Metrics_moy.to_excel('Moy_metrics_RF_smote.xlsx')    
 
    
 
   
#%%
lst_param = list(X.columns)
facteur_moy =pd.DataFrame(index=lst_param,columns=['Moyenne','STD','MIN','MAX'])
for i in range(np.size(list(X.columns))):
    facteur_moy.loc[lst_param[i]]['Moyenne'] = np.mean(facteur[lst_param[i]])
    facteur_moy.loc[lst_param[i]]['STD'] = np.std(facteur[lst_param[i]])
    facteur_moy.loc[lst_param[i]]['MIN'] = np.min(facteur[lst_param[i]])
    facteur_moy.loc[lst_param[i]]['MAX'] = np.max(facteur[lst_param[i]])
    
#%% 
facteur_moy['param']=facteur_moy.index
fig = plt.figure()
facteur_moy_sort = facteur_moy['Moyenne'].sort_values()
facteur_moy.plot(kind='barh',x='param',y='Moyenne',rot=45,yerr='STD',legend=False,colormap='plasma')
plt.savefig('fig2.pdf',figsize=40)


facteur_moy_sort.plot(kind='barh',rot=45,legend=False,colormap='plasma')


lst_code_monit = lst_code_total.str.lstrip('FULGUR-')
index_code = lst_code_total.str.lstrip('FULGUR-')

info_indiv =pd.read_csv('/Users/tj103659/Documents/Python/Data/BDD_histoBlessure_testPsy/BDD_historiqueBlessure_detail_psyFULL.csv', sep=';')

info_indiv = info_indiv[['ID Fulgur','Age','Poids','Taille','Genre','Discipline/position']]
info_indiv = info_indiv.dropna()
info_indiv = info_indiv.set_index(info_indiv['ID Fulgur'])

Concat_info_monit = pd.DataFrame(index=index_code,columns=['Age','Poids','Taille','Genre','Discipline/position'])
lst_param = ['Age','Poids','Taille','Genre','Discipline/position']
for i in index_code:
    if list(info_indiv.index).count(int(i)) == 1:
        for k in lst_param :
            Concat_info_monit.loc[i][k] = info_indiv.loc[int(i)][k]

#%% SUNBURST
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_excel('Data_sunburst.xlsx',header=0)
df= df.replace('H','Male')
df= df.replace('F','Female')
df = df.replace('Inconnu','Not specified')

fig = px.sunburst(df, path=['Genre','Discipline/position'], values='Code FULGUR',color='Discipline/position')

fig.update_traces(
    textinfo='percent root',
    hovertemplate='<b>%{label}</b> %root : (%{percentRoot:.2%f}) \n n_abs : (%{value:.2%f})',
    insidetextorientation='auto'
)

lst_ids = fig._data[0]['ids']
k=0
for i in lst_ids:    
      if i.endswith('Female') :
          fig._data[0]['marker']['colors'][k]='#FDD3D3'
      elif i.endswith('Male') :
          fig._data[0]['marker']['colors'][k]='#272B66'
      elif i.endswith('/Athletic') :
          fig._data[0]['marker']['colors'][k]='#F75150'
      elif i.endswith('/Bobsleigh') :
          fig._data[0]['marker']['colors'][k]='#9CA65F'
      elif i.endswith('/Rugby') :
          fig._data[0]['marker']['colors'][k]='#F3BE22'
      elif i.endswith('specified') :
          fig._data[0]['marker']['colors'][k]='#CCCED0'
      k=k+1
fig.update_layout(uniformtext=dict(minsize=20, mode='show'))

fig.write_image("sunburstPopulation.pdf",format='pdf',width=1200, height=1200)

