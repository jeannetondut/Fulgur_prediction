#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 08:56:37 2022

@author: tj103659


HOW TO USE IT : 
    
    import fonctions_definition as fulgur 
    nom_fichier = list(['fulgur-bob-21sept22.csv','fulgur-ffr-22sept22.csv','fulgur-ffa-21sept22.csv'])
    # Tri des données par fédération
    X_bob, Tab_bob,lst_code_bob,sortie_bob = fulgur.MAIN(nom_fichier[0])
    X_ffr, Tab_ffr,lst_code_ffr,sortie_ffr = fulgur.MAIN(nom_fichier[1])
    X_ffa, Tab_ffa,lst_code_ffa,sortie_ffa = fulgur.MAIN(nom_fichier[2])
    # Réunir l'ensemble des données dans une seule matrice  
    X,Y = fulgur.concat_X(X_bob,X_ffr,X_ffa,sortie_bob,sortie_ffr,sortie_ffa)

"""
 
 
import numpy as np
import pandas as pd
import math
import datetime
from datetime import datetime
 
"""
FONCTIONS
"""


def replace_txt_num (Dataframe):
    Dataframe = Dataframe.replace('Aucune douleur',0)
    Dataframe = Dataframe.replace('Douleurs modérées',50)
    Dataframe = Dataframe.replace('Toute la journée',100)
    Dataframe = Dataframe.replace('Jamais',0)
    Dataframe = Dataframe.replace('Pas fatigué du tout',0)
    Dataframe = Dataframe.replace('Oui, de façon parfaite',100)
    Dataframe = Dataframe.replace('Non, très mal',0)
    Dataframe = Dataframe.replace('Très confiant',100)
    Dataframe = Dataframe.replace('Très préoccupé',100)
    Dataframe = Dataframe.replace('Totalement fatigué',100)
    Dataframe = Dataframe.replace('Douleurs insupportables',100)
    Dataframe = Dataframe.replace('Je ne sais pas',0)
    Dataframe = Dataframe.replace("Pas du tout d'accord",0)
    Dataframe = Dataframe.replace('Pas préoccupé',0)
    Dataframe = Dataframe.replace('Pas tendu',0)
    Dataframe = Dataframe.replace('Très tendu',100)
    Dataframe = Dataframe.replace('Très mauvaises',0)
    Dataframe = Dataframe.replace('Pas du tout sûr',0)
    Dataframe = Dataframe.replace('Pas du tout',0)
    Dataframe = Dataframe.replace('Pas confiant',0)
    Dataframe = Dataframe.replace('NON, aucune blessure ni problème physique',0)
    Dataframe = Dataframe.replace("OUI, mais aucune participation possible à l'entraînement/compétition",1)
    Dataframe = Dataframe.replace("OUI, mais participation complète à l'entraînement/compétition",0.3)
    Dataframe = Dataframe.replace("OUI, mais participation réduite à l'entraînement/compétition",0.6)
    Dataframe = Dataframe.replace('Effort max',100)
    Dataframe = Dataframe.replace('Non',0)
    Dataframe = Dataframe.replace('Oui',1)
    Dataframe = Dataframe.replace('Pas du tout difficile',0)  
   
    return Dataframe

def export_AMS(nom_fichier,rugby=0):
 monitoring = pd.read_csv(nom_fichier,sep=',')
 #monitoring = monitoring.drop_duplicates()
 lst = list()
 for column in monitoring : 
    lst.append(column)
    
 lst2 = pd.Series.tolist(monitoring['Date'])
 
 monitoring = monitoring.set_axis(lst2, axis = 0)    
 monitoring.index = pd.to_datetime(monitoring.index,dayfirst=True)
 #monitoring = monitoring.drop(['Date','About','by','Nom'],axis=1)
 monitoring = replace_txt_num(monitoring)
 
 # Keep few parameters 
 Monitoring_evening = monitoring.loc[monitoring['Moment']== 'Le soir']
 Monitoring_morning = monitoring.loc[monitoring['Moment']== 'Le matin']
 Monitoring_training = monitoring.loc[monitoring['Moment']== "Après l'entraînement"]
 Monitoring_hebdo = monitoring.loc[monitoring['Moment']== "Questionnaire de fin de semaine"]

## Choose which parameters you want to keep for the model 
 lst_soir = ['Code FULGUR','Douleurs soir','Fatigue soir','Tendresse','Joie','Peur','Colère','Honte','Tristesse','Blessure','Stress']
 lst_matin = ['Code FULGUR','Qualité','Fatigue','Urines','Douleurs matin','Préoccupations','Tensions','Confiance']
 lst_training = ['Code FULGUR','Quelle a été la durée de cette séance ?',"Quelle a été l'intensité de cette séance ?"]#,'Type séance','Boisson sucrée']
 lst_blessure = ['Code FULGUR', 'Blessure']

 Monitoring_evening = Monitoring_evening[lst_soir]
 Monitoring_morning = Monitoring_morning[lst_matin]
 Monitoring_training = Monitoring_training[lst_training]
# Trier les données et remplacer le txt par 100 ou 0 
# Blessure = pd.concat([monitoring['Code FULGUR'],monitoring['Blessure']],axis=1)
 Monitoring_evening = Monitoring_evening.drop_duplicates()
 Blessure = Monitoring_evening[lst_blessure]
 
 Monitoring_morning = Monitoring_morning.drop_duplicates()
 Monitoring_training = Monitoring_training.drop_duplicates()
 Monitoring_evening = Monitoring_evening.drop(columns = 'Blessure')
 date_soir =  pd.to_datetime(Monitoring_evening.index,dayfirst=True) + pd.Timedelta(days=1)
 Monitoring_evening.index = pd.to_datetime(date_soir,dayfirst=True)

 return Monitoring_evening,Monitoring_morning,Monitoring_training,Monitoring_hebdo,Blessure

def blessure_export(Blessure):
    date = Blessure.index
    date = pd.to_datetime(date,dayfirst=True)
    Blessure['Date'] = date
    V = Blessure.drop_duplicates()
    Y = V.replace("OUI, mais participation complète à l'entraînement/compétition",0.33)
    Y = Y.replace("OUI, mais participation réduite à l'entraînement/compétition",0.66)
    Y = Y.replace("OUI, mais aucune participation possible à l'entraînement/compétition",1)
    
    lst_code = Y['Code FULGUR']
    lst_code = lst_code.drop_duplicates()
    Yblessure = pd.DataFrame()
    for code in range(np.size(lst_code)) : 
        K = Y.loc[Y['Code FULGUR']== lst_code[code]]
        K = K.sort_values('Date')
        for j in range(np.size(K,axis=0)-1,0,-1) : 
            # Si 2 blessures identiques et consécutives on retire la blessure
            if (K['Blessure'][j]== K['Blessure'][j-1]) & (K['Blessure'][j]!=0) & ((K['Date'][j]-K['Date'][j-1])==pd.Timedelta("1 days") ) : 
                K['Blessure'][j]=0
        Yblessure = Yblessure.append(K)
                
    return Yblessure

def path_to_data(Code_athlete,Monitoring_evening,Monitoring_hebdo,Monitoring_morning,Monitoring_training):
 Evening_code = Monitoring_evening.loc[Monitoring_evening['Code FULGUR']== Code_athlete]
 Hebdo_code = Monitoring_hebdo.loc[Monitoring_hebdo['Code FULGUR']== Code_athlete]
 Morning_code = Monitoring_morning.loc[Monitoring_morning['Code FULGUR']== Code_athlete]
 Training_code = Monitoring_training.loc[Monitoring_training['Code FULGUR']== Code_athlete]
 return Evening_code,Hebdo_code,Morning_code,Training_code

def reindex_par_date(dataframe_name,list_param):
    lst = pd.Series.tolist(dataframe_name['date'])
    lst2 = list()
    for i in range(np.size(lst)):
        lst2.append(lst[i][:10])

    dataframe_1 = dataframe_name.set_axis(lst2, axis = 0)
    dataframe_new = dataframe_1.drop(list_param,axis=1)
    return dataframe_new

def Creation_Matrice(evening,morning,training=None):
 
 morning = morning.drop(['Code FULGUR'],axis=1)
 morning = morning[~morning.index.duplicated(keep='first')]

 evening = evening.drop(['Code FULGUR'],axis=1)
 evening = evening[~evening.index.duplicated(keep='first')]

# training = training.drop(['Code FULGUR'],axis=1)
 MatriceTT = pd.concat([morning,evening],axis=1)

 #MatriceTT = morning.join(evening)
 #if training !=None :
 #    MatriceTT = MatriceTT.join(training)
 
 return MatriceTT

def serie_7j(MatriceTT) :
 V = MatriceTT.loc[~MatriceTT.index.duplicated(), :]
 # Supprimer les lignes où il y a une discontinuité des données
 MatriceTT2 = V.dropna(axis=0)
 # Convert index into datetime
 MatriceTT2.index = pd.to_datetime(MatriceTT2.index,dayfirst=True)
 #Ranger par date chronologique
 MatriceTT2 = MatriceTT2.sort_index(axis=0,ascending=True)
 example = pd.DataFrame()
 start_semaine=list()

 for i in range(np.size(MatriceTT2,axis=0)-6):
        day1 = MatriceTT2.index[i]
        #day1 = datetime.strptime(day1, '%d-%m-%Y').date()
        day7 = MatriceTT2.index[i+6]
       # day7 = datetime.strptime(day7, '%d-%m-%Y').date()
        Delt = pd.Timedelta(day7-day1)
        
        if Delt == pd.Timedelta("7 days") :
            start_semaine.append(MatriceTT2.index[i])
            semaine = pd.DataFrame()
            for k in range(7):
                jour = MatriceTT2.iloc[i+k]
                semaine = pd.concat([semaine, jour])
            semaine = semaine.transpose()
            example = example.append([semaine])
            #example.iloc[l] = example.set_(day1,axis=0)

 #example.set_index(start_semaine,inplace=True,)
 return example, start_semaine

def interpolation_donnees_manquantes(D):
 nb_param = np.size(D,axis=1)
 nb_jour = np.size(D,axis=0)
 for i in range(nb_param) :
    for j in range(1,nb_jour-2):
        if math.isnan(D.iloc[j,i]) == True :
            if math.isnan(D.iloc[j+1,i]) == True :
                D.iloc[j,i] = (D.iloc[j+2,i] + D.iloc[j-1,i]) / 2
                D.iloc[j+1,i] = (D.iloc[j+2,i] + D.iloc[j,i]) / 2
            else :
                D.iloc[j,i] = (D.iloc[j+1,i] + D.iloc[j-1,i]) / 2
 return D

def serie_5j(MatriceTT) :
 V = MatriceTT.loc[~MatriceTT.index.duplicated(), :]
 # Supprimer les lignes où il y a une discontinuité des données
 MatriceTT2 = V.dropna(axis=0)
 # Convert index into datetime
 MatriceTT2.index = pd.to_datetime(MatriceTT2.index,dayfirst=True)
 #Ranger par date chronologique
 MatriceTT2 = MatriceTT2.sort_index(axis=0,ascending=True)
 example = pd.DataFrame()
 start_semaine=list()

 for i in range(np.size(MatriceTT2,axis=0)-4):
        day1 = MatriceTT2.index[i]
        #day1 = datetime.strptime(day1, '%d-%m-%Y').date()
        day7 = MatriceTT2.index[i+4]
       # day7 = datetime.strptime(day7, '%d-%m-%Y').date()
        Delt = pd.Timedelta(day7-day1)
        
        if Delt == pd.Timedelta("5 days") :
            start_semaine.append(MatriceTT2.index[i])
            semaine = pd.DataFrame()
            for k in range(5):
                jour = MatriceTT2.iloc[i+k]
                semaine = pd.concat([semaine, jour])
            semaine = semaine.transpose()
            example = example.append([semaine])
            #example.iloc[l] = example.set_(day1,axis=0)

 #example.set_index(start_semaine,inplace=True,)
 return example, start_semaine

def serie_3j(MatriceTT) :
 V = MatriceTT.loc[~MatriceTT.index.duplicated(), :]
 # Supprimer les lignes où il y a une discontinuité des données
 MatriceTT2 = V.dropna(axis=0)
 # Convert index into datetime
 MatriceTT2.index = pd.to_datetime(MatriceTT2.index,dayfirst=True)
 #Ranger par date chronologique
 MatriceTT2 = MatriceTT2.sort_index(axis=0,ascending=True)
 example = pd.DataFrame()
 start_semaine=list()

 for i in range(np.size(MatriceTT2,axis=0)-3):
        day1 = MatriceTT2.index[i]
        #day1 = datetime.strptime(day1, '%d-%m-%Y').date()
        day7 = MatriceTT2.index[i+2]
       # day7 = datetime.strptime(day7, '%d-%m-%Y').date()
        Delt = pd.Timedelta(day7-day1)
        
        if Delt == pd.Timedelta("2 days") :
            start_semaine.append(MatriceTT2.index[i])
            semaine = pd.DataFrame()
            for k in range(3):
                jour = MatriceTT2.iloc[i+k]
                semaine = pd.concat([semaine, jour])
            semaine = semaine.transpose()
            example = example.append([semaine])
            #example.iloc[l] = example.set_(day1,axis=0)

 #example.set_index(start_semaine,inplace=True,)
 return example, start_semaine

def serie_1j(MatriceTT) :
 V = MatriceTT.loc[~MatriceTT.index.duplicated(), :]
 # Supprimer les lignes où il y a une discontinuité des données
 MatriceTT2 = V.dropna(axis=0)
 # Convert index into datetime
 MatriceTT2.index = pd.to_datetime(MatriceTT2.index,dayfirst=True)
 #Ranger par date chronologique
 MatriceTT2 = MatriceTT2.sort_index(axis=0,ascending=True)
 example = MatriceTT2
 start_semaine= MatriceTT2.index
 return example, start_semaine





def blessure_info(XX,Yblessure,start,code):
    # Vecteur de sortie : blessure=1 ou pas blessure=0
    
     y_f_semaine = pd.DataFrame(columns = ['Blessure'] ,index=start)
     Yblessure.index = pd.to_datetime(Yblessure.index,dayfirst=True)
     A = Yblessure.loc[Yblessure['Code FULGUR']==code]
     fin_semaine = A.index - pd.Timedelta(days=6)
     B = A.set_index(fin_semaine)
     B = B['Blessure']
    # Bles = pd.concat([XX, B],axis=1)
     Bles = XX.join(B)
     X = Bles.dropna()
     y_f_semaine = X['Blessure']
     X['CODE'] = code 
     return y_f_semaine,X

def blessure_info_jour_suivant(XX,Yblessure,start,code):
    # Vecteur de sortie : blessure=1 ou pas blessure=0
    
     y_f_semaine = pd.DataFrame(columns = ['Blessure'] ,index=start)
     Yblessure.index = pd.to_datetime(Yblessure.index,dayfirst=True)
     A = Yblessure.loc[Yblessure['Code FULGUR']==code]
     fin_semaine = A.index - pd.Timedelta(days=5)
     B = A.set_index(fin_semaine)
     B = B['Blessure']
    # Bles = pd.concat([XX, B],axis=1)
     Bles = XX.join(B)
     X = Bles.dropna()
     y_f_semaine = X['Blessure']
     X['CODE'] = code 
     return y_f_semaine,X


 


def create_example_code (Code_athlete,Monitoring_evening,Monitoring_hebdo,Monitoring_morning,Monitoring_training,iD):
    Evening_code,Hebdo_code,Morning_code,Training_code = path_to_data(Code_athlete,Monitoring_evening,Monitoring_hebdo,Monitoring_morning,Monitoring_training)
    # Sensations du Matin
    M = Morning_code.drop_duplicates()
    # Sensations du soir
    E = Evening_code.drop_duplicates()
    # Concatenation des sensations du matin et du soir 
    D = Creation_Matrice(M, E)
    D['id'] = iD
    # Série de plusieurs jours consécutifs: au choix serie_7j et serie_6j 
    #example_code , start = serie_3j(D)
    #example_code , start = serie_5j(D)
    # example_code , start = serie_7j(D)
    example_code , start = serie_1j(D)
    
    example_code.set_axis(start,inplace=True,axis=0)
   

    return example_code,start



def export_codes (Monitoring_evening,Monitoring_morning,training=0,Monitoring_training=None):
    code_evening = Monitoring_evening['Code FULGUR']
    code_morning = Monitoring_morning['Code FULGUR']   
    code_evening = code_evening.loc[~code_evening.duplicated()]
    code_morning = code_morning.loc[~code_morning.duplicated()]   
    code_evening = code_evening.set_axis(code_evening, axis = 0)
    code_morning = code_morning.set_axis(code_morning, axis = 0)
    V = pd.concat([code_morning,code_evening])
    if training ==1 : 
        code_training = Monitoring_training['Code FULGUR']    
        code_training = code_training.loc[~code_training.duplicated()]
        code_training = code_training.set_axis(code_training, axis = 0)
        V = pd.concat([code_morning,code_evening,code_training])
    V = V.loc[~V.duplicated()]
    V = V.dropna()       
      
    return V

def create_example_general(lst_code,Blessure,Monitoring_evening,Monitoring_hebdo,Monitoring_morning,Monitoring_training): 
    example = pd.DataFrame()
    sortie = pd.DataFrame()
    for i in range(np.size(lst_code,axis=0)):
        code = lst_code.iloc[i]['Code FULGUR']
        iD = lst_code.iloc[i]['id']
        # Créer matrice de 7 jours consécutifs 
        example1, start_semaine = create_example_code(code,Monitoring_evening,Monitoring_hebdo,Monitoring_morning,Monitoring_training,iD)
        # Associer une blessure à la fin de semaine 
    
        if np.size(example1,axis=0) != 0:
            # Blessure le jour suivant :
            Y_it,X = blessure_info_jour_suivant(example1,Blessure,start_semaine,code)
            # Blessure le jour J :
            # Y_it,X = blessure_info(example1,Blessure,start_semaine,code)
            example = pd.concat([example , X],axis=0)
            sortie = pd.concat([sortie,Y_it])
            
    return example, sortie

def calcul_taux_reponse(Monitoring,lst_code,training =0,Monitoring_evening=None):

    Dataframe = Monitoring.drop_duplicates()
    taux = pd.DataFrame(lst_code)
    taux['taux de réponse'] = ''
    for i in range(np.size(lst_code)) : 
        athlete = Dataframe.loc[Dataframe['Code FULGUR'] == lst_code[i]]
        if np.size(athlete,axis=0)==0:
            taux.loc[lst_code[i]]['taux de réponse']= 0
        elif np.size(athlete,axis=0)==1:
            taux.loc[lst_code[i]]['taux de réponse']= 0
        else :
         jour_fin = pd.to_datetime(athlete.index[0],dayfirst=True)
         jour_debut = pd.to_datetime(athlete.index[np.size(athlete,axis=0)-1],dayfirst=True)
         if training==1:
             athlete_nbseance = Monitoring_evening.loc[Monitoring_evening['Code FULGUR'] == lst_code[i]]
             nb_seance = Monitoring_evening["Combien de séances d'entrainement as-tu eu aujourd'hui ?"].sum(skipna=True)
             NBjour = nb_seance
         else :
             nbtotaljour = pd.Timedelta(jour_fin - jour_debut)
             NBjour = nbtotaljour.days
         taux_athlete = (np.size(athlete,axis=0) - 1 ) / NBjour
         taux.loc[lst_code[i]]['taux de réponse']= taux_athlete
    taux = pd.to_numeric(taux['taux de réponse'])
    moyenne_taux = np.mean(taux)
    std_taux = np.std(taux)
    return taux, moyenne_taux,std_taux

def separation_training_test_set(Xtotal,Ytotal,taux):
  coupure =np.round(taux*np.size(Xtotal,axis=0),decimals=0)
  coupure = int(coupure)
  end = np.size(Xtotal,axis=0)
  X_training = Xtotal.iloc[0:coupure, : ]
  X_test = Xtotal.iloc[coupure+1:end, : ]
  Y_training = Ytotal.iloc[0:coupure, : ]
  Y_test = Ytotal.iloc[coupure+1:end, : ]

  return X_training, Y_training , X_test, Y_test

def MAIN(name):
    Monitoring_evening,Monitoring_morning,Monitoring_training,Monitoring_hebdo,Blessure = export_AMS(name)
    Blessure = blessure_export(Blessure)
    Monitoring_morning = calcul_anxiete(Monitoring_morning)
    lst_code = export_codes(Monitoring_evening,Monitoring_morning,training=0,Monitoring_training=None)
    kk = np.arange(np.size(lst_code))
    lst_code = pd.DataFrame(lst_code) 
    lst_code['id'] = kk       
    example,sortie = create_example_general(lst_code,Blessure,Monitoring_evening,Monitoring_hebdo,Monitoring_morning,Monitoring_training)
    exemple = example.drop(columns=['Blessure'])
    
    #exemple = example
    return exemple,example,lst_code,sortie

def concat_X(X1,X2,X3,Y1,Y2,Y3):

    X = pd.concat([ X1, X2, X3],axis=0)
    Y = pd.concat([ Y1, Y2, Y3],axis=0)
    return X,Y

def exposition(X,Y):
    # The number of injury is a positive integer that can be modeled as a Poisson distribution. 
    # It is then assumed to be the number of discrete events occurring with a constant rate in a given time interval 
    # (Exposure, in units of years).
    # ie. le nombre de blessure / athlete sur une durée de monitoring
   
    a =  list( X.index )  
    b =  pd.Series.tolist(X['CODE'])
    Y['CODE']=b
    Y = multi_index_code_date(Y)
    
    c = [ a , b]
    tuples = list(zip(*c))
    
    index = pd.MultiIndex.from_tuples(tuples ,
                                       names=['Date','Code'])
    """
    Xx = X.reset_index()
    Yy = Y.reset_index()
    Xx= Xx.drop(columns='index')
    Yy= Yy.drop(columns='index')
    columns_name= X.columns
    X2 = pd.DataFrame(pd.DataFrame.to_numpy(Xx),index=index,columns=columns_name)
    X2 = X2.drop(columns='CODE')
    
    Y2 = pd.DataFrame(pd.DataFrame.to_numpy(Yy),index=index,columns=['Blessure'])
    """
    X = multi_index_code_date(X)
    
    db = Y.set_index ( X.index )
    df = X.set_index ( X.index)
    dx = df #.drop(columns=['Urines','CODE','Peur','Tristesse','Colère','Fatigue soir','Tendresse','Qualité','Fatigue','Confiance','Honte'])
    dx['Blessure'] =  db
   
    lst_code = b
    lst_code = list(set(lst_code))
    exposure = pd.DataFrame(np.zeros([np.size(df,axis=0)]),index=index,columns=list('E'))
    
    for i in range(np.size(lst_code)): 
        count_inj=0
        
        F = dx.iloc[dx.index.get_level_values('Code') == lst_code[i]]
        
        if type(F) != type(dx) : 
            count = 0 
            exposure.iloc[exposure.index.get_level_values('Code') == lst_code[i]] = 0.01
        else :
            count = np.size(F,axis=0)
            blessure = F['Blessure']
            for j in range(count): 
                if blessure[j]!=0 :
                    count_inj = count_inj + 1
            if count_inj == 0 :
                exposure.iloc[exposure.index.get_level_values('Code') == lst_code[i]]  = 0.01
            else :
                exposure.iloc[exposure.index.get_level_values('Code') == lst_code[i]] = count_inj/count
    
   
   
    dx['Exposition'] = exposure
    return exposure, dx, Y



def get_psycho_data():
   
     
    import urllib.request, json 
    import pandas as pd
    import numpy as np
    
    with urllib.request.urlopen("https://fulgur.infoshn.fr/services.ashx/listeTestAvecDonnees?cleAPI=SEP2021INSEPFulgur&test=Bilan%20Psychologique") as url:
        data = json.loads(url.read().decode())
        results = data['resultats']
        df = pd.DataFrame.from_dict(results)
        C = df['donnees']
        
    fulgur_psqs = pd.read_csv('fulgur_psqs2.csv',sep=';')
    fulgur_psqs = fulgur_psqs.dropna()
    fulgur_psqs['psqs'] = fulgur_psqs['psqs'].apply(int)
    
    lst_code_psqs = pd.Series.tolist(fulgur_psqs['psqs'])
    
    fulgur_psqs = fulgur_psqs.set_axis(lst_code_psqs, axis = 0)    
    #fulgur_psqs = fulgur_psqs.drop(columns=['CODE FULGUR'])
        
        
        # Get the value of the anxiete score
    def get_scores(df,C):
       lst_athlete = df['sportif']
       Score_psy = pd.DataFrame(columns = ['Anxiété','Determ','Efficacité','Optimisme'],index=lst_athlete)
       for i in range(np.size(lst_athlete)):
           D = C[i]
           E = json.loads(D)
           a = (E['scores']['anxiete']['value'])
           d = (E['scores']['determ']['value'])
           e = (E['scores']['efficacite']['value']) 
           o = (E['scores']['optimisme']['value'])
    
           Score_psy.loc[lst_athlete[i]] = [a, d , e , o]
            
       return Score_psy
    
    
    
    Score_psy = get_scores(df, C)
    # Associer code fulgur au résultat du test psy 
    Score_psy['Code FULGUR']=  fulgur_psqs['Code']
    return Score_psy 


def add_baseline(X,Y,Score_psy,exposure=0):
    
    db = Y.set_index ( X['CODE'])
    lst_code_total = list(X['CODE'])
    lst_code_total = list(set(lst_code_total))
    df = X.set_index ( X['CODE'])
    dx = df.drop(columns=['Urines','CODE','Peur','Tristesse','Colère','Fatigue soir','Tendresse','Qualité','Fatigue','Confiance','Honte'])
    dx['Blessure'] =  db
    if type(exposure)== type(pd.DataFrame()) : 
        dx['Exposition'] =  exposure['E']
        
    X2 = pd.DataFrame()

    for i in range(np.size(lst_code_total)): 
    
        # A  = X.loc[X['CODE']== lst_code_total[i]]
        A = dx.loc[lst_code_total[i]]
        if type(A) == type(pd.Series()) : 
            A = dx.loc[[lst_code_total[i]],:]
            
        Baseline = Score_psy.loc[Score_psy['Code FULGUR']== lst_code_total[i]]
        if np.size(Baseline,axis=0)!=0:
            a = np.full([np.size(A,axis=0),1],Baseline['Anxiété'])
            A['Anxiété'] = a
            
            d =  np.full([np.size(A,axis=0),1],Baseline['Determ'])
            A['Determ'] = d
    
            e =  np.full([np.size(A,axis=0),1],Baseline['Efficacité']) 
            A['Efficacité'] = e
    
            o = np.full([np.size(A,axis=0),1],Baseline['Optimisme']) 
            A['Optimisme'] = o
    
            X2 = pd.concat([X2 , A],axis=0)
         
    return X2



def predict_injury(X2,exposure=0,alpha=1e-10, max_iter=5000):
   
    from sklearn.linear_model import PoissonRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    
    # Split DATA into test and training data set
    X_training, X_test, Y_training,Y_test = train_test_split(X2.drop(columns='Blessure'),X2['Blessure'],test_size=0.3,shuffle=True,random_state=0)
    
    # Build the model 
    if type(exposure)== type(0) :
        model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
        model.fit(X_training, Y_training)
    else :
        model = PoissonRegressor(alpha=alpha, max_iter=max_iter)
        model.fit(X_training, Y_training, sample_weight=X_training["Exposition"])
    
    # Compute prediction 
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred)
    Y_pred = Y_pred.set_index ( X_test.index )
    Y_pred = Y_pred.rename(columns={0:'Predicted'})
    
    #lst_code_test = list(X_test.index.values) 

    #Prediction = pd.DataFrame(Y_pred,index=lst_code_test,columns=['Predict'])
     
    Y_pred['True value'] = Y_test

    bins = [0 , 0.2, 0.5, 0.75 , 1.15]

    counts = plt.hist( [Y_pred['Predicted'],Y_pred['True value']] , bins=bins)
    plt.legend(['Predict','Test data'])
    plt.xlabel('Risque de blessure')
    plt.ylabel('Nb Occurence')
    if type(exposure)== type(0):
        plt.title('Modèle : Douleurs soir/Joie/Douleurs matin/Préoccupations/ Baseline psy')
    else :
        plt.title('Modèle : Douleurs soir/Joie/Douleurs matin/Préoccupations/ Baseline psy / avec exposition')
    plt.grid()
    #plt.savefig('Histo_predict_avec_sans_exposition.pdf')
    r2 = r2_score(Y_pred['True value'], Y_pred['Predicted'])
    print('R2 score sans expo', r2_score(Y_pred['True value'], Y_pred['Predicted']))
    
    scores = cross_val_score(model, X2.drop(columns='Blessure'),X2['Blessure'], cv=5) 
    
    return counts,r2,scores, Y_pred



def concat_morning_evening_journalier(nom_fichier,lst_code):
    # Donne une matrice de taille (Nxp) avec N le nombre de jours monitorés et p le nombre de parametre suivi par jour
 Monitoring_evening,Monitoring_morning,Monitoring_training,Monitoring_hebdo,Blessure = export_AMS(nom_fichier)
 Blessure = blessure_export(Blessure)
 W = pd.DataFrame()
 for i in range(np.size(lst_code)): 
    A = Monitoring_evening.loc[Monitoring_evening['Code FULGUR']== lst_code[i]]
    M = Monitoring_morning.loc[Monitoring_morning['Code FULGUR']== lst_code[i]]
    A = A[~A.index.duplicated()]
    A = A.drop_duplicates(keep='first')
    M = M.drop_duplicates()
    M = M[~M.index.duplicated()]
    B = Blessure.loc[Blessure['Code FULGUR']== lst_code[i]]
    B = B.drop_duplicates()
    B = B[~B.index.duplicated()]    
    A['Fatigue matin'] = M['Fatigue']
    A['Qualité'] = M['Qualité']
    A['Préoccupations'] = M['Préoccupations']
    A['Douleurs matin'] = M['Douleurs matin']
    A['Tensions'] = M['Tensions']
    A['Confiance'] = M['Confiance']
    A['Blessure'] = B['Blessure']
    A = A.dropna()
    W = pd.concat([W , A], axis = 0 )
 return W   



def calcul_anxiete(X):
    anxiete= np.zeros([np.size(X,axis=0)])
    
    for i in range(np.size(X,axis=0)): 
        A = X.iloc[i,:]
        tension = float(A['Tensions'])
        confiance = float(A['Confiance'])
        preoccupation = float(A['Préoccupations'])
        anxiete[i] = (tension+preoccupation)/2 - confiance
    X = X.drop(columns=['Tensions','Confiance','Préoccupations'])
    X['Anxiete'] = anxiete
    return X


def param_to_remove(X,lst_param_a_retirer): 
    X = X.drop(columns=lst_param_a_retirer)
    return X



def predict_injury_lars(X2,exposure=0):
   
    from sklearn import linear_model
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    
    # Split DATA into test and training data set
    X_training, X_test, Y_training,Y_test = train_test_split(X2.drop(columns='Blessure'),X2['Blessure'],test_size=0.3,shuffle=True,random_state=0)
    
    # Build the model 
    #if type(exposure)== type(0) :
    model = linear_model.LassoLars(alpha=0.1, normalize=True)
    model.fit(X_training, Y_training)
    #else :
    #    model = linear_model.Lars(n_nonzero_coefs=500, normalize=False)
    #    model.fit(X_training, Y_training, sample_weight=X_training["Exposition"])
    
    # Compute prediction 
    Y_pred = model.predict(X_test)
    
    lst_code_test = list(X_test.index.values) 

    Prediction = pd.DataFrame(Y_pred,index=lst_code_test,columns=['Predict'])
     
    Prediction['Vraie valeur'] = Y_test

    bins = [0 , 0.2, 0.5, 0.75 , 1.15]

    counts = plt.hist( [Prediction['Predict'],Prediction['Vraie valeur']] , bins=bins)
    plt.legend(['Predict','Test data'])
    plt.xlabel('Risque de blessure')
    plt.ylabel('Nb Occurence')
    if type(exposure)== type(0):
        plt.title('Modèle : Douleurs soir/Joie/Douleurs matin/Préoccupations/ Baseline psy')
    else :
        plt.title('Modèle : Douleurs soir/Joie/Douleurs matin/Préoccupations/ Baseline psy / avec exposition')
    plt.grid()
    #plt.savefig('Histo_predict_avec_sans_exposition.pdf')
    
    
    if type(exposure)== type(0) :
        score_determ = model.score(X_test,Y_test)   
    else :
        score_determ = model.score(X_test,Y_test,sample_weight=X_test["Exposition"])   
        
    r2 = r2_score(Prediction['Vraie valeur'], Prediction['Predict'])
    print('R2 score sans expo', r2_score(Prediction['Vraie valeur'], Prediction['Predict']))
    scores = cross_val_score(model, X_test, Y_test, cv=5) 
    
    return counts,r2,scores, Y_pred, score_determ


def multi_index_code_date(X):
    a =  list( X.index )  
    b =  pd.Series.tolist(X['CODE'])
    c = [ a , b]
    tuples = list(zip(*c))
    
    index = pd.MultiIndex.from_tuples(tuples ,
                                       names=['Date','Code'])
    Xx = X.reset_index()
    Xx= Xx.drop(columns='index')
    columns_name= X.columns
    X2 = pd.DataFrame(pd.DataFrame.to_numpy(Xx),index=index,columns=columns_name)
    X2 = X2.drop(columns='CODE')
    return X2





def predict_injury_logistic(X2,exposure=0):
   
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    
    # Split DATA into test and training data set
    X_training, X_test, Y_training,Y_test = train_test_split(X2.drop(columns=['Blessure','CODE']),X2['Blessure'],test_size=0.3,shuffle=True,random_state=0)
    Y_training = Y_training.astype('int')
    # Build the model 
    if type(exposure)== type(0) :
        model = LogisticRegression(multi_class="multinomial", solver="lbfgs",max_iter=5000)
        model.fit(X_training, Y_training)
    else :
        model = LogisticRegression(multi_class="multinomial", solver="lbfgs",max_iter=5000)
        model.fit(X_training, Y_training, sample_weight=X_training["Exposition"])
    
    # Compute prediction 
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred)
    Y_pred = Y_pred.set_index ( X_test.index )
    Y_pred = Y_pred.rename(columns={0:'Predicted'})
    
    # USe predict proba
    Y_pred = model.predict_proba(X_test)
    #lst_code_test = list(X_test.index.values) 

    #Prediction = pd.DataFrame(Y_pred,index=lst_code_test,columns=['Predict'])
     
    Y_pred['True value'] = Y_test

    bins = [0 , 0.2, 0.5, 0.75 , 1.15]

    counts = plt.hist( [Y_pred['Predicted'],Y_pred['True value']] , bins=bins)
    plt.legend(['Predict','Test data'])
    plt.xlabel('Risque de blessure')
    plt.ylabel('Nb Occurence')
    if type(exposure)== type(0):
        plt.title('Modèle : Douleurs soir/Joie/Douleurs matin/Préoccupations/ Baseline psy')
    else :
        plt.title('Modèle : Douleurs soir/Joie/Douleurs matin/Préoccupations/ Baseline psy / avec exposition')
    plt.grid()
    #plt.savefig('Histo_predict_avec_sans_exposition.pdf')
    r2 = r2_score(Y_pred['True value'], Y_pred['Predicted'])
    print('R2 score sans expo', r2_score(Y_pred['True value'], Y_pred['Predicted']))
    
    scores = cross_val_score(model, X2.drop(columns='Blessure'),X2['Blessure'], cv=5) 
    
    return counts,r2,scores, Y_pred


def predict_logistic_CV(X2,Y,scoring):
   
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
    from sklearn.metrics import r2_score, recall_score
    import matplotlib.pyplot as plt
    
    # Split DATA into test and training data set
    X_training, X_test, Y_training,Y_test = train_test_split(X2.drop(columns=['CODE']),Y,test_size=0.3,shuffle=True,random_state=0)
    Y_training = Y_training.astype('int')
    # Build the model 
    scoring = ['precision_macro', 'recall_macro']
    model = LogisticRegression(multi_class="multinomial", solver="lbfgs",max_iter=5000,class_weight='balanced')
    scores = cross_validate(model, X_training, Y_training, scoring=scoring,cv=5)
    
    model.fit(X_training, Y_training)
    
    # Compute prediction 
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred)
    Y_pred = Y_pred.set_index ( X_test.index )
    Y_pred = Y_pred.rename(columns={0:'Predicted'})

    Y_pred = model.predict_proba(X_test)

    Y_pred['True value'] = Y_test


    scores = cross_val_score(model, X2.drop(columns='Blessure'),X2['Blessure'], cv=5) 
    
    return scores, Y_pred


def predict_injury_deepL(X2,exposure=0):
   
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    
    # Split DATA into test and training data set
    X_training, X_test, Y_training,Y_test = train_test_split(X2.drop(columns=['Blessure','CODE']),X2['Blessure'],test_size=0.3,shuffle=True,random_state=0)
    Y_training = Y_training.astype('int')
    # Build the model 
    if type(exposure)== type(0) :
        model = LogisticRegression(multi_class="multinomial", solver="lbfgs",max_iter=5000)
        model.fit(X_training, Y_training)
    else :
        model = LogisticRegression(multi_class="multinomial", solver="lbfgs",max_iter=5000)
        model.fit(X_training, Y_training, sample_weight=X_training["Exposition"])
    
    # Compute prediction 
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred)
    Y_pred = Y_pred.set_index ( X_test.index )
    Y_pred = Y_pred.rename(columns={0:'Predicted'})
    
    # USe predict proba
    Y_pred = model.predict_proba(X_test)
    #lst_code_test = list(X_test.index.values) 

    #Prediction = pd.DataFrame(Y_pred,index=lst_code_test,columns=['Predict'])
     
    Y_pred['True value'] = Y_test

    bins = [0 , 0.2, 0.5, 0.75 , 1.15]

    counts = plt.hist( [Y_pred['Predicted'],Y_pred['True value']] , bins=bins)
    plt.legend(['Predict','Test data'])
    plt.xlabel('Risque de blessure')
    plt.ylabel('Nb Occurence')
    if type(exposure)== type(0):
        plt.title('Modèle : Douleurs soir/Joie/Douleurs matin/Préoccupations/ Baseline psy')
    else :
        plt.title('Modèle : Douleurs soir/Joie/Douleurs matin/Préoccupations/ Baseline psy / avec exposition')
    plt.grid()
    #plt.savefig('Histo_predict_avec_sans_exposition.pdf')
    r2 = r2_score(Y_pred['True value'], Y_pred['Predicted'])
    print('R2 score sans expo', r2_score(Y_pred['True value'], Y_pred['Predicted']))
    
    scores = cross_val_score(model, X2.drop(columns='Blessure'),X2['Blessure'], cv=5) 
    
    return counts,r2,scores, Y_pred


def predict_svc(X,Y): 
    from sklearn import svm
    import allMetrics as allMetrics
    from sklearn.model_selection import train_test_split, cross_val_score
    X_training, X_test, Y_training,Y_test = train_test_split(X,Y,test_size=0.3,shuffle=True)
    Y_training = Y_training.astype('int')
    
    clf = svm.SVC(class_weight='balanced',probability=True)
    clf.fit(X_training,Y_training)
    Y_pred = clf.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred)
    Y_pred = Y_pred.set_index ( X_test.index )
    Y_pred = Y_pred.rename(columns={0:'Predicted'})

   # Y_pred = clf.predict_proba(X_test)

    Y_pred['True value'] = Y_test
    metrics = allMetrics.allMetrics(clf, X_test, Y_test)
    scores = cross_val_score(clf, X,Y, cv=5,scoring='test_precision') 
    
    return metrics, scores


def predict_svc_bagging(X,Y): 
    from sklearn import svm
    from sklearn.ensemble import BaggingClassifier
    import allMetrics as allMetrics
    from sklearn.model_selection import train_test_split, cross_val_score
    X_training, X_test, Y_training,Y_test = train_test_split(X,Y,test_size=0.3,shuffle=True)
    Y_training = Y_training.astype('int')
    
    
    clf = svm.SVC(class_weight='balanced', probability=True)
    bagging = BaggingClassifier(clf,n_estimators = 50, max_samples=500, max_features=14)
  
    bagging.fit(X_training,Y_training)
    Y_pred = bagging.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred)
    Y_pred = Y_pred.set_index ( X_test.index )
    Y_pred = Y_pred.rename(columns={0:'Predicted'})

   # Y_pred = clf.predict_proba(X_test)

    Y_pred['True value'] = Y_test
    metrics = allMetrics.allMetrics(bagging, X_test, Y_test)
    scores = cross_val_score(bagging, X,Y, cv=5,scoring='test_precision') 
    
    return metrics, scores


def SVC_gridsearch(X,Y): 
    import allMetrics as allMetrics
    from sklearn import svm
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform
    from sklearn.model_selection import train_test_split
    X_training, X_test, Y_training,Y_test = train_test_split(X,Y,test_size=0.3,shuffle=True)
    Y_training = Y_training.astype('int')
    
    clf0 = svm.SVC(class_weight='balanced', probability=True)
    
    distributions = {'C':[1,10,100],
                     'gamma':[1,0.1,0.001], 
                     'kernel':['linear','rbf','sigmoid']}
    
                     
    clf = RandomizedSearchCV(clf0, distributions, random_state=0, cv=5)
    search = clf.fit(X_training,Y_training)
    search.best_params_

    Y_pred = search.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred)
    Y_pred = Y_pred.set_index ( X_test.index )
    Y_pred = Y_pred.rename(columns={0:'Predicted'})
    Y_pred['True value'] = Y_test
    metrics = allMetrics.allMetrics(search, X_test, Y_test)
    return search, metrics


def DT_gridsearch(X,Y): 
    import allMetrics as allMetrics
    from sklearn import tree
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import uniform
    from sklearn.model_selection import train_test_split
    import imblearn
    from imblearn.over_sampling import RandomOverSampler, SMOTE

    # Splitting
    X_training, X_test, Y_training,Y_test = train_test_split(X,Y,test_size=0.3,shuffle=True)
    Y_training = Y_training.astype('int')
    # SMOTE
    oversampling = SMOTE()
    X_training,Y_training = oversampling.fit_resample(X_training, Y_training)
    
    # DT 
    #clf0 = tree.DecisionTreeClassifier(class_weight='balanced')
    clf0 = tree.DecisionTreeClassifier()
    
    distributions = {'max_depth':[2,4,6,8,10,12,14,16,18,20],
                     'max_features' : np.arange(np.size(X_training,axis=1)-1)+2,
                     'criterion' : ['gini','entropy']}
    
                     
    clf = RandomizedSearchCV(clf0, distributions, random_state=0, cv=5)
    search = clf.fit(X_training,Y_training)
    best_param = search.best_params_
    
    run = 50 
    lst_metrics = ['rocProba','tnPred','fpPred','fnpred','tpPred','accPred','recPred','spePred','prePred','fprPred']
    Metrics = pd.DataFrame(index=np.arange(run),columns=lst_metrics)
    facteur_influent = pd.DataFrame(index=np.arange(run),columns=list(X_training.columns))

    for r in range(run):
        clf_opt = tree.DecisionTreeClassifier(max_depth=best_param['max_depth'],
                                              max_features=best_param['max_features'],
                                              criterion=best_param['criterion'])
        X_training, X_test, Y_training,Y_test = train_test_split(X,Y,test_size=0.3,shuffle=True)
        Y_training = Y_training.astype('int')
        X_training,Y_training = oversampling.fit_resample(X_training, Y_training)
        c = clf_opt.fit(X_training,Y_training)
     
        metrics = allMetrics.allMetrics(c, X_test, Y_test)
      
        for k in range(np.size(lst_metrics)):
           Metrics.iloc[r][lst_metrics[k]] =  metrics[lst_metrics[k]]
           facteur_influent.iloc[r] = clf_opt.feature_importances_
    
    return best_param, Metrics, facteur_influent



def RF_gridsearch(X,Y): 
    import allMetrics as allMetrics
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV
  
    from sklearn.model_selection import train_test_split
    import imblearn
    from imblearn.over_sampling import RandomOverSampler, SMOTE

    # Splitting
    X_training, X_test, Y_training,Y_test = train_test_split(X,Y,test_size=0.3,shuffle=True)
    Y_training = Y_training.astype('int')
    # SMOTE
    oversampling = SMOTE()
    X_training,Y_training = oversampling.fit_resample(X_training, Y_training)
    
    # DT 
    #clf0 = tree.DecisionTreeClassifier(class_weight='balanced')
    clf0 = RandomForestClassifier()
    
    distributions = {'max_depth':[2,4,6,8,10,12,14,16,18,20],
                     'n_estimators':[20,40,60,80,100,120,140,160,180,200],
                     'max_features' : ['sqrt','log2'],
                     'criterion' : ['gini','entropy','log_loss']}
    
                     
    clf = RandomizedSearchCV(clf0, distributions, random_state=0, cv=5)
    search = clf.fit(X_training,Y_training)
    best_param = search.best_params_
    
    run = 50 
    lst_metrics = ['rocProba','tnPred','fpPred','fnpred','tpPred','accPred','recPred','spePred','prePred','fprPred']
    Metrics = pd.DataFrame(index=np.arange(run),columns=lst_metrics)

    for r in range(run):
        clf_opt = RandomForestClassifier(     max_depth=best_param['max_depth'],
                                              max_features=best_param['max_features'],
                                              criterion=best_param['criterion'])
        X_training, X_test, Y_training,Y_test = train_test_split(X,Y,test_size=0.3,shuffle=True)
        Y_training = Y_training.astype('int')
        X_training,Y_training = oversampling.fit_resample(X_training, Y_training)
        c = clf_opt.fit(X_training,Y_training)
     
        metrics = allMetrics.allMetrics(c, X_test, Y_test)
      
        for k in range(np.size(lst_metrics)):
           Metrics.iloc[r][lst_metrics[k]] =  metrics[lst_metrics[k]]
    
    
    return best_param, Metrics





