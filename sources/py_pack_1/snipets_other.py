# -*- coding: utf-8 -*-
__author__ = 'plemberger'

import pandas as pd
import numpy as np

# comptage du nombre de valeurs par groupe
df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'], 'key2' : ['one', 'two', 'one', 'two', 'one'], 'data1' : np.random.randn(5), 'data2' : np.random.randn(5)})
grouped = df['data1'].groupby(df['key1'])
count_serie = grouped.count()


# création d'un Series à partir d'un dict
dict = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
series = pd.Series(dict)


# création d'un DataFrame à partir d'un dict par colonne
dicts = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'], \
         'year': [2000, 2001, 2002, 2001, 2002], \
         'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(dicts)


# création d'un DataFrame de test avec une liste de colonnes et un index
df = pd.DataFrame(np.arange(16).reshape((4, 4)), \
                  index=['Ohio', 'Colorado', 'Utah', 'New York'], \
                  columns=['one', 'two', 'three', 'four'])
index = df.index  # l'objet Index
values = df.values # un array des valeurs


# concaténation des lignes de deux DataFrames (l'indexation n'est pas utilisée)
pd.DataFrame(np.concatenate([df_1.values, df_2.values], axis = 0))


# sampling échantillonnage aléatoire d'un DataFrame
import pandas
import random

df = pandas.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
rows = random.sample(df.index, 10)
df_10 = df.ix[rows]
df_90 = df.drop(rows)


# création d'un DataFrame de test avec des nombres aléatoires
df = pd.DataFrame(np.random.randn(4,7), index=['B', 'A', 'D', 'C'], columns=['col1','col2','col3','col4','col5','col6','col7'])


# récupérer une colonne d'un DataFrale sous forme d'un Series
col1 = df.year
col2 = df['pop']


# récupérer plusieurs colonnes sous forme d'un nouveau DataFrame
def_new = df[['pop','year']]


# définition d'un nom pour la liste des colonnes et pour l'index
df.index.name = 'state'
df.columns.name = 'number'


# définition d'une colonne d'un DataFrame comme index et suppression de la colonne en question
df.set_index('one', drop=True, inplace = True)


# réordonner des colonnes d'un DataFrame et rajout d'un colonne vide
df_new = pd.DataFrame(df, columns=['year', 'state', 'pop', 'debt'])


# réordonner des lignes d'un DataFrame en utilisant l'index
df_new = df.reindex(['New York', 'Utah', 'Colorado', 'Ohio'])


# insertion d'un enregistrement dans un DataFrame


# insertion d'une nouvelle colonne dans un DataFrame
df_new = df.insert(1, 'new_col', [666, 777, 888, 999])


# insertion de plusieurs colonnes dans un DataFrame ou concaténation de plusieurs DataFrame
df = pd.concat([df_1, df_2], axis=1)


# sélection d'un ou plusieurs enregistrements d'un DataFrame à partir de l'index
row1 = df.ix['Utah']
row2 = df.ix[['Utah', 'New York']]


# sélection de lignes et de colonnes par intervalles ou par liste d'indices ou par liste de numéros
df_new = df.ix[0:2, 1:3]


# mise à jour d'une ligne (enregistrement) d'un DataFrame
df.ix['Utah'] = [88, 99, 1010, 1111]


# mise à jour d'une colonne d'un DataFrame
data_new['debt']=888


# suppression d'une ligne d'un DataFrame
df_new = df.drop('Ohio')        # création d'un nouveau DataFrame
df.drop('Ohio', inplace = True) # suppression de la ligne sur le DataFrame courant


# suppression d'une colonne d'un DataFrame
df_new = df.drop('two', axis=1)


# modification du nom des colonnes d'un DataFrame


# génération d'un tableau de nombres aléatoires
data = np.random.randn(4,7)


# changement du nombre de lignes et de colonnes d'un tableau
arr = np.arange(15).reshape((3, 5))


# produit scalaire de deux vecteurs
u=np.array([1,2,3])   ; v=np.array([4,5,6])
produit_scalaire = np.dot(u.T,v)


# génération d'un nouvel array avec une condition (p99)
arr = np.random.randn(4, 4)
np.where(arr > 0, 1, -1)


# définition d'une lambda-fonction sur un array
f = lambda x: x.max() - x.min()


# application de la fonction f sur les lignes/colonnes çàd aggrégation selon axe 1/0
df.apply(f,1);  df.apply(f,0)


# classer les lignes d'un DataFrame selon les valeurs de l'index
df = pd.DataFrame(np.random.randn(4,7), \
                  index=['B', 'A', 'D', 'C'], \
                  columns=['col1','col2','col3','col4','col5','col6','col7'])
df_sorted = df.sort_index(axis=0)


# classer les lignes d'un DataFrame selon les valeurs d'une certaine colonne
df_sorted = df.sort_index(by='col3')


# suppression d'espaces dans un string
str.strip('aze   ')


# récupérer à la fois les indices et les valeurs d'une collection
a_collection = []
for i, value in enumerate(a_collection):
    pass #<do something with i & value>


# résumé des statistiques principales d'un DataFrame
df.describe()


# récupérer l'index du maximum dans un Series
series = pd.Series([3,2,5,6,8,0,2,9,1])
series.idxmax()


# taille d'une liste ou d'un ensemble
my_list = [1,2,2,3,3,3,7]
my_set = set(my_list)
length_list = len(my_list)
length_set = len(my_set)


# p413
seq1=[]; seq2=[]
for i, (a, b) in enumerate(zip(seq1, seq2)):
    pass #<do something with i, a, b>

# p403
def attempt_float(x):
    try:
        return float(x)
    except:
        return x


# p415
mapping = {}; key_list=[]; value_list=[]
for key, value in zip(key_list, value_list):
    mapping[key] = value


# p415
some_dict={}
default_value=0
key=0
value = some_dict.get(key, default_value)


# p423
def default_value():
    pass
states=[]
map(default_value, states)


# p424
def apply_to_list(some_list, f):
    return [f(x) for x in some_list]

ints = [4, 0, 1, 5, 6]
apply_to_list(ints, lambda x: x * 2)

# sélection de lignes dans un DataFrame avec une logique complexe
df_A[(df_A.col1 < 0) & (df_A.col1 > -1)]



# commandes utiles etc...

"""
    commandes utiles dans PyCharm
    -----------------------------
    <CTRL> + 'mouse hover' pour trouver les définitions
    <ALT><Maj><E> pour exécuter une ligne dans la console Python

    commandes utiles dans la console Python
    ---------------------------------------
    %run <>
    %reset


    fonction utiles
    ---------------
    type(<var>),
    isinstance(<var>, <type>)
    f = lambda x: x**3
    DataFrame.apply(<function>)
    DataFrame.describe()
    range(<n>)


"""


# essais divers et variés

"""
# les 5 premières colonnes d'un DataFrame
df_train_data[range(5)]

# noms des colonnes
# u'Id', u'City', u'City Group', u'Type', u'P1', ..., u'P37', u'year', u'day_in_year', u'day_of_week']

size_beginning = 4
first_records = df_train_data.head(4)

df_test_by_col_names = df_train_data.head(size_beginning)[['Id' ,'City', 'City Group','Open Date', 'Type', 'P1', 'P2', 'P3']]
df_test_by_col_numbers = df_train_data.head(size_beginning)[[0, 1, 2, 3]]

# TODO essayer de rajouter un index à un DataFrame simple
#dt_bidon = DataFrame(np.random(4,3), columns=list('bde'))

# sélection de colonne
print
print '****************************************'
print 'par noms de colonnes'
print df_test_by_col_names
print '****************************************'
print 'par numéros de colonnes'
print df_test_by_col_numbers
print '****************************************'
df_with_dates_converted = df_test_by_col_names.convert_objects(convert_dates=True)
print 'traitement des dates'
print df_with_dates_converted
print '****************************************'

# conversion d'une date string en date time
une_date_string = df_test_by_col_names['Open Date'][0]
un_vrai_datetime = dt.datetime.strptime(une_date_string, '%m/%d/%Y')

# conversion d'une colonne de strings qui représentent des dates en un Series de datetime
colonne_date_string = df_test_by_col_names['Open Date']
colonne_date_datetime = pd.to_datetime(colonne_date_string, infer_datetime_format=True)

# création de plusieurs Series qui contiennent les n° des jours dans l'année  l'année, le mois etc...
colonne_jours_dans_annee  = colonne_date_datetime.dt.dayofyear
colonne_mois_dans_annee   = colonne_date_datetime.dt.month
colonne_jour_dans_semaine = colonne_date_datetime.dt.dayofweek

# quelques informations statistiques
nb_records_train = df_train_data.shape[0]
nb_records_test = df_test_data.shape[0]
print
print "nombre d'enregistrements dans les donnees d\'entrainement :", nb_records_train
print "nombre d'enregistrements dans les donnees de test :", nb_records_test

# pour récupérer un enregistrement à partir de la valeur d'un index
item = df_sample_data.ix[3]



"""
