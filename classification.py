# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 20:14:12 2020

@author: Louis SANT'ANNA
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import seaborn as sns




#Récupération des données (dataset) depuis le repertoire local du fichier
dataset = pd.read_csv('C:/Users/Public/Projet ml/happiness_2019.csv', sep=',')

"""
Si on remarque bien le jeu de données, le rang est déterminé à partir du score. Le vecteur colonne rang 
n'est pas une information utile vu qu'on peut toujours le déterminer à partir du vecteur colonne score.
On peut donc enlever la variable rang lors de notre analyse.
Le vecteur score, ou la variable score si on se positionne sur une ligne est en fait donc
la variable à expliquer (à décrire), on l'obtient ainsi à partir des variables explicatives qui
sont les autres variables restantes à savoir le PIB par habitant, soutien social, espérance de vie... 
Du coup avant de faire une quelconque forme de classification, ce serait astucieux d'observer sur 
un même graphe la variation de ces variables explicatives entre ces pays
"""
data_frame = dataset.filter(['Country or region','GDP per capita','Social support','Healthy life expectancy',
'Freedom to make life choices','Generosity','Perceptions of corruption'])
data_frame = data_frame.set_index('Country or region')
data_frame.plot.barh(stacked=True,  figsize=(21,30))

"""
Après avoir mis en exergue graphiquement les variations pour ces différents pays, 
on peut calculer la matrice de corrélation de nos variables explicatives
"""
df = data_frame.loc[:, 'GDP per capita':'Perceptions of corruption']
figure, axis = plt.subplots(figsize=(18, 18))
sns.heatmap(data=df.corr(),annot=True, linewidths=.5, fmt= '.1f',ax = axis)
plt.show()

"""
Définition de l'algorithme de clustering K-Means 
"""
def Classification_Kmeans(data,numberOfClusters):
       #Initializing Kmeans.cluster object was imported from sklearn in begining.
       kmeans = cluster.KMeans(n_clusters=numberOfClusters)
       # Fitting the input data and getting the cluster labels
       cluster_labels = kmeans.fit_predict(data)
       # Getting the cluster centers
       cluster_centers = kmeans.cluster_centers_
       cluster_centers.shape
       return cluster_labels,cluster_centers


"""
Maintenant on recherche la valeur optimale de K, qui sera le nombre de clusters qu'on aura.
Pour trouver la valeur de K, on utilise l'algorithme du procédé coude (Elbow method)
Techniquement l'idée est de représenter les différentes valeurs du coût (somme des carrées 
des distances entre une observation et son centroid, le centroid est le centre du cluster) en fonction 
de la variation de k. Plus la valeur de K augmente, moins il y aura d'éléments dans le groupe. 
La distorsion moyenne diminuera donc. Moins il y a d'éléments, plus on se rapproche du centroïde. 
Ainsi, le point où cette distorsion diminue le plus est le point de coude.
"""
def recherche_valeur_K(data):
    somme_des_carrees_des_distances = []
    """
    on prend maintenant un ensemble de valeur possibles pour le nombre de clusters,
    de minimum 1 à maximum 15 c'est raisonnable
    """
    K = range(1,15)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(data)
        somme_des_carrees_des_distances.append(km.inertia_)
    plt.plot(K, somme_des_carrees_des_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('somme des carrées des distances')
    plt.title('Procédé du coude pour trouver le k optimal')
    plt.show()
    
"""
La fonction du procédé du coude étante construite, on construit le graphe,
je rappelle que ce graphe a l'allure d'un bras, et la valeur optimale de K est en fait prise
au niveau de l'abscisse du coude
"""    
recherche_valeur_K(data_frame)
#Après lecture du graphe, on obtient K=3, donc on choisira alors le nombre clusters égale à 3

#Construction des différents clusters 
def plot_cluster(labels,centroids,data_frame):
    #On récupère le nombre des colonnes
    nombre_de_dimensions = data_frame.columns.size
    #Nombre de tracés qu'il faut pour 6 variables explicatives avec 2 prises lors de chaque tracé
    nombre_de_constructions = int(nombre_de_dimensions/2)
    #Nombre de ligne et colonne pour chaque tracé
    fig,ax = plt.subplots(nombre_de_constructions,1, figsize=(10,10))
    for x,y in zip(range(0,nombre_de_dimensions,2),range(0,nombre_de_constructions)):
         ax[y].scatter(data_frame.iloc[:, x], data_frame.iloc[:, x+1], c=labels, s=50, cmap='viridis')
         ax[y].scatter(centroids[:,x], centroids[:, x+1], c='red', s=200, alpha=0.5)
    plt.subplots_adjust(bottom=-0.5, top=1.5)
    plt.show()

labels,centroids = Classification_Kmeans(data_frame,3)     
plot_cluster(labels,centroids,data_frame)

"""
On obtient donc à la fin la formation des différents clusters possibles avec le centroid en rouge
"""