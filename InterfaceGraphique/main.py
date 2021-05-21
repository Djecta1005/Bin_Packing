import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

st.title('Projet Optimisation Combinatoire ')

st.write("""
# EQUIPE 01
""")

dataset_name = st.sidebar.selectbox(
    'Sélectionner le Dataset',
    ('Facile', 'Normal', 'Difficile')
)

st.write(f"## {dataset_name} Dataset")

algo_name = st.sidebar.selectbox(
    'Selectionner l''algorithme',
    ('Best Fit', 'Best Fit Decreasing', 'Worst Fit','Next Fit', 'First Fit', 'First Fit Decreasing')
)
st.write(f"## {algo_name} Algorithme")

def get_dataset(name):
    data = None
    if name == 'Facile':
        data =  open("E:\\instances\\N4C3W4_T.txt","r")
    elif name == 'Normal':
        data = open("E:\\instances\\N1C1W1_A.txt","r")
    else:
        data = open("E:\\instances\\HARD0.txt","r")
    X = data
    #y = data.target
    return X

def get_algo(name,w,n,c):
    algo = None
    if name == 'Best Fit':
        algo = bestFit(w,n,c)
    elif name == 'Best Fit Decreasing':
        algo = bestFitDecreasing(w,n,c)
    elif name =='Worst Fit':
        algo = worstFit(w, c)
    elif name == 'Next Fit':
        algo = nextFit(w, c)
    elif name =='First Fit':
        algo = firstFit(w, n, c)
    else:
        algo = firstFitDecreasing(w, n, c)
    return algo

####### Heuristiques implémentées ###

def bestFit(weight, n, c):
    # nombre de bins au départ
    res = 0

    # tableau contenant l'espace restant dans chaque bin
    bin_rem = [0] * n
    obj_inBin = dict()
    # print(bin_rem)

    # Placer chaque objet i dans un bin selon son poids weight[i]
    for i in range(n):
        j = 0
        # print(bin_rem)

        # intialiser le minimum d'espace restant dans tous les bins à une valeur supérieure à la capacité
        min = c + 1

        # intialiser l'indexe du meilleur bin
        bi = 0
        #print("i = ",i)
        #print("weight = ", weight[i])
        # parcourir les bins et chercher le bin qui convient le mieux à l'objet qu'on veut placer
        for j in range(res):
            if ((bin_rem[j] >= weight[i]) and (bin_rem[j] - weight[i] < min)):
                bi = j
                min = bin_rem[j] - weight[i]

                # Si aucun bin ne convient, créer un nouveau bin
        if (min == c + 1):
            if (weight[i] <= c):
                bin_rem[res] = c - weight[i]
                obj_inBin[res] = list()
                (obj_inBin.get(res)).append(weight[i])
                res = res + 1
        else:  # Assigner l'objet au meilleur bin
            bin_rem[bi] -= weight[i]
            (obj_inBin.get(bi)).append(weight[i])

    return res,obj_inBin
def bestFitDecreasing(weight, n, c):
    weight.sort()
    weightS=weight[::-1]
    print(weightS)
    return bestFit(weightS, n, c)
def firstFit(weight, n, c):
    res = 0  # Number of Bins
    bin_rem = np.zeros(n)  # The remaining capacity of each bin which will be different with each iteration
    obj_inBin = dict()    # The Objects that we will put in each bin

    for i in range(n):  # for each object
        j = 0
        for j in range(res + 1):  # for each bin already existes

            if bin_rem[j] >= weight[i]:  # Testing if the bin "j" still have place to contain the object "i"
                # if yes:
                bin_rem[j] = bin_rem[j] - weight[i]  # Updaing the capacity of the bin in use
                (obj_inBin.get(j)).append(weight[i])
                break  # Stop iterating since we have found a bin

        if j == res:  # Testing if the bin is already full or cannot contain more objects for the moment
            obj_inBin[res] = list()
            bin_rem[res] = c - weight[i]  # Creating a new bin
            (obj_inBin.get(res)).append(weight[i]) # Updating bins content
            res = res + 1  # Updating the number of bins

    return res,obj_inBin
def firstFitDecreasing(weight, n, c):
    - np.sort(-weight)# Sorting Object's list
    return firstFit(weight, n, c)
def nextFit(weight, capacity):
    # Initialisation de la variable nb qui contiendra le nombre de bacs necéssaire pour contenir les objets
    nb = 0
    # La variable c contiendra la capacité libre restante dans le bac courant, elle est initialisée a 0 car aucun bac au depart
    c = 0
    # la variable Obj_inBin contiendra la liste des objets rangés dans chaque bac, c'est un dictionnaire avec pour clé le numero du bac
    obj_inBin = dict()
    # Parcours des poids des objets que l'on souhaite ranger
    for w in range(len(weight)):
        if c >= weight[w]:
            # si la capacité restante dans le bac actuel est suffisante pour contenir l'objet w alors on l'ajoute au bac
            # et on décrémonte de la capacité c le poids de l'objet w
            c = c - weight[w]
            # on joute l'objet a la liste correspondant au bac courant dans le quel il vient d'etre rangé
            (obj_inBin.get(nb)).append(weight[w])
        else:
            # si la capacité restante n'est pas suffisante alors on ajoute un nouveau bac de capacité "capacity"
            nb += 1
            c = capacity - weight[
                w]  # et on retranche le poids de l'objet w qui vient d'etre ajouté au bac pour avoir la capacité restante
            obj_inBin[nb] = list()  # on créer la liste vide qui correspond au nouveau bac ajouté
            (obj_inBin.get(nb)).append(
                weight[w])  # on ajoute a cette liste l'objet qu'on vient de ranger dans le bac nb
    return nb,obj_inBin
def worstFit(weight, capacity):
    # Initialisation de la variable nb qui contiendra le nombre de bacs necéssaire
    nb = 0;

    # le nombre d'objet en entré
    n = len(weight)

    # on a n objets en entré donc on aura au maximum n bacs si on place chaque objet dans un bac
    # capaBin est un tableau de n cases qui contiendra la capacité libre restante dans chaque bac,
    # il est initialisé a 0 car aucun bac prit au départ
    capaBin = np.zeros(n);

    # la variable Obj_inBin contiendra la liste des objets rangés dans chaque bac, c'est un dictionnaire avec pour clé le numero du bac
    obj_inBin = dict()
    # Parcours des poids des objets que l'on souhaite ranger
    for i in range(n):
        # Trouver le pire bac  pouvant contenir l'objet i :
        # (celui ou la capacité réstante apres avoir ranger i est la plus grande)
        j = 0;
        bi = 0;  # Indice du bac dans le quel on rangera i
        max = -1;  # ranger i dans le bac qui maximise max (l'espace réstant apres avoir ranger i)
        for j in range(nb):  # [0,nb[
            # parcours des bacs éxistant
            if (capaBin[j] >= weight[i] and capaBin[j] - weight[i] > max):
                # si on trouve un bac pouvant contenir i et dont l'espace restant est supérieur a max
                # alors mise a jour numero du bac élu et de la var max
                bi = j;
                max = capaBin[j] - weight[i];

        # Si il n'ya aucun bac ou alors qu'il n'ya plus de place dans tous les bac pour contenir i
        if (max == -1):
            # On ajoute un nouveau bac et on range i
            capaBin[nb] = capacity - weight[i];
            nb += 1;
            # on créer la liste vide qui correspond au nouveau bac ajouté
            obj_inBin[nb] = list()
            # on ajoute a cette liste l'objet qu'on vient de ranger dans le bac nb
            (obj_inBin.get(nb)).append(weight[i])
        else:  # Sinon alors un bac a été choisi pour contenir i, c'est celui qui maximise "max" (la capacité restante)
            # on accéde au tableau des capacités des bacs et on soustrait au bac choisi le poid de i
            capaBin[bi] -= weight[i];
            # on ajoute l'objet i à la liste d'objet du bac choisi bi+1
            # (bi+1 car bi [0,n-1] alors que les clés de notre dictionnaire obj_inBin varie [1,n]) l'objet qui vient d'y etre rangé
            (obj_inBin.get(bi + 1)).append(weight[i])
    return nb,obj_inBin

XT= get_dataset(dataset_name)
lignes = XT.readlines()
n=int(lignes[0])
c=int(lignes[1])
weight = [ int(x) for x in lignes[2:len(lignes)] ]
weight=np.asarray(weight)
#st.write('Shape of dataset:', lignes.shape(0))
st.write(lignes)
#print(weight)
st.write('Bin capacity : ', c)
st.write('Number ob objects :', n)
#l = bestFit(weight, n, c)
A, obj_inBin = get_algo(algo_name,weight,n,c)
st.write('Number of bins required : ', A)
st.write("Occupation des bins : ")
for (i, val) in obj_inBin.items():
    Wstring = ""
    for j in range(len(val)):
        Wstring = Wstring + ", " + str(val[j])
    st.write("Bin : ", i, " contient les objets avec les poids :", Wstring)
