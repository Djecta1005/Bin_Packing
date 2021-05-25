import streamlit as st
import numpy as np
import csv
import timeit
import pandas as pd
import copy
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

#st.title('Projet Optimisation Combinatoire ')

#st.write(""" # EQUIPE 01 """)

Fonctionnalite = st.sidebar.selectbox(
    'Sélectionner une fonctionnalité',
    ('Exécution', 'Historique', 'Comparaison')
)
st.write(f"""# {Fonctionnalite} """)
if Fonctionnalite=='Exécution':
    dataset_type = st.sidebar.selectbox(
        'Sélectionner le type du Dataset',
        ('Facile', 'Normal', 'Difficile','Spéciales Méthode Exacte')
    )
    if dataset_type=='Facile':
        dataset_name = st.sidebar.selectbox(
            'Sélectionner une instance facile',
            ('N4C1W2_H', 'N4C3W1_A', 'N3C1W1_A','N3C3W1_A','N2C1W2_Q','N2C3W1_A','N1C1W1_R','N1C3W1_A')
        )
    elif dataset_type=='Normal':
        dataset_name = st.sidebar.selectbox(
            'Sélectionner une instance normale',
            ('N4W1B1R0', 'N4W4B3R9', 'N3W1B1R0','N3W4B1R0','N2W1B1R0','N2W4B3R0','N1W1B1R0', 'N4W4B3R9')
        )
    elif dataset_type=='Difficile':
        dataset_name = st.sidebar.selectbox(
            'Sélectionner une instance difficile',
            ('HARD0', 'HARD2', 'Falkenauer_u1000_00','Falkenauer_u1000_19')
        )
    elif dataset_type=='Spéciales Méthode Exacte':
        dataset_name = st.sidebar.selectbox(
            'Sélectionner une instance aléatoire',
            ('ajouter ici les datasets kawther')
        )

    st.write(f"## {dataset_type} Dataset : Instance {dataset_name} ")

    algo_name = st.sidebar.selectbox(
        'Selectionner l''algorithme',
        ('Best Fit', 'Best Fit Decreasing', 'Worst Fit','Next Fit', 'First Fit', 'First Fit Decreasing')
    )
    st.write(f"## {algo_name} Algorithme")

def get_dataset(name):
    data = None
    if name == 'N4C1W2_H':
        data =  open("instances\\Facile\\T_Grande_500\\N4C1W2_H.txt","r")
    elif name == 'N4C3W1_A':
        data = open("instances\\Facile\\T_Grande_500\\N4C3W1_A.txt","r")
    elif name == 'N3C1W1_A':
        data = open("instances\\Facile\\T_Moyenne_200\\N3C1W1_A.txt","r")
    elif name == 'N3C3W1_A':
        data = open("instances\\Facile\\T_Moyenne_200\\N3C3W1_A.txt","r")
    elif name == 'N2C1W2_Q':
        data = open("instances\\Facile\\T_Petite_100\\N2C1W2_Q.txt","r")
    elif name == 'N2C3W1_A':
        data = open("instances\\Facile\\T_Petite_100\\N2C3W1_A.txt","r")
    elif name == 'N1C1W1_R':
        data = open("instances\\Facile\\T_Tres_Petite_50\\N1C1W1_R.txt","r")
    elif name == 'N1C3W1_R':
        data = open("instances\\Facile\\T_Tres_Petite_50\\N1C3W1_A.txt","r")
    elif name == 'N4W1B1R0':
        data = open("instances\\Moyenne\\T_Grande_500\\N4W1B1R0.txt","r")
    elif name == 'N4W4B3R9':
        data = open("instances\\Moyenne\\T_Grande_500\\N4W4B3R9.txt","r")
    elif name == 'N3W1B1R0':
        data = open("instances\\Moyenne\\T_Moyenne_200\\N3W1B1R0.txt","r")
    elif name == 'N3W4B1R0':
        data = open("instances\\Moyenne\\T_Moyenne_200\\N3W4B1R0.txt","r")
    elif name == 'N2W1B1R0':
        data = open("instances\\Moyenne\\T_Petite_100\\N2W1B1R0.txt","r")
    elif name == 'N2W4B3R0':
        data = open("instances\\Moyenne\\T_Petite_100\\N2W4B3R0.txt","r")
    elif name == 'N1W1B1R0':
        data = open("instances\\Moyenne\\T_Tres_Petite_50\\N1W1B1R0.txt","r")
    elif name == 'N4W4B3R9':
        data = open("instances\\Moyenne\\T_Tres_Petite_50\\N4W4B3R9.txt","r")
    elif name == 'HARD0':
        data = open("instances\\Difficile\\T_Moyenne_200\\HARD0.txt","r")
    elif name == 'HARD2':
        data = open("instances\\Difficile\\T_Moyenne_200\\HARD2.txt","r")
    elif name == 'Falkenauer_u1000_19':
        data = open("instances\\Difficile\\T_Tres_Grande_1000\\Falkenauer_u1000_19.txt","r")
    elif name == 'Falkenauer_u1000_00':
        data = open("instances\\Difficile\\T_Tres_Grande_1000\\Falkenauer_u1000_00.txt","r")
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
    elif name=='First Fit Decreasing':
        algo = firstFitDecreasing(w, n, c)


    return algo

####### Heuristiques implémentées ###

def bestFit(weight, n, c):
    temps_debut = timeit.default_timer()
    # nombre de bins au départ
    res = 0

    # tableau contenant l'espace restant dans chaque bin
    bin_rem = [0] * n
    #obj_inBin = dict()
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
     #           obj_inBin[res] = list()
      #          (obj_inBin.get(res)).append(weight[i])
                res = res + 1
        else:  # Assigner l'objet au meilleur bin
            bin_rem[bi] -= weight[i]
       #     (obj_inBin.get(bi)).append(weight[i])
    duree = timeit.default_timer() - temps_debut
    return res,duree
def bestFitDecreasing(weight, n, c):
    temps_debut = timeit.default_timer()
    weight.sort()
    weightS=weight[::-1]
    print(weightS)
    F,d= bestFit(weightS, n, c)
    duree = timeit.default_timer() - temps_debut
    return F,duree
def firstFit(weight, n, c):
    temps_debut = timeit.default_timer()
    res = 0  # Number of Bins
    bin_rem = np.zeros(n)  # The remaining capacity of each bin which will be different with each iteration
   # obj_inBin = dict()    # The Objects that we will put in each bin

    for i in range(n):  # for each object
        j = 0
        for j in range(res + 1):  # for each bin already existes

            if bin_rem[j] >= weight[i]:  # Testing if the bin "j" still have place to contain the object "i"
                # if yes:
                bin_rem[j] = bin_rem[j] - weight[i]  # Updaing the capacity of the bin in use
    #            (obj_inBin.get(j)).append(weight[i])
                break  # Stop iterating since we have found a bin

        if j == res:  # Testing if the bin is already full or cannot contain more objects for the moment
     #       obj_inBin[res] = list()
            bin_rem[res] = c - weight[i]  # Creating a new bin
      #      (obj_inBin.get(res)).append(weight[i]) # Updating bins content
            res = res + 1  # Updating the number of bins
    duree = timeit.default_timer() - temps_debut
    return res, duree
def firstFitDecreasing(weight, n, c):
    temps_debut = timeit.default_timer()

   # - np.sort(-weight)# Sorting Object's list
    weight.sort()
    weightS = weight[::-1]
    F,d=firstFit(weightS, n, c)
    duree = timeit.default_timer() - temps_debut
    return F,duree
def nextFit(weight, capacity):
    temps_debut = timeit.default_timer()
    # Initialisation de la variable nb qui contiendra le nombre de bacs necéssaire pour contenir les objets
    nb = 0
    # La variable c contiendra la capacité libre restante dans le bac courant, elle est initialisée a 0 car aucun bac au depart
    c = 0
    # la variable Obj_inBin contiendra la liste des objets rangés dans chaque bac, c'est un dictionnaire avec pour clé le numero du bac
    #obj_inBin = dict()
    # Parcours des poids des objets que l'on souhaite ranger
    for w in range(len(weight)):
        if c >= weight[w]:
            # si la capacité restante dans le bac actuel est suffisante pour contenir l'objet w alors on l'ajoute au bac
            # et on décrémonte de la capacité c le poids de l'objet w
            c = c - weight[w]
            # on joute l'objet a la liste correspondant au bac courant dans le quel il vient d'etre rangé
     #       (obj_inBin.get(nb)).append(weight[w])
        else:
            # si la capacité restante n'est pas suffisante alors on ajoute un nouveau bac de capacité "capacity"
            nb += 1
            c = capacity - weight[
                w]  # et on retranche le poids de l'objet w qui vient d'etre ajouté au bac pour avoir la capacité restante
      #      obj_inBin[nb] = list()  # on créer la liste vide qui correspond au nouveau bac ajouté
      #      (obj_inBin.get(nb)).append(         weight[w])  # on ajoute a cette liste l'objet qu'on vient de ranger dans le bac nb
    duree = timeit.default_timer() - temps_debut
    return nb, duree
def worstFit(weight, capacity):
    # Initialisation de la variable nb qui contiendra le nombre de bacs necéssaire
    temps_debut = timeit.default_timer()
    nb = 0;

    # le nombre d'objet en entré
    n = len(weight)

    # on a n objets en entré donc on aura au maximum n bacs si on place chaque objet dans un bac
    # capaBin est un tableau de n cases qui contiendra la capacité libre restante dans chaque bac,
    # il est initialisé a 0 car aucun bac prit au départ
    capaBin = np.zeros(n);

    # la variable Obj_inBin contiendra la liste des objets rangés dans chaque bac, c'est un dictionnaire avec pour clé le numero du bac
  #  obj_inBin = dict()
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
   #         obj_inBin[nb] = list()
            # on ajoute a cette liste l'objet qu'on vient de ranger dans le bac nb
    #        (obj_inBin.get(nb)).append(weight[i])
        else:  # Sinon alors un bac a été choisi pour contenir i, c'est celui qui maximise "max" (la capacité restante)
            # on accéde au tableau des capacités des bacs et on soustrait au bac choisi le poid de i
            capaBin[bi] -= weight[i];
            # on ajoute l'objet i à la liste d'objet du bac choisi bi+1
            # (bi+1 car bi [0,n-1] alors que les clés de notre dictionnaire obj_inBin varie [1,n]) l'objet qui vient d'y etre rangé
     #       (obj_inBin.get(bi + 1)).append(weight[i])
    duree = timeit.default_timer() - temps_debut
    return nb,duree

################################ Méthodes exactes #####################################
def current_milli_time():
    return round(time.time() * 1000)
class Node:

    def __init__(self, nbObjects, nbBins, cRemaining):
        self.nbObjects = nbObjects  # number of the first k objects packing
        self.nbBins = nbBins  # number of bins used to pack the first k objects
        self.cRemaining = cRemaining  # a table of remaining capacities of each one of nbBins used

    def getNbObjects(self):
        return self.nbObjects

    def getNbBins(self):
        return self.nbBins

    def getIcRemaining(self, i):
        return self.cRemaining[i]

    def getCremaining(self):
        return self.cRemaining

    def printNode(self):
        print("objects", self.nbObjects)
        print("nbBins", self.nbBins)
        print("cRem", self.cRemaining)
def branchAndBoundL(n, objects, c):
    # time= current_milli_time()
    time = timeit.default_timer()
    print(time)
    # n: number of objects which is the max number of bins we can use
    # objects: table of weights of objects
    # c: capacity of each bin
    cpt = 0
    minBins = 17  # initialize upper bound (Sup)
    usedBins = 0  # initialize the number of used Bins
    cRemaining = [c] * n  # initialize the table of remaining cpacaties of each bin
    nodes = []  # array that will contain created nodes and not processed

    # create the root node with 0 bins and 0 objects
    node = Node(0, usedBins, cRemaining)
    nodes.append(node)

    while len(nodes) > 0:
        node = nodes.pop()  # get a node to explore it
        nbObjects = node.getNbObjects()
        usedBins = node.getNbBins()
        if (nbObjects == n and usedBins < minBins):  # update the upper bound
            minBins = usedBins
            # node.printNode()

        else:
            if (
                    usedBins < minBins):  # evaluate the node if the number of bins used is more than the minBins we ignore it
                objectWeight = objects[nbObjects]

                for i in range(usedBins + 1):
                    if (nbObjects < n) and (node.getIcRemaining(
                            i) >= objectWeight):  # check if it is possible to add the object in the bin i
                        newCremaining = node.getCremaining().copy()
                        newCremaining[i] -= objectWeight
                        cpt = cpt + 1
                        if (i == usedBins):  # new Bin is added
                            newNode = Node(nbObjects + 1, usedBins + 1, newCremaining)
                        else:  # the bin is already added
                            newNode = Node(nbObjects + 1, usedBins, newCremaining)

                        nodes.append(newNode)
    time = timeit.default_timer() - time
    print(time)
    print("cpt", cpt)
    return minBins, time
def branchAndBound(n, objects, c):
    # n: number of objects which is the max number of bins we can use
    # objects: table of weights of objects
    # c: capacity of each bin
    cpt = 0
    time = timeit.default_timer()
    print(time)
    minBins = n  # initialize upper bound (Sup)
    usedBins = 0  # initialize the number of used Bins
    cRemaining = [c] * n  # initialize the table of remaining cpacaties of each bin
    nodes = []  # array that will contain created nodes and not processed
    curBin = 0
    cumWeight = []
    precWeight = 0
    newCremaining = cRemaining.copy()
    last = False
    for i in range(n):
        cumWeight.append(precWeight + objects[i])
        precWeight = precWeight + objects[i]
        j = 0
        while newCremaining[j] < objects[i]:
            j += 1

        if j == usedBins:
            last = True
            usedBins += 1
        else:
            last = False
        newCremaining[j] -= objects[i]
        nodes.append(Node(i, j, usedBins, newCremaining.copy(), last))
    cpt = n
    minBins = usedBins
    # print("min",minBins)
    #

    #
    while (len(nodes) > 0):
        #
        node = nodes.pop(-1)
        nbObjects = node.getNbObjects() + 1
        last = node.getLast()
        tRemaining = node.getCremaining().copy()
        usedBins = node.getNbBins()
        # print("use",usedBins)
        if (nbObjects == n):
            if (usedBins < minBins):
                minBins = usedBins
            while (last and len(nodes) > 0):
                node = nodes.pop(-1)
                last = node.getLast()
                # print(node.getNbBins())
            curBin = node.getCurBin() + 1

            node = nodes.pop(-1)
            nbObjects = node.getNbObjects() + 1
            tRemaining = node.getCremaining().copy()
            usedBins = node.getNbBins()

        while (tRemaining[curBin] < objects[nbObjects]):
            curBin += 1
        if (curBin == usedBins):
            last = True
            usedBins += 1
        else:
            last = False
        oRemaining = tRemaining[:usedBins]
        oRemaining[curBin] = oRemaining[curBin] - objects[nbObjects]
        large = max(oRemaining)
        j = 0
        if (nbObjects + 1 < n and large >= objects[n - 1]):
            j = 1
        # print(oRemaining, usedBins)
        ev = math.ceil(
            usedBins + (cumWeight[n - 1] - cumWeight[nbObjects] - j * (usedBins * c - cumWeight[nbObjects])) / c)
        while (ev > minBins and not last):
            curBin += 1
            while (tRemaining[curBin] < objects[nbObjects]):
                curBin += 1
            if (curBin == usedBins):
                last = True
                usedBins += 1
            else:

                last = False
            oRemaining = tRemaining[:usedBins]
            oRemaining[curBin] = oRemaining[curBin] - objects[nbObjects]
            large = max(oRemaining)
            j = 0
            if (nbObjects + 1 < n and large >= objects[n - 1]):
                j = 1
            ev = math.ceil(
                usedBins + (cumWeight[n - 1] - cumWeight[nbObjects] - j * (usedBins * c - cumWeight[nbObjects])) / c)

        if (ev <= minBins):
            cpt = cpt + 1
            nodes.append(node)
            tRemaining[curBin] = tRemaining[curBin] - objects[nbObjects]
            nodes.append(Node(nbObjects, curBin, usedBins, tRemaining.copy(), last))
            curBin = 0
            nbObjects = nbObjects + 1
        else:
            while (last and len(nodes) > 0):
                node = nodes.pop(-1)
                last = node.getLast()
            curBin = node.getCurBin() + 1

    time = timeit.default_timer() - time
    print("time", time)
    print("cpt", cpt)
    return minBins, time

#######################Méta Heuristiques #######################################"

###########Exuction#####################"
if Fonctionnalite =='Exécution':
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
    #temps_debut = timeit.default_timer()
    A,duree = get_algo(algo_name,weight,n,c)
    #duree = timeit.default_timer() - temps_debut
    with open('historique.csv','a',newline='',encoding='utf-8') as fichiercsv:
        writer=csv.writer(fichiercsv)
        writer.writerow([dataset_type,dataset_name, algo_name,n, A, duree, '/'])
    st.write('Number of bins required : ', A)
    st.write('Temps d''éxécution : ', duree)
##################Historique###########################"
if Fonctionnalite=='Historique':
    r = pd.read_csv("historique.csv",encoding = "ISO-8859-1" )
    st.write(r)


#with open('historique.csv','w',newline='') as fichiercsv:
 #   writer=csv.writer(fichiercsv)
  #  writer.writerow(['Type de l''instance','Instance', 'Algorithme', 'Nombre d''objets', 'Nombre de bins', 'Temps d''éxécution','Paramètres'])