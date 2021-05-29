import streamlit as st
import numpy as np
import csv
import math
import timeit
import pandas as pd
import copy
from datetime import datetime
import matplotlib.pyplot as plt
from numpy.random import randint
from numpy.random import rand
from collections import namedtuple
from random import *
import itertools


#st.title('Projet Optimisation Combinatoire ')

#st.write(""" # EQUIPE 01 """)
# solution exact
exacts = {
    "Falkenauer_u1000_00":399,
    "Falkenauer_u1000_19":400,
    "HARD0":56,
    "HARD2":56,
    "N1C1W1_R":25,
    "N1C3W1_A":16,
    "N1W1B1R0":18,
    "N1W4B3R9":6,
    "N2C1W2_Q":65,
    "N2C3W1_A":35,
    "N2W1B1R0":34,
    "N2W4B3R0":12,
    "N3C1W1_A":105,
    "N3C3W1_A":66,
    "N3W1B1R0":67,
    "N3W4B1R0":23,
    "N4C1W2_H":315,
    "N4C3W1_A":164,
    "N4W1B1R0":167,
    "N4W4B3R9":56
 
}
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
            ('ins5','ins7','ins10','ins12','ins14','ins18','ins20')
        )

    st.write(f"## {dataset_type} Dataset : Instance {dataset_name} ")

    algo_name = st.sidebar.selectbox(
        'Selectionner l''algorithme',
        ('Best Fit', 'Best Fit Decreasing', 'Worst Fit','Next Fit', 'First Fit', 'First Fit Decreasing','Branch and Bound L','Branch and Bound P','Genetic Algorithm','Simulated Annealing' )
    )
    st.write(f"## {algo_name} Algorithme")

def get_dataset(name):
    data = None
    if name == 'ins5':
        data =  open("instances/tres_facile/ins5.txt","r")
    elif name == 'ins7':
        data =  open("instances/tres_facile/ins7.txt","r")
    elif name == 'ins10':
        data =  open("instances/tres_facile/ins10.txt","r")
    elif name == 'ins12':
        data =  open("instances/tres_facile/ins12.txt","r")
    elif name == 'ins14':
        data =  open("instances/tres_facile/ins14.txt","r")
    elif name == 'ins18':
        data =  open("instances/tres_facile/ins18.txt","r")
    elif name == 'ins20':
        data =  open("instances/tres_facile/ins20.txt","r")
    elif name == 'N4C1W2_H':
        data =  open("instances/Facile/T_Grande_500/N4C1W2_H.txt","r")
    elif name == 'N4C3W1_A':
        data = open("instances/Facile/T_Grande_500/N4C3W1_A.txt","r")
    elif name == 'N3C1W1_A':
        data = open("instances/Facile/T_Moyenne_200/N3C1W1_A.txt","r")
    elif name == 'N3C3W1_A':
        data = open("instances/Facile/T_Moyenne_200/N3C3W1_A.txt","r")
    elif name == 'N2C1W2_Q':
        data = open("instances/Facile/T_Petite_100/N2C1W2_Q.txt","r")
    elif name == 'N2C3W1_A':
        data = open("instances/Facile/T_Petite_100/N2C3W1_A.txt","r")
    elif name == 'N1C1W1_R':
        data = open("instances/Facile/T_Tres_Petite_50/N1C1W1_R.txt","r")
    elif name == 'N1C3W1_R':
        data = open("instances/Facile/T_Tres_Petite_50/N1C3W1_A.txt","r")
    elif name == 'N4W1B1R0':
        data = open("instances/Moyenne/T_Grande_500/N4W1B1R0.txt","r")
    elif name == 'N4W4B3R9':
        data = open("instances/Moyenne/T_Grande_500/N4W4B3R9.txt","r")
    elif name == 'N3W1B1R0':
        data = open("instances/Moyenne/T_Moyenne_200/N3W1B1R0.txt","r")
    elif name == 'N3W4B1R0':
        data = open("instances/Moyenne/T_Moyenne_200/N3W4B1R0.txt","r")
    elif name == 'N2W1B1R0':
        data = open("instances/Moyenne/T_Petite_100/N2W1B1R0.txt","r")
    elif name == 'N2W4B3R0':
        data = open("instances/Moyenne/T_Petite_100/N2W4B3R0.txt","r")
    elif name == 'N1W1B1R0':
        data = open("instances/Moyenne/T_Tres_Petite_50/N1W1B1R0.txt","r")
    elif name == 'N4W4B3R9':
        data = open("instances/Moyenne/T_Tres_Petite_50/N4W4B3R9.txt","r")
    elif name == 'HARD0':
        data = open("instances/Difficile/T_Moyenne_200/HARD0.txt","r")
    elif name == 'HARD2':
        data = open("instances/Difficile/T_Moyenne_200/HARD2.txt","r")
    elif name == 'Falkenauer_u1000_19':
        data = open("instances/Difficile/T_Tres_Grande_1000/Falkenauer_u1000_19.txt","r")
    elif name == 'Falkenauer_u1000_00':
        data = open("instances/Difficile/T_Tres_Grande_1000/Falkenauer_u1000_00.txt","r")
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
    elif name=='Branch and Bound L':
        algo= branchAndBoundL(n,w,c)
    elif name=='Branch and Bound P':
        algo= branchAndBoundP(n,w,c)
    elif name=='Genetic Algorithm':
        algo= genetic_algorithm(w,c)
    elif name=='Simulated Annealing':
        items = [Objet(w[i]) for i in range(len(w))]
        sa=RS(c,items)
        algo=sa.executer()
                


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
class NodeL:

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
    objects=-np.sort(-objects)
    time = timeit.default_timer()
    # n: number of objects which is the max number of bins we can use
    # objects: table of weights of objects
    # c: capacity of each bin
    minBins = n  # initialize upper bound (Sup)
    usedBins = 0  # initialize the number of used Bins
    cRemaining = [c] * n  # initialize the table of remaining cpacaties of each bin
    nodes = []  # array that will contain created nodes and not processed

    # create the root node with 0 bins and 0 objects
    node = NodeL(0, usedBins, cRemaining)
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
                        if (i == usedBins):  # new Bin is added
                            newNode = NodeL(nbObjects + 1, usedBins + 1, newCremaining)
                        else:  # the bin is already added
                            newNode = NodeL(nbObjects + 1, usedBins, newCremaining)

                        nodes.append(newNode)
    time = timeit.default_timer() - time
    return minBins, time
##################################Profondeur####################################
class Node :
    
    def __init__(self,nbObjects,curBin,nbBins, cRemaining,last):
        
        self.nbObjects=nbObjects           # number of the first k objects packing
        self.nbBins=nbBins                 # number of bins used to pack the first k objects
        self.cRemaining=cRemaining         # a table of remaining capacities of each one of nbBins used
        self.curBin=curBin
        self.last=last

        
    def getNbObjects(self):
        return self.nbObjects
    
    def getNbBins(self):
        return self.nbBins
    
    def getCurBin(self):
        return self.curBin
        
    def getIcRemaining(self,i):
        return self.cRemaining[i]
    
    def getCremaining(self):
        return self.cRemaining
    
    def getLast(self):
        return self.last
    

    
    def printNode(self):
        print("objects",self.nbObjects)
        print("nbBins",self.nbBins)
        print("curBin",self.curBin)
        print("cRem",self.cRemaining)
        print("last",self.last)

    
def branchAndBoundP(n,objects,c):
    
    # n: number of objects which is the max number of bins we can use
    # objects: table of weights of objects
    # c: capacity of each bin
    objects=-np.sort(-objects)
    time = timeit.default_timer()
    minBins= n # initialize upper bound (Sup)
    usedBins=0 # initialize the number of used Bins
    cRemaining= [c]*n # initialize the table of remaining cpacaties of each bin
    nodes=[] # array that will contain created nodes and not processed
    curBin=0
    cumWeight=[]
    precWeight=0
    newCremaining=cRemaining.copy()
    last=False
    
    for i in range (n):
        cumWeight.append(precWeight+objects[i])
        precWeight=precWeight+objects[i]
        j=0
        while newCremaining[j]< objects[i]:
            j+=1
        
        if j==usedBins:
            last=True
            usedBins+=1
        else:
            last=False
        newCremaining[j]-=objects[i]
        nodes.append(Node(i,j,usedBins,newCremaining.copy(),last))
    minBins=usedBins
    
    while (len(nodes)>0):
 
            node=nodes.pop(-1)
            nbObjects=node.getNbObjects()+1
            last=node.getLast()
            tRemaining=node.getCremaining().copy()
            usedBins=node.getNbBins()
            if(nbObjects==n):
                    if(usedBins<minBins):
                        minBins=usedBins
                    while(last  and len(nodes)>0):
                        node=nodes.pop(-1)
                        last=node.getLast()
                    curBin=node.getCurBin()+1
                    
                    node=nodes.pop(-1)
                    nbObjects=node.getNbObjects()+1
                    tRemaining=node.getCremaining().copy()
                    usedBins=node.getNbBins()
                        
            while(tRemaining[curBin]< objects[nbObjects]):
                curBin+=1
            if (curBin==usedBins):
                last=True
                usedBins+=1
            else:
                last=False
            oRemaining=tRemaining[:usedBins]
            oRemaining[curBin]=oRemaining[curBin]-objects[nbObjects]
            large=max(oRemaining)
            j=0
            
            if(nbObjects+1<n and large>=objects[n-1]):
                j=1
            ev=math.ceil(usedBins+(cumWeight[n-1]-cumWeight[nbObjects]-j*(usedBins*c-cumWeight[nbObjects]))/c)
            while(ev>minBins and not last):
                        curBin+=1
                        while(tRemaining[curBin]< objects[nbObjects]):
                                curBin+=1
                        if (curBin==usedBins):
                                last=True
                                usedBins+=1
                        else:
                                
                                last=False
                        oRemaining=tRemaining[:usedBins]
                        oRemaining[curBin]=oRemaining[curBin]-objects[nbObjects]
                        large=max(oRemaining)
                        j=0
                        if(nbObjects+1<n and large>=objects[n-1] ):
                            j=1
                        ev=math.ceil(usedBins+(cumWeight[n-1]-cumWeight[nbObjects]-j*(usedBins*c-cumWeight[nbObjects]))/c)
            
            if(ev<=minBins):
                nodes.append(node)
                tRemaining[curBin]=tRemaining[curBin]-objects[nbObjects]
                nodes.append(Node(nbObjects,curBin,usedBins,tRemaining.copy(),last))
                curBin=0
                nbObjects=nbObjects+1
            else:
                while (last and len(nodes)>0):
                        node=nodes.pop(-1)
                        last=node.getLast()
                curBin=node.getCurBin()+1
    time = timeit.default_timer()-time
    return minBins, time
#######################Méta Heuristiques #######################################"
####################### GA #######################################"
# identifying the structure of Item (Object) and Candidate (A possible solution)
Item = namedtuple("Item", ['id', 'size'])
Candidate = namedtuple("Candidate", ['items', 'fitness'])
# The function cost returns the cost of a collection of bins that is represented by these bins length
def cost(bins):
    return len(bins)
#________________________________________The class Bin________________________________________________________
class Bin(object):
    count = itertools.count()
    
    #This procedure simply create a new bin
    #The items contained on a bin are defined by their index in its items table
    def __init__(self, capacity):
        self.id = next(Bin.count)
        self.capacity = capacity
        self.free_space = capacity
        self.items = []
        self.used_space = 0
    
    #This procedure adds a new item to a specific bin and update the values of used and free spaces
    def add_item(self, item):
        self.items.append(item)
        self.free_space -= item.size
        self.used_space += item.size
    
    #This procedure removes a specific item in a specific bin and update the values of used and free spaces
    def remove_item(self, item_index):
        item_to_remove = self.items[item_index]
        del self.items[item_index]
        self.free_space += item_to_remove.size
        self.used_space -= item_to_remove.size

    
    #This function allows us to know if we can add a specific item to a  specific bin or not
    def fits(self, item):
        return self.free_space >= item.size

    
    #This function display the content of a specific bin
    def __str__(self):
        items = [str(it) for it in self.items]
        items_string = '[' + ' '.join(items) + ']'
        return "Bin n° " + str(self.id) + " containing the " + \
               str(len(self.items)) + " following items : " + items_string + \
               " with " + str(self.free_space) + " free space."
    
    #This function copy a bin into a new one with keeping the old bin
    def __copy__(self):
        new_bin = Bin(self.capacity)
        new_bin.free_space = self.free_space
        new_bin.used_space = self.used_space
        new_bin.items = self.items[:]
        return new_bin
        
        #Now we will be introducing the different procedures and functions
#that allows us to build our Genetic Algorithm :
# ----> Different Heuritic methods to help us define the fitness function
# ----> Population generator
# ----> Parents selection algorithms
# ----> Crossover algorithm
# ----> Mutation algorithm
#___________________________________________Heuristics_________________________________________________
def firstfit(items, current_bins, capacity):
    bins = [copy.copy(b) for b in current_bins]
    if not bins:
        bins = [Bin(capacity)]
    for item in items:
        if item.size > capacity:
            continue
        first_bin = next((bin for bin in bins if bin.free_space >= item.size), None)
        if first_bin is None:
            bin = Bin(capacity)
            bin.add_item(item)
            bins.append(bin)
        else:
            first_bin.add_item(item)
    return bins
#_______________________________Population Generator____________________________
# This function generates the initial population randomly
def population_generator(items, capacity, population_size, greedy_solver):
    candidate = Candidate(items[:], fitness(items, capacity, greedy_solver))
    population = [candidate]
    #print(len(population[0].fitness))
    new_items = items[:]
    for i in range(population_size - 1):
        shuffle(new_items)
        candidate = Candidate(new_items[:], fitness(new_items, capacity, greedy_solver))
        if candidate not in population:
            population.append(candidate)
    return population

#____________________________________fitness___________________________
def fitness(candidate, capacity, greedy_solver):
    if greedy_solver == 'FF':
        return firstfit(candidate,[], capacity)
        
#________________________________Selection Methods__________________
# In K-Way tournament selection, we select K individuals from the population
# at random (using a geometric function)
# and select the best out of these to become a parent with a trournament_selection probability.
# The same process is repeated for selecting the next parent.
def tournament_selection(population, tournament_selection_probability, k):
    candidates = [population[(randint(0, len(population) - 1))]]
    while len(candidates) < k:
        new_indiv = population[(randint(0, len(population) - 1))]
        if new_indiv not in candidates:
            candidates.append(new_indiv)
    ind = int(np.random.geometric(tournament_selection_probability, 1))
    while ind >= k:
        ind = int(np.random.geometric(tournament_selection_probability, 1))
    return candidates[ind]
    
    
def roulette_wheel_selection(population):
    max = sum([len(e.fitness) for e in population])
    pick = uniform(0, max)
    current = max
    for item in population:
        current -= len(item.fitness)
        if current < pick:
            return item
            
            
def SUS(population, n):
    selected = []
    pointers = []
    max = sum([len(e.fitness) for e in population])
    distance = max / n
    start = uniform(0, distance)
    for i in range(n):
        pointers.append(start + i * distance)
    for pointer in pointers:
        current = 0
        for item in population:
            current += len(item.fitness)
            if current > pointer:
                selected.append(item)
    return selected


def rank_selection(population):
    length = len(population)
    rank_sum = length * (length + 1) / 2
    pick = uniform(0, rank_sum)
    current = 0
    i = length
    for item in population:
        current += i
        if current > pick:
            return item
        i -= 1

#______________________________Crossover____________________________
def crossover(parent1, parent2):
    taken = [False] * len(parent1)
    child = []
    i = 0
    while i < len(parent1):
        element = parent1[i]
        if not taken[element.id]:
            child.append(element)
            taken[element.id] = True
        element = parent2[i]
        if not taken[element.id]:
            child.append(element)
            taken[element.id] = True
        i += 1
    return child
    
#______________________________________Mutation_____________________
def mutation(member, capacity, greedy_solver):
    member_items = member.items
    a = randint(0, len(member_items) - 1)
    b = randint(0, len(member_items) - 1)
    while a == b:
        b = randint(0, len(member_items) - 1)
    c = member_items[a]
    member_items[a] = member_items[b]
    member_items[b] = c
    member = Candidate(member_items, fitness(member_items, capacity, greedy_solver))
    return member
    
    
#______________________________________Genetic Algorithm________________
def genetic_algorithm(weights, capacity, population_size=50, generations=250, max_no_change=50,k=20, tournament_selection_probability=0.7, crossover_probability=0.6, mutation_probability=0.3, greedy_solver="FF", allow_duplicate_parents=False, selection_method="RW"):
    time = timeit.default_timer()
    items = [Item]
    items = [ Item(i,weights[i]) for i in range(len(weights))]
    population = population_generator(items, capacity, population_size, greedy_solver)
    best_solution = fitness(items, capacity, greedy_solver)
    i = 0
    current_iteration = 0
    num_no_change = 0
    while current_iteration < generations and num_no_change < max_no_change:
        new_generation = []
        best_child = best_solution
        for j in range(population_size):


            if selection_method == 'SUS':
                first_parent = SUS(population, 1)[0].items
                second_parent = SUS(population, 1)[0].items
                if not allow_duplicate_parents:
                    while first_parent == second_parent:
                        second_parent = SUS(population, 1)[0].items
            elif selection_method == 'TS':
                first_parent = tournament_selection(population, tournament_selection_probability, k).items
                second_parent = tournament_selection(population, tournament_selection_probability, k).items
                if not allow_duplicate_parents:
                    while first_parent == second_parent:
                        second_parent = tournament_selection(population, tournament_selection_probability, k).items
            elif selection_method == 'RW':
                first_parent = roulette_wheel_selection(population).items
                second_parent = roulette_wheel_selection(population).items
                if not allow_duplicate_parents:
                    while first_parent == second_parent:
                        second_parent = roulette_wheel_selection(population).items
            elif selection_method == 'RS':
                first_parent = rank_selection(population).items
                second_parent = rank_selection(population).items
                if not allow_duplicate_parents:
                    while first_parent == second_parent:
                        second_parent = rank_selection(population).items
            else:
                return

            child = crossover(first_parent, second_parent)
            child = Candidate(child[:], fitness(child, capacity, greedy_solver))

            prob = random()
            if prob <= mutation_probability:
                child = mutation(child, capacity, greedy_solver)

            if len(child.fitness) < len(best_child):
                best_child = child.fitness
            new_generation.append(child)

        num_no_change += 1
        if len(best_child) < len(best_solution):
            best_solution = best_child
            num_no_change = 0
        population = [Candidate(p.items[:], p.fitness) for p in new_generation]
        population.sort(key=lambda candidate: len(candidate.fitness), reverse=True)
        #if i==30:
         #   print([len(population[l].fitness) for l in range(len(population))])
        current_iteration += 1
    #for d in range(len(best_solution)):
     #   print(best_solution[d].__str__())
    #print("iterations number ",current_iteration,"stagnation: ", num_no_change)
    time = timeit.default_timer()-time
    return len(best_solution), time
    
###################RS##########################
#Classe Objet
class Objet:
    def __init__(self, size):
        self.size = size
    def get_size(self):
        return self.size
    
#Classe Bin'''
class Boite:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = []

    def add_item(self, new_item):
        #Ajoute un objet au bin si la capacité restante de ce dernierle permet
        if self.can_add_item(new_item):
            self.items.append(new_item)
            return True
        return False

    def can_add_item(self, new_item):
        #Determine si l'objet peut etre ajouté au bin en comparant la capacité actuelle du bin avec la taille de l'objet
        return new_item.size <= self.open_space()

    def filled_space(self):
        #Reourne l'espace occupé du bin qui est la somme des tailles des objets contenus dans ce bin
        return sum(item.size for item in self.items)

    def open_space(self):
        
        return self.capacity - self.filled_space()
    def afficher_contenu(self):
        print(item.size for item in self.items)
    
    def get_items(self):
        return [item.get_size() for item in self.items]
    
class FirstFit():
    def apply(item, bins):
        b = next((b for b in bins if b.can_add_item(item)), None)
        if not b:
            b = Boite(bins[0].capacity)
            bins.append(b)
        b.add_item(item)
        return bins
#Le recuit simulé'''
class RS(object):
    def __init__(self,capacity, items,alpha=0.9,t_init=500,t_target=10,iter_nb=7):
        self.alpha = alpha
        self.items = items
        self.capacity = capacity
        self.bins = [Boite(capacity)]
        self.t_init = t_init
        self.t_target = t_target
        self.iter_nb = iter_nb
        
    def executer(self):
        time = timeit.default_timer()
        # Initial solution generated with first fit method
        for item in self.items:
            self.bins = FirstFit.apply(item, self.bins)
            # Initialize temperature
        
        t = self.t_init
            # Average to temprature to separate
        t_average = (self.t_init + self.t_target) / 2
            # iterate
        while t > self.t_target:
            for i in range(self.iter_nb):

                neighbour = self._swap_11()

                delta = self._objective_function(neighbour) - self._objective_function(self.bins)
                if delta > 0:
                    self.bins = copy.deepcopy(neighbour)
                else:
                    u = np.random.random()
                    if (u < np.exp(delta/t)):
                        self.bins = copy.deepcopy(neighbour)
            t = self.alpha * t
        time = timeit.default_timer()-time
        return len(self.bins), time

    # move a random element from a random bin and to another random bin
    def _swap_10(self):
        neighbour = copy.deepcopy(self.bins)
        b_index = np.random.randint(low=0,high=len(neighbour))
        bin_to_remove_from = neighbour[b_index]
        i_index = np.random.randint(low=0,high=len(bin_to_remove_from.items))
        item_to_move = bin_to_remove_from.items[i_index]
        del bin_to_remove_from.items[i_index] #Suppression de l'objet à déplacer dans le bin d'origine
        neighbour[b_index] = bin_to_remove_from
        cont = True
        while cont:
            bin = neighbour[np.random.randint(low=0,high=len(neighbour))]
            if bin.can_add_item(item_to_move):
                bin.add_item(item_to_move)
                cont = False
        if len(bin_to_remove_from.items) == 0:
            del neighbour[b_index]
        return neighbour

    # swap two random elements from two random bins
    def _swap_11(self):
        neighbour = copy.deepcopy(self.bins)
        cont = True
        while cont:
            b_index1, b_index2 = np.random.randint(low=0,high=len(self.bins),size=2)
            bin1 = neighbour[b_index1]
            bin2 = neighbour[b_index2]
            i_index1 = np.random.randint(low=0,high=len(bin1.items))
            i_index2 = np.random.randint(low=0,high=len(bin2.items))
            item1 = bin1.items[i_index1]
            item2 = bin2.items[i_index2]
            if (bin1.filled_space() - item1.size + item2.size <= self.capacity) and (bin2.filled_space() - item2.size + item1.size <= self.capacity) :
                cont = False
                bin1.items[i_index1] = item2
                bin2.items[i_index2] = item1
                neighbour[b_index1] = bin1
                neighbour[b_index2] = bin2
                break
        return neighbour

    def _objective_function(self,bins):
        S = 0
        for bin in bins:
            s = 0
            for item in bin.items:
                s += item.size
            S = ( S + s ** 2 )
        return S
###########Exuction#####################"
if Fonctionnalite =='Exécution':
    XT= get_dataset(dataset_name)
    lignes = XT.readlines()
    n=int(lignes[0])
    c=int(lignes[1])
    weight = [ int(x) for x in lignes[2:len(lignes)] ]
    weight=np.asarray(weight)
    #st.write('Shape of dataset:', lignes.shape(0))
    #st.write(lignes)
    #print(weight)
    st.write('Bin capacity : ', c)
    st.write('Number ob objects :', n)
    st.write('Solution exacte :',exacts[dataset_name])
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

###########Comparaison###########################"
if Fonctionnalite == 'Comparaison':
    comparaison_types = st.sidebar.selectbox(
    'Selectionner les méthodes',
    ('Méthode exacte','Méthodes heuristiques', 'Métaheuristique', 'Heuristiques Vs Métaheuristique'))
    if comparaison_types=='Méthode exacte':
        st.write(" ")
        st.write(" ")
        st.write("#### Temps d'éxéctuion Branch And Bound")
        st.write(" ")
        st.write(" ")
        df = pd.read_csv("Temps_Exec_BranchAndBound.csv", encoding="ISO-8859-1")
        instances = df['instances']
        y= df.iloc[:, 1]
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(instances, y, label='Branch and Bound')

        ax.legend()
        plt.xlabel('Instances')
        # naming the y-axis
        plt.ylabel('Temps dexécution')
        plt.legend()

        plt.show()
        st.pyplot()
    ############# 1-  Comparaison de la solution des heuristiques
    if comparaison_types=='Méthodes heuristiques':
        st.write(" ")
        st.write(" ")
        st.write("#### Comparaison des solutions des heuristiques et de la solution exacte")
        st.write(" ")
        st.write(" ")
        col_names = ["FF", "FFD","BF","BFD","NF","WF","Solution_Exacte"]
        df = pd.read_csv("NB_bin_heuristique.csv", encoding="ISO-8859-1")

        fig, ax = plt.subplots(figsize=(20, 5))
        # The x-values of the bars.
        instances = df['instances']
        x = np.arange(len(instances))

        # The width of the bars (1 = the whole width of the 'year group')
        width = 0.10

        # Create the bar charts!
        ax.bar(x - 3 * width / 2, df['FF'], width, label='FF', color='#0343df')
        ax.bar(x - width / 2, df['FFD'], width, label='FFD', color='#e50000')
        ax.bar(x + width / 2, df['BF'], width, label='BF', color='#ffff14')
        ax.bar(x + 3 * width / 2, df['BFD'], width, label='BFD', color='#929591')
        ax.bar(x - 5 * width / 2, df['NF'], width, label='NF', color='#00FF23')
        ax.bar(x + 5 * width / 2, df['WF'], width, label='WF', color='#FF00F0')
        ax.bar(x - 7 * width / 2, df['SE'], width, label='SE', color='#FF8000')
        # Notice that features like labels and titles are added in separate steps
        ax.set_ylabel('Nombre de bins')
        ax.set_title('Solutions des heuristiques')

        ax.set_xticks(x)  # This ensures we have one tick per year, otherwise we get fewer
        ax.set_xticklabels(instances.astype(str).values, rotation='vertical')
        st.set_option('deprecation.showPyplotGlobalUse', False)

        ax.legend()
        plt.show()
        st.pyplot()

        #####################2- comapraison temps d'execution heuristique
        st.write(" ")
        st.write(" ")
        st.write("#### Comparaison des temps d'éxéctuion des heuristiques")
        st.write(" ")
        st.write(" ")
        df = pd.read_csv("Temps_Exec_Heuristique.csv", encoding="ISO-8859-1")
        instances = df['instances']
        y = np.zeros((6, 12))
        y[0] = df.iloc[:, 1]
        y[1] = df.iloc[:, 2]
        y[2] = df.iloc[:, 3]
        y[3] = df.iloc[:, 4]
        y[4] = df.iloc[:, 5]
        y[5] = df.iloc[:, 6]
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(instances, y[0], label='FF')
        ax.plot(instances, y[1], label='FFD')
        ax.plot(instances, y[2], label='BF')
        ax.plot(instances, y[3], label='BF')
        ax.plot(instances, y[4], label='BF')
        ax.plot(instances, y[5], label='BF')

        ax.legend()
        plt.xlabel('Instances')
        # naming the y-axis
        plt.ylabel('Temps dexécution')
        plt.legend()

        plt.show()
        st.pyplot()
        
    ##################################3- Nb bins méta heuristiques
    if comparaison_types=='Métaheuristique':
        st.write(" ")
        st.write(" ")
        st.write("#### Comparaison des solutions des métaheuristiques et de la solution exacte")
        st.write(" ")
        st.write(" ")
        col_names = ["AG", "RS", "SE"]
        df = pd.read_csv("NB_bin_Metaheuristique.csv", encoding="ISO-8859-1")

        fig, ax = plt.subplots(figsize=(20, 5))
        # The x-values of the bars.
        instances = df['instances']
        x = np.arange(len(instances))

        # The width of the bars (1 = the whole width of the 'year group')
        width = 0.10

        # Create the bar charts!
        ax.bar(x - 3 * width / 2, df['AG'], width, label='AG', color='#0343df')
        ax.bar(x - width / 2, df['RS'], width, label='RS', color='#e50000')
        ax.bar(x + width / 2, df['SE'], width, label='SE', color='#ffff14')
        # Notice that features like labels and titles are added in separate steps
        ax.set_ylabel('Nombre de bins')
        ax.set_title('Solutions des métaheuristiques')

        ax.set_xticks(x)  # This ensures we have one tick per year, otherwise we get fewer
        ax.set_xticklabels(instances.astype(str).values, rotation='vertical')
        st.set_option('deprecation.showPyplotGlobalUse', False)

        ax.legend()
        plt.show()
        st.pyplot()
    ####################4- comapraison temps d'execution des métaheuristique
        st.write(" ")
        st.write(" ")
        st.write("#### Comparaison des temps d'éxéctuion des métaheuristiques")
        st.write(" ")
        st.write(" ")
        df = pd.read_csv("Temps_Exec_MetaHeuristique.csv", encoding="ISO-8859-1")
        instances = df['instances']
        y = np.zeros((2, 12))
        y[0] = df.iloc[:, 1]
        y[1] = df.iloc[:, 2]
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(instances, y[0], label='AG')
        ax.plot(instances, y[1], label='RS')

        ax.legend()
        plt.xlabel('Instances')
        # naming the y-axis
        plt.ylabel('Temps dexécution')
        plt.legend()

        plt.show()
        st.pyplot()
    if comparaison_types=='Heuristiques Vs Métaheuristique':
    ############# 5-  Comparaison de la solution des méthodes approchées
        st.write(" ")
        st.write(" ")
        st.write("#### Comparaison des solutions des méthodes approchées et de la solution exacte")
        st.write(" ")
        st.write(" ")
        col_names = ["BF","BFD","AG","Solution_Exacte"]
        df = pd.read_csv("NB_bin_heuristique_metaheuristique.csv", encoding="ISO-8859-1")

        fig, ax = plt.subplots(figsize=(20, 5))
        # The x-values of the bars.
        instances = df['instances']
        x = np.arange(len(instances))

        # The width of the bars (1 = the whole width of the 'year group')
        width = 0.10

        # Create the bar charts!
        ax.bar(x - 3 * width / 2, df['BF'], width, label='BF', color='#0343df')
        ax.bar(x - width / 2, df['BFD'], width, label='BFD', color='#e50000')
        ax.bar(x + width / 2, df['AG'], width, label='AG', color='#ffff14')
        ax.bar(x + 3 * width / 2, df['SE'], width, label='SE', color='#929591')
        # Notice that features like labels and titles are added in separate steps
        ax.set_ylabel('Nombre de bins')
        ax.set_title('Solutions des méthodes approchées')

        ax.set_xticks(x)  # This ensures we have one tick per year, otherwise we get fewer
        ax.set_xticklabels(instances.astype(str).values, rotation='vertical')
        st.set_option('deprecation.showPyplotGlobalUse', False)

        ax.legend()
        plt.show()
        st.pyplot()

#with open('historique.csv','w',newline='') as fichiercsv:
 #   writer=csv.writer(fichiercsv)
  #  writer.writerow(['Type de l''instance','Instance', 'Algorithme', 'Nombre d''objets', 'Nombre de bins', 'Temps d''éxécution','Paramètres'])
