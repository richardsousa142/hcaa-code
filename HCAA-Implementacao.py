'''
HCAA SIMPLIFICADO, SEM LIMIT ON GROWTH
'''
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as hr
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
from matplotlib import pyplot as plt

class Tree:
    '''
    Classe Tree serve para criar a árvore que será usada no estágio adicional e nos ajudará a
    atribuir peso para os clusters e ativos contidos nesses clusters.

    Parameters
    ---------------
    value: int
           É o valor que será guardado por nó, no nosso caso esse valor guardará o indice do cluster criado
           ou o indice dos ativos, que são as folhas.
    weight: float
           É o peso que será atribuido a cada cluster ou ativo
    '''
    def __init__(self, value, weight = None) -> None:
        self.left = None
        self.right = None
        self.data = value
        self.weight = weight

def get_stocks(asset, s_date, e_date):
  '''
    Essa função como o próprio nome já diz tem como objetivo pegar os dados dos ativos que iremos usar
    para performar todo o processo de criação do portfolio.

    Para obter esses dados usamos a API de finanças do Yahoo, yfinance

    Parameters
    ------------
    asset: list
           Lista de strings onde cada string é o ticker do ativo que desejamos obter os dados
    s_date: str
            É a string que usamos para informar a data de inicio para obter os dados, ou seja, queremos
            que os dados obtidos seja a partir dessa data.
    e_date: str
            Semelhante ao s_date, e_date é a data final que usamos para informar até o momento que
            queremos os dados.

    return
    ------------
    data : dataframe pandas
            Dataframe contendo todos os dados dos ativos que passamos como parametro para a API
  '''
  data = yf.download(asset, start = s_date, end = e_date)['Adj Close']
  return data

def get_correlation(data):
  '''
  Obtem a correlação da matriz data, para obter a correlação usamos o metodo  Pearson

  Parameters
  ------------
  data: dataframe pandas
          É a matriz obtida através da API do Yahoo e contem todos os dados dos ativos passados

  Return
  ------------
  data.corr : dataframe pandas
          Matriz contendo a correlação entre os ativos
  '''
  return data.corr(method='pearson')

def calc_distance(correlation):
    '''
    O processo de hierarchical clustering precisa de uma medida de distance, então para isso
    usaremos a medida de Mantegna 1999.

    Di,j = √0.5 * (1 - Pi,j) *P = representação de rho

    Parameters
    ------------
    correlation: dataframe pandas
        Dataframe contendo a correlação entre os ativos

    Return
    ------------
    distance: ndarray
        Matriz contendo a distancia calculada com base na correlação fornecida
    '''
  distance = np.sqrt(0.5 * (1 - correlation))
  return distance

def hierarchical_clustering(euclidean_distance, linkage):
    '''
    Performa o calculo da matriz de linkage usando a biblioteca scipy e o metodo linkage

    Parameters
    ------------
    euclidean_distance: ndarray
        Matriz contendo a distancia euclidiana
    linkage: str
        String para informar qual será o metodo de linkage utilizado

    Return
    ------------
    clustering_matrix: ndarray
        A matriz de hierarchical clustering codificada como uma matriz de link
    '''
  clustering_matrix = hr.linkage(euclidean_distance, method = linkage, optimal_ordering = True)
  return clustering_matrix

def get_idx_cluster_merged(clustering, len_stocks):
    '''
    Através da matriz de linkage podemos obter os indices dos clusters que foram combinados
    usando para isso a coluna 1 e 2 dessa matriz.

    Essa função retorna um dicionario onde a chave é um indice do cluster criado para comportar
    os dois outros clusters que foram combinados, já os valores são os clusters que foram
    combinados.

    Caso tenha um ponto de cutoff o indice que antes teria como valor dois clusters
    agora será formado portodas as folhas que pertenciam aos clusters anteriores ao indice em
    questão.

    Parameters
    -----------------
    clustering : ndarray
                É a matriz de linkage resultante do processo de hierarchical clustering
    len_stocks: int
                É o tamanho da lista contendo os ativos de interesse

    return
    -----------------
    cluster_merged: dict
                Dicionario contendo como chave os indices dos clusters que foram criados através
                da combinação de outros, e valores os indices dos clusters que foram envolvidos na
                criação do clusters
    '''
  cluster_merged = {}
  #idxs_cla e idxs_clb sao os indices dos clusters na matriz de link(clustering)
  idxs_cla = clustering[:, 0].tolist(); idxs_clb = clustering[:, 1].tolist()
  for i in range(len(clustering)):
    cluster_merged[f"{len_stocks + i}"] = [int(idxs_cla[i]) if idxs_cla[i] < 10 else int(idxs_cla[i]), int(idxs_clb[i]) if int(idxs_clb[i]) < 10 else int(idxs_clb[i])]
  return cluster_merged

def create_tree_from_clusters(cluster, dicio, raiz):
    '''
    Nessa funcao temos como objetivo pegar o dicionario e usar os dados contidos ali para criar a arvore.
    Para isso, recebemos os valores da ultima chave no parametro 'cluster', com isso verificamos o valor
    de cluster[0] >= 10, se for maior de 10 significa que foi um cluster criado no processo de linkage e
    portanto tem ativos ou clusters associados a ele. O mesmo vale para o cluster[1].

    Usamos cluster[0] para representar os ativos ou clusters que estão contidos na subarvore esquerda, e
    cluster[1] para representar os ativos ou clusters que estão contidos na subarvore direita.

    Tendo essas duas explicações prévias, começamos o codigo de fato.

    Se cluster[0] >= 10 criamos o nó esquerdo da arvore contendo esse valor, pois abaixo dele terá dois ativos
    ou outros clusters. O mesmo vale para o cluster[1].

    A função art_leaf_node sera chamada quando o valor contido em cluster[0] ou cluster[1] for uma lista

    A cada chamada recursiva da função passamos o novo valor de cluster, e também passamos a raiz.left ou
    raiz.right para que nas proximas execuções a arvore continue sendo criada

    Exemplo esquematico
    ------------------------------------------------------------------------------------------------
                [14, 17]
        [14]                 [17]
     [8]    [4]        [16]         [15]
                    [3]    [13]  [7]     [12]
                        [9]    [5]   [11]    [2]
                                  [1]    [10]
                                      [6]    [0]

    Parameters
    ----------
    cluster : list
        Lista contendo dois elementos, esses elementos são os indices dos clusters que foram combinados na
        criação do ultimo cluster
    dicio : dict
        Dicionario onde a chave é o indice dos clusters que foram criados através da combinação de outros
        clusters, e os valores são os indices desses clusters
    raiz : Tree
        Arvore que iremos criar com base no dicionario e os clusters que sao seus valores

    Return
    ------
    cluster : list
        Lista contendo dois elementos ou apenas um, esses elementos são os indices dos clusters que
        foram combinados na criação do ultimo cluster
    '''
    if cluster[0] < 10 and cluster[1] < 10:
        raiz.left = Tree(cluster[0])
        raiz.right = Tree(cluster[1])
        return cluster

    if cluster[0] >= 10:
        raiz.left = Tree(cluster[0])
        cluster[0] = dicio[f"{int(cluster[0])}"];
        atr_leaf_node(cluster[0], raiz.left)
        create_tree_from_clusters(cluster[0], dicio, raiz.left)
    if cluster[1] >= 10:
        raiz.right = Tree(cluster[1])
        cluster[1] = dicio[f"{int(cluster[1])}"];
        atr_leaf_node(cluster[1], raiz.right)
        create_tree_from_clusters(cluster[1], dicio, raiz.right)
    return cluster

def atr_leaf_node(cluster, raiz):
    '''
    Na função create_tree temos um problema que os valores menores que 10 não seriam colocados na arvore,
    isso acontece devido ao fato de so querermos valores maiores que 10, portanto essa função foi criada
    ela verifica se um dos valores é menor que 10 e se o outro é maior que 10, e guarda o valor menor que
    10 na arvore

    Parameters
    ----------
    cluster : list
        Lista contendo dois elementos, esses elementos são os indices dos clusters que foram combinados na
        criação do ultimo cluster
    raiz : Tree
        Arvore para ser adicionado o valor menor que 10, que nesse caso será uma folha
    '''
    temp = cluster
    if temp[0] <= 10 and temp[1] >= 10:
        raiz.left = Tree(temp[0])
    if temp[0] >= 10 and temp[1] <= 10:
        raiz.right = Tree(temp[1])

def weight_tree(arvore):
    '''
    Essa função é usada para dar peso para os ativos e para os cluster, como um dendrogram é uma arvore binaria
    criamos uma arvore e definimos o peso para os nós.

    Parameters
    ----------
    arvore : Tree
        Raiz da arvore

    Return
    --------------
    vet_weight : list
        vetor contendo o peso de cada folha da arvore
    '''
    arvore_aux = arvore
    pilha = list()
    weight = 0
    vet_weight = []
    while ( (arvore_aux != None) or (not pilha_vazia(pilha)) ):
        while arvore_aux != None:
            pilha.append(arvore_aux)
            weight = arvore_aux.weight
            arvore_aux = arvore_aux.left
            set_weight(arvore_aux, weight)
        arvore_aux = pilha.pop()
        if arvore_aux.right == None and arvore_aux.left == None: #Se no for folha
          vet_weight.append(arvore_aux.weight)
        weight = arvore_aux.weight
        arvore_aux = arvore_aux.right
        set_weight(arvore_aux, weight)
    return vet_weight

def set_weight(arvore, weight):
    '''
    Função auxiliar para atribuir o peso para o novo cluster ou para um ativo caso seja um nó folha.
    Nesse caso como estamos seguindo o paper de Thomas Raffinot apenas pegamos o peso do nó acima na
    arvore e dividimos por 2 para termos o novo peso.

    Parameters
    arvore : Tree
        raiz da arvore que usaremos para guardar o novo peso
    weight : float
        Peso do nó acima na arvore do nó que recebemos. Dessa forma, conseguimos calcular o peso para o nó arvore recebido,
        como no caso do paper do Thomas Raffinot ele usa o sistema de peso igual entre os clusters, aqui apenas pegamos o
        peso do nó acima e dividimos por 2.
    '''
    if arvore != None: arvore.weight = weight / 2

def pilha_vazia(pilha):
    '''
    Função auxiliar que serve somente para sabermos se a stack esta vazia.

    Parameters
    ----------
    pilha : list
        Pilha é uma lista que esta sendo usada como stack

    Return
    len(pilha) : bool
        Retorna verdadeiro caso a pilha esteja vazia, ou falso caso tenha pelo menos 1 elemento
    '''
    return len(pilha) == 0

def main():
  asset = ['MSFT', 'PCAR', 'JPM', 'AAPL', 'GOOGL', 'AMZN', 'ITUB', 'VALE', 'SHEL', 'INTC']
  start = '2016-01-01'; end = '2022-01-01'
  data_stocks = get_stocks(asset, start,  end)

  # Stage 1: Hierarchical Clustering

  correlation = get_correlation(data_stocks)
  distance = calc_distance(correlation)
  distance_euclidean = euclidean_distance_improve(len(asset), distance)
  clustering = hierarchical_clustering(distance_euclidean, 'ward')

  # Additional Stage to aux the weight stage

  #dicionario contendo os indices dos clusters gerados da combinação de outros 2
  cluster_merged = get_idx_cluster_merged(clustering, len(asset))
  #lista contendo as chaves do dicionario
  keys = list(cluster_merged.keys())
  #ultimos clusters merged
  cluster = cluster_merged[keys[-1]]
  #Cria arvore onde a raiz é os dois ultimos clusters combinados
  raiz = Tree([13, 17], 100)
  #criando a arvore atravez dos dois ultimos clusters combinados
  #usando o dicionario para mapear os clusters que foram criados atraves da combinação
  #passando a arvore para prencher o campo data de cada no
  create_tree_from_clusters(cluster, cluster_merged, raiz)

  #Stage 2: Assigning weights to clusters

  #vetor com o peso de cada ativo e cluster
  vet_weight = weight_tree(raiz)
  #dicionario mapeando o ativo e seu respectivo peso no portfolio
  dict_asset_weight = {key: f'{value}%' for key, value in zip(list(hr.leaves_list(clustering)), vet_weight)}

  #Print of dendrogram
  print_cluster(clustering)
  print(dict_asset_weight)
  dendrogram(clustering)

if __name__ == "__main__":
    main()