import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import tkinter as tk
from tkinter import ttk, messagebox
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter


# Função para calcular a entropia
def calcular_entropia(y):
    contador = Counter(y)
    total = len(y)
    entropia = -sum((count / total) * np.log2(count / total) for count in contador.values())
    return entropia


# Função para calcular a informação
def calcular_informacao(y, y_esquerda, y_direita):
    entropia_total = calcular_entropia(y)
    proporcao_esquerda = len(y_esquerda) / len(y)
    proporcao_direita = len(y_direita) / len(y)
    entropia_esquerda = calcular_entropia(y_esquerda)
    entropia_direita = calcular_entropia(y_direita)
    informacao = entropia_total - (proporcao_esquerda * entropia_esquerda + proporcao_direita * entropia_direita)
    return informacao


# Definindo os dados de treinamento
X = [
    [0, 1, 0, 0],  # Ruim, Alta, Nenhuma, < 15.000
    [1, 1, 0, 1],  # Desconhecida, Alta, Nenhuma, >= 15.000 a <= 35.000
    [1, 0, 0, 1],  # Desconhecida, Baixa, Nenhuma, >= 15.000 a <= 35.000
    [1, 0, 0, 2],  # Desconhecida, Baixa, Nenhuma, > 35.000
    [1, 0, 0, 2],  # Desconhecida, Baixa, Nenhuma, > 35.000
    [1, 0, 1, 2],  # Desconhecida, Baixa, Adequada, > 35.000
    [0, 0, 0, 0],  # Ruim, Baixa, Nenhuma, < 15.000
    [0, 0, 1, 2],  # Ruim, Baixa, Adequada, > 35.000
    [2, 0, 0, 2],  # Boa, Baixa, Nenhuma, > 35.000
    [2, 1, 1, 2],  # Boa, Alta, Adequada, > 35.000
    [2, 1, 0, 0],  # Boa, Alta, Nenhuma, < 15.000
    [2, 1, 0, 1],  # Boa, Alta, Nenhuma, >= 15.000 a <= 35.000
    [2, 1, 0, 2],  # Boa, Alta, Nenhuma, > 35.000
]


y = [2, 2, 1, 2, 0, 0, 2, 1, 0, 0, 2, 1, 0]  # Classe de risco: Baixo, Moderado, Alto


# Definindo os nomes das variáveis
nomes_variaveis = ['História de crédito', 'Dívida', 'Garantias', 'Renda']


# Definindo as classes de risco
classes_risco = ['Baixo', 'Moderado', 'Alto']


# Criando o modelo de árvore de decisão binária
modelo_binario = DecisionTreeClassifier(criterion='entropy', max_features=1)


# Treinando o modelo
modelo_binario.fit(X, y)


# Função para prever o risco com base nos dados de entrada
def prever_risco(dados):
    # Convertendo os dados de entrada para uma lista
    dados_lista = list(dados)


    # Validando os valores de entrada
    if not all(isinstance(valor, int) for valor in dados_lista):
        raise ValueError("Todos os valores de entrada devem ser inteiros.")


    # Checando se os valores de entrada estão dentro do intervalo esperado
    for i, valor in enumerate(dados_lista):
        if valor < 0 or valor > 3:
            raise ValueError(f"O valor para {nomes_variaveis[i]} deve estar entre 0 e 3.")


    # Prevendo a classe de risco
    risco_predito = modelo_binario.predict([dados_lista])[0]


    # Retornando a classe de risco predita
    return classes_risco[risco_predito]


# Função para criar a árvore de decisão fixa
def criar_arvore_decisao_fixa():
    grafo = graphviz.Digraph(format='png')
    grafo.attr(rankdir='LR')


    # Definindo os nós
    grafo.node('1', label='História de crédito')
    grafo.node('2', label='Dívida')
    grafo.node('3', label='Renda')
    grafo.node('4', label='Garantias')
   
    # Folhas
    grafo.node('5', label='Risco = Alto', shape='box', color='lightblue')
    grafo.node('6', label='Risco = Moderado', shape='box', color='lightblue')
    grafo.node('7', label='Risco = Baixo', shape='box', color='lightblue')
    grafo.node('8', label='Risco = Baixo', shape='box', color='lightblue')
    grafo.node('9', label='Risco = Moderado', shape='box', color='lightblue')
    grafo.node('10', label='Risco = Baixo', shape='box', color='lightblue')


    # Conexões
    grafo.edge('1', '5', label='Ruim')
    grafo.edge('1', '2', label='Desconhecida')
    grafo.edge('1', '3', label='Boa')


    grafo.edge('2', '6', label='Alta')
    grafo.edge('2', '7', label='Baixa')


    grafo.edge('3', '8', label='> 35.000')
    grafo.edge('3', '4', label='<= 35.000')


    grafo.edge('4', '10', label='Adequada')
    grafo.edge('4', '9', label='Nenhuma')


    return grafo


# Função para criar a árvore de decisão manualmente com entropia e informação
def criar_arvore_decisao(valores_entrada, risco_predito):
    # Criando o grafo da árvore de decisão
    grafo = graphviz.Digraph(format='png')
    grafo.attr(rankdir='LR')


    # Adicionando o nó raiz
    grafo.node('raiz', label='Início')


    # Criando os nós internos e folhas com base nos valores de entrada e no risco previsto
    for i, valor_entrada in enumerate(valores_entrada):
        no_pai = 'raiz' if i == 0 else f'no_{i - 1}'
        no_filho = f'no_{i}'
        rotulo_no_filho = f'{nomes_variaveis[i]} = {valor_entrada}'


        # Adicionando o nó filho
        grafo.node(no_filho, label=rotulo_no_filho)


        # Adicionando a aresta entre o nó pai e o nó filho
        grafo.edge(no_pai, no_filho)


        # Checando se o nó filho é uma folha (risco previsto)
        if i == len(valores_entrada) - 1:
            rotulo_no_folha = f'Risco = {risco_predito}'


            # Adicionando o nó folha
            grafo.node(f'folha_{i}', label=rotulo_no_folha, shape='box', color='lightblue')


            # Adicionando a aresta entre o nó filho e o nó folha
            grafo.edge(no_filho, f'folha_{i}')


    # Retornando o grafo da árvore de decisão
    return grafo


# Função para criar e exibir a interface gráfica com Tkinter
def interface_usuario():
    # Função para prever o risco e exibir o resultado na interface
    def prever():
        try:
            valores_entrada = [int(entry.get()) for entry in entries]
            risco_predito = prever_risco(valores_entrada)
            resultado_label.config(text=f"Risco: {risco_predito}")


            # Criar a árvore de decisão e visualizar
            arvore_decisao = criar_arvore_decisao(valores_entrada, risco_predito)
            arvore_decisao.view()
        except ValueError as e:
            messagebox.showerror("Erro", str(e))


    # Função para visualizar a árvore de decisão fixa
    def visualizar_arvore_fixa():
        arvore_fixa = criar_arvore_decisao_fixa()
        arvore_fixa.view()


    # Inicializando a janela principal
    root = tk.Tk()
    root.title("Árvore de Decisão")


    # Criando os widgets
    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E))


    ttk.Label(frame, text="Insira os valores para as variáveis:").grid(row=0, column=0, columnspan=2)


    entries = []
    for i, nome_variavel in enumerate(nomes_variaveis):
        ttk.Label(frame, text=nome_variavel).grid(row=i + 1, column=0, sticky=tk.W)
        entry = ttk.Entry(frame)
        entry.grid(row=i + 1, column=1)
        entries.append(entry)


    prever_button = ttk.Button(frame, text="Prever Risco", command=prever)
    prever_button.grid(row=len(nomes_variaveis) + 1, column=0, columnspan=2)


    visualizar_fixa_button = ttk.Button(frame, text="Visualizar Árvore Fixa", command=visualizar_arvore_fixa)
    visualizar_fixa_button.grid(row=len(nomes_variaveis) + 2, column=0, columnspan=2)


    resultado_label = ttk.Label(frame, text="Risco: ")
    resultado_label.grid(row=len(nomes_variaveis) + 3, column=0, columnspan=2)


    # Iniciando o loop da interface gráfica
    root.mainloop()


# Chamando a interface de usuário
interface_usuario()



