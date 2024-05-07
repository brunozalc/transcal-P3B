import numpy as np
import matplotlib.pyplot as plt


def get_data():
    # get how many nodes with have in the system
    n = int(input("Quantos nós temos no sistema? "))

    # for each node, get the x and y coordinates
    nodes = []
    for i in range(n):
        x, y = map(int, input(f"Digite as coordenadas do nó {i+1}: ").split())
        nodes.append((x, y))

    # get the incidence of each element
    elements = []
    for i in range(n):
        elements.append(
            list(map(int, input(f"Digite a incidência do nó {i+1}: ").split())))

    # get the properties of each element (A, E, L)