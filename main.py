import numpy as np
import matplotlib.pyplot as plt


class Node:
    def __init__(self, id, x, y, constraints, forces):
        self.id = id
        self.x = x
        self.y = y
        self.constraints = constraints                  # indica se o nó é restrito
        self.forces = forces                            # forças aplicadas no nó
        # graus de liberdade do nó
        self.dof = [2 * (id - 1), 2 * (id - 1) + 1]


class Element:
    def __init__(self, id, node1, node2, elasticity, area):
        self.id = id
        self.node1 = node1
        self.node2 = node2
        self.elasticity = elasticity
        self.area = area

    def length(self):
        return np.sqrt((self.node2.x - self.node1.x)**2 + (self.node2.y - self.node1.y)**2)

    def sine(self):
        return (self.node2.y - self.node1.y) / self.length()

    def cosine(self):
        return (self.node2.x - self.node1.x) / self.length()

    def stiffness_matrix(self):
        L = self.length()
        k = self.elasticity * self.area / L

        # montamos a matriz de rigidez do elemento
        c = self.cosine()
        s = self.sine()
        k_element = np.array([[c**2, c*s, -c**2, -c*s],
                              [c*s, s**2, -c*s, -s**2],
                              [-c**2, -c*s, c**2, c*s],
                              [-c*s, -s**2, c*s, s**2]]) * k

        return k_element


class Truss:
    def __init__(self):
        self.nodes = []
        self.elements = []
        self.yield_stress = 250e6  # tensão de escoamento
        self.buckling_factor = 1.0  # fator de segurança

    def add_node(self, id, x, y, constraints=[False, False], forces=[0, 0]):
        self.nodes.append(Node(id, x, y, constraints, forces))

    def add_element(self, id, node1_id, node2_id, elasticity=200e9, area=6e-5):
        node1 = next(node for node in self.nodes if node.id ==
                     node1_id)  # busca o nó com id node1_id
        node2 = next(node for node in self.nodes if node.id ==
                     node2_id)  # busca o nó com id node2_id
        self.elements.append(
            Element(id, node1, node2, elasticity, area))

    def assemble_global_stiffness_matrix(self):
        dimension = 2 * len(self.nodes)
        k_global = np.zeros((dimension, dimension))

        for element in self.elements:
            k_element = element.stiffness_matrix()
            idx = element.node1.dof + element.node2.dof

            for i in range(4):
                for j in range(4):
                    k_global[idx[i]][idx[j]] += k_element[i][j]

        return k_global

    def apply_constraints(self, k_global, f_global):
        for node in self.nodes:
            if node.constraints[0]:  # restrição no grau de liberdade X
                idx = node.dof[0]
                k_global[idx, :] = 0
                k_global[:, idx] = 0
                k_global[idx, idx] = 1
                f_global[idx] = 0

            if node.constraints[1]:  # restrição no grau de liberdade Y
                idx = node.dof[1]
                k_global[idx, :] = 0
                k_global[:, idx] = 0
                k_global[idx, idx] = 1
                f_global[idx] = 0

        return k_global, f_global

    def solve(self):
        k_global = self.assemble_global_stiffness_matrix()
        f_global = np.array(
            [force for node in self.nodes for force in node.forces])

        k_global, f_global = self.apply_constraints(k_global, f_global)

        rank = np.linalg.matrix_rank(k_global)
        if rank < k_global.shape[0]:
            print(f"A matriz de rigidez global é singular. Rank: {
                rank}, Esperado: {k_global.shape[0]}")
            raise ValueError(
                "A matriz de rigidez global é singular e não pode ser resolvida.")

        u = np.linalg.solve(k_global, f_global)
        print("Vetor de deslocamentos (u):\n", u)

        stresses, reactions = self.post_process(u, k_global, f_global)
        print("Forças de reação (R):\n", reactions)

        return u, stresses, reactions

    def post_process(self, u, k_global, f_global):
        stresses = []
        reactions = np.zeros_like(f_global)

        for element in self.elements:
            node1_dof = element.node1.dof
            node2_dof = element.node2.dof
            dof_indices = node1_dof + node2_dof
            u_element = u[dof_indices]
            k_element = element.stiffness_matrix()
            f_element = np.dot(k_element, u_element)
            stress = element.elasticity / \
                element.length() * (f_element[2] - f_element[0])
            stresses.append(stress)

            for i in range(2):
                idx1 = node1_dof[i]
                idx2 = node2_dof[i]
                reactions[idx1] += f_element[i]
                reactions[idx2] += f_element[i+2]

        for i, node in enumerate(self.nodes):
            if node.constraints[0]:  # restrição no grau de liberdade X
                idx = node.dof[0]
                reactions[idx] -= f_global[idx]

            if node.constraints[1]:  # restrição no grau de liberdade Y
                idx = node.dof[1]
                reactions[idx] -= f_global[idx]

        self.analyze_stresses(stresses)

        return stresses, reactions

    def analyze_stresses(self, stresses):
        for i, stress in enumerate(stresses):
            if abs(stress) > self.yield_stress:
                print(
                    f"Elemento {i + 1} falha por tensão: Stress = {stress:.2e} Pa, Yield Stress = {self.yield_stress:.2e} Pa")

            if stress < 0:  # sob compressão
                critical_load = (np.pi ** 2 * self.elements[i].elasticity *
                                 self.elements[i].area) / (self.buckling_factor * self.elements[i].length() ** 2)
                if abs(stress) > critical_load:
                    print(
                        f"Elemento {i + 1} falha por flambagem: Stress = {stress:.2e} Pa, Critical Load = {critical_load:.2e} Pa")

    def plot(self, u, stresses, reactions):
        plt.figure(figsize=(12, 8))

        # Plotar os elementos da treliça
        for element in self.elements:
            x = [element.node1.x, element.node2.x]
            y = [element.node1.y, element.node2.y]
            plt.plot(x, y, 'b', label='Elementos' if element.id == 1 else "")

        # Plotar os nós e deslocamentos
        for node in self.nodes:
            idx = node.id - 1
            plt.plot(node.x, node.y, 'ro', label='Nós' if node.id == 1 else "")
            plt.text(node.x, node.y, f'u{idx+1}\n({u[2*idx]:.2e}, {u[2*idx+1]:.2e})',
                     fontsize=8, color='green', ha='center', va='bottom', rotation=45)

        # Adicionar tensões nos elementos
        for i, element in enumerate(self.elements):
            x_mid = (element.node1.x + element.node2.x) / 2
            y_mid = (element.node1.y + element.node2.y) / 2
            plt.text(x_mid, y_mid, f'σ{i+1} = {stresses[i]:.2e}',
                     fontsize=8, color='blue', ha='center', va='center', rotation=45)

        # Adicionar reações nos apoios
        for i, node in enumerate(self.nodes):
            if node.constraints[0] or node.constraints[1]:
                if node.constraints[0]:
                    plt.text(node.x - 0.03, node.y, f'Rx = {reactions[node.dof[0]]:.2e}',
                             fontsize=8, color='red', ha='center', va='top', rotation=45)
                if node.constraints[1]:
                    plt.text(node.x - 0.03, node.y, f'Ry = {reactions[node.dof[1]]:.2e}',
                             fontsize=8, color='red', ha='left', va='top', rotation=45)

        # Adicionar legenda
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Treliça')
        plt.legend(loc='upper right')
        plt.grid(True)

        # Desenhar a legenda personalizada para as cores
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='b', lw=2),
                        Line2D([0], [0], color='red',
                               marker='o', linestyle='None'),
                        Line2D([0], [0], color='green',
                               linestyle='None', marker='o'),
                        Line2D([0], [0], color='blue', linestyle='None', marker='o')]

        plt.legend(custom_lines, ['Elementos', 'Nós',
                                  'Deslocamentos', 'Tensões'])

        plt.show()


def main():
    truss = Truss()

    dx = 0.05  # distância entre os nós

    # nós do eixo y = 0 (9 no total)
    truss.add_node(1, 0, 0, constraints=[True, True])
    truss.add_node(2, dx, 0)
    truss.add_node(3, dx*2, 0)
    truss.add_node(4, dx*3, 0)
    truss.add_node(5, dx*4, 0)
    truss.add_node(6, dx*5, 0)
    truss.add_node(7, dx*6, 0)
    truss.add_node(8, dx*7, 0)
    truss.add_node(9, dx*8, 0, constraints=[False, True])

    # nós do eixo y = 0.05 (7 no total)
    truss.add_node(10, dx, dx)
    truss.add_node(11, dx*2, dx)
    truss.add_node(12, dx*3, dx, forces=[0, -100])
    truss.add_node(13, dx*4, dx, forces=[0, -100])
    truss.add_node(14, dx*5, dx, forces=[0, -100])
    truss.add_node(15, dx*6, dx)
    truss.add_node(16, dx*7, dx)

    # elementos horizontais
    truss.add_element(1, 1, 2)
    truss.add_element(2, 2, 3)
    truss.add_element(3, 3, 4)
    truss.add_element(4, 4, 5)
    truss.add_element(5, 5, 6)
    truss.add_element(6, 6, 7)
    truss.add_element(7, 7, 8)
    truss.add_element(8, 8, 9)
    truss.add_element(9, 10, 11)
    truss.add_element(10, 11, 12)
    truss.add_element(11, 12, 13)
    truss.add_element(12, 13, 14)
    truss.add_element(13, 14, 15)
    truss.add_element(14, 15, 16)

    # elementos verticais
    truss.add_element(15, 2, 10)
    truss.add_element(16, 3, 11)
    truss.add_element(17, 4, 12)
    truss.add_element(18, 5, 13)
    truss.add_element(19, 6, 14)
    truss.add_element(20, 7, 15)
    truss.add_element(21, 8, 16)

    # elementos diagonais
    truss.add_element(22, 1, 10)
    truss.add_element(23, 10, 3)
    truss.add_element(24, 11, 4)
    truss.add_element(25, 12, 5)
    truss.add_element(26, 5, 14)
    truss.add_element(27, 6, 15)
    truss.add_element(28, 7, 16)
    truss.add_element(29, 16, 9)

    u, stresses, reactions = truss.solve()

    truss.plot(u, stresses, reactions)

    return 0


if __name__ == "__main__":
    main()
