import netket as nk

def edges_2D(Lx, Ly, next_neighbours=True):
    edges = []
    for i in range(Ly):
        for j in range(Lx):
            edges.append([i * Lx + j, i * Lx + (j+1)%Lx, 0])
            edges.append([i * Lx + j, ((i+1)%Ly) * Lx + j, 0])
            if next_neighbours:
                edges.append([i * Lx + j, ((i+1)%Ly) * Lx + (j+1)%Lx, 1])
                edges.append([i * Lx + j, ((i+1)%Ly) * Lx + (j-1)%Lx, 1])
    return edges

def edges_1D(L, next_neighbours=True):
    edges = []
    for i in range(L):
        edges.append([i, (i + 1)%L, 0])
        if next_neighbours:
            edges.append([i, (i + 2)%L, 1])
    return edges

def get_J1_J2_Hamiltonian(Lx, Ly=None, J1=1., J2=0., sign_rule=True):
    if J2 != 0.:
        next_neighbours = True
    else:
        next_neighbours = False

    if Ly is None:
        edges = edges_1D(Lx, next_neighbours=next_neighbours)
    else:
        edges = edges_2D(Lx, Ly, next_neighbours=next_neighbours)

    g = nk.graph.Graph(edges=edges)
    hilbert = nk.hilbert.Spin(0.5, N=g.n_nodes)

    if J2 != 0:
        hamiltonian = nk.operator.Heisenberg(hilbert, g, J=[J1/4, J2/4], sign_rule=sign_rule)
    else:
        hamiltonian = nk.operator.Heisenberg(hilbert, g, J=J1/4, sign_rule=sign_rule)

    return hamiltonian
