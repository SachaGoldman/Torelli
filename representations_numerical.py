import itertools
import math

import numpy

# This hyperparamter represents the genus of the underlying surface who's Torelli Lie Algebra we're interested in
# so all underlying vector spaces have dimension 2g
g = 3


def tensor(obs):
    # Return the n-fold tensor product of some objects

    n = len(obs)

    if n == 1:
        return obs[0]

    a = obs[0]
    b = tensor(obs[1:])

    return numpy.kron(a, b)


def alt(vectors):
    # The n-fold alternating product of list vectors
    n = len(vectors)

    if n == 1:
        return vectors[0]

    a = vectors[0]
    b = alt(vectors[1:])

    return tensor([a, b]) + tensor([b, a])


def E(a, b, n):
    matrix = numpy.zeros([n, n])
    matrix[a, b] = 1
    return matrix


def H(i):
    return E(i, i, 2 * g) - E(i + g, i + g, 2 * g)


def X(i, j):
    return E(i, j, 2 * g) - E(j + g, i + g, 2 * g)


def Y(i, j):
    return E(i, j + g, 2 * g) - E(j, i + g, 2 * g)


def Z(i, j):
    return E(i + g, j, 2 * g) - E(j + g, i, 2 * g)


def U(i):
    return E(i, i + g, 2 * g)


def V(i):
    return E(i + g, i, 2 * g)


def derivation_action(matrix, n):
    # Convert from a standard matrix to the matrix that acts as on the nth tensor power
    n = matrix.shape[0]

    return sum([tensor([numpy.identity(n)] * (i) + [matrix] + [numpy.identity(n)] * (n - i - 1))
                for i in range(n)])


if __name__ == '__main__':
    standard_rep_symp_basis = list(numpy.identity(2 * g))

    rep_basis = []

    for (i, j, k) in itertools.combinations(range(g), 3):
        for (s_i, s_j, s_k) in itertools.product(*([range(2)] * 3)):
            rep_basis.append(alt([standard_rep_symp_basis[i + g * s_i],
                             standard_rep_symp_basis[j + g * s_j], standard_rep_symp_basis[k + g * s_k]]))

    for i in range(g):
        j = (i + 1) % g
        for k in list(range(g)):
            if k not in (i, j):
                for s_k in range(2):
                    rep_basis.append(alt(
                        [standard_rep_symp_basis[i], standard_rep_symp_basis[i + g], standard_rep_symp_basis[k + g * s_k]]) - alt(
                        [standard_rep_symp_basis[j], standard_rep_symp_basis[j + g], standard_rep_symp_basis[k + g * s_k]]))

    sp_generators = []
    h_generators = []

    for i in range(g):
        h_generators.append(H(i))

        sp_generators.append(U(i))
        sp_generators.append(V(i))

        for j in range(g):
            if j != i:
                sp_generators.append(X(i, j))
                sp_generators.append(Y(i, j))
                sp_generators.append(Z(i, j))
