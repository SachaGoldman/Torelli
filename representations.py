import itertools

import numpy
import sympy
from scipy import sparse
from sympy.combinatorics.permutations import Permutation

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

    return sparse.kron(a, b)


def alt(vectors):
    # The n-fold alternating product of list vectors
    first = True

    for perm in itertools.permutations(list(range(len(vectors)))):
        new_part = Permutation(perm).signature() * tensor([vectors[i] for i in perm])
        if first:
            vector = new_part
            first = False
        else:
            vector += new_part

    return vector


def E(a, b, n):
    matrix = numpy.zeros((n, n))
    matrix[a, b] = 1
    return sparse.csr_matrix(matrix)


def H(i):
    return E(i, i, 2 * g) - E(i + g, i + g, 2 * g)


def X(i, j):
    return E(i, j, 2 * g) - E(j + g, i + g, 2 * g)


def Y(i, j):
    return E(i, j + g, 2 * g) + E(j, i + g, 2 * g)


def Z(i, j):
    return E(i + g, j, 2 * g) + E(j + g, i, 2 * g)


def U(i):
    return E(i, i + g, 2 * g)


def V(i):
    return E(i + g, i, 2 * g)


def norm_weight(vector, lie_algebra):
    return numpy.linalg.norm(numpy.array(weight_of_vector(vector, lie_algebra)))


def weight_of_vector(vector, lie_algebra):
    weight = []

    for matrix in lie_algebra:
        product = matrix @ vector

        if product.data.any():
            weight.append(product.data[0] / vector.data[0])
        else:
            weight.append(0)

    return tuple(weight)


def derivation_action(matrix, n):
    # Convert from a standard matrix to the matrix that acts as on the nth tensor power
    d = matrix.shape[0]

    return sum([tensor([sparse.csr_matrix(numpy.identity(d)) for _ in range(i)] + [matrix] + [sparse.csr_matrix(numpy.identity(d)) for _ in range(n - i - 1)]) for i in range(n)])


def find_smallest(array):
    new_array = array[numpy.nonzero(array)]
    idx = numpy.abs(new_array).argmin()
    return new_array[idx]


def dimension(highest_weight):
    dim = 1

    for i in range(g):
        for j in range(i + 1):
            if i != j:
                dim *= (highest_weight[j] - highest_weight[i] + i - j) / (i - j)

            dim *= (highest_weight[i] + highest_weight[j] - i - j + 2 * g) / (2 * g - i - j)

    return dim


def orbit_size(orbit):
    size = 0

    for key in orbit:
        size += len(orbit[key][0])

    return size


def find_orbit(vector, rep, lie_algebra):
    max_weight = weight_of_vector(max_weight_space_element, h_generators_derivation)

    orbit = {max_weight: [
        [max_weight_space_element], numpy.array([numpy.squeeze(max_weight_space_element.toarray())])]}
    stack = [(max_weight_space_element, max_weight)]

    print(len(wedge_basis))

    while (stack):
        print(f'orbit size: {orbit_size(orbit)}, stack size: {len(stack)}')
        vector, weight = stack.pop(-1)

        for matrix in sp_generators_derivation:
            new_vector = matrix @ vector
            new_weight = weight_of_vector(new_vector, h_generators_derivation)

            if new_vector.data.any():
                if new_weight in orbit:
                    weight_space_orbit_matrix = orbit[new_weight][1]
                    new_weight_space_orbit_matrix = numpy.append(
                        weight_space_orbit_matrix, [numpy.squeeze(new_vector.toarray())], axis=0)
                    rank = numpy.linalg.matrix_rank(new_weight_space_orbit_matrix)

                    if len(orbit[new_weight][0]) == rank - 1:
                        print(f'orbit size: {orbit_size(orbit)}, stack size: {len(stack)}')
                        orbit[new_weight][1] = new_weight_space_orbit_matrix
                        orbit[new_weight][0].append(new_vector)
                        stack.append((new_vector, new_weight))
                else:
                    orbit[new_weight] = [[new_vector], numpy.array(
                        [numpy.squeeze(new_vector.toarray())])]
                    stack.append((new_vector, new_weight))

    return orbit


if __name__ == '__main__':
    standard_rep_symp_basis = [sparse.csr_matrix(v).T for v in list(numpy.identity(2 * g))]

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
                if j < i:
                    sp_generators.append(Y(i, j))
                    sp_generators.append(Z(i, j))

    wedge_basis = []

    for i in range(len(rep_basis)):
        for j in range(i):
            wedge_basis.append(alt([rep_basis[i], rep_basis[j]]))

    h_generators_derivation = [derivation_action(derivation_action(
        generator, 3), 2) for generator in h_generators]

    sp_generators_derivation = [derivation_action(derivation_action(
        generator, 3), 2) for generator in sp_generators]

    max_norm_weight = -1

    for new_weight_space_element in wedge_basis:
        new_norm_weight = norm_weight(new_weight_space_element, h_generators_derivation)
        if new_norm_weight > max_norm_weight:
            max_weight_space_element = new_weight_space_element
            max_norm_weight = new_norm_weight

    orbit = find_orbit(new_weight_space_element, wedge_basis, sp_generators_derivation)

    print(len(wedge_basis))
    print(orbit_size(orbit))
