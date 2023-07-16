import itertools

import numpy
from sympy import Matrix, SparseMatrix, eye
from sympy.combinatorics.permutations import Permutation
from sympy.physics.quantum.matrixutils import matrix_tensor_product

# This paramter represents the genus of the underlying surface who's Torelli Lie Algebra we're interested in
# so all underlying vector spaces have dimension 2g
g = 4


def tensor(obs):
    # Return the n-fold tensor product of some objects

    n = len(obs)

    if n == 1:
        return obs[0]

    a = obs[0]
    b = tensor(obs[1:])

    return matrix_tensor_product(a, b)


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
    matrix = SparseMatrix(n, n, [0] * (n ** 2))
    matrix[a, b] = 1
    return matrix


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


def symplectic_generators():
    # We don't return the cartan-subalgebra, just the rest
    sp_generators = []

    for i in range(g):
        sp_generators.append(U(i))
        sp_generators.append(V(i))

        for j in range(g):
            if j != i:
                sp_generators.append(X(i, j))
                if j < i:
                    sp_generators.append(Y(i, j))
                    sp_generators.append(Z(i, j))

    return sp_generators


def symplectic_cartan_subalgebra_generators():
    return [H(i) for i in range(g)]


def derivation_action(matrix, n):
    # Convert from a standard matrix to the matrix that acts as on the nth tensor power
    d = matrix.shape[0]

    partial_actions = [tensor([SparseMatrix(eye(d)) for _ in range(i)] + [matrix] +
                              [SparseMatrix(eye(d)) for _ in range(n - i - 1)]) for i in range(n)]

    action = partial_actions[0]

    for partial_action in partial_actions[1:]:
        action += partial_action

    return action


def weight_of_vector(vector, cartan_subalgebra):
    weight = []

    for matrix in cartan_subalgebra:
        product = matrix @ vector

        if product.CL:
            weight.append(product.CL[0][2] / vector.CL[0][2])
        else:
            weight.append(0)

    return tuple([int(num) for num in weight])


def norm_weight(vector, lie_algebra):
    return numpy.linalg.norm(numpy.array(weight_of_vector(vector, lie_algebra)))


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


def project(basis):
    inclusion = basis[0]
    for basis_element in basis[1:]:
        inclusion = inclusion.row_join(basis_element)

    inclusion = SparseMatrix(inclusion)
    projection = psuedo_inverse(inclusion)

    return inclusion, projection


def psuedo_inverse(matrix):
    return (matrix.H * matrix) ** -1 * matrix.H


def pullback(matricies, function, inverse):
    return [inverse @ matrix @ function for matrix in matricies]


def standard_basis(dim):
    basis = []

    for i in range(dim):
        basis_element = SparseMatrix(Matrix([0] * dim))
        basis_element[i] = 1
        basis.append(basis_element)

    return basis


def find_orbit(vector, rep, lie_algebra, cartan_subalgebra):
    max_weight = weight_of_vector(max_weight_space_element, cartan_subalgebra)

    orbit = {max_weight: [
        [max_weight_space_element], max_weight_space_element]}
    stack = [(max_weight_space_element, max_weight)]

    print(len(wedge_basis))

    while (stack):
        print(f'orbit size: {orbit_size(orbit)}, stack size: {len(stack)}')
        vector, weight = stack.pop(-1)

        for matrix in lie_algebra:
            new_vector = matrix @ vector
            new_weight = weight_of_vector(new_vector, cartan_subalgebra)

            if new_vector.CL:
                if new_weight in orbit:
                    weight_space_orbit_matrix = orbit[new_weight][1]
                    new_weight_space_orbit_matrix = weight_space_orbit_matrix.row_join(new_vector)

                    rank = rank = numpy.linalg.matrix_rank(
                        numpy.array(new_weight_space_orbit_matrix).astype(numpy.float64))

                    if len(orbit[new_weight][0]) == rank - 1:
                        print(f'orbit size: {orbit_size(orbit)}, stack size: {len(stack)}')
                        orbit[new_weight][1] = new_weight_space_orbit_matrix
                        orbit[new_weight][0].append(new_vector)
                        stack.append((new_vector, new_weight))
                else:
                    orbit[new_weight] = [[new_vector], new_vector]
                    stack.append((new_vector, new_weight))

    return orbit


if __name__ == '__main__':
    standard_rep_symp_basis = standard_basis(2 * g)
    sp_generators = symplectic_generators()
    h_generators = symplectic_cartan_subalgebra_generators()

    print("Created basic objects")

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

    rep_inclusion, rep_projection = project(rep_basis)

    h_generators_rep = [derivation_action(
        generator, 3) for generator in h_generators]

    sp_generators_rep = [derivation_action(
        generator, 3) for generator in sp_generators]

    h_generators_rep_pullback = pullback(h_generators_rep, rep_inclusion, rep_projection)
    sp_generators_rep_pullback = pullback(sp_generators_rep, rep_inclusion, rep_projection)

    pullback_rep_basis = standard_basis(len(rep_basis))

    print("Found Representation")

    wedge_basis = []

    for i in range(len(pullback_rep_basis)):
        for j in range(i):
            wedge_basis.append(alt([pullback_rep_basis[i], pullback_rep_basis[j]]))

    wedge_inclusion, wedge_projection = project(wedge_basis)

    h_generators_wedge = [derivation_action(
        generator, 2) for generator in h_generators_rep_pullback]

    sp_generators_wedge = [derivation_action(
        generator, 2) for generator in sp_generators_rep_pullback]

    h_generators_wedge_pullback = pullback(h_generators_wedge, wedge_inclusion, wedge_projection)
    sp_generators_wedge_pullback = pullback(sp_generators_wedge, wedge_inclusion, wedge_projection)

    pullback_wedge_basis = standard_basis(len(wedge_basis))

    print("Moved to second exterior power")

    max_norm_weight = -1

    for new_weight_space_element in pullback_wedge_basis:
        new_norm_weight = norm_weight(new_weight_space_element, h_generators_wedge_pullback)
        if new_norm_weight > max_norm_weight:
            max_weight_space_element = new_weight_space_element
            max_norm_weight = new_norm_weight

    print("Found highest weight vector")

    orbit = find_orbit(max_weight_space_element, pullback_wedge_basis,
                       sp_generators_wedge_pullback, h_generators_wedge_pullback)

    print(len(pullback_wedge_basis))
    print(orbit_size(orbit))
