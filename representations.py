import itertools

import numpy
from sympy import Matrix, SparseMatrix, eye
from sympy.combinatorics.permutations import Permutation
from sympy.physics.quantum.matrixutils import matrix_tensor_product

# This paramter represents the genus of the underlying surface who's Torelli Lie Algebra we're interested in
# so all underlying vector spaces have dimension 2g
g = 3


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


def derivation_action(matrix, n):
    # Convert from a standard matrix to the matrix that acts as on the nth tensor power
    d = matrix.shape[0]

    partial_actions = [tensor([SparseMatrix(eye(d)) for _ in range(i)] + [matrix] +
                              [SparseMatrix(eye(d)) for _ in range(n - i - 1)]) for i in range(n)]

    action = partial_actions[0]

    for partial_action in partial_actions[1:]:
        action += partial_action

    return action


def E(a, b, n):
    # Create an n by n basis matrix with 1 only at (a,b)
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
    # Create the generating matricies for the symplectic Lie algebra
    # We don't return the cartan-subalgebra, just the rest
    generators = []

    for i in range(g):
        generators.append(U(i))
        generators.append(V(i))

        for j in range(g):
            if j != i:
                generators.append(X(i, j))
                if j < i:
                    generators.append(Y(i, j))
                    generators.append(Z(i, j))

    return generators


def symplectic_cartan_subalgebra_generators():
    # Create the generating matricies for the cartan subagebrasymplectic Lie algebra
    return [H(i) for i in range(g)]


def weight_of_vector(vector, cartan_subalgebra):
    # Calculate the weight of a vector in the dual basis to the given basis for the cartan_subalgebra
    weight = []

    for matrix in cartan_subalgebra:
        new_vector = matrix @ vector

        if new_vector.CL:
            weight.append(new_vector.CL[0][2] / vector.CL[0][2])
        else:
            weight.append(0)

    return tuple([int(num) for num in weight])


def norm_weight(vector, lie_algebra):
    # The norm of the weight of a vector
    return numpy.linalg.norm(numpy.array(weight_of_vector(vector, lie_algebra)))


def representation_dimension(highest_weight):
    # Find the dimesion of a representation with highest weight highest_weight
    dim = 1

    for i in range(g):
        for j in range(i + 1):
            if i != j:
                dim *= (highest_weight[j] - highest_weight[i] + i - j) / (i - j)

            dim *= (highest_weight[i] + highest_weight[j] - i - j + 2 * g) / (2 * g - i - j)

    return dim


def psuedo_inverse(matrix):
    return (matrix.H * matrix) ** -1 * matrix.H


def project(basis):
    # Given a k dimensional subspace of an n dimensional vector space, find the inclusion and projection mapping basis to the standard basis for a k dimensional vector space
    inclusion = basis[0]
    for basis_element in basis[1:]:
        inclusion = inclusion.row_join(basis_element)

    inclusion = SparseMatrix(inclusion)
    projection = psuedo_inverse(inclusion)

    return inclusion, projection


def pullback(matricies, function, left_inverse):
    # Given a matrix, function, its left inverse, left_inverse, and a matrix that maps the cokernel of function to itself, pullback the matrix
    return [left_inverse @ matrix @ function for matrix in matricies]


def standard_basis(n):
    # Create the standard basis for an n-dimensional vector space
    basis = []

    for i in range(n):
        basis_element = SparseMatrix(Matrix([0] * n))
        basis_element[i] = 1
        basis.append(basis_element)

    return basis


def sub_rep_size(sub_rep):
    # Calculate the size of an subrepresentation
    size = 0

    for weight in sub_rep:
        size += len(sub_rep[weight][0])

    return size


def find_sub_rep(vector, rep, lie_algebra, cartan_subalgebra, expected_size):
    # Given a weight vector, vector, in a representation, rep, of a Lie algebra, lie_algebra find the smallest subrepresentation containing vector
    # If we expect vector to be in a representation of a certain size, expected_size, then once the subrepresentation reaches this size we can stop checking if its bigger
    max_weight = weight_of_vector(max_weight_space_element, cartan_subalgebra)

    # We break down the subrepresentation by weight to speed up computations
    sub_rep = {max_weight: [
        [max_weight_space_element], max_weight_space_element]}
    stack = [(max_weight_space_element, max_weight)]

    while (stack):
        print(f'sub_rep size: {sub_rep_size(sub_rep)}, stack size: {len(stack)}')
        vector, weight = stack.pop(-1)

        for matrix in lie_algebra:
            new_vector = matrix @ vector
            new_weight = weight_of_vector(new_vector, cartan_subalgebra)

            if new_vector.CL:
                if new_weight in sub_rep:
                    weight_space_sub_rep_matrix = sub_rep[new_weight][1]
                    new_weight_space_sub_rep_matrix = weight_space_sub_rep_matrix.row_join(
                        new_vector)

                    # This is done in numpy because its faster and we can afford the numerical imprecision
                    rank = numpy.linalg.matrix_rank(
                        numpy.array(new_weight_space_sub_rep_matrix).astype(numpy.float64))

                    if len(sub_rep[new_weight][0]) == rank - 1:
                        print(f'sub_rep size: {sub_rep_size(sub_rep)}, stack size: {len(stack)}')
                        sub_rep[new_weight][1] = new_weight_space_sub_rep_matrix
                        sub_rep[new_weight][0].append(new_vector)
                        stack.append((new_vector, new_weight))
                else:
                    print(f'sub_rep size: {sub_rep_size(sub_rep)}, stack size: {len(stack)}')
                    sub_rep[new_weight] = [[new_vector], new_vector]
                    stack.append((new_vector, new_weight))

            # Checking if we've already found everything
            if expected_size and sub_rep_size(sub_rep) == expected_size:
                stack = None
                break

    return sub_rep


if __name__ == '__main__':
    # First we create the standard representation
    standard_rep = standard_basis(2 * g)
    sp_generators = symplectic_generators()
    h_generators = symplectic_cartan_subalgebra_generators()

    print('Create basic objects')

    # Now we find a basis for the representation V_{1,1,1}
    base_rep = []

    for (i, j, k) in itertools.combinations(range(g), 3):
        for (s_i, s_j, s_k) in itertools.product(*([range(2)] * 3)):
            base_rep.append(alt([standard_rep[i + g * s_i],
                                 standard_rep[j + g * s_j], standard_rep[k + g * s_k]]))

    for i in range(g):
        j = (i + 1) % g
        for k in list(range(g)):
            if k not in (i, j):
                for s_k in range(2):
                    base_rep.append(alt(
                        [standard_rep[i], standard_rep[i + g], standard_rep[k + g * s_k]]) - alt(
                        [standard_rep[j], standard_rep[j + g], standard_rep[k + g * s_k]]))

    print('Found basis for base represenation')

    # We now compute how the Lie algebra acts base_rep
    inclusion_base, projection_base = project(base_rep)

    h_generators_base = [derivation_action(
        generator, 3) for generator in h_generators]

    sp_generators_base = [derivation_action(
        generator, 3) for generator in sp_generators]

    h_generators_rep_pullback = pullback(h_generators_base, inclusion_base, projection_base)
    sp_generators_rep_pullback = pullback(sp_generators_base, inclusion_base, projection_base)

    pullback_base_rep = standard_basis(len(base_rep))

    print('Found Lie algebra action on base representation')

    # Now we compute the second exterior power of base_rep including how the Lie algebra acts on it
    wedge_rep = []

    for i in range(len(pullback_base_rep)):
        for j in range(i):
            wedge_rep.append(alt([pullback_base_rep[i], pullback_base_rep[j]]))

    print('Found basis for wedge represenation')

    inclusion_wedge, projection_wedge = project(wedge_rep)

    h_generators_wedge = [derivation_action(
        generator, 2) for generator in h_generators_rep_pullback]

    sp_generators_wedge = [derivation_action(
        generator, 2) for generator in sp_generators_rep_pullback]

    h_generators_wedge_pullback = pullback(h_generators_wedge, inclusion_wedge, projection_wedge)
    sp_generators_wedge_pullback = pullback(sp_generators_wedge, inclusion_wedge, projection_wedge)

    pullback_wedge_rep = standard_basis(len(wedge_rep))

    print('Found Lie algebra action on wedge representation')

    # This is just for testing as of now

    max_norm_weight = -1

    for new_weight_space_element in pullback_wedge_rep:
        new_norm_weight = norm_weight(new_weight_space_element, h_generators_wedge_pullback)
        if new_norm_weight > max_norm_weight:
            max_weight_space_element = new_weight_space_element
            max_norm_weight = new_norm_weight

    sub_rep = find_sub_rep(max_weight_space_element, pullback_wedge_rep,
                           sp_generators_wedge_pullback, h_generators_wedge_pullback, None)

    print(len(pullback_wedge_rep))
    print(sub_rep_size(sub_rep))
