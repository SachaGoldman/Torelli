import itertools
import json
import os.path

import numpy
from sympy import Matrix, SparseMatrix, eye
from sympy.combinatorics.permutations import Permutation
from sympy.parsing.sympy_parser import parse_expr
from sympy.physics.quantum.matrixutils import matrix_tensor_product
from sympy.polys.matrices import DomainMatrix

# This paramter represents the genus of the underlying surface who's Torelli Lie Algebra we're interested in
# so all underlying vector spaces have dimension 2g
g = 3
# The represents the degree of depth to calculate the graded Lie algebra to
k = 2

# Flags
debug = False
overwrite_json = False


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
        generators.append(H(i))
        generators.append(U(i))
        generators.append(V(i))

        for j in range(g):
            if j != i:
                generators.append(X(i, j))
                if j < i:
                    generators.append(Y(i, j))
                    generators.append(Z(i, j))

    return generators


def dual_symplectic_generators():
    # Create the generating matricies for the symplectic Lie algebra
    # We don't return the cartan-subalgebra, just the rest
    generators = []

    for i in range(g):
        generators.append(H(i))
        generators.append(2 * V(i))
        generators.append(2 * U(i))

        for j in range(g):
            if j != i:
                generators.append(X(j, i))
                if j < i:
                    generators.append(Z(j, i))
                    generators.append(Y(j, i))

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


def norm_weight(weight):
    # The norm of the weight of a vector
    return numpy.linalg.norm(numpy.array(weight))


def representation_dimension(highest_weight):
    # Find the dimesion of a representation with highest weight highest_weight
    highest_weight = [abs(n) for n in highest_weight]
    highest_weight.sort(reverse=True)

    dim = 1

    for i in range(g):
        for j in range(i + 1):
            if i != j:
                dim *= (highest_weight[j] - highest_weight[i] + i - j) / (i - j)

            dim *= (highest_weight[i] + highest_weight[j] - i - j + 2 * g) / (2 * g - i - j)

    return round(dim)


def wedge_rep_size_breakdown():
    # Return the expected sizes of the repsentations of the second exterior power
    # This is from Hain's paper
    if g == 6:
        return [representation_dimension(highest_weight) for highest_weight in [[2, 2, 1, 1, 0, 0], [2, 2, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]]
    elif g == 5:
        return [representation_dimension(highest_weight) for highest_weight in [[2, 2, 1, 1, 0], [2, 2, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]]]
    elif g == 4:
        return [representation_dimension(highest_weight) for highest_weight in [[2, 2, 1, 1], [2, 2, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]]]
    elif g == 3:
        return [representation_dimension(highest_weight) for highest_weight in [[2, 2, 0], [0, 0, 0]]]


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


def casimir_element(basis, dual_basis):
    k = basis[0].shape[0]

    element = SparseMatrix(k, k, [0] * (k ** 2))
    for matrix, dual in zip(basis, dual_basis):
        element += matrix @ dual

    return element


def eigenvalue(matrix, vector):
    new_vector = matrix @ vector

    if not new_vector.CL:
        return 0

    return new_vector.CL[0][2] / vector.CL[0][2]


def serialize_sparse_matrix(matrix):
    return {'entries': f'{matrix.CL}', 'dimension': matrix.shape}


def deserialize_sparse_matrix(dictionary_representation):
    return SparseMatrix(*dictionary_representation['dimension'], {(row, column): entry for (row, column, entry) in parse_expr(dictionary_representation['entries'])})


def decompose(representation, casimir_matrix, h_generators):
    # decompose a representation by subrepresentation, we assume the represenation is given by a basis of weights
    representation_by_weight = {}

    for vector in representation:
        weight = weight_of_vector(vector, h_generators)

        if not weight in representation_by_weight:
            representation_by_weight[weight] = []

        representation_by_weight[weight].append(vector)

    irriducible_sub_reps = {}

    print('Decomposed representation by weight')

    eigenvalues = casimir_matrix.eigenvals().keys()

    print('Found eigenvalues ' + f'{list(eigenvalues)}'[1:-1])

    for eigenvalue in eigenvalues:
        irriducible_sub_reps[eigenvalue] = []

    for weight, weight_space in representation_by_weight.items():
        k = len(weight_space)

        weight_space_inclusion, weight_space_projection = project(weight_space)

        casimir_matrix_pullback = pullback([casimir_matrix], weight_space_inclusion,
                                           weight_space_projection)[0]

        for eigenvalue in eigenvalues:
            kernel_matrix = SparseMatrix((DomainMatrix.from_Matrix(
                (casimir_matrix_pullback - eye(k) * eigenvalue)).to_field().nullspace()).to_Matrix())

            kernel = [weight_space_inclusion @
                      kernel_matrix.row(n).T for n in range(kernel_matrix.shape[0])]

            irriducible_sub_reps[eigenvalue].extend(kernel)

    return irriducible_sub_reps.values()


def lyndon_words(s, n):
    w = [-1]
    while w:
        w[-1] += 1
        yield w
        m = len(w)
        while len(w) < n:
            w.append(w[-m])
        while w and w[-1] == s - 1:
            w.pop()


def free_lie_algebra(basis, max_degree):
    pass  # Use lyndon words


if __name__ == '__main__':
    if not overwrite_json and os.path.isfile(f'g_equals_{g}.json'):
        json_file = open(f'g_equals_{g}.json', 'r')
        deserializable_data = json.loads(json_file.read())
        json_file.close()

        h_generators_wedge_pullback = [deserialize_sparse_matrix(
            dictionary_representation) for dictionary_representation in deserializable_data['h_generators_wedge_pullback']]
        sp_generators_wedge_pullback = [deserialize_sparse_matrix(
            dictionary_representation) for dictionary_representation in deserializable_data['sp_generators_wedge_pullback']]
        dual_sp_generators_wedge_pullback = [deserialize_sparse_matrix(
            dictionary_representation) for dictionary_representation in deserializable_data['dual_sp_generators_wedge_pullback']]
        pullback_wedge_rep = [deserialize_sparse_matrix(
            dictionary_representation) for dictionary_representation in deserializable_data['pullback_wedge_rep']]
        h_generators_rep_pullback = [deserialize_sparse_matrix(
            dictionary_representation) for dictionary_representation in deserializable_data['h_generators_rep_pullback']]
        sp_generators_rep_pullback = [deserialize_sparse_matrix(
            dictionary_representation) for dictionary_representation in deserializable_data['dual_sp_generators_rep_pullback']]
        dual_sp_generators_rep_pullback = [deserialize_sparse_matrix(
            dictionary_representation) for dictionary_representation in deserializable_data['sp_generators_rep_pullback']]
        pullback_base_rep = [deserialize_sparse_matrix(
            dictionary_representation) for dictionary_representation in deserializable_data['pullback_base_rep']]
        irriducible_sub_reps = [[deserialize_sparse_matrix(
            dictionary_representation) for dictionary_representation in irriducible_sub_rep] for irriducible_sub_rep in deserializable_data['irriducible_sub_reps']]
    else:
        # Expected Sizes
        print('Expected subrep sizes are ' + f'{wedge_rep_size_breakdown()}'[1:-1])

        # First we create the standard representation
        standard_rep = standard_basis(2 * g)
        sp_generators = symplectic_generators()
        dual_sp_generators = dual_symplectic_generators()
        h_generators = symplectic_cartan_subalgebra_generators()

        print('Created basic objects')

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

        print('Found basis for base rep')

        # We now compute how the Lie algebra acts base_rep
        inclusion_base, projection_base = project(base_rep)

        h_generators_base = [derivation_action(
            generator, 3) for generator in h_generators]

        sp_generators_base = [derivation_action(
            generator, 3) for generator in sp_generators]

        dual_sp_generators_base = [derivation_action(
            generator, 3) for generator in dual_sp_generators]

        h_generators_rep_pullback = pullback(h_generators_base, inclusion_base, projection_base)
        sp_generators_rep_pullback = pullback(sp_generators_base, inclusion_base, projection_base)
        dual_sp_generators_rep_pullback = pullback(
            dual_sp_generators_base, inclusion_base, projection_base)

        pullback_base_rep = standard_basis(len(base_rep))

        print('Found Lie algebra action on base rep')
        # Now we compute the second exterior power of base_rep including how the Lie algebra acts on it
        wedge_rep = []

        for i in range(len(pullback_base_rep)):
            for j in range(i):
                wedge_rep.append(alt([pullback_base_rep[i], pullback_base_rep[j]]))

        print('Found basis for wedge rep')

        inclusion_wedge, projection_wedge = project(wedge_rep)

        h_generators_wedge = [derivation_action(
            generator, 2) for generator in h_generators_rep_pullback]

        sp_generators_wedge = [derivation_action(
            generator, 2) for generator in sp_generators_rep_pullback]

        dual_sp_generators_wedge = [derivation_action(
            generator, 2) for generator in dual_sp_generators_rep_pullback]

        h_generators_wedge_pullback = pullback(
            h_generators_wedge, inclusion_wedge, projection_wedge)
        sp_generators_wedge_pullback = pullback(
            sp_generators_wedge, inclusion_wedge, projection_wedge)
        dual_sp_generators_wedge_pullback = pullback(
            dual_sp_generators_wedge, inclusion_wedge, projection_wedge)

        pullback_wedge_rep = standard_basis(len(wedge_rep))

        print('Found Lie algebra action on wedge rep')

        casimir_matrix = casimir_element(
            sp_generators_wedge_pullback, dual_sp_generators_wedge_pullback)

        irriducible_sub_reps = decompose(
            pullback_wedge_rep, casimir_matrix, h_generators_wedge_pullback)

        print('Found subreps of sizes ' +
              f'{[len(sub_rep) for sub_rep in irriducible_sub_reps]}'[1:-1])

        serializable_data = {
            'h_generators_wedge_pullback': [serialize_sparse_matrix(matrix) for matrix in h_generators_wedge_pullback],
            'sp_generators_wedge_pullback': [serialize_sparse_matrix(matrix) for matrix in sp_generators_wedge_pullback],
            'dual_sp_generators_wedge_pullback': [serialize_sparse_matrix(matrix) for matrix in dual_sp_generators_wedge_pullback],
            'pullback_wedge_rep': [serialize_sparse_matrix(matrix) for matrix in pullback_wedge_rep],
            'h_generators_rep_pullback': [serialize_sparse_matrix(matrix) for matrix in h_generators_rep_pullback],
            'sp_generators_rep_pullback': [serialize_sparse_matrix(matrix) for matrix in sp_generators_rep_pullback],
            'dual_sp_generators_rep_pullback': [serialize_sparse_matrix(matrix) for matrix in dual_sp_generators_rep_pullback],
            'pullback_base_rep': [serialize_sparse_matrix(matrix) for matrix in pullback_base_rep],
            'irriducible_sub_reps': [[serialize_sparse_matrix(matrix) for matrix in irriducible_sub_rep] for irriducible_sub_rep in irriducible_sub_reps]
        }

        json_file = open(f'g_equals_{g}.json', 'w')
        json_object = json.dumps(serializable_data, indent=4)
        json_file.write(json_object)
        json_file.close()

    print("Finished (de)serializing")

    # print(len([w for w in lyndon_words(110, 4)]))
