if __name__ == '__main__':
    g = 4

    lam = [2, 2, 1, 1]

    dim = 1

    for i in range(g):
        for j in range(i + 1):
            if i != j:
                dim *= (lam[j] - lam[i] + i - j) / (i - j)

            dim *= (lam[i] + lam[j] - i - j + 2 * g) / (2 * g - i - j)

    print(dim)
