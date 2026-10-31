
import math


def off_diagonal_norm(A):
    """Норма внедиагональных элементов матрицы."""
    n = len(A)
    total = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            total += A[i][j] ** 2

    return math.sqrt(total)


def jacobi_rotation(A, eps):
    n = len(A)

    A = [row[:] for row in A]

    V = [[0.0] * n for _ in range(n)]

    for i in range(n):
        V[i][i] = 1.0

    iterations = 0
    history = [off_diagonal_norm(A)]

    while off_diagonal_norm(A) > eps:
        p, q = 0, 1

        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i][j]) > abs(A[p][q]):
                    p, q = i, j

        app = A[p][p]
        aqq = A[q][q]
        apq = A[p][q]

        phi = 0.5 * math.atan2(2.0 * apq, app - aqq)

        c = math.cos(phi)
        s = math.sin(phi)

        for k in range(n):
            if k != p and k != q:
                akp = A[k][p]
                akq = A[k][q]

                A[k][p] = c * akp + s * akq
                A[p][k] = A[k][p]

                A[k][q] = -s * akp + c * akq
                A[q][k] = A[k][q]

        A[p][p] = c * c * app + 2.0 * c * s * apq + s * s * aqq
        A[q][q] = s * s * app - 2.0 * c * s * apq + c * c * aqq

        A[p][q] = 0.0
        A[q][p] = 0.0

        for k in range(n):
            vkp = V[k][p]
            vkq = V[k][q]

            V[k][p] = c * vkp + s * vkq
            V[k][q] = -s * vkp + c * vkq

        iterations += 1
        history.append(off_diagonal_norm(A))

    eigenvalues = [A[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: eigenvalues[i])

    eigenvalues = [eigenvalues[i] for i in order]
    eigenvectors = [
        [V[row][i] for row in range(n)]
        for i in order
    ]

    return eigenvalues, eigenvectors, iterations, history


def main():
    n = int(input("Введите размер матрицы n: "))

    A = []
    print("Введите элементы матрицы построчно:")

    for i in range(n):
        row = list(map(float, input().split()))

        if len(row) != n:
            print("Ошибка: в каждой строке должно быть n элементов.")
            return

        A.append(row)

    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > 1e-12:
                print("Ошибка: матрица должна быть симметричной.")
                return

    eps = float(input("Введите точность eps: "))

    if eps <= 0:
        print("Ошибка: точность должна быть положительной.")
        return

    eigenvalues, eigenvectors, iterations, history = (
        jacobi_rotation(A, eps)
    )

    print("\nСобственные значения:")
    for i, value in enumerate(eigenvalues):
        print(f"lambda_{i + 1} = {value:.10f}")

    print("\nСобственные векторы (по одному в строке):")
    for i, vector in enumerate(eigenvectors):
        print(f"v_{i + 1} = " +
              " ".join(f"{x:.10f}" for x in vector))

    print(f"\nКоличество итераций: {iterations}")
    print(f"Итоговая норма: {history[-1]:.12e}")

    print("\nЗависимость нормы от числа итераций:")
    print("Итерация       t(A)")
    for k, value in enumerate(history):
        print(f"{k:5d}       {value:.12e}")


if __name__ == "__main__":
    main()
