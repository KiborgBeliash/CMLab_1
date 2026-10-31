A = [
    [-19,   2,  -1,  -8],
    [  2,  14,   0,  -4],
    [  6,  -5, -20,  -6],
    [ -6,   4,  -2,  15]
]
b = [38, 20, 52, 43]
n = 4
EPS = 1e-6
MAX_ITER = 100000


# Вспомогательные функции
def vec_norm_c(v):
    return max(abs(x) for x in v)


def mat_norm_c(M):
    return max(sum(abs(M[i][j]) for j in range(len(M))) for i in range(len(M)))


def mat_vec_mul(M, v):
    n = len(v)
    return [sum(M[i][j] * v[j] for j in range(n)) for i in range(n)]


def vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def vec_sub(a, b):
    return [a[i] - b[i] for i in range(len(a))]


# Построение а и b
def build_alpha_beta(A, b):
    n = len(b)
    alpha = [[0.0] * n for _ in range(n)]
    beta = [0.0] * n
    for i in range(n):
        aii = A[i][i]
        beta[i] = b[i] / aii
        for j in range(n):
            if i != j:
                alpha[i][j] = -A[i][j] / aii
    return alpha, beta


# Метод простых итераций 
def mpi(alpha, beta, eps):
    norm_alpha = mat_norm_c(alpha)
    factor = norm_alpha / (1.0 - norm_alpha)
    x_prev = beta[:]
    iterations = 0

    for _ in range(MAX_ITER):
        x_curr = vec_add(mat_vec_mul(alpha, x_prev), beta)
        iterations += 1
        diff = vec_norm_c(vec_sub(x_curr, x_prev))
        if factor * diff <= eps:
            return x_curr, iterations, factor * diff
        x_prev = x_curr[:]

    return x_prev, iterations, factor * vec_norm_c(vec_sub(
        vec_add(mat_vec_mul(alpha, x_prev), beta), x_prev))


# Метод Зейделя
def zeidel(alpha, beta, eps):
    n = len(beta)
    norm_alpha = mat_norm_c(alpha)
    factor = norm_alpha / (1.0 - norm_alpha)
    x = beta[:]
    iterations = 0

    for _ in range(MAX_ITER):
        x_new = x[:]
        for i in range(n):
            s = beta[i]
            for j in range(n):
                if j != i:
                    s += alpha[i][j] * x_new[j]
            x_new[i] = s
        iterations += 1
        diff = vec_norm_c(vec_sub(x_new, x))
        if factor * diff <= eps:
            return x_new, iterations, factor * diff
        x = x_new[:]

    return x, iterations, factor * vec_norm_c(vec_sub(x, beta))


def main():
    alpha, beta = build_alpha_beta(A, b)
    q = mat_norm_c(alpha)

    print(f"Точность eps = {EPS}")
    print()

    # МПИ
    x_mpi, it_mpi, err_mpi = mpi(alpha, beta, EPS)
    print("Метод простых итераций (Якоби)")
    print(f"Норма матрицы итераций q = {q:.6f}")
    for i in range(n):
        print(f"x{i+1} = {x_mpi[i]:.10f}")
    print(f"Количество итераций: {it_mpi}")
    print(f"Оценка погрешности: {err_mpi:.2e}")
    print()

    # Зейдель
    x_z, it_z, err_z = zeidel(alpha, beta, EPS)
    print("Метод Зейделя")
    print(f"Норма матрицы итераций q = {q:.6f}")
    for i in range(n):
        print(f"x{i+1} = {x_z[i]:.10f}")
    print(f"Количество итераций: {it_z}")
    print(f"Оценка погрешности: {err_z:.2e}")
    print()

    print("СРАВНЕНИЕ МЕТОДОВ")
    print(f"Итерации Якоби: {it_mpi}")
    print(f"Итерации Зейделя: {it_z}")
    print(f"Разница: {it_mpi - it_z} итераций ")


if __name__ == "__main__":
    main()