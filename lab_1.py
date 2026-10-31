def copy_matrix(A):
    """Копия матрицы"""
    return [row[:] for row in A]


def print_matrix(A, name=""):
    """Вывод матрицы"""
    if name:
        print(name)
    for row in A:
        print("  [", end="")
        for j, val in enumerate(row):
            print(f"{val:10.6f}", end="  " if j < len(row) - 1 else "")
        print("]")
    print()


def print_vector(v, name=""):
    """Вывод вектора"""
    if name:
        print(name, end=" ")
    print("[", end="")
    for j, val in enumerate(v):
        print(f"{val:10.6f}", end="  " if j < len(v) - 1 else "")
    print("]")
    print()


def mat_mul(A, B):
    """Умножение матриц A * B"""
    n = len(A)
    m = len(B[0])
    k = len(B)
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            for p in range(k):
                C[i][j] += A[i][p] * B[p][j]
    return C


def mat_vec_mul(A, x):
    """Умножение матрицы на вектор"""
    n = len(A)
    m = len(x)
    b = [0.0] * n
    for i in range(n):
        for j in range(m):
            b[i] += A[i][j] * x[j]
    return b


def identity_matrix(n):
    """Единичная матрица n×n"""
    I = [[0.0] * n for _ in range(n)]
    for i in range(n):
        I[i][i] = 1.0
    return I


def swap_rows(M, i, j):
    """Поменять местами строки i и j"""
    M[i], M[j] = M[j], M[i]


# 1. LU-разложение с выбором главного элемента
def lu_decomposition(A):
    n = len(A)
    U = copy_matrix(A)
    L = identity_matrix(n)
    P = identity_matrix(n)

    for k in range(n - 1):
        # Поиск главного элемента в столбце k
        max_val = abs(U[k][k])
        max_row = k
        for i in range(k + 1, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                max_row = i

        if max_val < 1e-12:
            raise ValueError("Матрица вырожденная")

        # Перестановка строк
        if max_row != k:
            swap_rows(U, k, max_row)
            swap_rows(P, k, max_row)
            for col in range(k):
                L[k][col], L[max_row][col] = L[max_row][col], L[k][col]

        # Исключение Гаусса
        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]
            for j in range(k, n):
                U[i][j] -= L[i][k] * U[k][j]

    return P, L, U


# 2. Решение СЛАУ: Ly = Pb, затем Ux = y
def solve_lu(L, U, P, b):
    n = len(L)

    # Pb
    Pb = [0.0] * n
    for i in range(n):
        for j in range(n):
            Pb[i] += P[i][j] * b[j]

    # Прямая подстановка: Ly = Pb
    y = [0.0] * n
    for i in range(n):
        s = Pb[i]
        for j in range(i):
            s -= L[i][j] * y[j]
        y[i] = s

    # Обратная подстановка: Ux = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= U[i][j] * x[j]
        x[i] = s / U[i][i]

    return x


# 3. Определитель
def determinant(P, U):
    n = len(P)

    # Находим перестановку
    perm = []
    for i in range(n):
        for j in range(n):
            if P[i][j] == 1.0:
                perm.append(j)
                break

    visited = [False] * n
    num_swaps = 0
    for i in range(n):
        if not visited[i]:
            j = i
            cycle_len = 0
            while not visited[j]:
                visited[j] = True
                j = perm[j]
                cycle_len += 1
            num_swaps += cycle_len - 1

    sign = 1 if num_swaps % 2 == 0 else -1

    det_U = 1.0
    for i in range(n):
        det_U *= U[i][i]

    return sign * det_U


# 4. Обратная матрица
def inverse_matrix(P, L, U):
    n = len(L)
    A_inv = [[0.0] * n for _ in range(n)]

    for j in range(n):
        e_j = [0.0] * n
        e_j[j] = 1.0
        col = solve_lu(L, U, P, e_j)
        for i in range(n):
            A_inv[i][j] = col[i]

    return A_inv


# 5. Основная программа
if __name__ == "__main__":

    # Исходные данные
    A = [
        [ 8.0,  8.0, -5.0, -8.0],
        [ 8.0, -5.0,  9.0, -8.0],
        [ 5.0, -4.0, -6.0, -2.0],
        [ 8.0,  3.0,  6.0,  6.0]
    ]

    b = [13.0, 38.0, 14.0, -95.0]

    n = len(A)

    print("=" * 60)
    print("  ИСХОДНЫЕ ДАННЫЕ")
    print("=" * 60)
    print_matrix(A, "Матрица A:")
    print_vector(b, "Вектор b:")

    # LU-разложение
    P, L, U = lu_decomposition(A)

    print("\n" + "=" * 60)
    print("  LU-РАЗЛОЖЕНИЕ (PA = LU)")
    print("=" * 60)
    print_matrix(P, "P =")
    print_matrix(L, "L =")
    print_matrix(U, "U =")

    # Решение СЛАУ
    x = solve_lu(L, U, P, b)

    print("\n" + "=" * 60)
    print("  РЕШЕНИЕ СЛАУ")
    print("=" * 60)
    for i in range(n):
        print(f"  x{i+1} = {x[i]:.6f}")

    # Проверка: подстановка в исходную систему
    print("\n" + "=" * 60)
    print("  ПРОВЕРКА РЕШЕНИЯ (подстановка Ax)")
    print("=" * 60)
    Ax = mat_vec_mul(A, x)
    print_vector(Ax, "Ax = ")
    print_vector(b,  "b  = ")
    
    # Считаем максимальную разницу
    max_diff = 0.0
    for i in range(n):
        diff = abs(Ax[i] - b[i])
        if diff > max_diff:
            max_diff = diff
    
    if max_diff < 1e-10:
        print("  => Решение ВЕРНОЕ")
    else:
        print("  => Решение НЕВЕРНОЕ!")

    # Определитель
    det = determinant(P, U)
    print("\n" + "=" * 60)
    print("  ОПРЕДЕЛИТЕЛЬ")
    print("=" * 60)
    print(f"  det(A) = {det:.6f}")

    # Обратная матрица
    A_inv = inverse_matrix(P, L, U)

    print("\n" + "=" * 60)
    print("  ОБРАТНАЯ МАТРИЦА")
    print("=" * 60)
    print_matrix(A_inv, "A⁻¹ =")

    # Проверка
    print("\n" + "=" * 60)
    print("  ПРОВЕРКА ОБРАТНОЙ МАТРИЦЫ (A·A⁻¹)")
    print("=" * 60)
    I_check = mat_mul(A, A_inv)
    print_matrix(I_check, "A·A⁻¹ =")
    print("  => Должна получиться единичная матрица")