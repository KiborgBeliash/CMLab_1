import math

def dot_product(v1, v2):
    """Скалярное произведение двух векторов"""
    return sum(x * y for x, y in zip(v1, v2))

def vector_norm(v):
    """Евклидова норма вектора"""
    return math.sqrt(sum(x * x for x in v))

def mat_mul(A, B):
    """Умножение двух матриц"""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    C = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(rows_A):
                C[i][j] += A[i][k] * B[k][j]
    return C

def qr_decomposition(A):
    """
    QR-разложение матрицы с использованием преобразований Хаусхолдера.
    Возвращает ортогональную матрицу Q и верхнюю треугольную R.
    """
    n = len(A)
    # Копируем матрицу A в R
    R = [row[:] for row in A]
    # Инициализируем Q единичной матрицей
    Q = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    
    for k in range(n - 1):
        # Извлекаем k-й столбец от диагонали до конца
        x = [R[i][k] for i in range(k, n)]
        norm_x = vector_norm(x)
        
        # Формируем вектор Хаусхолдера v
        v = [0.0] * n
        sign = 1.0 if x[0] >= 0 else -1.0
        v[k] = x[0] + sign * norm_x
        
        # ИСПРАВЛЕНИЕ: используем i - k для корректного индекса в векторе x
        for i in range(k + 1, n):
            v[i] = x[i - k] 
            
        norm_v_sq = dot_product(v, v)
        if norm_v_sq < 1e-12:
            continue
            
        # Формируем матрицу Хаусхолдера H = E - 2 * (v * v^T) / (v^T * v)
        H = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for i in range(k, n):
            for j in range(k, n):
                H[i][j] -= 2.0 * v[i] * v[j] / norm_v_sq
                
        # Обновляем R и Q
        R = mat_mul(H, R)
        Q = mat_mul(Q, H)
        
    return Q, R

def qr_algorithm(A, epsilon=1e-6, max_iter=1000):
    """
    QR-алгоритм для нахождения собственных значений.
    """
    n = len(A)
    Ak = [row[:] for row in A]
    
    for iteration in range(max_iter):
        Q, R = qr_decomposition(Ak)
        Ak_next = mat_mul(R, Q)
        
        # Критерий сходимости: норма поддиагональных элементов < epsilon
        sub_diag_sum = 0.0
        for i in range(1, n):
            for j in range(i):
                sub_diag_sum += Ak_next[i][j] ** 2
                
        if math.sqrt(sub_diag_sum) < epsilon:
            Ak = Ak_next
            break
            
        Ak = Ak_next
        
    return Ak

# Входные данные
A = [
    [-6.0, 1.0, -4.0],
    [-6.0, 8.0, -2.0],
    [2.0, -9.0, 5.0]
]

epsilon = 1e-6
eigenvalues_matrix = qr_algorithm(A, epsilon=epsilon)

print(f"Матрица после схождения (с точностью {epsilon}):")
for row in eigenvalues_matrix:
    print([f"{val:8.4f}" for val in row])

print("\nСобственные значения матрицы (диагональные элементы):")
for i in range(len(eigenvalues_matrix)):
    print(f"lambda_{i+1} = {eigenvalues_matrix[i][i]:.6f}")