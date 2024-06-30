"""
    Descripción:
        Este programa demuestra el uso del patrón de diseño Strategy para implementar diferentes métodos
        de álgebra lineal (Descomposición LU, Cholesky, QR, Gauss-Seidel, etc.) de manera modular y extensible.
        Cada clase de solución actúa como una estrategia (objeto método) para resolver un sistema de ecuaciones
        lineales con un método específico, lo que permite cambiar fácilmente el método de resolución sin modificar
        el código cliente. Cada método de álgebra lineal se implementa en una clase separada que sigue la misma
        interfaz, lo que facilita la incorporación de nuevos métodos sin alterar la estructura existente.

    Autor: Yadir Vega Espinoza.

    Description:
        This program demonstrates the use of the Strategy design pattern to implement different linear algebra
        methods (LU Decomposition, Cholesky, QR Decomposition, Gauss-Seidel, etc.) in a modular and extensible way.
        Each solver class acts as a strategy (method object) to solve a system of linear equations using a specific
        method, allowing the solver method to be easily changed without modifying the client code. Each linear algebra
        method is implemented in a separate class that follows the same interface, making it easy to add new methods
        without altering the existing structure.

    Author: Yadir Vega Espinoza.

"""
import numpy as np

class LUSolver:
    def solve(self, A, b):
        """
        Resuelve un sistema de ecuaciones lineales Ax = b utilizando la descomposición LU.

        Args:
            A (numpy.ndarray): La matriz de coeficientes.
            b (numpy.ndarray): El vector de términos independientes.

        Returns:
            numpy.ndarray: El vector solución x.
        """
        P, L, U = self.lu_decomposition(A)
        y = self.forward_substitution(L, P.dot(b))
        x = self.backward_substitution(U, y)
        return x

    def lu_decomposition(self, A):
        """
        Realiza la descomposición LU de una matriz A utilizando eliminación gaussiana con pivoteo parcial.

        Args:
            A (numpy.ndarray): La matriz a descomponer.

        Returns:
            tuple: Una tupla (P, L, U) donde P es la matriz de permutación, L es la matriz triangular inferior unitaria y U es la matriz triangular superior resultante de la descomposición.
        """
        n = len(A)
        P = np.identity(n)
        L = np.zeros((n, n))
        U = A.astype(np.float64)

        for i in range(n):
            pivot_row = i + np.argmax(np.abs(U[i:, i]))
            if pivot_row != i:
                P[[i, pivot_row]] = P[[pivot_row, i]]
                L[[i, pivot_row], :i] = L[[pivot_row, i], :i]
                U[[i, pivot_row]] = U[[pivot_row, i]]
            L[i, i] = 1
            for j in range(i + 1, n):
                factor = U[j, i] / U[i, i]
                L[j, i] = factor
                U[j, i:] -= factor * U[i, i:]
        return P, L, U

    def forward_substitution(self, L, b):
        """
        Realiza la sustitución hacia adelante para resolver Ly = Pb.

        Args:
            L (numpy.ndarray): La matriz triangular inferior.
            b (numpy.ndarray): El vector de términos independientes reordenado.

        Returns:
            numpy.ndarray: El vector solución y.
        """
        n = len(L)
        y = np.zeros(n)
        for i in range(n):
            y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
        return y

    def backward_substitution(self, U, y):
        """
        Realiza la sustitución hacia atrás para resolver Ux = y.

        Args:
            U (numpy.ndarray): La matriz triangular superior.
            y (numpy.ndarray): El vector solución de Ly = Pb.

        Returns:
            numpy.ndarray: El vector solución x.
        """
        n = len(U)
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
        return x

class CholeskySolver:
    def solve(self, A, b):
        """
        Resuelve un sistema de ecuaciones lineales Ax = b utilizando la factorización de Cholesky.

        Args:
            A (numpy.ndarray): La matriz de coeficientes.
            b (numpy.ndarray): El vector de términos independientes.

        Returns:
            numpy.ndarray: El vector solución x.
        """
        L = self.cholesky_factorization(A)
        y = self.forward_substitution(L, b)
        x = self.backward_substitution(L.T, y)
        return x

    def cholesky_factorization(self, A):
        """
        Realiza la factorización de Cholesky para una matriz simétrica definida positiva A.

        Args:
            A (numpy.ndarray): La matriz a factorizar.

        Returns:
            numpy.ndarray: La matriz triangular inferior L.
        """
        n = len(A)
        L = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1):
                if i == j:
                    L[i, i] = np.sqrt(A[i, i] - np.sum(L[i, :i]**2))
                else:
                    L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]

        return L

    def forward_substitution(self, L, b):
        """
        Realiza la sustitución hacia adelante para resolver Ly = b.

        Args:
            L (numpy.ndarray): La matriz triangular inferior.
            b (numpy.ndarray): El vector de términos independientes reordenado.

        Returns:
            numpy.ndarray: El vector solución y.
        """
        n = len(L)
        y = np.zeros(n)
        for i in range(n):
            y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
        return y

    def backward_substitution(self, U, y):
        """
        Realiza la sustitución hacia atrás para resolver Ux = y.

        Args:
            U (numpy.ndarray): La matriz triangular superior.
            y (numpy.ndarray): El vector solución de Ly = b.

        Returns:
            numpy.ndarray: El vector solución x.
        """
        n = len(U)
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
        return x

class QRSolver:
    def solve(self, A, b):
        """
        Resuelve un sistema de ecuaciones lineales Ax = b utilizando la factorización QR.

        Args:
            A (numpy.ndarray): La matriz de coeficientes.
            b (numpy.ndarray): El vector de términos independientes.

        Returns:
            numpy.ndarray: El vector solución x.
        """
        Q, R = np.linalg.qr(A)
        x = np.linalg.solve(R, Q.T.dot(b))
        return x

class GaussSeidelSolver:
    def solve(self, A, b):
        """
        Resuelve un sistema de ecuaciones lineales Ax = b utilizando el método de Gauss-Seidel.

        Args:
            A (numpy.ndarray): La matriz de coeficientes.
            b (numpy.ndarray): El vector de términos independientes.

        Returns:
            numpy.ndarray: El vector solución x.
        """
        n = len(A)
        x = np.zeros(n)
        max_iter = 1000
        tol = 1e-6

        for _ in range(max_iter):
            x_new = np.zeros(n)
            for i in range(n):
                x_new[i] = (b[i] - np.dot(A[i, :i], x_new[:i]) - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
            if np.linalg.norm(x_new - x) < tol:
                return x_new
            x = x_new

        raise NoConvergenceError("Method did not converge within the maximum number of iterations.")

class NoConvergenceError(Exception):
    """Excepción lanzada cuando un método no converge dentro del número máximo de iteraciones."""
    pass

# Ejemplo de uso
A = np.array([[4, 2, 1], [2, 9, 3], [1, 3, 6]])
b = np.array([1, 2, 3])

lu_solver = LUSolver()
cholesky_solver = CholeskySolver()
qr_solver = QRSolver()
gauss_seidel_solver = GaussSeidelSolver()

print("LU Solver:", lu_solver.solve(A, b))
print("Cholesky Solver:", cholesky_solver.solve(A, b))
print("QR Solver:", qr_solver.solve(A, b))
print("Gauss-Seidel Solver:", gauss_seidel_solver.solve(A, b))
