"""
    Descripción:
        Este programa demuestra el uso del patrón de diseño Strategy para implementar diferentes métodos
        de interpolación (Newton, Lagrange, spline cúbico, etc.) de manera modular y extensible. La clase
        Interpolation actúa como contexto y utiliza una estrategia (objeto método) para interpolar una función
        a partir de un conjunto de puntos dados, lo que permite cambiar fácilmente el método de interpolación
        sin modificar el código cliente. Cada método de interpolación se implementa en una clase separada que
        sigue la misma interfaz, lo que facilita la incorporación de nuevos métodos sin alterar la estructura
        existente.

    Autor: Yadir Vega Espinoza.

    Description:
        This program demonstrates the use of the Strategy design pattern to implement different interpolation
        methods (Newton, Lagrange, cubic spline, etc.) in a modular and extensible way. The Interpolation class
        acts as a context and uses a strategy (method object) to interpolate a function from a given set of points,
        allowing the interpolation method to be easily changed without modifying the client code. Each interpolation
        method is implemented in a separate class that follows the same interface, making it easy to add new methods
        without altering the existing structure.

    Author: Yadir Vega Espinoza.

"""
import numpy as np

class Interpolation:
    """
    Main class for interpolation using different methods.

    Attributes:
        method (object): The interpolation method object.
    """
    def __init__(self, method):
        self.method = method

    def interpolate(self, x, y, x_new):
        """
        Interpolates the function.

        Args:
            x (array): The x-coordinates of the data points.
            y (array): The y-coordinates of the data points.
            x_new (array): The x-coordinates where the interpolation is evaluated.

        Returns:
            array: The interpolated values at x_new.
        """
        return self.method.interpolate(x, y, x_new)

class NewtonInterpolation:
    """
    Class for Newton interpolation.

    Methods:
        interpolate(x, y, x_new): Interpolates the function using Newton's method.
    """
    def interpolate(self, x, y, x_new):
        n = len(x)
        coefficients = self.compute_coefficients(x, y)
        result = []
        for x_val in x_new:
            y_val = sum(coefficients[i] * self._newton_basis(x, i, x_val) for i in range(n))
            result.append(y_val)
        return result

    def compute_coefficients(self, x, y):
        n = len(x)
        coefficients = y.copy()
        for i in range(1, n):
            for j in range(n - 1, i - 1, -1):
                coefficients[j] = (coefficients[j] - coefficients[j - 1]) / (x[j] - x[j - i])
        return coefficients

    def _newton_basis(self, x, k, x_val):
        basis = 1
        for i in range(k):
            basis *= (x_val - x[i])
        return basis

class LagrangeInterpolation:
    """
    Class for Lagrange interpolation.

    Methods:
        interpolate(x, y, x_new): Interpolates the function using Lagrange's method.
    """
    def interpolate(self, x, y, x_new):
        result = []
        for x_val in x_new:
            y_val = sum(y[j] * self._lagrange_basis(x, j, x_val) for j in range(len(x)))
            result.append(y_val)
        return result

    def _lagrange_basis(self, x, j, x_val):
        basis = 1
        for m in range(len(x)):
            if m != j:
                basis *= (x_val - x[m]) / (x[j] - x[m])
        return basis


# Ejemplo de uso
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 8, 27, 64, 125])
x_new = np.linspace(0, 5, 1000)

newton_interpolation = Interpolation(NewtonInterpolation())
lagrange_interpolation = Interpolation(LagrangeInterpolation())

print("Newton Interpolation:", newton_interpolation.interpolate(x, y, x_new))
print("\n\nLagrange Interpolation:", lagrange_interpolation.interpolate(x, y, x_new))

