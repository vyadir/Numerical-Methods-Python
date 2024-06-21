"""
    Descripción:
        Este programa demuestra el uso del patrón de diseño Strategy para implementar diferentes métodos
        de búsqueda de raíces (Newton-Raphson, secante, falsa posición, etc.) de manera modular y extensible.
        La clase RootFinding actúa como contexto y utiliza una estrategia (objeto método) para encontrar la raíz
        de una función dada, lo que permite cambiar fácilmente el método de búsqueda de raíces sin modificar
        el código cliente. Cada método de búsqueda de raíces se implementa en una clase separada que sigue
        la misma interfaz, lo que facilita la incorporación de nuevos métodos sin alterar la estructura existente.

    Autor: Yadir Vega Espinoza.

    Description:
        This program demonstrates the use of the Strategy design pattern to implement different root-finding
        methods (Newton-Raphson, secant, false position, etc.) in a modular and extensible way. The RootFinding
        class acts as a context and uses a strategy (method object) to find the root of a given function, allowing
        the root-finding method to be easily changed without modifying the client code. Each root-finding method
        is implemented in a separate class that follows the same interface, making it easy to add new methods
        without altering the existing structure.

    Author: Yadir Vega Espinoza.

"""
from sympy import symbols, diff

class RootFinding:
    """
    Main class for finding roots using different methods.

    Attributes:
        function (function): The function for which the root is sought.
        method (object): The root-finding method object.
        tolerance (float): The tolerance for convergence.
        max_iter (int): The maximum number of iterations.
    """
    def __init__(self, function, method, tolerance=1e-6, max_iter=100):
        self.function = function
        self.method = method
        self.tolerance = tolerance
        self.max_iter = max_iter

    def find_root(self, initial):
        """
        Finds the root of the function.

        Args:
            initial (float or tuple): The initial guess or interval for the root.

        Returns:
            float: The root found.

        Raises:
            NoConvergenceError: If the method does not converge within the maximum number of iterations.
        """
        try:
            return self.method.find_root(self.function, initial, self.tolerance, self.max_iter)
        except NoConvergenceError as e:
            print(f"Error: {e}")
            return None

class NewtonRaphsonMethod:
    """
    Class for the Newton-Raphson method.

    Methods:
        find_root(function, initial, tolerance, max_iter): Finds the root of the function using the Newton-Raphson method.
    """
    def find_root(self, function, initial, tolerance, max_iter):
        x = symbols('x')
        f = function(x)
        df = diff(f, x)
        x0 = initial
        for iteration in range(max_iter):
            try:
                x1 = x0 - f.subs(x, x0) / df.subs(x, x0)
            except ZeroDivisionError:
                raise NoConvergenceError("Derivative is zero at some point.")
            if abs(x1 - x0) < tolerance:
                return x1
            x0 = x1
        raise NoConvergenceError("Method did not converge within the maximum number of iterations.")

class SecantMethod:
    """
    Class for the secant method.

    Methods:
        find_root(function, initial, tolerance, max_iter): Finds the root of the function using the secant method.
    """
    def find_root(self, function, initial, tolerance, max_iter):
        x = symbols('x')
        f = function(x)
        iteration = 0
        x0, x1 = initial
        while iteration < max_iter:
            x2 = x1 - f.subs(x, x1) * (x1 - x0) / (f.subs(x, x1) - f.subs(x, x0))
            if abs(x2 - x1) < tolerance:
                return x2
            x0, x1 = x1, x2
            iteration += 1
        raise NoConvergenceError("Method did not converge within the maximum number of iterations.")

class FalsePositionMethod:
    """
    Class for the false position method.

    Methods:
        find_root(function, interval, tolerance, max_iter): Finds the root of the function using the false position method.
    """
    def find_root(self, function, interval, tolerance, max_iter):
        a, b = interval
        iteration = 0
        while iteration < max_iter:
            c = b - (function(b) * (b - a)) / (function(b) - function(a))
            if abs(function(c)) < tolerance:
                return c
            if function(a) * function(c) < 0:
                b = c
            else:
                a = c
            iteration += 1
        raise NoConvergenceError("Method did not converge within the maximum number of iterations.")

class NoConvergenceError(Exception):
    """Exception raised when a method does not converge within the maximum number of iterations."""
    pass

# Usage example
f = lambda x: x**2 - 4

newton_raphson_method = NewtonRaphsonMethod()
secant_method = SecantMethod()
false_position_method = FalsePositionMethod()

root_finding_newton = RootFinding(f, newton_raphson_method, tolerance=1e-8, max_iter=50)
root_finding_secant = RootFinding(f, secant_method, tolerance=1e-8, max_iter=50)
root_finding_false_position = RootFinding(f, false_position_method, tolerance=1e-8, max_iter=50)

print("Newton-Raphson:", root_finding_newton.find_root(2))
print("Secant:", float(root_finding_secant.find_root((1, 3))))
print("False Position:", root_finding_false_position.find_root((1, 3)))