"""
    Description:
        This program demonstrates the use of the Strategy design pattern to implement different optimization
        methods (Gradient Descent, Genetic Algorithm, Simulated Annealing, etc.) in a modular and extensible way.
        The class Optimization acts as a context and uses a strategy (method object) to optimize a given function,
        allowing the optimization method to be easily changed without modifying the client code. Each optimization
        method is implemented in a separate class that follows the same interface, making it easy to add new methods
        without altering the existing structure.

    Author: Yadir Vega Espinoza.

    Descripción:
        Este programa demuestra el uso del patrón de diseño Strategy para implementar diferentes métodos
        de optimización (Descenso del gradiente, Algoritmo genético, Recocido simulado, etc.) de manera modular
        y extensible. La clase Optimization actúa como contexto y utiliza una estrategia (objeto método) para
        optimizar una función dada, lo que permite cambiar fácilmente el método de optimización sin modificar
        el código cliente. Cada método de optimización se implementa en una clase separada que sigue la misma
        interfaz, lo que facilita la incorporación de nuevos métodos sin alterar la estructura existente.

    Autor: Yadir Vega Espinoza.
"""


from sympy import symbols, diff
import random
import math

class Optimization:
    """
    Main class for optimization using different methods.

    Attributes:
        function (function): The function to be optimized.
        method (object): The optimization method object.
        tolerance (float): The tolerance for convergence.
        max_iter (int): The maximum number of iterations.
    """
    def __init__(self, function, method, tolerance=1e-6, max_iter=100):
        self.function = function
        self.method = method
        self.tolerance = tolerance
        self.max_iter = max_iter

    def optimize(self, initial):
        """
        Optimizes the function.

        Args:
            initial (float or tuple): The initial guess or parameters for optimization.

        Returns:
            float: The optimized value.

        Raises:
            NoConvergenceError: If the method does not converge within the maximum number of iterations.
        """
        try:
            return self.method.optimize(self.function, initial, self.tolerance, self.max_iter)
        except NoConvergenceError as e:
            print(f"Error: {e}")
            return None

class GradientDescent:
    """
    Class for the gradient descent method.

    Methods:
        optimize(function, initial, tolerance, max_iter): Optimizes the function using the gradient descent method.
    """
    def optimize(self, function, initial, tolerance, max_iter):
        x, y = symbols('x y')
        f = function(x, y)
        dfx = diff(f, x)
        dfy = diff(f, y)
        params = initial
        for iteration in range(max_iter):
            try:
                gradient = [dfx.subs({x: params[0], y: params[1]}), dfy.subs({x: params[0], y: params[1]})]
                params = [p - 0.01 * g for p, g in zip(params, gradient)]  # Gradient descent update rule
            except ZeroDivisionError:
                raise NoConvergenceError("Gradient is zero at some point.")
            if all(abs(p - params[i]) < tolerance for i, p in enumerate(params)):
                return params
        raise NoConvergenceError("Method did not converge within the maximum number of iterations.")

class GeneticAlgorithm:
    """
    Class for the genetic algorithm.

    Methods:
        optimize(function, initial, tolerance, max_iter): Optimizes the function using the genetic algorithm.
    """
    def optimize(self, function, initial, tolerance, max_iter):
        population_size = 10
        mutation_rate = 0.1
        crossover_rate = 0.8
        n_params = len(initial)

        def create_individual():
            return [random.uniform(-10, 10) for _ in range(n_params)]

        def fitness(individual):
            return -function(*individual)

        def crossover(parent1, parent2):
            crossover_point = random.randint(1, n_params - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
            return child1, child2

        def mutate(individual):
            mutated = [param + random.uniform(-1, 1) for param in individual]
            return mutated

        population = [create_individual() for _ in range(population_size)]
        best_individual = None
        best_fitness = float('-inf')

        for _ in range(max_iter):
            next_generation = []
            for _ in range(population_size // 2):
                parent1, parent2 = random.choices(population, k=2)
                if random.random() < crossover_rate:
                    child1, child2 = crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                if random.random() < mutation_rate:
                    child1 = mutate(child1)
                if random.random() < mutation_rate:
                    child2 = mutate(child2)
                next_generation.extend([child1, child2])

            population = next_generation
            for individual in population:
                fit = fitness(individual)
                if fit > best_fitness:
                    best_individual = individual
                    best_fitness = fit

        return best_individual

class SimulatedAnnealing:
    """
    Class for the simulated annealing method.

    Methods:
        optimize(function, initial, tolerance, max_iter): Optimizes the function using simulated annealing.
    """
    def optimize(self, function, initial, tolerance, max_iter):
        x, y = symbols('x y')
        f = function(x, y)
        params = initial
        best_params = params
        best_value = -float('inf')
        temperature = 1.0
        cooling_rate = 0.01

        def acceptance_probability(old_value, new_value, temperature):
            if new_value > old_value:
                return 1.0
            return math.exp((new_value - old_value) / temperature)

        for iteration in range(max_iter):
            new_params = [p + random.uniform(-0.1, 0.1) for p in params]
            new_value = -function(*new_params)
            if acceptance_probability(-function(*params), new_value, temperature) > random.random():
                params = new_params
            if new_value > best_value:
                best_value = new_value
                best_params = new_params
            temperature *= 1 - cooling_rate
            if temperature < tolerance:
                break

        return best_params

class NoConvergenceError(Exception):
    """Exception raised when a method does not converge within the maximum number of iterations."""
    pass

# Usage example
f = lambda x, y: -(x**2 + y**2)  # Example function to minimize

gradient_descent = GradientDescent()
genetic_algorithm = GeneticAlgorithm()
simulated_annealing = SimulatedAnnealing()

optimization_gradient_descent = Optimization(f, gradient_descent, tolerance=1e-8, max_iter=50)
optimization_genetic_algorithm = Optimization(f, genetic_algorithm, tolerance=1e-8, max_iter=50)
optimization_simulated_annealing = Optimization(f, simulated_annealing, tolerance=1e-8, max_iter=50)

print("Gradient Descent:", optimization_gradient_descent.optimize([2, 2]))
print("Genetic Algorithm:", optimization_genetic_algorithm.optimize([2, 2]))
print("Simulated Annealing:", optimization_simulated_annealing.optimize([2, 2]))

