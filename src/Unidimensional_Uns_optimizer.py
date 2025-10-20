import matplotlib.pyplot as plt
import numpy as np
from Optimizer_Uns_Base import UnconstrainedOptimizerBase
class UnidimensionalUnconstrainedOptimizerBase(UnconstrainedOptimizerBase):
    def __init__(self, function, init_value, epsilon,derivative=None,second_derivative=None):
        super().__init__(function, init_value, epsilon,derivative,second_derivative)
  
    def newton_algorithm(self,stop_rule='grad',approx_derivative=True):
        '''
        Newton method (iterative algorithm)

        1. Determine a initial value x_0 ∈ R**n
        2. Repeat while ||x_k+1 - x_k|| > epsilon
        3. Propose as solution x_k

        In Newton method not only besides the differenciability in the objective function ,
        we need double diferenciability (Taylor aproximation):

                f'(x) = f'(x_0) + f''(x_0)(x-x_0)

        The minimum has to verify the null derivative:

                x_(k+1) = x_k - f'(x_k)/f''(x_k) k = 0,1,...

        Stop rule is based in chosing x_k as optimum value when:

                ||x_k - x_(k-1)|| < epsilon
        or:
                |f'(x_k)| < epsilon

        '''
        STOP_RULES = {
            "grad": self.stop_by_grad,
            "x_diff": self.stop_by_x_diff,
            "f_diff": self.stop_by_f_diff,
        }

        stopping_condition = STOP_RULES.get(stop_rule)
        if stopping_condition is None:
            raise ValueError(f"Unknown stop_rule '{stop_rule}'")

        if approx_derivative:
            derivative = self.derivative_approx
            derivative_second = self.second_derivative_approx

        elif self.derivative is not None:
            derivative = self.derivative
            derivative_second = self.derivative_second
        else:
            raise ValueError("No derivative function was provided, nor was an approximation requested.")
        
        k = 1
        data = {}
        x_k = self.init_value
        grad = derivative(x_k)
        grad_2 = derivative_second(x_k)
        x_k_1 = x_k - grad / grad_2
        grad_new = derivative(x_k_1)
        data[k] = (x_k,x_k_1)
        print(f"Iter {k}: x_k = {x_k:.6f}, grad = {grad:.6f}, grad_2 = {grad_2:.6f}, x_k+1 = {x_k_1:.6f}")

        while stopping_condition(grad_new, x_k_1, x_k):
            k += 1
            x_k = x_k_1
            grad = derivative(x_k)
            grad_2 = derivative_second(x_k)
            x_k_1 = x_k - grad / grad_2
            grad_new = derivative(x_k_1)
            data[k] = (x_k,x_k_1)
            print(f"Iter {k}: x_k = {x_k:.6f}, grad = {grad:.6f}, grad_2 = {grad_2:.6f}, x_k+1 = {x_k_1:.6f}")

        print(f"Stopped after {k} iterations. Final approximation: x = {x_k_1:.6f}, grad_x {derivative(x_k_1):.6f}")
        return x_k_1,k,data

    
    def plot_iterarion_iterative_2D(self, data):
        """
        Plots the function and the iteration points x_k from Newton's method.

        Parameters:
        - data: dictionary from the newton_algorithm method
                where keys are iteration numbers and values are tuples (x_k, x_k+1)
        """
        # Obtener los valores de x en cada iteración
        x_values = [data[k][0] for k in data]
        
        # Crear un rango para graficar la función
        x_min = min(x_values) - 1
        x_max = max(x_values) + 1
        x_plot = np.linspace(x_min, x_max, 400)
        y_plot = [self.function(xi) for xi in x_plot]

        # Evaluar la función en los puntos de iteración
        y_values = [self.function(xi) for xi in x_values]

        # Graficar la función
        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, y_plot, label='f(x)', color='blue')
        
        # Graficar los puntos de iteración
        plt.plot(x_values, y_values, 'ro--', label='Iterations')

        # Etiquetas
        for i, (xk, yk) in enumerate(zip(x_values, y_values)):
            plt.text(xk, yk, f'$x_{i}$', fontsize=9, ha='right')

        plt.title("Newton's Method Iterations Over Function")
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_function(self, x_min, x_max, num_points=400):
        """
        Plots the function self.function over the interval [x_min, x_max].

        Parameters:
        - x_min: float, start of interval
        - x_max: float, end of interval
        - num_points: int, number of points to plot
        """
        x_vals = np.linspace(x_min, x_max, num_points)
        y_vals = [self.function(x) for x in x_vals]

        plt.figure(figsize=(8,5))
        plt.plot(x_vals, y_vals, label='f(x)')
        plt.title("Function plot")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True)
        plt.legend()
        plt.show()


    
