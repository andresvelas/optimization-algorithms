import numpy as np
from contextlib import redirect_stdout
import os
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go
from Optimizer_Uns_Base import UnconstrainedOptimizerBase
from  Unidimensional_Uns_optimizer import UnidimensionalUnconstrainedOptimizerBase


class MultidimensionalUnconstrainedOptimizerBase(UnconstrainedOptimizerBase):
    def __init__(self, function, init_value, epsilon,gradient=None,hessian=None,derivative=None,second_derivative=None):
        super().__init__(function, init_value, epsilon,gradient,hessian,derivative,second_derivative)
        self.x_k = None
        self.d_k = None
  
    def g(self,lambda_k):
        return self.function(self.x_k+ lambda_k*self.d_k)
    
    def steepest_descent(self,N,derivative_approx=True):
        '''
        Steepest descent 

        This algorithm suppose the differentiability of the function to minimize.

        This iterative algorithm set off in the iteration k=0, where we suggest a initial value
        x_0 ∈ R**(n).

        For the iteration k we want to search the direction d_k ∈ R**(n) that maximizes the function descent
        (gradient vector)

        The steepest descent is follow by the gradient vector with unit module:
                d_k = - (gradient_f)/||gradient_f||
        so the step to one solution to another is:
                x_(K+1) = h(x_k) = x_k + lambda_k * d_k
        
        To the determination of lambda_k ∈ R solve

               lambda_k = argmin f(x_k + lambda * d_k)
        
        We can observe that x_k and d_k are known variables, and the variable to optimize
        is lambda.

        The stop rule for this algorithm is verified in the k iteration when:
                ||gradient_f|| < epsilon
        '''
        data = {}
        k = 1
        if derivative_approx:
            gradient = self.gradient_approx
        elif self.gradient is not None:
            gradient = self.gradient

        else:
            raise ValueError("No derivative function was provided, nor was an approximation requested.")
        
        x_k = self.init_value
        gradient_f_x_k = gradient(x_k)
        d_k = - gradient_f_x_k / np.linalg.norm(gradient_f_x_k)

        self.x_k = x_k
        self.d_k = d_k
        lambda_k = UnidimensionalUnconstrainedOptimizerBase(self.g,1,1e-5)
        print(50*'-')
        print('Lambda aproximation (first iteration)')
        lambda_k ,_,_ = lambda_k.newton_algorithm()
        print(50*'-')
        x_k_1 = x_k + lambda_k * d_k
        print(f"Iter {k}: x_k = {x_k}, lambda_k = {lambda_k:.6f}, d_k = {d_k}, x_k+1 = {x_k_1}")
        gradient_f_x_k = gradient(x_k_1)
        data[k] = {'x_k':x_k,
                   'lambda_k':lambda_k,
                   'd_k':d_k}
        while self.stop_by_grad(gradient_f_x_k,_,_) and k < N:
            k+=1
            d_k = - gradient_f_x_k / np.linalg.norm(gradient_f_x_k)
            x_k = x_k_1
            self.x_k = x_k
            self.d_k = d_k

            lambda_k = UnidimensionalUnconstrainedOptimizerBase(self.g,1,1e-5)
            with open(os.devnull, 'w') as f, redirect_stdout(f):
                lambda_k ,_,_ = lambda_k.newton_algorithm()
            x_k_1 = x_k + lambda_k * d_k
            gradient_f_x_k = gradient(x_k_1)
            print(f"Iter {k}: x_k = {x_k}, lambda_k = {lambda_k:.6f}, d_k = {d_k}, x_k+1 = {x_k_1}")
            data[k] = {'x_k':x_k,
                   'lambda_k':lambda_k,
                   'd_k':d_k}
            
        print(f"Stopped after {k} iterations. Final approximation: x = {x_k_1}")

        return x_k_1, k ,data
    def newton_method(self,N,stop_rule='grad',derivative_approx=True):
        '''
        Newton algorithm

        Extension to n dimensions of the one-dimensional Newton algorithm.
        It is based on approximating the gradient of a function f: ℝⁿ → ℝ using the Taylor expansion
        up to the first term, assuming the function is twice differentiable.
        Given a point x_k ∈ ℝⁿ, the gradient can be approximated as:

        ∇f(x) ≈ ∇f(x_k) + H(x_k)(x - x_k)

        where H(x_k) is the Hessian matrix of f at x_k:

        H(x_k) = [
        [∂²f(x_k)/∂x₁∂x₁, ..., ∂²f(x_k)/∂x₁∂xₙ],
        [...                      ...               ],
        [∂²f(x_k)/∂xₙ∂x₁, ..., ∂²f(x_k)/∂xₙ∂xₙ]
        ]

        Since the optimum of the function f is reached at a stationary point, we have:
        ∇f(x_{k+1}) = 0 ≈ ∇f(x_k) + H(x_k)(x_{k+1} - x_k)

        Rearranging, we obtain the update rule:

        x_{k+1} = x_k - H(x_k)^{-1} ∇f(x_k)

        This iteration is repeated until convergence to find the function's minimum.
        '''
        STOP_RULES = {
            "grad": self.stop_by_grad,
            "x_diff": self.stop_by_x_diff,
            "f_diff": self.stop_by_f_diff,
        }
        stopping_condition = STOP_RULES.get(stop_rule)
        if stopping_condition is None:
            raise ValueError(f"Unknown stop_rule '{stop_rule}'")
        if derivative_approx:
            gradient = self.gradient_approx
            hessian = self.hessian_approx

        elif self.gradient is not None:
            gradient = self.gradient
            hessian = self.hessian
        else:
            raise ValueError("No derivative function was provided, nor was an approximation requested.")
        
        k = 1
        data = {}

        x_k = self.init_value
        data[k-1] = {'x_k':x_k}
        grad = gradient(x_k)
        hessian = hessian(x_k)
        x_k_1 = x_k - np.linalg.inv(hessian) @ grad
        grad_new = gradient(x_k_1)
        data[k] = {'x_k':x_k_1}
        print(f"Iter {k}: x_k = {x_k}, grad = {grad_new}, hessian = {hessian}, x_k+1 = {x_k_1}")
        while stopping_condition(grad_new, x_k_1, x_k) and k < N:               
            k += 1
            grad = gradient(x_k)
            hessian = hessian(x_k)
            x_k_1 = x_k - np.linalg.inv(hessian) @ grad
            x_k = x_k_1
            grad_new = gradient(x_k_1)
            data[k] = {'x_k':x_k}
            print(f"Iter {k}: x_k = {x_k}, grad = {grad}, hessian = {hessian}, x_k+1 = {x_k_1}")

        print(f"Stopped after {k} iterations. Final approximation: x = {x_k_1}")
        return x_k_1,k,data


    def plot_contour_with_points_and_path(self, data, grid_size=150, cmap='Greys', levels=80,
                                        margin_x=3, margin_y=1):

        xs = np.array([v['x_k'] for v in data.values()])


        x_min = xs[:, 0].min() - margin_x
        x_max = xs[:, 0].max() + margin_x
        y_min = xs[:, 1].min() - margin_y
        y_max = xs[:, 1].max() + margin_y

        x = np.linspace(x_min, x_max, grid_size)
        y = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x, y)

        Z = np.array([[self.function(np.array([xi, yi])) for xi in x] for yi in y])


        fig, ax = plt.subplots(figsize=(10, 8))

  
        contour = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.9)
        ax.contour(X, Y, Z, levels=levels, colors='black', linewidths=0.4)


        ax.scatter(xs[:, 0], xs[:, 1], color='limegreen', edgecolor='black', s=20, label="Puntos $x_k$")


        ax.plot(xs[:, 0], xs[:, 1], color='red', linewidth=1, label='Trayectoria')


        ax.legend(loc='upper right')
        ax.set_title("Contorno de $f(x, y)$ con trayectoria del método")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Barra de color
        fig.colorbar(contour, ax=ax, label="f(x, y)")

        plt.tight_layout()
        plt.show()

    def plot_function_3d(self, grid_size=100):
        '''
        Plotea la función self.function (o self.g) en 3D usando Plotly.
        La función debe recibir un np.array([x, y]) y retornar un escalar.
        '''
        # Definir el rango del gráfico
        x_range = y_range = 5  # ajustable
        x = np.linspace(-x_range, x_range, grid_size)
        y = np.linspace(-y_range, y_range, grid_size)
        X, Y = np.meshgrid(x, y)

        # Evaluar la función en la grilla
        Z = np.array([[self.function(np.array([xi, yi])) for xi in x] for yi in y])

        # Crear gráfico 3D
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
        fig.update_layout(
            title='Superficie de f(x, y)',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='f(x, y)'
            ),
            width=800,
            height=700
        )
        fig.show()
        