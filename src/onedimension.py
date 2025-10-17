import matplotlib.pyplot as plt
import numpy as np

class UnconstrainedUnidimensionalOptimization:
    def __init__(self, function,init_value, tol, derivative=None,second_derivative=None):
        self.function = function
        self.init_value = init_value
        self.tol = tol
        self.derivative = derivative
        self.derivative_second = second_derivative
    def derivative_approx(self,f,x,h=1e-5):

        '''
        Central-difference approximation
        '''
        return (f(x + h) - f(x - h)) / (2 * h)

    def stop_by_grad(self,grad, x_new, x):
        return np.linalg.norm(grad) > self.tol

    def stop_by_x_diff(self,grad, x_new, x):
        return np.linalg.norm(x_new - x) > self.tol

    def stop_by_f_diff(self,grad, x_new, x):
        return abs(self.function(x_new) - self.function(x)) > self.tol

    def newton_algorithm(self,stop_rule='grad',aprox_derivative=True):
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
        # Lookup the stopping function
        stopping_condition = STOP_RULES.get(stop_rule)
        if stopping_condition is None:
            raise ValueError(f"Unknown stop_rule '{stop_rule}'")

        if aprox_derivative:
            derivative = lambda x: self.derivative_approx(self.function,x)
            derivative_second = lambda x: self.derivative_approx(derivative, x)

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

        print(f"Stopped after {k} iterations. Final approximation: x = {x_k_1:.6f}")
        return x_k_1,k
    

    
