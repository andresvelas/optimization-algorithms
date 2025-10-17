import numpy as np

class Optimizer():
    def __init__(self,function,init_value,epsilon):
        self.function=function
        self.init_value = init_value
        self.epsilon = epsilon

    def steepest_descent(self):
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
        gradient_f_x_k = self.derivative_aprox(x_k)
        d_k = - gradient_f_x_k / np.linalg.norm(np.linalg.norm())

        x_k_1 = x_k + lambda_k * d_k
        np.linalg.norm()


        return None
    def derivative_aprox(self,v,h=1e-5):
        v = np.array(v, dtype=float)
        n = len(v)

        perturbations = np.eye(n) * h

        
        # v has shape (n,) → it's a 1D row vector
        # perturbations has shape (n, n) → identity matrix scaled by h

        # Thanks to NumPy broadcasting:
        # - v is implicitly broadcasted to shape (n, n), where each row is a copy of v
        # - This allows row-wise addition:
        #     - Row 0 of perturbations adds h to v[0]
        #     - Row 1 of perturbations adds h to v[1]
        #     - ...
        #     - Row n-1 of perturbations adds h to v[n-1]

        # As a result: 
        # v_forward becomes a matrix (n, n) where each row is v with +h added to one component
        v_forward = v + perturbations

        # Same logic for subtraction: each row has -h added to a different component
        v_backward = v - perturbations

        f_forward = np.apply_along_axis(self.function, 1, v_forward)
        f_backward = np.apply_along_axis(self.function, 1, v_backward)
        
        grad = (f_forward - f_backward) / (2 * h)
        return grad
    