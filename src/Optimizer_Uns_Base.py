import numpy as np

class UnconstrainedOptimizerBase():
    def __init__(self, function,init_value, epsilon=1e-6, gradient= None, hessian= None,derivative=None,second_derivative=None):
        self.function = function
        self.init_value = init_value
        self.gradient = gradient
        self.hessian = hessian
        self.derivative = derivative
        self.second_derivative = second_derivative
        self.epsilon = epsilon

    def stop_by_grad(self,grad, x_new, x):
        return np.linalg.norm(grad) > self.epsilon

    def stop_by_x_diff(self,grad, x_new, x):
        return np.linalg.norm(x_new - x) > self.epsilon

    def stop_by_f_diff(self,grad, x_new, x):
        return abs(self.function(x_new) - self.function(x)) > self.epsilon


    def hessian_approx(self, x, h=1e-5, regularization=True, eps_reg=1e-6):
        """
        Aproximación de la Hessiana con opción de regularización automática
        para evitar singularidad o matrices mal condicionadas.
        
        Args:
            x: vector de parámetros
            h: paso para diferencias finitas
            regularization: si True, se agrega un eps a la diagonal si la matriz es mal condicionada
            eps_reg: valor de regularización (agregado a la diagonal)
        """
        x = np.asarray(x, dtype=float).flatten()
        n = x.size
        H = np.zeros((n, n))
        fx = self.function(x)
        I = np.eye(n)

        # Calcular Hessiana por diferencias finitas
        for i in range(n):
            f_forward = self.function(x + h * I[i])
            f_backward = self.function(x - h * I[i])
            H[i, i] = (f_forward - 2 * fx + f_backward) / h**2

            for j in range(i + 1, n):
                f1 = self.function(x + h * I[i] + h * I[j])
                f2 = self.function(x + h * I[i] - h * I[j])
                f3 = self.function(x - h * I[i] + h * I[j])
                f4 = self.function(x - h * I[i] - h * I[j])
                H[i, j] = H[j, i] = (f1 - f2 - f3 + f4) / (4 * h**2)

        # Regularización si está mal condicionada
        if regularization:
            cond_number = np.linalg.cond(H)
            if cond_number > 1e12:
                H += eps_reg * np.eye(n)  # evita singularidad

        return H


    def gradient_approx(self,v,h=1e-5):
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
        v_backward = v - perturbations
        f_forward = np.apply_along_axis(self.function, 1, v_forward)
        f_backward = np.apply_along_axis(self.function, 1, v_backward)
        return (f_forward - f_backward) / (2 * h)

    def second_derivative_approx(self,x,h=1e-5):
        return (self.function(x+h)-2*self.function(x)+self.function(x-h))/(h**2)

    def derivative_approx(self,x,h=1e-5):

        '''
        Central-difference approximation
        '''
        return (self.function(x + h) - self.function(x - h)) / (2 * h)

