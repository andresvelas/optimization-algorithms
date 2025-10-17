import matplotlib.pyplot as plt
import numpy as np

class UnconstrainedUnidimensionalOptimization:
    def __init__(self, function,interval,init_value, tol, derivative=None,second_derivative=None):
        self.function = function
        self.interval = interval
        self.init_value = 0
        self.I_0 = interval[1]-interval[0]
        self.tol = tol
        self.derivative = derivative
        self.derivative_second = second_derivative
    def fibonacci(self):
        '''
        Fibonacci method for unidimensional optimization

        We want to guarantee the best result in the worst case, so we require:
                    I_(n-1)= 2 * I_n
        and considering  the ecuation:
                    I_k = I_(k+1) + I_(k+2)
        we can deduce that:
            - I_(n-1) = 2*I_n
            - I_(n-2) = I_(n-1) + I_n = 3*I_n
            - I_(n-3) = I_(n-2) + I_(n-1) = 5*I_n
            ...
            - I_k = F_(n-k) * I_n
            ...
            - I_0 = F_n * I_n
        
        I_0 / F_n < tol => I_n < tol * F_n

        We can find the minimum n that satisfies this condition, 
        and if we find n, we can find all the I_k, k in [0,n]

        Problems:
        In the last iteration, we will have x_a == x_b, so we won't be able to define an interval.
        We define delta = tol - I_n, and adjust x_a and x_b by subtracting and adding delta/2, respectively,
        to ensure the final interval is strictly less than the tolerance.
        '''
        print("Método de Fibonacci")

        # We generate a list of Fibonacci numbers
        fibonacci = [1,1]
        [fibonacci.append(fibonacci[i-1] + fibonacci[i-2]) for i in range(2,100)]
        fibonacci = fibonacci[1:]

        n=0
        l = self.interval[0]
        u = self.interval[1]

        k=0
        while self.I_0/fibonacci[k] >= self.tol:
            k += 1
        data = {}
        while self.I_0/fibonacci[n] >= self.tol:
            n += 1
            I_n = self.I_0 * (fibonacci[k-n]/fibonacci[k])
 
            x_a = u - I_n
            x_b = l + I_n
            if n ==1:
                data[0]= {
                    'u':u,
                    'x_a':x_a,
                    'x_b':x_b,
                    'l':l   
                }
                print(f'Iteración {0}: I_n = {self.I_0}, l = {l}, u = {u}')
            if not(x_a==x_b):
                u,l = self.condition(x_a,x_b,u,l)


            print(f'Iteración {n}: I_n = {I_n}, l = {l}, u = {u}')
            data[n]= {
                'u':u,
                'x_a':x_a,
                'x_b':x_b,
                'l':l
            }
            if self.I_0/fibonacci[n] < self.tol:
                delta = self.tol - self.I_0/fibonacci[n]
                x_a = x_a - (delta - delta/2)
                x_b = x_b + (delta - delta/2)
                u,l = self.condition(x_a,x_b,u,l)

                data['solution']= (l,u)
                print(100*'-')
                print('\n')
                print(f"Solución encontrada en el intervalo: [{l}, {u}]")
                print('\n')
        return data, (l,u) , k
    def seccion_aurea(self):
        '''
        Golden section method for unidimensional optimization

        We want the ratio between the intervals to be the same in each iteration, that is:
                I_k / I_(k+1) = I_(k+1) / I_(k+2) = constant
        From this relationship we can deduce that:
                constant = (I_(k+2) + I_(k+3)) / I_(k+2) = 1 + 1/constant
                constant^2 - constant - 1 = 0
        Solving the equation we get:
                constant = (1 + sqrt(5)) / 2 = phi (golden ratio)
        Therefore, in each iteration it holds that:
                I_k = phi * I_(k+1)
        
        Knowing the ratio, we can define the points x_a, x_b in [l, u] in each iteration as:
                x_a = u - (u - l) / phi
                x_b = l + (u - l) / phi
        '''
        print("Método de la sección áurea")
        iteraciones = (np.log(self.interval[1]-self.interval[0]) - np.log(self.tol)) / np.log((1 + np.sqrt(5)) / 2)
        iteraciones = np.ceil(iteraciones)
        phi = (1 + np.sqrt(5)) / 2  
        u = self.interval[1]
        l = self.interval[0]
        x_a =  u - (u - l) / phi
        x_b = l + (u - l) / phi
        n= 0
        data = {}
        data[n]= {
            'u':u,
            'x_a':x_a,
            'x_b':x_b,
            'l':l   
            }
        print(f"Iteración {n}:intervalo = {u - l}, l = {l}, u = {u}")
        while abs(u - l) > self.tol:
            n= n + 1
            u,l = self.condition(x_a,x_b,u,l)
            x_a =  u - (u - l) / phi
            x_b = l + (u - l) / phi
            print(f"Iteración {n}:intervalo = {u - l}, l = {l}, u = {u}")
            data[n]= {
                'u':u,
                'x_a':x_a,
                'x_b':x_b,
                'l':l
            }
            if abs(u - l) < self.tol:
                data['solution']= (l,u)
                print(100*'-')
                print('\n')
                print(f"Solución encontrada en el intervalo: [{l}, {u}]")
                print('\n')


        return data, (float(l), float(u)), iteraciones
    def biseccion(self,aprox_derivative = False):
        '''
        Bisection method for unidimensional optimization

        We need the derivative of the function to apply this method.
        We will find the midpoint c = (u + l) / 2 and evaluate f'(c).
        - if f'(c) ~= 0 then c is a stationary point (we return it)
        - if f'(c) > 0 then the function is increasing at c, so the minimizer is
          to the left: set u = c
        - if f'(c) < 0 then the function is decreasing at c, so the minimizer is
          to the right: set l = c

        '''
        print("Método de Bisección")
        if aprox_derivative:
            derivative = self.derivative_approx
        elif self.derivative is not None:
            derivative = self.derivative
        else:
            raise ValueError("No se proporcionó una función derivada ni se solicitó aproximación.")
        iteraciones =  (np.log2(self.I_0/self.tol))
        iteraciones = np.ceil(iteraciones)
        u = self.interval[1]
        l = self.interval[0]
        n=0
        data = {}
        while abs(u - l) > self.tol:
            n= n + 1
            c = (u + l) / 2
            if n==1:
                data[0]= {
                    'u':u,
                    'c':c,
                    'l':l   
                }
                print(f"Iteración {0}:intervalo = {u - l}, l = {l}, u = {u}")
            if derivative(c) == 0:
                return data, (c,c), n 
                break
            elif derivative(c) > 0:
                u = c
            else:
                l = c

            print(f"Iteración {n}:intervalo = {u - l}, l = {l}, u = {u}")
            data[n]= {
                'u':u,
                'c':(u + l) / 2,
                'l':l
            }
            if abs(u - l) < self.tol:
                data['solution']= (l,u)
                print(100*'-')
                print('\n')
                print(f"Solución encontrada en el intervalo: [{l}, {u}]")
                print('\n')
        return data, (l,u) , iteraciones       
    def derivative_approx(self,f,x,h=1e-5):

        '''
        Central-difference approximation
        '''
        return (f(x + h) - f(x - h)) / (2 * h)
    def condition(self,x_a,x_b,u,l):
        '''
        Condition to update the interval for Fibonacci and Aurea methods.

        l < x_a < x_b < u
        If f(x_a) < f(x_b) => u = x_b
        else l = x_a

        Problem:
        We are not considering the case f(x_a) == f(x_b) because it is very unlikely to happen
        But if it happens, we can choose to update either u or l, because by the Rolle's theorem,
        there is at least one point in (x_a,x_b) where the derivative is 0, so we can reduce the interval anyway.
        '''
        if self.function(x_a) < self.function(x_b):
            u = x_b
        else:
            l= x_a
        return (u,l)

    def stop_by_grad(self,grad, x_new, x,  tol):
        return np.linalg.norm(grad) < tol

    def stop_by_x_diff(self,grad, x_new, x,  tol):
        return np.linalg.norm(x_new - x) < tol

    def stop_by_f_diff(self,grad, x_new, x,  tol):
        return abs(self.function(x_new) - self.function(x)) < tol

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

        x = self.ini
        while stopping_condition(derivative(x))
    



        
    


    def plot_iterations(self, data):
        intervalos = []
        etiquetas = []
        puntos_por_iter = []
        iter_nums = sorted([k for k in data.keys() if isinstance(k, int)])

        for k in iter_nums:
            iter_data = data[k]
            l = iter_data['l']
            u = iter_data['u']
            intervalos.append((l, u))
            etiquetas.append(f"Iter {k}")
            puntos = {
                key: val for key, val in iter_data.items()
                if key not in ['l', 'u']
            }
            puntos_por_iter.append(puntos)
        l_final, u_final = data['solution']
        l_vals = [l for l, _ in intervalos]
        anchos = [u - l for l, u in intervalos]

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, (l, ancho, puntos) in enumerate(zip(l_vals, anchos, puntos_por_iter)):
            ax.broken_barh([(l, ancho)], (i - 0.4, 0.8), facecolors='skyblue', edgecolor='black')
            for j, (nombre, valor) in enumerate(puntos.items()):
                color = f'C{j}' 
                ax.plot(valor, i, 'o', color=color, label=nombre if i == 0 else "")

        ax.broken_barh([(l_final, u_final - l_final)], (len(etiquetas) - 0.4, 0.8),
                    facecolors='lightgreen', edgecolor='black', label='Intervalo Final')

        etiquetas.append('Final')

        ax.set_yticks(range(len(etiquetas)))
        ax.set_yticklabels(etiquetas)
        ax.set_xlabel("x")
        ax.set_title("Evolución del Intervalo por Iteración")
        ax.grid(True)
        

        
        unique_keys = set()
        for puntos in puntos_por_iter:
            unique_keys.update(puntos.keys())
        handles = [
            plt.Line2D([], [], color='skyblue', lw=10, label='Intervalos'),
            plt.Line2D([], [], color='lightgreen', lw=10, label='Intervalo Final')
        ]
        for j, key in enumerate(sorted(unique_keys)):
            color = f'C{j}'
            handles.append(plt.Line2D([], [], marker='o', color=color, linestyle='None', label=key))

        ax.legend(handles=handles, loc='upper right')
        plt.tight_layout()
        plt.show()
    



