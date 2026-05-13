import numpy as np

class MLP:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1  
        self.W = []
        self.b = []
        
        for l in range(self.L):
            W_l = np.random.randn(layer_sizes[l], layer_sizes[l+1]) * np.sqrt(2 / layer_sizes[l]) #Cada fila es una muestra con las probabilidades de que pertenezca a cada clase
            b_l = np.zeros((1, layer_sizes[l+1]))
            self.W.append(W_l)
            self.b.append(b_l)
    

    def relu(self, z):
        return np.maximum(0, z)
        
    
    def relu_derivative(self, z):
        return (z>0).astype(float)

    
    def softmax(self, z):
        exp_Z = np.exp(z) # ver si hay que estabilizar el vector de Z

        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    
    def forward(self, X): #para la red X tiene que estar flat
        X = X.reshape(X.shape[0], -1)
        vals = {'A': [X], 'Z': []}
        A = X
        #capas ocultas
        for l in range(self.L - 1):
            Z = A @ self.W[l] + self.b[l]
            A = self.relu(Z)
            vals['Z'].append(Z)
            vals['A'].append(A)
        
        #capa de salida con softmax
        Z = A @ self.W[-1] + self.b[-1]
        A = self.softmax(Z)
        vals['Z'].append(Z)
        vals['A'].append(A)
        
        return A, vals
    
    def cross_entropy(self, y_pred, y_true):
        n = y_true.shape[0]

        log_probs = -np.log(y_pred[np.arange(n), y_true].astype(float))
        return np.mean(log_probs)
        
    
    def backward(self, y_true, lay_vals):
    
        n = y_true.shape[0]
        grads_W = []
        grads_b = []
        
        # gradiente de la capa de salida
        A_out = lay_vals['A'][-1] 
        dZ = A_out.copy()
        dZ[np.arange(n), y_true] -= 1  # derivada combinada softmax y cross-entropy
        dZ /= n
        
        # gradientes de W y b de la capa de salida
        grads_W.append(lay_vals['A'][-2].T @ dZ)   
        grads_b.append(np.sum(dZ, axis=0, keepdims=True)) 
        
        # propagar para atras por las capas ocultas
        for l in range(self.L - 2, -1, -1):
            dA = dZ @ self.W[l+1].T 
            dZ = dA * self.relu_derivative(lay_vals['Z'][l]) 
            
            grads_W.append(lay_vals['A'][l].T @ dZ)
            grads_b.append(np.sum(dZ, axis=0, keepdims=True))
        
        # invertir para que el indice 0 corresponda a la capa 0
        grads_W.reverse()
        grads_b.reverse()
        
        return grads_W, grads_b
    
    def update_params(self, grads_W, grads_b, learning_rate):
        for l in range(self.L):
            self.W[l] -= learning_rate * grads_W[l]
            self.b[l] -= learning_rate * grads_b[l]
    
    def fit(self, X_train, y_train, X_val, y_val, epochs, learning_rate):
        train_costs = []
        val_costs = []
        
        for epoch in range(epochs):
            # forward
            A_out, lay_vals = self.forward(X_train)
            
            # costo train
            train_cost = self.cross_entropy(A_out, y_train)
            train_costs.append(train_cost)
            
            # backward y update
            grads_W, grads_b = self.backward(y_train, lay_vals)
            self.update_params(grads_W, grads_b, learning_rate)
            
            # costo validacion (solo forward, sin actualizar)
            A_val, _ = self.forward(X_val)
            val_cost = self.cross_entropy(A_val, y_val)
            val_costs.append(val_cost)
            
        return train_costs, val_costs


    def predict(self, X):
        A_out, _ = self.forward(X)
        return np.argmax(A_out, axis=1)  # devuelve el indice de la clase con mayor probabilidad