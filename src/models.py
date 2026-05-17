import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MLP:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1  
        self.W = []
        self.b = []

        # inicializaciones para ADAM
        self.m_W = [np.zeros_like(w) for w in self.W]
        self.v_W = [np.zeros_like(w) for w in self.W]
        self.m_b = [np.zeros_like(b) for b in self.b]
        self.v_b = [np.zeros_like(b) for b in self.b]
                
        for l in range(self.L):
            W_l = np.random.randn(layer_sizes[l], layer_sizes[l+1]) * np.sqrt(2 / layer_sizes[l]) #Cada fila es una muestra con las probabilidades de que pertenezca a cada clase
            b_l = np.zeros((1, layer_sizes[l+1]))
            self.W.append(W_l)
            self.b.append(b_l)
        
        self.m_W = [np.zeros_like(w) for w in self.W]
        self.v_W = [np.zeros_like(w) for w in self.W]
        self.m_b = [np.zeros_like(b) for b in self.b]
        self.v_b = [np.zeros_like(b) for b in self.b]
                

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
        
    
    def backward(self, y_true, lay_vals, l2_lambda=0.0):
    
        n = y_true.shape[0]
        grads_W = []
        grads_b = []
        
        # gradiente de la capa de salida
        A_out = lay_vals['A'][-1] 
        dZ = A_out.copy()
        dZ[np.arange(n), y_true] -= 1  # derivada combinada softmax y cross-entropy
        dZ /= n
        
        # gradientes de W y b de la capa de salida
        gW_out = lay_vals['A'][-2].T @ dZ
        if l2_lambda > 0:
            gW_out += l2_lambda * self.W[-1]
        grads_W.append(gW_out)
        grads_b.append(np.sum(dZ, axis=0, keepdims=True)) 
        
        # propagar para atras por las capas ocultas
        for l in range(self.L - 2, -1, -1):
            dA = dZ @ self.W[l+1].T 
            dZ = dA * self.relu_derivative(lay_vals['Z'][l]) 
            grads_W_l = lay_vals['A'][l].T @ dZ
            
            if l2_lambda > 0:
                grads_W_l += l2_lambda * self.W[l]
            grads_W.append(grads_W_l)
        
            grads_b.append(np.sum(dZ, axis=0, keepdims=True))
        
        # invertir para que el indice 0 corresponda a la capa 0
        grads_W.reverse()
        grads_b.reverse()
        
        return grads_W, grads_b
    
    def update_params(self, grads_W, grads_b, learning_rate):
        for l in range(self.L):
            self.W[l] -= learning_rate * grads_W[l]
            self.b[l] -= learning_rate * grads_b[l]
    
    def fit(self, X_train, y_train, X_val, y_val, epochs, lr,
        batch_size=None, optimizer='sgd', lr_schedule=None,
        l2_lambda=0.0, early_stopping=False, patience=10,
        lr_min=1e-5, decay_rate=0.999):
    
        train_costs = []
        val_costs = []
        best_val_cost = np.inf
        epochs_no_improve = 0
        best_W = [w.copy() for w in self.W]
        best_b = [b.copy() for b in self.b]
        t = 1  # paso para adam
        lr_init = lr
        
        #VER QUE NO SE PISEN LOS LR

        for epoch in range(epochs):
            
            # learning rate scheduling
            current_lr = self.get_lr(epoch, lr_init, epochs, lr_schedule, lr_min, decay_rate)
            
            # mini-batch o batch completo
            if batch_size is not None:
                indices = np.random.permutation(len(X_train))
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]
                
                for start in range(0, len(X_train), batch_size):
                    X_batch = X_shuffled[start:start + batch_size]
                    y_batch = y_shuffled[start:start + batch_size]
                    
                    A_batch, cache_batch = self.forward(X_batch)
                    grads_W, grads_b = self.backward(y_batch, cache_batch, l2_lambda)
                    
                    if optimizer == 'adam':
                        self.update_params_adam(grads_W, grads_b, current_lr, t)
                        t += 1
                    else:
                        self.update_params(grads_W, grads_b, current_lr)
            else:
                # batch completo
                A_out, cache = self.forward(X_train)
                grads_W, grads_b = self.backward(y_train, cache, l2_lambda)
                
                if optimizer == 'adam':
                    self.update_params_adam(grads_W, grads_b, current_lr, t)
                    t += 1
                else:
                    self.update_params(grads_W, grads_b, current_lr)
            
            # costo train y val
            A_out, _ = self.forward(X_train)
            train_cost = self.cross_entropy(A_out, y_train)
            train_costs.append(train_cost)
            
            A_val, _ = self.forward(X_val)
            val_cost = self.cross_entropy(A_val, y_val)
            val_costs.append(val_cost)
            
            
            # early stopping
            if early_stopping:
                if val_cost < best_val_cost:
                    best_val_cost = val_cost
                    epochs_no_improve = 0
                    best_W = [w.copy() for w in self.W]
                    best_b = [b.copy() for b in self.b]
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        self.W = best_W  # restaurar los pesos
                        self.b = best_b
                        break
        
        return train_costs, val_costs


    def predict(self, X):
        A_out, _ = self.forward(X)
        return np.argmax(A_out, axis=1)  # devuelve el indice de la clase con mayor probabilidad
    
    def update_params_adam(self, grads_W, grads_b, lr, t, b1=0.9, b2=0.999, eps=1e-8):
        for l in range(self.L):
            self.m_W[l] = b1 * self.m_W[l] + (1 - b1) * grads_W[l]
            self.v_W[l] = b2 * self.v_W[l] + (1 - b2) * grads_W[l]**2

            m_W_corr = self.m_W[l] / (1 - b1**t)
            v_W_corr = self.v_W[l] / (1 - b2**t)
            self.W[l] -= lr * m_W_corr / (np.sqrt(v_W_corr) + eps)

            # actualizar momentos de b
            self.m_b[l] = b1 * self.m_b[l] + (1 - b1) * grads_b[l]
            self.v_b[l] = b2 * self.v_b[l] + (1 - b2) * grads_b[l]**2
            m_b_corr = self.m_b[l] / (1 - b1**t)
            v_b_corr = self.v_b[l] / (1 - b2**t)
            self.b[l] -= lr * m_b_corr / (np.sqrt(v_b_corr) + eps)


    def get_lr(self, epoch, lr_0, epochs, schedule=None, lr_min=1e-5, decay_rate=0.999):
        if schedule == 'linear':
            lr = lr_0 - (lr_0 - lr_min) * epoch / epochs
            return max(lr_min, lr)
        elif schedule == 'exponential':
            return max(lr_min, lr_0 * (decay_rate ** epoch))
        else:
            return lr_0 
        
class MLP_PyTorch(nn.Module):
    def __init__(self, layer_sizes):

        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:   # ReLU en capas ocultas
                layers.append(nn.ReLU())
            # cross entropy de pytorch incluye softmax
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
def train_pytorch_model(model, train_loader, val_loader, device,
                        lr, l2_lambda, epochs, patience=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_weights = None

    def evaluate(loader):
        model.eval()
        total_loss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b)
                total_loss += criterion(logits, y_b).item() * len(y_b)
                correct    += (logits.argmax(1) == y_b).sum().item()
                n          += len(y_b)
        return total_loss / n, correct / n

    for epoch in range(1, epochs + 1):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            optimizer.step()

        train_loss, train_acc = evaluate(train_loader)
        val_loss,   val_acc   = evaluate(val_loader)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Early stopping 
        if patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping en época {epoch}")
                    break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history
    

class MLP_M3(nn.Module):
    def __init__(self, layer_sizes, activation="relu", dropout=0.0):
        super().__init__()
        
        activaciones = {
            "relu":      nn.ReLU(),
            "leakyrelu": nn.LeakyReLU(0.1),
            "silu":      nn.SiLU(),
            "gelu":      nn.GELU(),
        }
        
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            if i < len(layer_sizes) - 2:  # capas ocultas
                layers.append(activaciones[activation])
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
