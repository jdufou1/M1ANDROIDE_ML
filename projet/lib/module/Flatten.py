from lib.module.Module import Module

class Flatten(Module):

    def __init__(self):
        """Initialisation des parametres"""
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        """Annule gradient"""
        pass

    def forward(self, X):
        """Calcule la passe forward"""
        self._forward = X.reshape(X.shape[0], -1)
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        """Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step"""
        pass

    def backward_update_gradient(self, input, delta):
        """Met a jour la valeur du gradient"""
        pass

    def backward_delta(self, input, delta):
        """Calcul la derivee de l'erreur"""
        self._delta = delta.reshape(input.shape)
        return self._delta