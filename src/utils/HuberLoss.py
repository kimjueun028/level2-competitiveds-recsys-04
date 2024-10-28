import numpy as np

class AsymmetricHuberLoss:
    def __init__(self, delta=1.0, beta=1.05):   # 작게 예측하는 값에 대해 loss를 키운다.(default : 5%)
        self.delta = delta
        self.beta = beta

    def _calculate_loss(self, y_true, y_pred):

        # 식 구현
        error = y_true - y_pred
        abs_error = np.abs(error) / y_true
        quadratic = np.minimum(abs_error, self.delta)   
        linear = abs_error - quadratic 
        loss = 0.5 * quadratic**2 + self.delta * linear

        # loss penalty 추가
        underestimation_mask = y_pred < y_true * 0.95 
        loss[underestimation_mask] *= self.beta
        return loss   # eval_metric에 사용

    def gradient(self, y_true, y_pred):   # 1차 미분값
        error = y_pred - y_true
        abs_error = np.abs(error) / y_true
        grad = np.where(abs_error <= self.delta, error, self.delta * np.sign(error))  
        grad[y_pred < y_true * 0.95] *= self.beta   
        return grad

    def hessian(self, y_true, y_pred):   # 2차 미분값
        abs_error = np.abs(y_pred - y_true) / y_true
        hess = np.where(abs_error <= self.delta, 1.0, 0.0)
        hess[y_pred < y_true * 0.95] *= self.beta   
        return hess
    
def custom_loss(y_true, y_pred):
    loss = AsymmetricHuberLoss()
    grad = loss.gradient(y_true, y_pred)
    hess = loss.hessian(y_true, y_pred)
    return grad, hess

def custom_metric(y_true, y_pred):
    loss = AsymmetricHuberLoss()
    return np.mean(loss._calculate_loss(y_true, y_pred)) * 1000