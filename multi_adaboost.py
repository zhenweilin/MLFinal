import torch
import numpy as np
from utils import check_accuracy
from DataProcess import Datasets
from run import selectModel
batch_size = 128

if torch.cuda.is_available():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainloader, testloader, num_classes = Datasets(batch_size)


class multi_adaboost():
    def __init__(self,K,M):
        '''
        K: category number
        M: training circle number, epoch
        n: the number of model
        '''
        self.K = K
        self.M = M
        pass

    def caculate_e(self,err,w):
        e = w*err
        return e

    def alpha_update(self,e):
        '''
        e: model's error
        '''

        alpha = np.log((1-e)/(e+1e-5))+np.log(self.K-1)
        return alpha

    def update_w(self,w,alpha,err):
        # print('update once')
        w = w*np.exp(alpha*err)
        return w

    def SAMME(self,modelset):
        self.modelset = modelset
        weight = [1/len(modelset)]*len(modelset)
        for m in range(self.M):#epoch
            for i,model in enumerate(modelset):
                accuracy = check_accuracy(model,trainloader)
                err = 1 - accuracy
                if err > (1-1/self.K):
                    raise ValueError('This model is too stupid')
                e = self.caculate_e(err,weight[i])
                alpha = self.alpha_update(e)
                weight[i] = self.update_w(weight[i],alpha,e)
            sumvalue = sum(weight)
            weight = [i/sumvalue for i in weight]
            self.weight = weight
            print(self.weight)
        return self.weight
    
    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for (X,y) in testloader:
                # pred = torch.zeros(self.K,len(testloader))
                for i,model in enumerate(self.modelset):
                    device = next(model.parameters()).device
                    X = X.to(device)
                    y_pred = model(X)
                    # print(y_pred.shape)
                    if i == 0:
                        pred = torch.zeros_like(y_pred)
                    pred.add_(y_pred,alpha = self.weight[i])
                    # _,predicted = torch.max(y_pred.data,1)
                    # print(predicted.shape)
                _,predicted = torch.max(pred,1)
                y = y.to(device)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy = correct / total
        print('accuracy is :{}'.format(accuracy))
        return accuracy
    


if __name__ == "__main__":
    checkpoint1 = torch.load('./checkpoints/SGD_l1_mobilenetv1_chestX_10_edi_2.0_theta_10_lam_2.0e-04.pt',map_location='cpu')
    checkpoint2 = torch.load('./checkpoints/SGD_mcp_mobilenetv1_chestX_10_edi_2.0_theta_10_lam_4.0e-04.pt',map_location='cpu')
    checkpoint3 = torch.load('./checkpoints/SGD_l1_resnet50_chestX_10_edi_2.0_theta_10_lam_4.0e-04.pt',map_location='cpu')
    # checkpoint4 = torch.load('./checkpoints/SGD_l1_resnet18_chestX_10_edi_2.0_theta_10_lam_1.0e-04.pt',map_location='cpu')
    model1 = selectModel('mobilenetv1')
    model2 = selectModel('mobilenetv1')
    model3 = selectModel('resnet50')
    # model4 = selectModel('resnet18')
    model1.load_state_dict(checkpoint1['model_state_dict'])
    model2.load_state_dict(checkpoint2['model_state_dict'])
    model3.load_state_dict(checkpoint3['model_state_dict'])
    # model4.load_state_dict(checkpoint4['model_state_dict'])
    modelset = [model1,model2,model3]
    adaboost = multi_adaboost(3,5)
    adaboost.SAMME(modelset)
    adaboost.test()


    