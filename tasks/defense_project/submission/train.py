"""
The template for the students to train the model.
Please do not change the name of the functions in Adv_Training.
"""
import sys
sys.path.append("../../../")
import torch
import random
import numpy as np
from predict import LeNet
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from utils import get_dataset
import importlib.util

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class VirtualModel:
    def __init__(self, device, model) -> None:
        self.device = device
        self.model = model

    def get_batch_output(self, images):
        predictions = []
        # for image in images:
        predictions = self.model(images).to(self.device)
            # predictions.append(prediction)
        # predictions = torch.tensor(predictions)
        return predictions

    def get_batch_input_gradient(self, original_images, labels):
        original_images.requires_grad = True
        self.model.eval()
        outputs = self.model(original_images)
        loss = F.nll_loss(outputs, labels)
        self.model.zero_grad()
        loss.backward()
        data_grad = original_images.grad.data
        return data_grad

class Adv_Training():
    """
    The class is used to set the defense related to adversarial training and adjust the loss function. Please design your own training methods and add some adversarial examples for training.
    The perturb function is used to generate the adversarial examples for training.
    """
    def __init__(self, device, file_path, target_label=None, epsilon=0.3, min_val=0, max_val=1):
        sys.path.append(file_path)
        # from predict import LeNet
        self.model = LeNet().to(device)
        self.epsilon = epsilon
        self.device = device
        self.min_val = min_val
        self.max_val = max_val
        self.target_label = target_label
        self.perturb = self.load_perturb("../attacker_list/nontarget_FGSM")
        self.perturb_by_target_FGSM = self.load_perturb("../attacker_list/target_FGSM")
        self.perturb_by_target_PGD  = self.load_perturb("../attacker_list/target_PGD")

    def load_perturb(self, attack_path):
        spec = importlib.util.spec_from_file_location('attack', attack_path + '/attack.py')
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        # for attack methods evaluator, the Attack class name should be fixed
        attacker = foo.Attack(VirtualModel(self.device, self.model), self.device, attack_path)
        return attacker


    def train(self, trainset, valset, device, epoches=30):
        self.model.to(device)
        self.model.train()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=10)
        dataset_size = len(trainset)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters())
        for epoch in range(epoches):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                # get the attacked inputs total 3:
                # adv_inputs1, _ = self.perturb.attack(inputs, labels.detach().cpu().tolist())
                # adv_inputs1 = torch.tensor(adv_inputs1).to(device)
                target_label = 0
                # adv_inputs2, _ = self.perturb_by_target_FGSM.attack(inputs, labels.detach().cpu().tolist(), target_label)
                # adv_inputs2 = torch.tensor(adv_inputs2).to(device)
                adv_inputs3, _ = self.perturb_by_target_PGD.attack(inputs, labels.detach().cpu().tolist(), target_label)
                adv_inputs3 = torch.tensor(adv_inputs3).to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # use three different attackers to get adv_inputs
                # optimizer.zero_grad()
                # adv_outputs = self.model(adv_inputs1)
                # loss = criterion(adv_outputs, labels)*0.8
                # loss.backward()
                # optimizer.step()
                # running_loss += loss.item()
                
                # optimizer.zero_grad()
                # adv_outputs = self.model(adv_inputs2)
                # loss = criterion(adv_outputs, labels)*0.8
                # loss.backward()
                # optimizer.step()
                # running_loss += loss.item()

                optimizer.zero_grad()
                adv_outputs = self.model(adv_inputs3)
                loss = criterion(adv_outputs, labels)*0.6
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                
                # pred = torch.max(outputs, 1)
                # correct += pred.eq(labels.view_as(pred)).sum().item()
                
                
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / dataset_size))
            running_loss = 0.0
        valloader = torch.utils.data.DataLoader(valset, batch_size=100, shuffle=True, num_workers=10)
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print("Accuracy of the network on the val images: %.3f %%" % (100 * correct / total))
        return


def main():
    ############################################################
    # teacher model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    adv_training = Adv_Training(device, file_path='.')
    dataset_configs = {
                "name": "CIFAR10",
                "binary": True,
                "dataset_path": "../datasets/CIFAR10/student/",
                "student_train_number": 10000,
                "student_val_number": 1000,
                "student_test_number": 100,
    }

    dataset = get_dataset(dataset_configs)
    trainset = dataset['train']
    valset = dataset['val']
    # use this testset to generate soft pred labels for the teacher model
    testset = dataset['test']
    adv_training.train(trainset, valset, device)
    torch.save(adv_training.model.state_dict(), "defense_project-model.pth")
    ############################################################
    # student model
    
    # teacher_model = LeNet().to(device)
    # teacher_model.load_state_dict(torch.load("defense_homework-model.pth", map_location = 'cpu'))
    # teacher_model.eval()
    # for param in teacher_model.parameters():
    #     print(param)
    


if __name__ == "__main__":
    main()
