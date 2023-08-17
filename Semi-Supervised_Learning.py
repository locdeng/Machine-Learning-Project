import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler




import os
from PIL import Image
import argparse

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = [d for d in os.listdir(root) if not d.startswith('.')]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.imgs = []
        for c in self.classes:
            class_dir = os.path.join(root, c)
            for filename in os.listdir(class_dir):
                path = os.path.join(class_dir, filename)
                if os.path.isfile(path):  # Check if the item is a file
                    self.imgs.append((path, self.class_to_idx[c]))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target

"""
class CustomDataset_Nolabel(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        ImageList = os.listdir(root)
        self.imgs = []
        for filename in ImageList:
            path = os.path.join(root, filename)
            self.imgs.append(path)
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
"""

class Custom_model(nn.Module):
    def __init__(self):
        super(Custom_model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32768, 50)  # Adjust the input size based on the output shape of the last layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x
    
def model_selection(selection):
    if selection == "resnet":
        model = models.resnet18()
        model.conv1 =  nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False)
        model.layer4 = Identity()
        model.fc = nn.Linear(256, 50)
    elif selection == "vgg":
        model = models.vgg11_bn()
        model.features = nn.Sequential(*list(model.features.children())[:-7])
        model.classifier = nn.Sequential( nn.Linear(in_features=25088, out_features=50, bias=True))
    elif selection == "mobilenet":
        model = models.mobilenet_v2()
        model.classifier = nn.Sequential(nn.Linear(in_features=1280, out_features=50, bias=True))
   """
    elif  selection =='custom':
        model = Custom_model()
   """ 
    else:
        raise ValueError("Invalid model selection")
    return model

def cotrain(net1, net2, labeled_loader, unlabeled_loader, optimizer1_1, optimizer1_2, optimizer2_1, optimizer2_2, criterion, scaler):
    net1.train()
    net2.train()
    train_loss = 0
    correct = 0
    total = 0
    k = 0.8
    
    # labeled_training
    for batch_idx, (inputs, targets) in enumerate(labeled_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer1_1.zero_grad()
        optimizer2_1.zero_grad()

        with autocast():
            outputs1 = net1(inputs)
            loss1 = criterion(outputs1, targets)
        
        scaler.scale(loss1).backward()
        scaler.step(optimizer1_1)
        scaler.update()

        with autocast():
            outputs2 = net2(inputs)
            loss2 = criterion(outputs2, targets)
        
        scaler.scale(loss2).backward()
        scaler.step(optimizer2_1)
        scaler.update()

        train_loss += loss1.item() + loss2.item()
        _, predicted1 = outputs1.max(1)
        _, predicted2 = outputs2.max(1)
        total += targets.size(0)
        correct += (predicted1 == targets).sum().item() + (predicted2 == targets).sum().item()

    # unlabeled_training
    for batch_idx, inputs in enumerate(unlabeled_loader):
        inputs = inputs.cuda()

        with autocast():
            outputs1 = net1(inputs)
            _, predicted1 = outputs1.max(1)

        with autocast():
            outputs2 = net2(inputs)
            _, predicted2 = outputs2.max(1)

        agree = predicted1 == predicted2

        pseudo_labels = predicted1.clone()
        pseudo_labels[agree] = predicted1[agree]

        total += pseudo_labels.size(0)
        correct += (pseudo_labels == predicted1).sum().item()

        optimizer1_2.zero_grad()
        with autocast():
            outputs1_pseudo = net1(inputs)
            loss1_pseudo = criterion(outputs1_pseudo, pseudo_labels)
        scaler.scale(loss1_pseudo).backward()
        scaler.step(optimizer1_2)
        scaler.update()

        optimizer2_2.zero_grad()
        with autocast():
            outputs2_pseudo = net2(inputs)
            loss2_pseudo = criterion(outputs2_pseudo, pseudo_labels)
        scaler.scale(loss2_pseudo).backward()
        scaler.step(optimizer2_2)
        scaler.update()

        train_loss += loss1_pseudo.item() + loss2_pseudo.item()

    return train_loss, correct, total

def test(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return 100. * correct / total
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test',  type=str,  default='False')
    parser.add_argument('--student_abs_path',  type=str,  default='./')
    #args = parser.parse_args()
    args = parser.parse_known_args()[0]


    batch_size = 512   #Input the number of batch size
    if args.test == 'False':
        train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(64, scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

        dataset = CustomDataset(root = './data/Semi-Supervised_Learning/labeled', transform = train_transform)
        labeled_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        dataset = CustomDataset_Nolabel(root = './data/Semi-Supervised_Learning/unlabeled', transform = train_transform)
        unlabeled_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        dataset = CustomDataset(root = './data/Semi-Supervised_Learning/val', transform = test_transform)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    else :
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    if not os.path.exists(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning')):
        os.makedirs(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning'))



    model_sel_1 = "vgg" #write your choice of model (e.g., 'vgg')
    model_sel_2 = "mobilenet" #write your choice of model (e.g., 'resnet)


    model1 = model_selection(model_sel_1)
    model2 = model_selection(model_sel_2)

    params_1 = sum(p.numel() for p in model1.parameters() if p.requires_grad) / 1e6
    params_2 = sum(p.numel() for p in model2.parameters() if p.requires_grad) / 1e6

    if torch.cuda.is_available():
        model1 = model1.cuda()
    if torch.cuda.is_available():
        model2 = model2.cuda()

    #You may want to write a loader code that loads the model state to continue the learning process
    #Since this learning process may take a while.


    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss().cuda()
    else :
        criterion = nn.CrossEntropyLoss()


    optimizer1_1 = torch.optim.SGD(model1.parameters(), lr=0.01, momentum=0.9) #Optimizer for model 1 in labeled training
    optimizer2_1 = torch.optim.Adam(model2.parameters(), lr=0.001)

    optimizer1_2 = torch.optim.SGD(model1.parameters(), lr=0.01, momentum=0.9)
    optimizer2_2 = torch.optim.Adam(model2.parameters(), lr=0.001)

    epoch = 50 #Input the number of epochs
    

    if args.test == 'False':
        assert params_1 < 7.0, "Exceed the limit on the number of model_1 parameters"
        assert params_2 < 7.0, "Exceed the limit on the number of model_2 parameters"

        best_result_1 = 0
        best_result_2 = 0
        scaler = GradScaler()

        for e in range(0, epoch):
            train_loss, correct, total = cotrain(model1, model2, labeled_loader, unlabeled_loader, optimizer1_1, optimizer1_2,
                                     optimizer2_1, optimizer2_2, criterion, scaler)
            #cotrain(model1, model2, labeled_loader, unlabeled_loader, optimizer1_1, optimizer1_2, optimizer2_1, optimizer2_2, criterion)
            tmp_res_1 = test(model1, val_loader)
            # You can change the saving strategy, but you can't change file name/path for each model
            print ("[{}th epoch, model_1] ACC : {}".format(e, tmp_res_1))
            if best_result_1 < tmp_res_1:
                best_result_1 = tmp_res_1
                torch.save(model1.state_dict(),  os.path.join('./logs', 'Semi-Supervised_Learning', 'best_model_1.pt'))

            tmp_res_2 = test(model2, val_loader)
            # You can change save strategy, but you can't change file name/path for each model
            print ("[{}th epoch, model_2] ACC : {}".format(e, tmp_res_2))
            if best_result_2 < tmp_res_2:
                best_result_2 = tmp_res_2
                torch.save(model2.state_dict(),  os.path.join('./logs', 'Semi-Supervised_Learning', 'best_model_2.pt'))
        print('Final performance {} - {}  // {} - {}', best_result_1, params_1, best_result_2, params_2)


    else:
        dataset = CustomDataset(root = '/data/23_1_ML_challenge/Semi-Supervised_Learning/test', transform = test_transform)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model1.load_state_dict(torch.load(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning', 'best_model_1.pt'), map_location=torch.device('cuda')))
        res1 = test(model1, test_loader)

        model2.load_state_dict(torch.load(os.path.join(args.student_abs_path, 'logs', 'Semi-Supervised_Learning', 'best_model_2.pt'), map_location=torch.device('cuda')))
        res2 = test(model2, test_loader)

        if res1>res2:
            best_res = res1
            best_params = params_1
        else :
            best_res = res2
            best_params = params_2

        print(best_res, ' - ', best_params)
