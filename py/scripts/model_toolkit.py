import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import time
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from multiprocessing import set_start_method


class ModelToolkit:

    def __init__(self, model, name, checkpoint=None, batch_size=4, num_workers=4):
        self.model = model
        self.name = name
        self.lr = 5e-4
        self.model.fc = torch.nn.Linear(model.fc.in_features, 5)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=3,
                                                                    verbose=True)
        self.epoch = 0
        self.best_loss = float('inf')
        self.losses = {phase: [] for phase in ['train', 'val']}
        self.accuracy = {phase: [] for phase in ['train', 'val']}

        self.dataloaders = {phase: DataLoader(
            SteelDataset(phase),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True
        ) for phase in ['train', 'val']}

        if checkpoint is not None:
            self.load_model(checkpoint)

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            #torch.set_default_tensor_type('torch.cuda.FloatTensor')
            print(self.device, torch.cuda.get_device_name(0))
        else:
            self.device = torch.device('cpu')
            print(self.device)
        self.model = self.model.to(self.device)

        torch.backends.cudnn.benchmark = True
        set_start_method('spawn')

    def save_model(self):
        file_name = '{}-{}-{:.4f}'.format(self.name, self.epoch, self.best_loss)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimazer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'losses': self.losses,
            'accuracy': self.accuracy
        }, '~/workspace/steel_defect_recognition/checkpoints/{}'.format(file_name))
        print('saving model with name: "{}"'.format(file_name))

    def load_model(self, file_name):
        checkpoint = torch.load('~/workspace/steel_defect_recognition/checkpoints/{}'.format(file_name))
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimazer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.losses = checkpoint['losses']
        self.accuracy = checkpoint['accuracy']
        print('loading model with name: "{}"'.format(file_name))

    def forward(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(images)
        loss = self.criterion(outputs, targets)
        return loss, outputs

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.epoch += 1
            self.run_epoch('train')
            with torch.no_grad():
                val_loss = self.run_epoch('val')
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                print('******** New optimal found, saving state ********')
                self.save_model()
            print()
        self.plot_scores()

    def run_epoch(self, phase):
        start = time.strftime('%H:%M:%S')
        print(f'Starting epoch: {self.epoch} | phase: {phase} | ⏰: {start}')
        self.model.train(phase == 'train')
        running_loss = 0.0
        running_acc = 0.0
        total_batches = len(self.dataloaders[phase])
        tk0 = tqdm(self.dataloaders[phase], total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(tk0):
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            if phase == 'train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            outputs = outputs.detach().cpu()
            running_loss += loss.item()
            running_acc += (outputs.argmax(dim=1) == targets).float().mean()

            tk0.set_postfix(loss=(running_loss / (itr + 1)))
        epoch_loss = running_loss / total_batches
        '''logging the metrics at the end of an epoch'''
        self.losses[phase].append(epoch_loss)
        self.accuracy[phase] = running_acc / total_batches
        torch.cuda.empty_cache()
        return epoch_loss

    def plot_scores(self):
        plt.figure(figsize=(30, 20))
        plt.subplot(1, 2, 1)
        self.plot_score(self.losses, 'loss')
        plt.subplot(1, 2, 2)
        self.plot_score(self.accuracy, 'accuracy')
        plt.show()

    @staticmethod
    def plot_score(score, name):
        plt.plot(score['train'], label=f'train {name}')
        plt.plot(score['val'], label=f'val {name}')
        plt.title(f'{name} plot')
        plt.xlabel('Epoch')
        plt.ylabel(f'{name}')
        plt.legend()

class SteelDataset(Dataset):
    def __init__(self, phase):
        self.df = pd.read_csv('/home/void/workspace/steel_defect_recognition/data/' + phase + '.csv')
        self.root = '/home/void/workspace/steel_defect_recognition/data/images'
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, row_id):
        image_name = self.df.loc[row_id, 'img']
        image_path = os.path.join(self.root, image_name)
        img = cv2.imread(image_path)
        img = self.transforms(img)
        class_id = self.df.loc[row_id, 'ClassId']
        return img, class_id

    def __len__(self):
        return len(self.df.index)


