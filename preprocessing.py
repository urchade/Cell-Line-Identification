from torch.utils.data import DataLoader
from model import Classifier
from dataset import TrainDataset, Preprocessor, TestDataset
from trainer import ClassifierTrainer
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
import numpy as np

process = Preprocessor()

train_data = TrainDataset(process.get_train())
val_data = TrainDataset(process.get_val())
test_data = TestDataset(process.get_test())
test_loader = DataLoader(test_data, batch_size=16)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=100)

model = Classifier().cuda()

trainers = ClassifierTrainer(model, train_loader, val_loader,
                             metric=balanced_accuracy_score)

trainers.train(max_epoch=7, patience=1)

trainers.optimal_num_epochs = 5
trainers.plot_metric_curve()
trainers.plot_loss_curve()

trainers.model.load_state_dict(trainers.best_state_dict)

trainers.plot_cm()
trainers.report_result()

model.load_state_dict(trainers.best_state_dict)

test_data = TestDataset(process.get_test())
test_loader = DataLoader(test_data, batch_size=16)

preds = []

trainers.model.eval()

from tqdm import tqdm

for x in tqdm(test_loader):
    preds.append(trainers.model.predict(x.float().cuda()))

preds = [a for i in preds for a in i]

preds = process.enc.inverse_transform(preds)

import pandas as pd

sub = pd.read_csv('data/sample_submission.csv')

sub['cell_line'] = preds

sub.to_csv('save_6.csv', index=False)

trainers.report_result()

from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
sns.palplot(sns.xkcd_palette(colors))

_, _, (y_true, y_pred) = trainers.evaluate(val_loader)
cm = metrics.confusion_matrix(y_true, y_pred, normalize='true')
idx = sorted(np.unique(y_true))
confusion_matrix = pd.DataFrame(cm, index=process.enc.classes_, columns=process.enc.classes_)
plt.figure(figsize=(7, 7))
sns.heatmap(confusion_matrix, annot=True, fmt=".2f", square=True, cbar=False, cmap=sns.color_palette("Blues"))
plt.show()

plot_cm(val_loader)

confusion_matrix.columns = process.enc.classes_

confusion_matrix = confusion_matrix.set_index(process.enc.classes_)

import glob, os
from PIL import Image

files = sorted(glob.glob(os.path.join('images_train', '*.png')))

fig = plt.axes()
plt.imshow(Image.open(files[0]), )
plt.show()

plt.subplot(1, 3, 1)
plt.imshow(Image.open(files[0]))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(Image.open(files[1]))
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(Image.open(files[2]))
plt.axis('off')
plt.show()

gg = pd.read_csv('data/y_train.csv')
gg['cell_line'].plot()
plt.show()

sns.countplot(gg['cell_line'])
plt.show()