"""Author: Urchade Z.
"""
from torch.utils.data import DataLoader
from model import Classifier
from dataset import TrainDataset, Preprocessor, TestDataset

from trainer import ClassifierTrainer
from sklearn.metrics import balanced_accuracy_score
import pandas as pd

process = Preprocessor()

# Dataset
train_data = TrainDataset(process.get_train())
val_data = TrainDataset(process.get_val())
test_data = TestDataset(process.get_test())

# Dataloader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(val_data, batch_size=100)
test_loader = DataLoader(test_data, batch_size=16)

model = Classifier().cuda()

# Training
trainers = ClassifierTrainer(model, train_loader, val_loader,
                             metric=balanced_accuracy_score)

trainers.train(max_epoch=7, patience=1)

# Visualization
trainers.plot_cm()
trainers.plot_metric_curve()
trainers.plot_loss_curve()

# Evaluation
trainers.model.eval()

pred = []
for x in test_loader:
    pred.append(trainers.model.predict(x.float().cuda()))

pred = [a for i in pred for a in i]

pred = process.enc.inverse_transform(pred)

# Saving
sub = pd.read_csv('data/sample_submission.csv')
sub['cell_line'] = pred
sub.to_csv('submission.csv', index=False)
