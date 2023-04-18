#!/usr/bin/env python3

import os
import random
import yaml
from easyfsl.data_tools import TaskSampler

import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm import tqdm
from torch import optim
from easyfsl.utils import sliding_average

training_config_path = 'config/training_config.yaml'



from utilities.util_functions import config_loader

training_config = config_loader(training_config_path)

IMAGE_SIZE = 224

data_2023 = r'C:\DTU\fsl_paper\maritime-fsl\data\data_2023\data_2023'

model_save_path = r'C:\DTU\fsl_paper\maritime-fsl\models\pretrained\proto_1_v2.pth'

data_mean = [0.4609, 0.4467, 0.4413]
data_std = [0.1314, 0.1239, 0.1198]

train_data = datasets.ImageFolder(root = data_2023, transform = transforms.Compose(
        [
            transforms.Resize(
                size=IMAGE_SIZE,
                interpolation=transforms.functional.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std),
        ]
    ),)


def set_determinism():
    random_seed = 0
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
train_classes = os.listdir(data_2023)


n_way_train = len(train_classes) # Number of classes
N_SHOT = 1 # Number of images per class
N_QUERY = 1 # Number of images per class in the query set
N_TRAINING_EPISODES = 2000


train_data.labels = train_data.targets
train_sampler = TaskSampler(
    train_data, n_way=n_way_train, n_shot=N_SHOT,
    n_query=N_QUERY, n_tasks=N_TRAINING_EPISODES
)

train_loader = DataLoader(
    train_data,
    batch_sampler=train_sampler,
    #num_workers=8,
    pin_memory=True,
    worker_init_fn= seed_worker,
    collate_fn=train_sampler.episodic_collate_fn,
)


def fit(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:

    optimizer.zero_grad()
    classification_scores = model(
        support_images.cuda(), support_labels.cuda(), query_images.cuda()
    )
    loss = criterion(classification_scores, query_labels.cuda())
    loss.backward()
    optimizer.step()
    predictions = classification_scores.argmax(dim=1, keepdim=True).squeeze()
    correct_preds = (predictions.int() == query_labels.cuda().int()).float()

    return loss.item(), correct_preds


log_update_frequency = 20


class PrototypicalNetworkModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworkModel, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """

        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)

        n_way = len(torch.unique(support_labels))

        z_proto_median = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].median(dim = 0)[0]
                for label in range(n_way)
            ]
        )

        z_proto_mean = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(dim = 0)
                for label in range(n_way)
            ]
        )

        z_total = torch.div(torch.add(z_proto_median, z_proto_mean), 2)
        #Eucledian distance metric
        
        def pairwise(z_query, z_proto):
            pdist = nn.PairwiseDistance(p=2)
            d1 = []
            for j in range(0, len(z_query)):
                d2 = []
                for i in range(0,len(z_proto)):
                    d2.append(pdist(z_query[j], z_proto[i]))
                d1.append(d2)
            return(torch.FloatTensor(d1).to(device))

        def cosinesimilarity(z_query, z_proto):
            cos1 = nn.CosineSimilarity(dim=0, eps=1e-6)
            d1 = []
            for j in range(0, len(z_query)):
                d2 = []
                for i in range(0,len(z_proto)):
                    d2.append(cos1(z_query[j], z_proto[i]))
                d1.append(d2)
            return(torch.FloatTensor(d1).to(device))
        #dists = torch.cdist(z_query, z_total)
        #dists = pairwise(z_query, z_total)
        dists = torch.cdist(z_query, z_proto_mean)
        #dists = cosinesimilarity(z_query, z_total)
        scores = -dists
        
        return scores


if training_config['BACKBONE'] == 'custom':
    filename_pth = training_config['BACKBONE_PATH']

    convolutional_network = resnet18(pretrained=False)
    convolutional_network.fc = nn.Flatten()
    convolutional_network.load_state_dict(torch.load(filename_pth))
else:
    convolutional_network = resnet18(pretrained=True)
    convolutional_network.fc = nn.Flatten()

model = PrototypicalNetworkModel(convolutional_network)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for param in model.parameters():
   param.requires_grad = True

best_vloss = 1_000_000.
acc = []
all_loss = []
all_acc = []
model.train()
with tqdm(enumerate(train_loader), total=len(train_loader)) as tqdm_train:
    for episode_index, (
        support_images,
        support_labels,
        query_images,
        query_labels,
        _,
    ) in tqdm_train:
        loss_value, correct_preds = fit(
            support_images, support_labels, query_images, query_labels
        )
        all_loss.append(loss_value)
        acc.append(correct_preds)
        accuracy = torch.cat(acc, dim=0).mean().cpu()
        all_acc.append(accuracy)
        if episode_index % log_update_frequency == 0:
            tqdm_train.set_postfix(
                loss=sliding_average(all_loss, log_update_frequency),
                accuracy = float("{:.4f}".format(100.0 * accuracy))
            )
            if loss_value < best_vloss:
                print(f'{loss_value} < {best_vloss}')
                print('saving model')
                best_vloss = loss_value
                model_path = model_save_path
                torch.save(model.state_dict(), model_path)
accuracy = torch.cat(acc, dim=0).mean().cpu()
print(accuracy)