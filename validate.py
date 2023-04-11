import os

import torch
from torchvision.models import resnet18
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from torchvision import datasets, transforms
import random
import numpy as np
from easyfsl.data_tools import TaskSampler

from sklearn.metrics import precision_recall_fscore_support
from prettytable import PrettyTable

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_size = 224

proto_model_path = r'models\pretrained\proto_5_v1.pth'

N_SHOT = 5 # Number of images per class
N_QUERY = 5 # Number of images per class in the query set
N_EVALUATION_TASKS = 100
val_dir = r'C:\DTU\fsl_paper\maritime-fsl\data\data_2023\val_2023_v2'

test_classes = os.listdir(val_dir)
N_WAY_TEST = len(test_classes) # Number of classes

data_mean = [0.4609, 0.4467, 0.4413]
data_std = [0.1314, 0.1239, 0.1198]


val_data = datasets.ImageFolder(root = val_dir, transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.functional.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std),
        ]
    ),)


val_data.labels = val_data.targets
test_sampler = TaskSampler(
    val_data, n_way=N_WAY_TEST, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS
)

def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

test_loader = DataLoader(
    val_data,
    batch_sampler=test_sampler,
    #num_workers=8,
    pin_memory=True,
    worker_init_fn= seed_worker,
    collate_fn=test_sampler.episodic_collate_fn,
)

class PrototypicalNetworkModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworkModel, self).__init__()
        self.backbone = backbone


    def get_prototypes(self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor):
        
        z_support = self.backbone.forward(support_images)
        n_way = len(torch.unique(support_labels))

        mean_prototypes = torch.cat(
            [
                z_support[torch.nonzero(support_labels == label)].mean(dim = 0)
                
                for label in range(n_way)
            ]
        )
        return mean_prototypes

    def calculate_distance(self, mean_prototypes, query_images):
        z_query = self.backbone.forward(query_images)
        dist1 = torch.cdist(z_query, mean_prototypes)
        return -dist1
        
    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        mean_prototypes: None
    ) -> torch.Tensor:
        """
        Predict query labels using labeled support images.
        """

        z_query = self.backbone.forward(query_images)

        if not mean_prototypes:
            mean_prototypes = self.get_prototypes(support_images,support_labels)

        dists = torch.cdist(z_query, mean_prototypes)

        scores = -dists
        
        return scores, mean_prototypes


convolutional_network = resnet18(pretrained=True)
convolutional_network.fc = nn.Flatten()
#convolutional_network.load_state_dict(torch.load(resnet_path))
model = PrototypicalNetworkModel(convolutional_network)
model.to(device)
model.load_state_dict(torch.load(proto_model_path))
model.eval()

def evaluate_on_one_task(
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
    class_prototypes: torch.Tensor
):
    """
    evaluation function
    """
    model_scores, comp_prototypes =  model(
            support_images,
            support_labels,
            query_images,
            class_prototypes
        )
    class_inf = torch.max(model_scores.data
       ,1,
    )[1].tolist()

    return (
        torch.max(model_scores, 1,)[1] == query_labels
    ).sum().item(), len(query_labels), class_inf, query_labels.tolist(), comp_prototypes




def find_classes(val_dir):
    '''Finds the classes and their corresponding indexing id'''

    classes = os.listdir(val_dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return class_to_idx


def compute_prototypes(data_loader):
    best_prototypes = None
    total_predictions = 0
    correct_predictions = 0
    exact = []
    predicted = []
    pred_list = []
    class_prototypes = None
    model.eval()

    precision_total = [0] * len(find_classes(val_dir).keys())
    recall_total = [0] * len(find_classes(val_dir).keys())
    f1_score_total = [0] * len(find_classes(val_dir).keys())
    accuracy_total = 0
    with torch.no_grad():
        for episode_index, (support_images,support_labels,query_images,query_labels,class_ids,) in tqdm(enumerate(data_loader), total=len(data_loader)):
            correct, total, predicted_classes, exact_classes, comp_prototypes = evaluate_on_one_task(support_images.to(device), support_labels.to(device), query_images.to(device), query_labels.to(device), class_prototypes)
            #opt_prototypes = comp_prototypes
            exact.extend(exact_classes)
            predicted.extend(predicted_classes)
            pred_list.append(correct)
            total_predictions += total
            correct_predictions += correct
            faa = precision_recall_fscore_support(exact_classes, predicted_classes, average=None)
            precision_total = [sum(i) for i in zip(precision_total, faa[0] )]
            recall_total = [sum(i) for i in zip(recall_total, faa[1] )]
            f1_score_total = [sum(i) for i in zip(f1_score_total, faa[2] )]
            
            model_accuracy = (100 * correct_predictions/total_predictions)
            accuracy_total += model_accuracy
            


        precision_final = np.divide(precision_total, N_EVALUATION_TASKS)
        recall_final = np.divide(recall_total, N_EVALUATION_TASKS)
        f1_score_final = np.divide(f1_score_total, N_EVALUATION_TASKS)
        accuracy_final = accuracy_total/N_EVALUATION_TASKS
        accuracy_formatted = np.around(accuracy_final, 4)

        precision_formatted = list(np.around(np.array(precision_final), 4))
        recall_formatted = list(np.around(np.array(recall_final), 4))
        f1_score_formatted = list(np.around(np.array(f1_score_final), 4))

        print('accuracy: ', accuracy_formatted)
        
        x = PrettyTable()
        x.field_names = list(find_classes(val_dir).keys())
        x.add_row(precision_formatted)
        x.add_row(recall_formatted)
        x.add_row(f1_score_formatted)
        print(x)


if __name__ == "__main__":
    best_prototypes = compute_prototypes(test_loader)
    
    pass

