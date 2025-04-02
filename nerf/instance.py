import torch
import numpy as np
import copy

class Instance:
    def __init__(self, device="cuda"):
        self.criterion = torch.nn.BCEWithLogitsLoss().to(device) #r/p 92/94
        self.instance_ray_wise_prediction  = None
        self.instance_supervision = None

        self.tpr = [[a, None, None, None] for a in np.linspace(0.0, 1.0, 11)]
        self.tprs = []
        self.pos_ratios = []

    def calculate_loss(self, instance_supervision, max_instance):
        # instance: Nr Ns
        # instance_supervision Nr
        instance_ray_wise_prediction = max_instance
        # instance_ray_wise_prediction = max_instance
        loss = self.criterion(instance_ray_wise_prediction, instance_supervision.float())
        # instance_ray_wise_prediction = instance.mean(dim=1)
        self.instance_ray_wise_prediction = instance_ray_wise_prediction.cpu().detach().numpy()
        self.max_instance = max_instance.cpu().detach().numpy()
        self.instance_supervision = instance_supervision.cpu().detach().numpy()
        return loss

    def compute_metrics(self):
        for i in range(len(self.tpr)):
            threshold = self.tpr[i][0]
            predictions = torch.from_numpy(self.instance_ray_wise_prediction).float().numpy() > threshold
            tp = (predictions * self.instance_supervision).sum()
            fp = (predictions * (1-self.instance_supervision)).sum()
            fn = ((1-predictions) * self.instance_supervision).sum()
            precision = tp/(fp+tp+0.00001)
            recall = tp/(tp+fn)
            iou = tp/(tp+fp+fn)
            if np.isnan(iou):
                iou = 0.0
            if np.isnan(recall):
                recall = 0.0
            if np.isnan(precision):
                precision = 0.0
            self.tpr[i][1] = precision
            self.tpr[i][2] = recall
            self.tpr[i][3] = iou

        self.tprs.append(copy.deepcopy(self.tpr))
        self.pos_ratios.append(self.instance_supervision.sum() / self.instance_supervision.shape[0])

    def print_metrics(self, last=100):
        if len(self.pos_ratios) <= 0:
            return
        pos_ratio = np.array(self.pos_ratios[-last:]).mean()
        print('  positive ratio: {}'.format(pos_ratio))
        print('|   thresh    |     prec    |    recall  |    iou  |')
        tprs = np.array(self.tprs[:])
        for i in range(len(self.tpr)):
            print("|    {:.3f}    |".format(self.tpr[i][0]), end='')
            print("    {:.3f}    |".format(tprs[:, i, 1].mean()), end='')
            print("    {:.3f}   |".format(tprs[:, i, 2].mean()), end='')
            print("    {:.3f}   |".format(tprs[:, i, 3].mean()))
        self.tprs = []
        return tprs[:, -2, 3].mean()


