import numpy as np
import scipy.linalg
# from models.model import create_model, load_model, save_model
import torch
import os


class DecoderRNN(torch.nn.Module):
    def __init__(self, num_hidden, dataset=None):
        super(DecoderRNN, self).__init__()
        self.num_hidden = num_hidden
        if dataset == "nuscenes":
            self.lstm = torch.nn.LSTM(18, self.num_hidden)
            self.out1 = torch.nn.Linear(self.num_hidden, 64)
            self.out2 = torch.nn.Linear(64, 4 * 4)
        else:
            self.lstm = torch.nn.LSTM(11, self.num_hidden)
            self.out1 = torch.nn.Linear(self.num_hidden, 64)
            self.out2 = torch.nn.Linear(64, 4 * 5)

    def forward(self, input_traj):
        # Fully connected
        input_traj = input_traj.permute(1, 0, 2)

        output, (hn, cn) = self.lstm(input_traj)
        x = self.out1(output[-1])
        x = self.out2(x)
        return x


class LSTMPredictor(object):
    def __init__(self):

        self.model = DecoderRNN(128)
#         self.model = load_model(self.model, opt.load_model_traj)
        self.model.load_state_dict(torch.load('lstm.pth')['state_dict'])
        self.model = self.model.to("cuda")
        self.model.eval()

#         if opt.dataset == "nuscenes":
#             self.MAX_dis_fut = 4
#         else:
#             self.MAX_dis_fut = 5
        self.MAX_dis_fut = 5

    def predict(self, h0, c0, new_features):
        new_features = new_features.permute(1, 0, 2)

        output, (hn, cn) = self.model.lstm(new_features, (h0, c0))
        x = self.model.out1(output[-1])
        x = self.model.out2(x)

        x = x.view(self.MAX_dis_fut, -1).cpu().detach().numpy()
        predictions = {}
        for i in range(self.MAX_dis_fut):
            predictions[1 + i] = x[i]

        return hn, cn, predictions

    def gating_distance(
        self, mean, covariance, measurements, only_position=False, metric="maha"
    ):

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean

        if metric == "gaussian":

            d = measurements[:, 3:-1] - mean[3:-1]
            return np.sqrt(np.sum(d * d, axis=1))
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
            )
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError("invalid distance metric")