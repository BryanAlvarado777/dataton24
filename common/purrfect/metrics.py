import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

# ======= Metric used for scoring in the challenge ========


def WMAPE(T: torch.Tensor, P: torch.Tensor, delta=torch.tensor(1e-5)):
    T = torch.where(T == 0.0, delta, T)
    P = torch.where(P == 0.0, delta, P)
    W_fun = lambda T, max_T: 1 + (T / max_T)
    max_T = torch.max(T)
    W = W_fun(T, max_T)
    Numerator = W * (torch.abs(P - T) / torch.abs(T))
    sum_W = torch.sum(W)
    sum_Numerator = torch.sum(Numerator)
    return sum_Numerator / sum_W


def DICE_binary_mask(Tensor: torch.tensor):
    Tensor_mask = Tensor - torch.mean(Tensor)
    Tensor_mask[torch.where(Tensor_mask > 0)] = 1
    Tensor_mask[torch.where(Tensor_mask <= 0)] = 0
    return Tensor_mask


def DICEE(T: torch.Tensor, P: torch.Tensor, alpha: float = 0.5, beta: float = 0.5):
    if T.shape != P.shape:
        return None
    G = DICE_binary_mask(T)
    A = DICE_binary_mask(P)
    sum_G_A = torch.sum(G * A)
    sum_alpha = torch.sum(A * (1 - G))
    sum_beta = torch.sum(G * (1 - A))
    return 1 - (sum_G_A / (sum_G_A + alpha * sum_alpha + beta * sum_beta))


def find_top_k_peaks(im, sigma=3, N=3):
    im = im.astype('float32') 
    smoothed = gaussian_filter(im, sigma=sigma)
    coordinates = peak_local_max(smoothed, threshold_abs=None, num_peaks=N)
    while len(coordinates) < 3:
        coordinates = np.vstack([coordinates, np.array([0, 0])])
    return coordinates


def DPEAKS(T: torch.Tensor, P: torch.Tensor, num_peaks=3):
    T = T.numpy()
    P = P.numpy()
    PEAKS_T = find_top_k_peaks(T, N=num_peaks)
    PEAKS_P = find_top_k_peaks(P, N=num_peaks)
    sum_DPEAKS = np.sum(np.abs(PEAKS_T - PEAKS_P))
    return sum_DPEAKS


class MetricAccumulator:
    @staticmethod
    def create_default():
        return MetricAccumulator(
            {
                "WMAPE": WMAPE,
                "DICE": DICEE,
                "DPEAKS": DPEAKS,
            }
        )

    def __init__(self, metrics: dict):
        self.metrics = metrics
        self.reset()

    def reset(self):
        self.values = {metric_name: [] for metric_name in self.metrics.keys()}

    def update(self, T: torch.Tensor, P: torch.Tensor):
        for metric_name, metric_fun in self.metrics.items():
            for i in range(T.shape[0]):  # assume is in batches
                self.values[metric_name].append(metric_fun(T[i], P[i]))

    def get_metrics(self):
        return {
            metric_name: np.mean(metric_values)
            for metric_name, metric_values in self.values.items()
        }

    def get_metrics_str(self):
        return " ".join(
            [
                f"{metric_name}: {metric_value:.4f}"
                for metric_name, metric_value in self.get_metrics().items()
            ]
        )
