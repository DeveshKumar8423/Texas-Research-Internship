import time
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from data_processing import load_data, process_data_for_motion_codes, split_train_test_forecasting
from motion_code_utils import optimize_motion_codes, classify_predict_helper
from sparse_gp import sigmoid, q
from utils import accuracy, RMSE

class MotionCode:
    def __init__(self, m=10, Q=1, latent_dim=2, sigma_y=0.1):
        self.m = m
        self.Q = Q
        self.latent_dim = latent_dim
        self.sigma_y = sigma_y

    def fit(self, X_train, Y_train, labels_train, model_path):
        start_time = time.time()
        self.model_path = model_path
        writer = SummaryWriter(log_dir=f"tensorboard_logs/{model_path}_train")

        optimize_motion_codes(X_train, Y_train, labels_train, model_path=model_path, 
              m=self.m, Q=self.Q, latent_dim=self.latent_dim, sigma_y=self.sigma_y)

        # Log training duration
        self.train_time = time.time() - start_time
        writer.add_scalar("Training Time", self.train_time, 0)
        writer.close()

    def load(self, model_path=''):
        if len(model_path) == 0 and self.model_path is not None:
            model_path = self.model_path
        params = np.load(model_path + '.npy', allow_pickle=True).item()
        self.X_m, self.Z, self.Sigma, self.W = params.get('X_m'), params.get('Z'), params.get('Sigma'), params.get('W') 
        self.mu_ms, self.A_ms, self.K_mm_invs = params.get('mu_ms'), params.get('A_ms'), params.get('K_mm_invs')
        self.num_motion = self.Z.shape[0]
        self.kernel_params = []
        for k in range(self.num_motion):
            self.kernel_params.append((self.Sigma[k], self.W[k]))

    def classify_predict(self, X_test, Y_test):
        return classify_predict_helper(X_test, Y_test, self.kernel_params, 
                                       self.X_m, self.Z, self.mu_ms, self.A_ms, self.K_mm_invs)

    def classify_predict_on_batches(self, X_test_list, Y_test_list, true_labels):
        writer = SummaryWriter(log_dir="tensorboard_logs/classification")
        num_predicted = 0
        pred = []; gt = []
        if isinstance(Y_test_list, list):
            num_test = len(Y_test_list)
        else:
            num_test = Y_test_list.shape[0]
        pbar = tqdm(zip(X_test_list, Y_test_list), total=num_test, leave=False)
        num_predicted = 0
        for X_test, Y_test in pbar:
            pred_label = self.classify_predict(X_test, Y_test)
            gt_label = true_labels[num_predicted]
            pbar.set_description(f'Predict: {pred_label}; gt: {gt_label}')
            pred.append(pred_label); gt.append(gt_label)
            num_predicted += 1

        acc = accuracy(pred, gt)
        writer.add_scalar("Classification Accuracy", acc, 0)
        writer.close()
        return acc

    def forecast_predict(self, test_time_horizon, label):
        k = label
        return q(test_time_horizon, sigmoid(self.X_m @ self.Z[k]), 
                 self.kernel_params[k], self.mu_ms[k], self.A_ms[k], self.K_mm_invs[k])

    def forecast_predict_on_batches(self, test_time_horizon, Y_test_list, labels):
        mean_preds = []
        for k in range(self.num_motion):
            mean, _ = self.forecast_predict(test_time_horizon, label=k)
            mean_preds.append(mean)

        all_errors = [[] for _ in range(self.num_motion)]

        for i in range(len(Y_test_list)):
            label = labels[i]
            all_errors[label].append(RMSE(mean_preds[label], Y_test_list[i]))

        errs = np.zeros(self.num_motion)
        for i in range(self.num_motion):
            errs[i] = np.mean(np.array(all_errors[i]))

        writer = SummaryWriter(log_dir="tensorboard_logs/forecast")
        for i, err in enumerate(errs):
            writer.add_scalar(f"Forecast_RMSE/class_{i}", err, 0)
        writer.close()

        return errs

def motion_code_classify(model, name,
                        X_train, Y_train, labels_train,
                        X_test, Y_test, labels_test,
                        load_existing_model=False):
    model_path = 'saved_models/' + name + '_classify'
    if not load_existing_model:
        model.fit(X_train, Y_train, labels_train, model_path)
    model.load(model_path)
    acc = model.classify_predict_on_batches(X_test, Y_test, labels_test)
    return acc

def motion_code_forecast(model, name, X_train, Y_train, labels,
                         test_time_horizon, Y_test, load_existing_model=False):
    model_path = 'saved_models/' + name + '_forecast'
    if not load_existing_model:
        model.fit(X_train, Y_train, labels, model_path)
    model.load(model_path)
    err = model.forecast_predict_on_batches(test_time_horizon, Y_test, labels)

    return err

if __name__ == "__main__":
    dataset_name = "parkinson_pd1"

    train_data = np.load("data/parkinson_pd1/train.npy", allow_pickle=True).item()
    test_data = np.load("data/parkinson_pd1/test.npy", allow_pickle=True).item()

    X_train, Y_train, labels_train = train_data["X"], train_data["Y"], train_data["labels"]
    X_test, Y_test, labels_test = test_data["X"], test_data["Y"], test_data["labels"]

    model = MotionCode()
    acc = motion_code_classify(model, name=dataset_name,
                               X_train=X_train, Y_train=Y_train, labels_train=labels_train,
                               X_test=X_test, Y_test=Y_test, labels_test=labels_test)

    print(f"Classification Accuracy on {dataset_name}:", acc)






