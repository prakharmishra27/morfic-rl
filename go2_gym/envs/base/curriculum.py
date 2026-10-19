
####RNN refresh  curriculum
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from matplotlib import pyplot as plt

def is_met(scale, l2_err, threshold):
    return (l2_err / scale) < threshold

def key_is_met(metric_cache, config, ep_len, target_key, env_id, threshold):
    scale = 1
    l2_err = 0
    return is_met(scale, l2_err, threshold)

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0):
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class Curriculum:
    def __init__(self, seed, **key_ranges):
        self.rng = np.random.RandomState(seed)

        # build per-dimension centroids
        cfg = {}
        for key, (low, high, n_bins) in key_ranges.items():
            size = (high - low) / n_bins
            cfg[key] = np.linspace(low + size/2, high - size/2, n_bins)

        # record raw grid
        self._raw_grid = np.stack(np.meshgrid(*cfg.values(), indexing='ij'))
        self.keys     = list(key_ranges)
        self.grid     = self._raw_grid.reshape(len(self.keys), -1)
        self._l       = self.grid.shape[1]

        # track weights
        self.weights = np.zeros(self._l)
        self.indices = np.arange(self._l)

        # store bin sizes (for uniform sampling)
        self.bin_sizes = {
            key: (high - low) / n_bins
            for key, (low, high, n_bins) in key_ranges.items()
        }

    def __len__(self):
        return self._l

    def set_to(self, low, high, value=1.0):
        mask = (
            (self.grid >= low[:, None]) &
            (self.grid <= high[:, None])
        ).all(axis=0)
        self.weights[mask] = value

    def sample_bins(self, batch_size, low=None, high=None):
        # (you can keep your bounds logic here)
        p = self.weights / self.weights.sum()
        inds = self.rng.choice(self.indices, batch_size, p=p)
        return self.grid.T[inds], inds

    def sample_uniform_from_cell(self, centroids):
        bs = np.array(list(self.bin_sizes.values()))
        lo = centroids - bs/2
        hi = centroids + bs/2
        return self.rng.uniform(lo, hi)

    def sample(self, batch_size):
        centroids, inds = self.sample_bins(batch_size)
        samples = np.stack([self.sample_uniform_from_cell(c) for c in centroids])
        return samples, inds

    # def get_local_bins(self, bin_inds, radius=0.1):
    #     if bin_inds.size == 0:
    #         return []
    #     grid = self.grid
    #     # diffs = np.abs(grid[:, :, None] - grid[:, bin_inds, None])
    #     diffs = np.abs(grid[:, :, None] - grid[:, bin_inds][:, None, :])

    #     mask = (diffs <= radius).all(axis=0)
    #     return mask
    def get_local_bins(self, bin_inds, radius=0.1):
        # bin_inds: indices of "successful" bins
        bin_inds = np.asarray(bin_inds, dtype=int)
        if bin_inds.size == 0:
            # return empty index array
            return np.array([], dtype=int)

        grid = self.grid        # shape (D, B)
        D, B = grid.shape
        K = bin_inds.size

        # --- make radius per-dimension ---
        radius_arr = np.asarray(radius)
        if radius_arr.ndim == 0:
            # scalar → same radius for all dims
            radius_arr = np.full((D,), radius_arr, dtype=grid.dtype)
        else:
            # should be one radius per dim
            assert radius_arr.shape[0] == D, \
                f"radius/local_range has shape {radius_arr.shape}, expected ({D},)"

        # --- distances from each bin to each selected bin ---
        # grid[:, :, None]          -> (D, B, 1)
        # grid[:, bin_inds]         -> (D, K)
        # grid[:, bin_inds][:,None,:] -> (D, 1, K)
        # diffs                     -> (D, B, K)
        diffs = np.abs(grid[:, :, None] - grid[:, bin_inds][:, None, :])

        # broadcast radius (D,) -> (D, 1, 1) to compare with (D, B, K)
        mask = (diffs <= radius_arr[:, None, None]).all(axis=0)  # (B, K)

        # bins that are within radius of ANY successful bin
        adjacents = np.where(mask.any(axis=1))[0]                # (<= B,)

        return adjacents



    

        


class RewardThresholdCurriculum(Curriculum):
    def __init__(self,
                 seed,
                 input_size,       # catches the 4000
                 hidden_size,
                 output_size,      # catches the 8000
                 num_layers=1,
                 **key_ranges):
        # 1) build the bin grid
        super().__init__(seed, **key_ranges)

        # 2) figure out your real bin count
        n_bins = len(self)

        # 3) wire up the RNN to predict (lin, ang) for each bin
        self.rnn_model = RNNModel(
            input_size  = n_bins,
            hidden_size = hidden_size,
            output_size = n_bins * 2,
            num_layers  = num_layers
        ).to('cuda')

        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.rnn_model.parameters(), lr=1e-3)

        # 4) per-bin logs
        self.episode_reward_lin = np.zeros(n_bins)
        self.episode_reward_ang = np.zeros(n_bins)
        self.episode_lin_raw    = np.zeros(n_bins)
        self.episode_ang_raw    = np.zeros(n_bins)
        self.episode_duration   = np.zeros(n_bins)

    def train_rnn(self, input_data, target_data):
        self.rnn_model.train()
        self.optimizer.zero_grad()
        # hidden initial state:
        h0 = torch.zeros(
            self.rnn_model.num_layers,
            input_data.size(0),
            self.rnn_model.hidden_size,
            device=input_data.device
        )
        outputs = self.rnn_model(input_data, h0)
        # reshape & compute loss:
        outputs = outputs.view(-1, 2)
        target  = target_data.view(-1, 2)
        loss = self.criterion(outputs, target)
        if loss.requires_grad:
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def compute_active_coverage(self):
        """
        Compute active exploration coverage as the fraction of bins updated at the current timestep.
        Returns:
            active_coverage (float): Fraction of bins updated in the current timestep.
        """
        active_bins = np.sum(self.weights_delta > 0)  # Bins where weights were updated
        total_bins = len(self.weights)
        return active_bins / total_bins


    def compute_cumulative_coverage(self):
        """
        Compute cumulative exploration coverage as the fraction of bins explored up to the current timestep.
        Returns:
            cumulative_coverage (float): Fraction of bins explored cumulatively.
        """
        explored_bins = np.sum(self.weights > 0)  # Bins with non-zero weights
        total_bins = len(self.weights)
        return explored_bins / total_bins
    

    def update(self,
            bin_inds,
            task_rewards,
            success_thresholds,
            local_range=0.5):
        
        lin_vel_rewards, ang_vel_rewards       = task_rewards[:2]
        lin_vel_threshold, ang_vel_threshold   = success_thresholds[:2]

        # bring to CPU
        lin_vel_rewards = lin_vel_rewards.cpu().numpy()
        ang_vel_rewards = ang_vel_rewards.cpu().numpy()
        lin_vel_threshold = float(lin_vel_threshold)
        ang_vel_threshold = float(ang_vel_threshold)

        previous_weights = self.weights.copy()

        # log raw rewards
        self.episode_reward_lin[bin_inds] = lin_vel_rewards
        self.episode_reward_ang[bin_inds] = ang_vel_rewards

        # determine which bins “succeeded”
        is_success      = (lin_vel_rewards > lin_vel_threshold) & \
                        (ang_vel_rewards > ang_vel_threshold)
        successful_bins = bin_inds[is_success]

        # build one-hot input for the RNN
        n_bins = self._l  # total number of bins
        input_data = torch.zeros((1, 1, n_bins), dtype=torch.float32, device='cuda')
        input_data[0, 0, bin_inds] = 1.0
        input_data.requires_grad_()

        # hidden state
        h0 = torch.zeros(self.rnn_model.num_layers, input_data.size(0), self.rnn_model.hidden_size).to(input_data.device)


        # predict rewards
        rnn_output = self.rnn_model(input_data, h0).cpu().detach().numpy().flatten()


        # bump weights on successes
        for idx in successful_bins:
            i_lin = 2*idx
            i_ang = i_lin + 1
            if i_ang < len(self.weights)*2:
                pred_lin = rnn_output[i_lin]
                pred_ang = rnn_output[i_ang]
                inc = 0.2 * (pred_lin + pred_ang) / 2
                self.weights[idx] = np.clip(self.weights[idx] + inc, 0, 1)

        # only do neighbor-bump if we have successes
        if successful_bins.size > 0:
            adjacents = self.get_local_bins(successful_bins, local_range)
            for mask in adjacents:
                for idx in mask.nonzero()[0]:
                    i_lin = 2*idx
                    i_ang = i_lin + 1
                    if i_ang < len(self.weights)*2:
                        pred_lin = rnn_output[i_lin]
                        pred_ang = rnn_output[i_ang]
                        inc = 0.2 * (pred_lin + pred_ang) / 2
                        self.weights[idx] = np.clip(self.weights[idx] + inc, 0, 1)

        
        agg = np.zeros((n_bins, 2), dtype=np.float32)
        counts = np.zeros(n_bins, dtype=np.int32)
        for env_idx, bin_idx in enumerate(bin_inds):
            agg[bin_idx, 0] += lin_vel_rewards[env_idx]
            agg[bin_idx, 1] += ang_vel_rewards[env_idx]
            counts[bin_idx] += 1

        # avoid divide-by-zero: only normalize bins that were hit
        nonzero = counts > 0
        agg[nonzero] /= counts[nonzero][:, None]

        # 2) build the torch target tensor of shape (n_bins, 2)
        target = torch.from_numpy(agg).to(input_data.device)

        # 3) train the RNN
        self.train_rnn(input_data, target)
        
        # 

        # optional coverage logging
        self.weights_delta = self.weights - previous_weights
        active_coverage = self.compute_active_coverage()
        cumulative_coverage = self.compute_cumulative_coverage()
        # print(f"Active {active_coverage:.3f}, Cum. {cumulative_coverage:.3f}")



    def log(self, bin_inds, lin_vel_raw=None, ang_vel_raw=None, episode_duration=None):
        self.episode_lin_vel_raw[bin_inds] = lin_vel_raw.cpu().numpy()
        self.episode_ang_vel_raw[bin_inds] = ang_vel_raw.cpu().numpy()
        self.episode_duration[bin_inds] = episode_duration.cpu().numpy()


if __name__ == '__main__':
    r = RewardThresholdCurriculum(
        seed       = 100,
        hidden_size= 128,
        num_layers = 1,
        x          = (-1, 1, 20),
        y          = (-1, 1, 10),
        z          = (-1, 1, 20),
    )

    assert r._raw_grid.shape == (3, 20, 10, 20), \
        f"grid shape is wrong: {r._raw_grid.shape}"

    # dry‐run an update
    low, high = np.array([-1.0, -0.6, -1.0]), np.array([1.0, 0.6, 1.0])
    r.set_to(low, high, value=1.0)

    # sample some bins and do an update
    samples, bins   = r.sample(100)
    fake_lin, fake_ang = np.ones_like(bins), np.ones_like(bins)
    r.update(
        bin_inds         = bins,
        task_rewards     = (torch.tensor(fake_lin), torch.tensor(fake_ang)),
        success_thresholds = (0.0, 0.0),
        local_range      = 0.5,
    )
    print("Update ok, weights sum:", r.weights.sum())



