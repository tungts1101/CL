import numpy as np
import torch
import random
from collections import OrderedDict
from typing import Optional
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class Buffer(ABC):
    def __init__(self, capacity: int, decay: float = 1.0, seed: Optional[int] = None):
        self._capacity = capacity
        self._buffer = []
        self._weights = []
        self._total_seen = 0
        self._decay = decay
        self._seed = seed
    
    @abstractmethod
    def add(self, x: torch.Tensor, z: torch.Tensor, y: torch.Tensor):
        pass
    
    def add_capacity(self, new_capacity: int):
        self._capacity += new_capacity
    
    def update_weights(self):
        if not self._buffer:
            return
        self._weights = [w * self._decay for w in self._weights]
    
    def sample(self, batch_size: int = 32, seed: Optional[int] = None):
        if not self._buffer:
            return None

        if seed is not None:
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(self._buffer), size=min(batch_size, len(self._buffer)))
        elif self._seed is not None:
            rng = np.random.RandomState(self._seed + self._total_seen)
            indices = rng.choice(len(self._buffer), size=min(batch_size, len(self._buffer)))
        else:
            indices = np.random.choice(len(self._buffer), size=min(batch_size, len(self._buffer)))

        xs, zs, ys = zip(*[self._buffer[i] for i in indices])
        x_batch = torch.stack(xs)
        z_batch = torch.stack(zs)
        y_batch = torch.tensor(ys, dtype=torch.long)
        return x_batch, z_batch, y_batch
    
    def __iter__(self, batch_size: int = 32, seed: Optional[int] = None):
        buffer = self._buffer.copy()
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(buffer)
        elif self._seed is not None:
            rng = random.Random(self._seed + self._total_seen)
            rng.shuffle(buffer)
        else:
            random.shuffle(buffer)

        for i in range(0, len(buffer), batch_size):
            batch = buffer[i: i + batch_size]
            x_batch = torch.stack([x for x, _, _ in batch])
            z_batch = torch.stack([z for _, z, _ in batch])
            y_batch = torch.tensor([y for _, _, y in batch], dtype=torch.long)
            yield x_batch, z_batch, y_batch
    
    @property
    def size(self):
        return len(self._buffer)

    @property
    def size_by_class(self):
        class_counts = {}
        for _, _, y in self._buffer:
            y_value = y.item()
            class_counts[y_value] = class_counts.get(y_value, 0) + 1
        return OrderedDict(sorted(class_counts.items()))


class RandomReplayBuffer(Buffer):
    def add(self, x: torch.Tensor, z: torch.Tensor, y: torch.Tensor):
        for i in range(y.size(0)):
            self._total_seen += 1
            entry = (x[i].cpu(), z[i].cpu(), y[i].cpu())

            if len(self._buffer) < self._capacity:
                self._buffer.append(entry)
                self._weights.append(1.0)
            else:
                if random.random() >= (self._capacity / self._total_seen):
                    continue

                probs = torch.tensor(self._weights, dtype=torch.float32)
                inv_probs = 1.0 / (probs + 1e-6)
                inv_probs = inv_probs / inv_probs.sum()

                if self._seed is not None:
                    torch.manual_seed(self._seed + self._total_seen)  # vary by sample
                idx = torch.multinomial(inv_probs.cpu(), 1).item()

                if 1.0 >= self._weights[idx]:
                    self._buffer[idx] = entry
                    self._weights[idx] = 1.0


class BalanceReplayBuffer(Buffer):
    def add(self, x: torch.Tensor, z: torch.Tensor, y: torch.Tensor):
        for i in range(y.size(0)):
            self._total_seen += 1
            entry = (x[i].cpu(), z[i].cpu(), y[i].cpu())
            y_value = y[i].item()

            class_counts = self.size_by_class
            num_classes = len(class_counts) + (1 if y_value not in class_counts else 0)
            max_per_class = self._capacity // num_classes if num_classes > 0 else self._capacity

            if len(self._buffer) < self._capacity:
                self._buffer.append(entry)
                self._weights.append(1.0)
            else:
                current_count = class_counts.get(y_value, 0)
                if current_count < max_per_class:
                    self._buffer.append(entry)
                    self._weights.append(1.0)

                    # Remove an entry from the most represented class
                    class_counts = self.size_by_class
                    most_represented_class = max(class_counts, key=class_counts.get)
                    for idx, (_, _, label) in enumerate(self._buffer):
                        if label.item() == most_represented_class:
                            del self._buffer[idx]
                            del self._weights[idx]
                            break
                else:
                    if random.random() >= (max_per_class / (current_count + 1e-6)):
                        continue

                    probs = torch.tensor(self._weights, dtype=torch.float32)
                    inv_probs = 1.0 / (probs + 1e-6)
                    inv_probs = inv_probs / inv_probs.sum()

                    if self._seed is not None:
                        torch.manual_seed(self._seed + self._total_seen)  # vary by sample
                    idx = torch.multinomial(inv_probs.cpu(), 1).item()

                    if y_value == self._buffer[idx][2].item() and 1.0 >= self._weights[idx]:
                        self._buffer[idx] = entry
                        self._weights[idx] = 1.0


class MaxEntropyReplayBuffer(Buffer):
    def add(self, x: torch.Tensor, z: torch.Tensor, y: torch.Tensor):
        probs = torch.softmax(z, dim=-1)
        entropies = -torch.sum(probs * torch.log(probs + 1e-6), dim=-1)
        
        for i in range(y.size(0)):
            self._total_seen += 1
            entry = (x[i].cpu(), z[i].cpu(), y[i].cpu())
            entropy = entropies[i].item()

            if len(self._buffer) < self._capacity:
                self._buffer.append(entry)
                self._weights.append(entropy)
            else:
                if random.random() >= (self._capacity / self._total_seen):
                    continue

                probs = torch.tensor(self._weights, dtype=torch.float32)
                inv_probs = 1.0 / (probs + 1e-6)
                inv_probs = inv_probs / inv_probs.sum()

                if self._seed is not None:
                    torch.manual_seed(self._seed + self._total_seen)  # vary by sample
                idx = torch.multinomial(inv_probs.cpu(), 1).item()

                if entropy >= self._weights[idx]:
                    self._buffer[idx] = entry
                    self._weights[idx] = entropy

class CoresetReplayBuffer(Buffer):
    def add_model(self, model):
        if not hasattr(self, "_models"):
            self._models = []
        
        self._models.append(model)
    
    @torch.no_grad()
    def get_feature(self, x):
        features = [m(x) for m in self._models]
        return torch.cat(features, dim=-1)
    
    def add_cls_mean(self, data_manager, known_classes, total_classes):
        feature_dim = 768 * len(self._models)
                
        if not hasattr(self, "_cls_means"):
            self._cls_means = torch.zeros((total_classes, feature_dim))
        else:
            new_cls_means = torch.zeros((total_classes, feature_dim))
            new_cls_means[: known_classes] = self._cls_means
            self._cls_means = new_cls_means

        for cls_idx in range(known_classes, total_classes):
            proto_set = data_manager.get_dataset(
                np.arange(cls_idx, cls_idx + 1), source="train", mode="test"
            )
            proto_loader = DataLoader(
                proto_set, batch_size=512, shuffle=False, num_workers=4
            )

            features_list = []
            with torch.no_grad():
                for _, (_, x, _) in enumerate(proto_loader):
                    x = x.cuda()
                    f = self.get_feature(x)
                    features_list.append(f.cpu())

            features_list = torch.cat(features_list, dim=0)
            class_mean = torch.mean(features_list, dim=0)
            self._cls_means[cls_idx, :] = class_mean
    
    def add(self, x: torch.Tensor, z: torch.Tensor, y: torch.Tensor):
        # weights = []
        # cls_to_dist = {}
        # for (xi, yi) in zip(x, y):
        #     f = self.get_feature(xi.unsqueeze(0).cuda()).cpu()
        #     cls_mean = self._cls_means[yi.item()].unsqueeze(0)
        #     dist = torch.norm(f - cls_mean, p=2).item()
        #     weights.append(dist)

        #     if yi.item() not in cls_to_dist:
        #         cls_to_dist[yi.item()] = 0
        #     cls_to_dist[yi.item()] += dist
        
        # weights = [w / (cls_to_dist[y[i].item()] + 1e-6) for i, w in enumerate(weights)]
        
        eps = 1e-6
        x, y = x.cuda(), y.cuda()
        f = self.get_feature(x)  # (B, D)
        cls_means = self._cls_means.to(x.device)  # (C, D)
        sample_means = cls_means.index_select(0, y)  # (B, D)
        dists = torch.norm(f - sample_means, p=2, dim=1)  # (B,)
        C = cls_means.shape[0]
        cls_sums = torch.zeros(C, device=x.device).scatter_add_(0, y, dists)  # (C,)
        denom = cls_sums.index_select(0, y) + eps  # (B,)
        weights = dists / denom  # (B,)
        
        for i in range(y.size(0)):
            self._total_seen += 1
            entry = (x[i].cpu(), z[i].cpu(), y[i].cpu())
            y_value = y[i].item()

            class_counts = self.size_by_class
            num_classes = len(class_counts) + (1 if y_value not in class_counts else 0)
            max_per_class = self._capacity // num_classes if num_classes > 0 else self._capacity

            if len(self._buffer) < self._capacity:
                self._buffer.append(entry)
                self._weights.append(weights[i])  # higher weight for closer samples
            else:
                current_count = class_counts.get(y_value, 0)
                if current_count < max_per_class:
                    self._buffer.append(entry)
                    self._weights.append(weights[i])

                    # Remove an entry from the most represented class
                    class_counts = self.size_by_class
                    most_represented_class = max(class_counts, key=class_counts.get)
                    for idx, (_, _, label) in enumerate(self._buffer):
                        if label.item() == most_represented_class:
                            del self._buffer[idx]
                            del self._weights[idx]
                            break
                else:
                    if random.random() >= (max_per_class / (current_count + 1e-6)):
                        continue

                    probs = torch.tensor(self._weights, dtype=torch.float32)
                    probs = probs / probs.sum()

                    if self._seed is not None:
                        torch.manual_seed(self._seed + self._total_seen)  # vary by sample
                    idx = torch.multinomial(probs.cpu(), 1).item()

                    if y_value == self._buffer[idx][2].item() and weights[i] < self._weights[idx]:
                        self._buffer[idx] = entry
                        self._weights[idx] = weights[i]