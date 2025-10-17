import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import SimpleVitNet
from backbone.linears import SimpleContinualLinear
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from collections import OrderedDict
import random
import os


import random
import numpy as np
import torch
from collections import OrderedDict
from typing import Optional

class RandomReplayBuffer:
    def __init__(self, capacity: int, decay=1.0, seed: Optional[int] = None):
        self._capacity = capacity
        self._buffer = []
        self._weights = []
        self._total_seen = 0
        self._decay = decay
        self._seed = seed

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

    def update_weights(self):
        if not self._buffer:
            return
        self._weights = [w * self._decay for w in self._weights]

    def sample(self, batch_size: int = 32, seed: Optional[int] = None):
        if not self._buffer:
            return None

        weights_np = np.array(self._weights, dtype=np.float32)
        probs = weights_np / weights_np.sum()

        if seed is not None:
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(self._buffer), size=min(batch_size, len(self._buffer)), p=probs)
        elif self._seed is not None:
            rng = np.random.RandomState(self._seed + self._total_seen)
            indices = rng.choice(len(self._buffer), size=min(batch_size, len(self._buffer)), p=probs)
        else:
            indices = np.random.choice(len(self._buffer), size=min(batch_size, len(self._buffer)), p=probs)

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


def trim(tensor, topk=100):
    flattened = tensor.view(-1)
    magnitudes = torch.abs(flattened)
    num_keep = max(1, int(len(flattened) * topk / 100))
    threshold = torch.topk(magnitudes, num_keep, largest=True, sorted=True).values[-1]
    mask = magnitudes >= threshold
    trimmed = torch.where(mask, flattened, torch.tensor(0.0, dtype=tensor.dtype))

    gamma = torch.sign(trimmed)
    mu = torch.abs(trimmed)

    return (trimmed.view_as(tensor), gamma.view_as(tensor), mu.view_as(tensor))


def merge_task_vectors(trimmed_task_vectors):
    gamma_tvs = torch.stack([tv[1] for tv in trimmed_task_vectors], dim=0)
    gamma = torch.sign(gamma_tvs.sum(dim=0))
    mask = gamma_tvs == gamma
    tau_tvs = torch.stack([tv[0] for tv in trimmed_task_vectors], dim=0)
    mean_tvs = torch.where(mask, tau_tvs, torch.tensor(0.0, dtype=tau_tvs.dtype)).sum(
        dim=0
    ) / mask.sum(dim=0).clamp(min=1)

    return mean_tvs


def merge(base_params, tasks_params, method="ties", lamb=1.0, topk=100):
    params = {}
    for name in base_params:
        base_tv = base_params[name].clone()
        task_vectors = [task_params[name] for task_params in tasks_params]

        tvs = [task_vectors[i] - base_tv for i in range(len(task_vectors))]

        if method == "ties":
            tvs = [trim(tv, topk) for tv in tvs]
            merged_tv = merge_task_vectors(tvs)
        elif method == "max":
            merged_tv = torch.max(torch.stack(tvs, dim=0), dim=0)[0]
        elif method == "min":
            merged_tv = torch.min(torch.stack(tvs, dim=0), dim=0)[0]
        elif method == "max_abs":
            stacked = torch.stack(tvs, dim=0)
            abs_stacked = torch.abs(stacked)
            max_idx = torch.argmax(abs_stacked, dim=0)
            merged_tv = torch.gather(stacked, 0, max_idx.unsqueeze(0)).squeeze(0)

        params[name] = base_tv + lamb * merged_tv

    return params


# tune the model at first session with adapter, and then conduct simplecil.
num_workers = 8
EPSILON = 1e-8

class Learner(BaseLearner):
    CHECKPOINT_DIR = "checkpoints"
    
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet(args, True)
        self._linear = SimpleContinualLinear(self._network.backbone.embed_dim, args["init_cls"])
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

        self._network.to(self._device)
        torch.save(
            self._network.get_backbone_trainable_params(),
            self.backbone_checkpoint(self._cur_task),
        )

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        y_pred, y_true = self._eval_nme(self.test_loader)
        nme_accy = self._evaluate(y_pred, y_true)

        return cnn_accy, nme_accy

    def _eval_cnn(self, loader):
        self._network.eval()
        self._linear.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                features = self._network.extract_vector(inputs)
                outputs = self._linear(features)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means=None):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]


    def after_task(self):
        self._known_classes = self._total_classes

    def replace_fc(self, trainloader, model, args):       
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.extract_vector(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        Y = target2onehot(label_list, self.args["nb_classes"])
        Features_h = F.relu(embedding_list @ self.W_rand.cpu())
        self.Q = self.Q + Features_h.T @ Y
        self.G = self.G + Features_h.T @ Features_h
        ridge = self.optimise_ridge_parameter(Features_h, Y)
        Wo = torch.linalg.solve(self.G + ridge*torch.eye(self.G.size(dim=0)), self.Q).T # better nmerical stability than .invv
        self._network.fc.weight.data = Wo[0:self._network.fc.weight.shape[0],:].to(self._device)
        
        return model

    def setup_RP(self):
        M = self.args['M']
        self._network.fc.weight = nn.Parameter(torch.Tensor(self._network.fc.out_features, M).to(self._device)).requires_grad_(False) # num classes in task x M
        self._network.RP_dim = M
        self.W_rand = torch.randn(self._network.fc.in_features, M).to(self._device)
        self._network.W_rand = self.W_rand

        self.Q = torch.zeros(M, self.args["nb_classes"])
        self.G = torch.zeros(M, M)

    def optimise_ridge_parameter(self, Features, Y):
        ridges = 10.0 ** np.arange(-8, 9)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge*torch.eye(G_val.size(dim=0)), Q_val).T #better nmerical stability than .inv
            Y_train_pred = Features[num_val_samples::,:] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]))
        ridge = ridges[np.argmin(np.array(losses))]
        print('selected lambda =',ridge)
        return ridge
    
    def sample_cluster_centers(self, num_clusters, method="random"):
        if method == "random":
            max_val = 1.0
            min_val = -1.0
            centers = (
                torch.rand(num_clusters, self._network.backbone.embed_dim) * (max_val - min_val)
                + min_val
            )
            return centers.cuda()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        if self._cur_task == 0:
            self._total_classnum = data_manager.nb_classes
            self._centers = self.sample_cluster_centers(self._total_classnum * self.args["train_local_robustness_num_clusters"])
            logging.info(f"[Local Robustness] Cluster centers: {self._centers.shape}")

            if self.args["train_with_buffer"]:
                buffer_size = min(
                    int(data_manager.train_set_size * self.args["train_buffer_percent"]),
                    self.args["train_buffer_size"],
                )
                logging.info(f"[Replay Buffer] Maximum Size: {buffer_size}")
                self.buffer = RandomReplayBuffer(
                    buffer_size, self.args["train_buffer_decay"], seed=self.args["seed"]
                )
            else:
                self.buffer = None

        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", )
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)

        if self.buffer is not None:
            self.buffer.update_weights()

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        self._network.to(self._device)

        if (
            not os.path.exists(self.backbone_checkpoint(self._cur_task))
            or not os.path.exists(self.head_checkpoint(self._cur_task))
            or self.args["reset"]
        ):
            self._linear.update(
                self._total_classes - self._known_classes,
                freeze_old=self.args["train_freeze_old"],
            )
            self._linear.to(self._device)

            epochs = self.args['tuned_epoch']
            base_lr = self.init_lr
            weight_decay = self.weight_decay

            parameters = [
                {
                    "params": [
                        p for p in self._network.backbone.parameters() if p.requires_grad
                    ],
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p
                        for p in self._linear.heads[self._cur_task].parameters()
                        if p.requires_grad
                    ],
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                },
            ]
            if not self.args["train_freeze_old"]:
                parameters.append(
                    {
                        "params": [
                            p
                            for i, head in enumerate(self._linear.heads)
                            if i != self._cur_task
                            for p in head.parameters()
                            if p.requires_grad
                        ],
                        "lr": base_lr * 1e-2,
                        "weight_decay": weight_decay * 1e-2,
                    },
                )

            optimizer = optim.SGD(parameters, momentum=0.9)
            scheduler = optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=epochs, eta_min=self.min_lr)

            self._network.train()
            self._linear.train()
            logging.info(f"[Training] Task {self._cur_task + 1}")
            logging.info(f"[Training] {self._network}")
            logging.info(f"[Training] {self._linear}")

            rb_weight = self.args["train_base_reg_weight"]
            for epoch in range(epochs):
                total_ce_loss, total_rb_loss, total_acc, total = 0, 0, 0, 0

                for _, (_, x, y) in enumerate(train_loader):
                    x, y = x.to(self._device), y.to(self._device)

                    if self.args["train_with_buffer"] and self.buffer.size > 0:
                        x_buf, z_buf, y_buf = self.buffer.sample(batch_size=x.size(0))
                        x_buf, z_buf, y_buf = x_buf.to(self._device), z_buf.to(self._device), y_buf.to(self._device)
                        x = torch.cat([x, x_buf], dim=0)
                        y = torch.cat([y, y_buf], dim=0)

                    features = self._network.extract_vector(x)

                    if self.args["train_freeze_old"]:
                        y = torch.where(
                            y - self._known_classes >= 0, y - self._known_classes, -100
                        )
                        logits = self._linear.heads[self._cur_task](features)
                    else:
                        logits = self._linear(features)["logits"]

                    ce_loss = F.cross_entropy(logits, y)

                    # calculate local robustness loss ===
                    rb_loss = torch.tensor(0.0, device=x.device)
                    with torch.no_grad():
                        sim_matrix = torch.matmul(features, self._centers.T)  # dot product
                        nearest_idx = sim_matrix.argmax(dim=1)  # [B]

                    num_clusters = self._centers.shape[0]
                    total_samples = x.size(0)
                    # select a subset of clusters to compute the loss
                    subset_size = min(self._total_classes - self._known_classes, num_clusters)
                    selected_clusters = torch.randperm(num_clusters, device=x.device)[:subset_size]

                    for i in selected_clusters:
                        cluster_mask = nearest_idx == i
                        if cluster_mask.sum() == 0:
                            continue
                        cluster_features = features[cluster_mask]
                        center_features = self._centers[i].unsqueeze(0).expand_as(cluster_features)

                        cluster_size = cluster_mask.sum()
                        cluster_logits = logits[cluster_mask]
                        if self.args["train_freeze_old"]:
                            center_logits = self._linear.heads[-1](center_features)
                        else:
                            center_logits = self._linear(center_features)["logits"]
                        rb_loss += (1 / cluster_size) * F.mse_loss(cluster_logits, center_logits)

                    rb_loss = rb_loss / total_samples
                    loss = ce_loss + rb_weight * rb_loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_ce_loss += ce_loss.item() * len(y)
                    total_rb_loss += rb_loss.item() * len(y)
                    total_acc += (logits.argmax(dim=1) == y).sum().item()
                    total += len(y)

                scheduler.step()

                if epoch % 5 == 4:
                    logging.info(
                        f"[Training] Epoch {epoch + 1}/{epochs}, "
                        f"CE Loss: {total_ce_loss / total:.4f}, "
                        f"Reg Loss: {total_rb_loss / total:.4f}, "
                        f"Total Loss: {(total_ce_loss + rb_weight * total_rb_loss) / total:.4f}, "
                        f"Acc: {total_acc / total:.4f}"
                    )

            torch.save(
                self._network.get_backbone_trainable_params(),
                self.backbone_checkpoint(self._cur_task),
            )
            torch.save(
                self._linear.heads[-1].state_dict(),
                self.head_checkpoint(self._cur_task),
            )
        else:
            logging.info(f"[Training] Loading existing model for task {self._cur_task}")
            self._network.backbone.load_state_dict(
                torch.load(self.backbone_checkpoint(self._cur_task)), strict=False
            )
            self._network.to(self._device)
            
            self._linear.update(
                self._total_classes - self._known_classes,
                freeze_old=self.args["train_freeze_old"],
            )
            self._linear.heads[-1].load_state_dict(
                torch.load(self.head_checkpoint(self._cur_task)), strict=True
            )
            self._linear.to(self._device)

        if self.args["model_merge"] != "none":
            self.merge()
        
        if self.args["train_with_buffer"]:
            self._network.eval()
            with torch.no_grad():
                for _, batch in enumerate(train_loader):
                    _, x, y = batch
                    x, y = x.to(self._device), y.to(self._device)
                    features = self._network.extract_vector(x)
                    self.buffer.add(x, features, y)
            logging.info(f"[Replay Buffer] Size: {self.buffer.size}")
            logging.info(f"[Replay Buffer] Size by class: {self.buffer.size_by_class}")

        if self._cur_task == 0 and self.args["use_RP"]:
            self.setup_RP()
        self.replace_fc(train_loader_for_protonet, self._network, None)
    
    def merge(self):
        logging.info(f"[Model Merging] Method {self.args['model_merge']}")
        base_params = torch.load(self.backbone_checkpoint(-1))
        num_merged_params = sum(param.numel() for param in base_params.values())
        logging.info(f"[Model Merging] Merging with {num_merged_params:,} total parameters")

        task_params = [torch.load(self.backbone_checkpoint(task)) for task in range(self._cur_task + 1)]
        backbone_params = merge(
            base_params,
            task_params,
            method=self.args["model_merge"],
            lamb=self.args["model_merge_coef"],
            topk=self.args["model_merge_topk"],
        )
        self._network.backbone.load_state_dict(backbone_params, strict=False)
    
    def prefix(self):
        prefix_parts = [
            str(self.args['seed']),
            self.args['dataset'], 
            self.args['model_name'],
            self.args['backbone_type'],
            self.args['train_ca_method']
        ]
        return "_".join(prefix_parts)

    def backbone_checkpoint(self, task=-1):
        filename = f"{self.prefix()}_backbone" + (f"_{task}.pt" if task >= 0 else "_base.pt")
        return os.path.join(self.CHECKPOINT_DIR, filename)

    def head_checkpoint(self, task):
        filename = f"{self.prefix()}_head_{task}.pt"
        return os.path.join(self.CHECKPOINT_DIR, filename)