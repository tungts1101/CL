import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
import os
from utils.inc_net import SimpleVitNet, EaseNet, SLCANet, MOSNet


EPSILON = 1e-8
batch_size = 64

CHECKPOINT_DIR = "./checkpoints/"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self.args = args
        self._cls2task = {}

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim
    
    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def tsne(self,showcenters=False,Normalize=False):
        import umap
        import matplotlib.pyplot as plt
        print('now draw tsne results of extracted features.')
        tot_classes=self._total_classes
        test_dataset = self.data_manager.get_dataset(np.arange(0, tot_classes), source='test', mode='test')
        valloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        vectors, y_true = self._extract_vectors(valloader)
        if showcenters:
            fc_weight=self._network.fc.proj.cpu().detach().numpy()[:tot_classes]
            print(fc_weight.shape)
            vectors=np.vstack([vectors,fc_weight])
        
        if Normalize:
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit_transform(vectors)
        
        if showcenters:
            clssscenters=embedding[-tot_classes:,:]
            centerlabels=np.arange(tot_classes)
            embedding=embedding[:-tot_classes,:]
        scatter=plt.scatter(embedding[:,0],embedding[:,1],c=y_true,s=20,cmap=plt.cm.get_cmap("tab20"))
        plt.legend(*scatter.legend_elements())
        if showcenters:
            plt.scatter(clssscenters[:,0],clssscenters[:,1],marker='*',s=50,c=centerlabels,cmap=plt.cm.get_cmap("tab20"),edgecolors='black')
        
        plt.savefig(str(self.args['model_name'])+str(tot_classes)+'tsne.pdf')
        plt.close()

    def prefix(self):
        prefix_parts = [
            str(self.args["seed"]),
            self.args["dataset"],
            str(self.args["init_cls"]),
            self.args["model_name"],
            "lca",
        ]
        return "_".join(prefix_parts)

    def checkpoint_path(self, task):
        filename = "{}_{}.pkl".format(self.prefix(), task)
        return os.path.join(CHECKPOINT_DIR, filename)

    def save_checkpoint(self, filename, inc_save_dict=None):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        if inc_save_dict is not None:
            save_dict.update(inc_save_dict)
        torch.save(save_dict, filename)

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, self.args["init_cls"], self.args["increment"])
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass
    
    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
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

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []

        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy(
                        self._network.module.extract_vector(_inputs.to(self._device))
                    )
                else:
                    _vectors = tensor2numpy(
                        self._network.extract_vector(_inputs.to(self._device))
                    )

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
    
    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means

        def _compute_class_mean(self, data_manager, check_diff=False, oracle=False):
            if hasattr(self, '_class_means') and self._class_means is not None and not check_diff:
                ori_classes = self._class_means.shape[0]
                assert ori_classes == self._known_classes
                new_class_means = np.zeros((self._total_classes, self.feature_dim))
                new_class_means[:self._known_classes] = self._class_means
                self._class_means = new_class_means
                # new_class_cov = np.zeros((self._total_classes, self.feature_dim, self.feature_dim))
                new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
                new_class_cov[:self._known_classes] = self._class_covs
                self._class_covs = new_class_cov
            elif not check_diff:
                self._class_means = np.zeros((self._total_classes, self.feature_dim))
                # self._class_covs = np.zeros((self._total_classes, self.feature_dim, self.feature_dim))
                self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))

            for class_idx in range(self._known_classes, self._total_classes):

                data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx + 1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)

                try:
                    assert vectors.shape[0] > 1
                except AssertionError as e:
                    print("Size of the {}-th class is: {}, repeat it for twice.".format(class_idx, vectors.shape[0]))
                    vectors = np.tile(vectors, (2, 1))
                    print("Shape of vectors after repeating: {}".format(vectors.shape))

                # vectors = np.concatenate([vectors_aug, vectors])

                class_mean = np.mean(vectors, axis=0)
                # class_cov = np.cov(vectors.T)
                # try:
                #     class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T) + torch.eye(class_mean.shape[-1]) * 1e-4
                # except UserWarning as e:
                #     logging.warning("Caught UserWarning: ", e)
               
                self._class_means[class_idx, :] = class_mean
                self._class_covs[class_idx, ...] = class_cov

    def classifier_alignment(self, data_manager):
        self._network.to(self._device)
        # if len(self._multiple_gpus) > 1:
        #     self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._network.eval()

        logging.info(
            f"[Alignment] Compute class mean and cov for classes {self._known_classes} - {self._total_classes - 1}"
        )
        total_class = self._total_classes
        feature_dim = self._network.feature_dim
        if not hasattr(self, "_ca_class_means") or not hasattr(self, "_ca_class_covs"):
            self._ca_class_means = torch.zeros((total_class, feature_dim))
            self._ca_class_covs = torch.zeros((total_class, feature_dim, feature_dim))
        else:
            new_ca_class_means = torch.zeros((total_class, feature_dim))
            new_ca_class_means[: self._known_classes] = self._ca_class_means
            self._ca_class_means = new_ca_class_means
            new_ca_class_covs = torch.zeros((total_class, feature_dim, feature_dim))
            new_ca_class_covs[: self._known_classes] = self._ca_class_covs
            self._ca_class_covs = new_ca_class_covs

        for cls_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(cls_idx, cls_idx+1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=64, shuffle=False, num_workers=4)
            # vectors, _ = self._extract_vectors(idx_loader)

            vectors = []
            for _, _inputs, _targets in idx_loader:
                if isinstance(self._network, MOSNet):
                    _vectors = self._network(_inputs.to(self._device), adapter_id=self._cur_task, train=True)["features"]
                elif isinstance(self._network, SLCANet):
                    _vectors = self._network(_inputs.to(self._device), bcb_no_grad=True, fc_only=False)["features"]
                elif isinstance(self._network, EaseNet):
                    _vectors = self._network(_inputs.to(self._device))["features"]
                elif isinstance(self._network, SimpleVitNet):
                    _vectors = self._network(_inputs.to(self._device))["features"]
                _vectors = _vectors.detach().cpu().numpy()
                vectors.append(_vectors)

            vectors = np.concatenate(vectors)
            class_mean = np.mean(vectors, axis=0)
            class_cov = np.cov(vectors.T) + np.eye(class_mean.shape[-1]) * 1e-4            

            class_mean = torch.tensor(class_mean, dtype=torch.float64)
            class_cov = torch.tensor(class_cov, dtype=torch.float64)

            self._ca_class_means[cls_idx, :] = class_mean
            self._ca_class_covs[cls_idx, ...] = class_cov
        
        for clz in range(self._known_classes, self._total_classes):
            self._cls2task[clz] = self._cur_task
        
        if self._cur_task == 0:
            return
        
        from torch.distributions import MultivariateNormal
        import torch.nn.functional as F
        from torch import optim
        from tqdm import tqdm

        # Create optimizer
        ca_epochs = self.args.get("crct_epochs", 10)
        ca_lr = self.args.get("ca_lr", 0.005)

        if isinstance(self._network, MOSNet):
            logging.info("[Alignment] Finetune the backbone (ViT + Classifier) with MOS")
            param_list = [p for n, p in self._network.backbone.named_parameters() if p.requires_grad and 'adapter' not in n]
        else:
            for p in self._network.fc.parameters():
                p.requires_grad=True
            logging.info("[Alignment] Finetune the classifier only")
            param_list = [p for p in self._network.fc.parameters() if p.requires_grad]

        logging.info(f"[Alignment] Total trainable parameters: {sum(p.numel() for p in param_list):,}")
        network_params = [{'params': param_list, 'lr': ca_lr, 'weight_decay': 5e-4}]
        optimizer = optim.SGD(network_params, lr=ca_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=ca_epochs)

        robust_weight_base = self.args.get("ca_robust_weight", 0.0)
        entropy_weight = self.args.get("ca_entropy_weight", 0.0)
        logit_norm = self.args.get("ca_logit_norm", 0.0)

        # Sample data by using gaussian dist
        sampled_data = []
        sampled_label = []
        num_sampled_pcls = self.args.get("ca_sample_per_cls", 256)
        batch_size = self.args.get("ca_batch_size", 64)

        for class_idx in range(self._total_classes):
            mean = self._ca_class_means[class_idx].to(self._device)
            cov = self._ca_class_covs[class_idx].to(self._device)

            # m = MultivariateNormal(mean.float(), cov.float())
            # sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))

            try:
                m = MultivariateNormal(mean, covariance_matrix=cov)
                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
            except Exception as e:
                logging.warning(f"[Sampling] Invalid covariance at class {class_idx}, fallback to diag. Error: {e}")
                diag_cov = torch.diag(torch.clamp(torch.diag(cov), min=1e-6))
                m = MultivariateNormal(mean, covariance_matrix=diag_cov)
                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))

            sampled_data_single = torch.nan_to_num(
                sampled_data_single, nan=0.0, posinf=1e6, neginf=-1e6
            )

            sampled_data.append(sampled_data_single)
            sampled_label.extend([class_idx] * num_sampled_pcls)

        sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
        sampled_label = torch.tensor(sampled_label).long().to(self._device)

        inputs = sampled_data
        targets = sampled_label

        prog_bar = tqdm(range(ca_epochs))
        for _ in prog_bar:
            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            total_loss = total = 0
            total_ce_loss = total_rb_loss = total_entropy_loss = 0
            total_acc = 0

            for i in range(0, len(sampled_data), batch_size):
                x = sampled_data[i : i + batch_size]
                y = sampled_label[i : i + batch_size]

                if isinstance(self._network, MOSNet):
                    outputs = self._network(x, fc_only=True)
                    logits = outputs['logits'][:, :self._total_classes]
                elif isinstance(self._network, SLCANet):
                    outputs = self._network(x, bcb_no_grad=True, fc_only=True)
                    logits = outputs['logits']
                elif isinstance(self._network, EaseNet):
                    outputs = self._network(x)
                    logits = outputs['logits']
                elif isinstance(self._network, SimpleVitNet):
                    outputs = self._network(x)
                    logits = outputs['logits']
                
                logits = torch.clamp(logits, -5, 5)

                if logit_norm != 0:
                    batch_size = logits.size(0)
                    num_tasks = self._cur_task + 1
                    
                    # Compute per-task norms for averaging
                    task_norms = torch.zeros(batch_size, num_tasks, device=logits.device)
                    
                    for task in range(num_tasks):
                        # Get class indices for this task
                        cls_indices = [clz for clz in self._cls2task if self._cls2task[clz] == task]
                        if cls_indices:
                            task_logits = logits[:, cls_indices]  # (batch_size, num_classes_in_task)
                            task_norms[:, task] = torch.norm(task_logits, p=2, dim=-1) + 1e-7
                    
                    # Average norms across all tasks
                    avg_norms = task_norms.sum(dim=-1) / num_tasks  # Average across all tasks
                    avg_norms = avg_norms.unsqueeze(-1)  # (batch_size, 1)
                    
                    # Apply normalization: logits / avg_norm / logit_norm_factor
                    normalized_logits = logits / (avg_norms + 1e-7) / logit_norm
                    loss_vec = F.cross_entropy(normalized_logits, y, reduction="none")
                else:
                    loss_vec = F.cross_entropy(logits, y, reduction="none")

                if robust_weight_base == 0 and entropy_weight == 0:
                    loss = loss_vec.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    bs = len(y)
                    total_loss += loss.item() * bs
                    total_ce_loss += loss.item() * bs
                    total_rb_loss += 0
                    total_entropy_loss += 0
                    total_acc += (logits.argmax(dim=1) == y).sum().item()
                    total += bs
                else:
                    L_total = torch.tensor(0.0, device=x.device)  # L = Σ Li
                    total_term1 = torch.tensor(0.0, device=x.device)  # For logging: sum of all term1
                    total_term2 = torch.tensor(0.0, device=x.device)  # For logging: sum of all term2
                    total_term3 = torch.tensor(0.0, device=x.device)  # For logging: sum of all term3 (entropy)
                    
                    unique_classes = torch.unique(y)
                    class_dist = torch.cdist(x, self._ca_class_means[:self._total_classes].to(self._device))
                    class_indices = torch.argmin(class_dist, dim=1)
                    for class_i in unique_classes:
                        label_mask = (y == class_i)
                        distance_mask = (class_indices == class_i)
                        class_mask = distance_mask & label_mask
                        
                        # Get the samples that belong to this class
                        class_samples = torch.where(class_mask)[0]
                        
                        # If no samples meet the conditions, fall back to label-only (term1 only)
                        if len(class_samples) == 0:
                            # Fall back to using only label condition for term1
                            label_only_samples = torch.where(label_mask)[0]
                            if len(label_only_samples) == 0:
                                continue  # Skip if no samples with this label at all
                            
                            label_losses = loss_vec[label_mask]
                            term1 = label_losses.mean()
                            term2 = torch.tensor(0.0).cuda()
                            term3 = torch.tensor(0.0).cuda()
                        else:
                            class_losses = loss_vec[class_mask]
                            term1 = class_losses.mean()
                            
                            # Second term: E_{x,x'~Ni}[|ℓ(yi, ht+1(x)) - ℓ(yi, ht+1(x'))|] where x,x' ∈ Ai
                            if len(class_samples) >= 2:
                                pairwise_diffs = torch.abs(
                                    class_losses.unsqueeze(1) - class_losses.unsqueeze(0)
                                )
                                # Remove diagonal (self-comparisons)
                                mask = ~torch.eye(len(class_losses), dtype=torch.bool, device=x.device)
                                pairwise_diffs = pairwise_diffs[mask]
                                term2 = pairwise_diffs.mean()
                            else:
                                term2 = torch.tensor(0.0, device=x.device)
                            
                            # Third term: Cluster entropy minimization
                            if len(class_samples) >= 1 and entropy_weight != 0:
                                cluster_logits = logits[class_mask]  # Shape: (n_cluster_samples, n_classes)
                                cluster_probs = F.softmax(cluster_logits, dim=1)  # Shape: (n_cluster_samples, n_classes)
                                
                                # Compute entropy for each sample: -Σ p_i * log(p_i)
                                # Add small epsilon to prevent log(0)
                                cluster_entropy = -torch.sum(cluster_probs * torch.log(cluster_probs + 1e-8), dim=1)
                                term3 = cluster_entropy.mean()  # Average entropy across cluster samples
                            else:
                                term3 = torch.tensor(0.0, device=x.device)
                        
                        Li = term1 + robust_weight_base * term2 + entropy_weight * term3
                        L_total += Li
                        total_term1 += term1
                        total_term2 += robust_weight_base * term2
                        total_term3 += entropy_weight * term3

                    num_classes_in_batch = len(unique_classes)
                    if num_classes_in_batch > 0:
                        loss = L_total / num_classes_in_batch
                    else:
                        loss = loss_vec.mean()  # fallback
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    bs = len(y)
                    
                    # Average the terms by number of classes to get per-sample equivalent
                    if num_classes_in_batch > 0:
                        avg_term1 = total_term1 / num_classes_in_batch
                        avg_term2 = total_term2 / num_classes_in_batch
                        avg_term3 = total_term3 / num_classes_in_batch
                        avg_loss = L_total / num_classes_in_batch
                    else:
                        avg_term1 = torch.tensor(0.0, device=x.device)
                        avg_term2 = torch.tensor(0.0, device=x.device)
                        avg_term3 = torch.tensor(0.0, device=x.device)
                        avg_loss = loss_vec.mean()
                    
                    total_loss += avg_loss.item() * bs
                    total_ce_loss += avg_term1.item() * bs
                    total_rb_loss += avg_term2.item() * bs
                    total_entropy_loss += avg_term3.item() * bs
                    total_acc += (logits.argmax(dim=1) == y).sum().item()
                    total += bs

            scheduler.step()

            info = f"[Alignment] "
            info += f"Base Loss: {total_ce_loss/total:.4f}, "
            info += f"Robust Term: {total_rb_loss/total:.4f}, "
            info += f"Entropy Term: {total_entropy_loss/total:.4f}, "
            info += f"Total LCA Loss: {total_loss/total:.4f}, "
            info += f"Accuracy: {total_acc/total:.4f}"

            prog_bar.set_description(info)

        logging.info(info)
