import logging
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os
from .utils import merge
from .merging import BaseMergingLearner
import copy
from utils.data_manager import DataManager


num_workers = 8
EPSILON = 1e-8


class AlignmentLearner(BaseMergingLearner):
    def get_subset_per_class(self, dataset, percent=0.1, seed=42):
        """
        Returns a Subset of the dataset containing percent (e.g., 0.1 for 10%) of samples per class.
        Assumes dataset[i][-1] is the class label.
        """
        from collections import defaultdict
        import random
        from torch.utils.data import Subset
        random.seed(seed)
        class_indices = defaultdict(list)
        for i in range(len(dataset)):
            label = dataset[i][-1]
            class_indices[label].append(i)
        subset_indices = []
        for indices in class_indices.values():
            n = max(1, int(len(indices) * percent))
            subset_indices.extend(random.sample(indices, n))
        return Subset(dataset, subset_indices)
    
    def post_training(self):
        train_ca_method = self.args.get("train_ca_method", "sgd")
        if train_ca_method in ["sgd", "nes"]:
            logging.info(
                f"[Alignment] Compute class mean and cov for classes {self._known_classes} - {self._total_classes - 1}"
            )
            total_class = self._total_classes
            feature_dim = self._network.feature_dim
            if not hasattr(self, "_cls_means") or not hasattr(self, "_cls_covs"):
                self._cls_means = torch.zeros((total_class, feature_dim))
                self._cls_covs = torch.zeros((total_class, feature_dim, feature_dim))
            else:
                new_cls_means = torch.zeros((total_class, feature_dim))
                new_cls_means[: self._known_classes] = self._cls_means
                self._cls_means = new_cls_means
                new_cls_covs = torch.zeros((total_class, feature_dim, feature_dim))
                new_cls_covs[: self._known_classes] = self._cls_covs
                self._cls_covs = new_cls_covs

            for cls_idx in range(self._known_classes, self._total_classes):
                proto_set = self.data_manager.get_dataset(
                    np.arange(cls_idx, cls_idx + 1), source="train", mode="test"
                )
                proto_loader = DataLoader(
                    proto_set, batch_size=512, shuffle=False, num_workers=4
                )

                features_list = []
                self._network.eval()
                with torch.no_grad():
                    for _, (_, x, _) in enumerate(proto_loader):
                        x = x.cuda()
                        f = self._network.get_features(x)
                        features_list.append(f.cpu())

                features_list = torch.cat(features_list, dim=0)
                class_mean = torch.mean(features_list, dim=0)
                class_cov = (
                    torch.cov(features_list.T) + torch.eye(class_mean.shape[-1]) * 1e-4
                )

                self._cls_means[cls_idx, :] = class_mean
                self._cls_covs[cls_idx, ...] = class_cov
        
        if self.args["train_merge_method"] != "none":
            self.merge()
        
        if self._cur_task == 0:
            return

        if (
            os.path.exists(self.head_alignment_checkpoint(self._cur_task))
            and not self.args["reset"]
        ):
            logging.info(
                f"[Merging] Load existing alignment checkpoint"
            )
            self._network.fc.load_state_dict(
                torch.load(self.head_alignment_checkpoint(self._cur_task)), strict=True
            )
        else:
            if train_ca_method == "sgd":
                self._align_sgd()
            elif train_ca_method == "nes":
                self._align_nes()
            elif train_ca_method == "projection":
                self._align_projection()
            else:
                raise ValueError(f"Unknown alignment method {train_ca_method}")

            torch.save(
                self._network.fc.state_dict(),
                self.head_alignment_checkpoint(self._cur_task),
            )

    def _align_sgd(self):
        epochs = self.args.get("train_ca_epochs", 10)
        samples_per_class = self.args.get("train_ca_samples_per_class", 256)
        batch_size = self.args.get("train_ca_batch_size", 64)
        lr = self.args.get("train_ca_lr", 1e-2)
        weight_decay = self.args.get("train_ca_weight_decay", 5e-4)
        task_decay_factor = self.args.get("train_ca_task_decay_factor", 0.5)

        for p in self._network.fc.parameters():
            p.requires_grad = True

        param_groups = []
        total_head_params = 0
        for task_id in range(self._cur_task + 1):
            task_params = [p for p in self._network.fc.heads[task_id].parameters() if p.requires_grad]
            if task_params:
                task_age = self._cur_task - task_id
                task_lr_multiplier = task_decay_factor ** task_age
                task_lr = lr * task_lr_multiplier
                
                param_groups.append({
                    "params": task_params,
                    "lr": task_lr,
                    "weight_decay": weight_decay
                })
                
                task_param_count = sum(p.numel() for p in task_params)
                total_head_params += task_param_count

        logging.info(f"[Alignment] Training {total_head_params:,} head parameters")

        self._network.train()
        optimizer = optim.SGD(
            param_groups, lr=lr, weight_decay=weight_decay, momentum=0.9
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=epochs
        )

        # Sample data from Gaussian distribution ------------------------------
        sampled_data, sampled_label = [], []
        for cls_idx in range(self._total_classes):
            cls_mean = self._cls_means[cls_idx].cuda()
            cls_cov = self._cls_covs[cls_idx].cuda()

            m = MultivariateNormal(cls_mean.float(), cls_cov.float())

            sampled_features = m.sample((samples_per_class,))
            sampled_data.append(sampled_features)
            sampled_label.extend([cls_idx] * samples_per_class)

        sampled_data = torch.cat(sampled_data, dim=0).float().cuda()
        sampled_label = torch.tensor(sampled_label).long().cuda()

        for epoch in range(epochs):
            indexes = torch.randperm(sampled_data.size(0))
            sampled_data = sampled_data[indexes]
            sampled_label = sampled_label[indexes]

            total_loss, total, total_acc = 0, 0, 0

            for i in range(0, len(sampled_data), batch_size):
                x = sampled_data[i : i + batch_size]
                y = sampled_label[i : i + batch_size]

                logits = self._network.fc(x)["logits"]
                loss = F.cross_entropy(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = len(y)
                total_loss += loss.item() * bs
                total_acc += (logits.argmax(dim=1) == y).sum().item()
                total += bs

            scheduler.step()

            if epoch % 5 == 4 or epoch == epochs - 1:
                logging.info(
                    f"[Alignment] Epoch {epoch+1}/{epochs}, "
                    f"Total Loss: {total_loss/total:.4f}, "
                    f"Total Accuracy: {total_acc/total:.4f}"
                )

    def _align_nes(self):
        samples_per_class = self.args.get("train_ca_samples_per_class", 256)

        # Sample data from Gaussian distribution
        sampled_data, sampled_label = [], []
        for cls_idx in range(self._total_classes):
            cls_mean = self._cls_means[cls_idx].cuda()
            cls_cov = self._cls_covs[cls_idx].cuda()

            m = MultivariateNormal(cls_mean.float(), cls_cov.float())

            sampled_features = m.sample((samples_per_class,))
            sampled_features += torch.randn_like(sampled_features) * 0.1
            sampled_data.append(sampled_features)
            sampled_label.extend([cls_idx] * samples_per_class)

        sampled_data = torch.cat(sampled_data, dim=0).float().cuda()
        sampled_label = torch.tensor(sampled_label).long().cuda()

        @torch.no_grad()
        def get_accuracy():
            total_acc = 0
            for i in range(0, len(sampled_data), 512):
                logits = self._network.fc(sampled_data[i : i + 512])["logits"]
                total_acc += (logits.argmax(dim=1) == sampled_label[i : i + 512]).sum().item()
            return total_acc / len(sampled_data)

        lambda_cfg  = self.args.get("train_ca_lambda_task", {})
        default_old = self.args.get("train_ca_lambda_old_default", 5e-3)
        default_cur = self.args.get("train_ca_lambda_cur_default", 1e-3)

        lambda_task = {}
        for t in range(self._cur_task + 1):
            lam = lambda_cfg.get(str(t), lambda_cfg.get(t, default_cur if t == self._cur_task else default_old))
            lambda_task[t] = float(lam)

        # Build initial genome ------------------------------------------------
        head_params = []
        param_shapes = []
        head_param_slices = {} 
        offset = 0

        for task_idx in range(self._cur_task + 1):
            head = self._network.fc.heads[task_idx]
            head_start = offset
            for p in head.parameters():
                arr = p.data.detach().cpu().numpy().flatten()
                head_params.append(arr)
                param_shapes.append(p.shape)
                offset += arr.size
            head_end = offset
            head_param_slices[task_idx] = (head_start, head_end)

        theta = np.concatenate(head_params).astype(np.float32)
        original_solution = theta.copy()

        def objective_function(params_flat: np.ndarray) -> float:
            try:
                param_idx = 0
                original_params_snap = {}
                for task_idx in range(self._cur_task + 1):
                    head = self._network.fc.heads[task_idx]
                    original_params_snap[task_idx] = []
                    for _, p in head.named_parameters():
                        original_params_snap[task_idx].append(p.data.clone())
                        sz = p.numel()
                        new_data = params_flat[param_idx:param_idx + sz]
                        param_idx += sz
                        p.data = torch.from_numpy(new_data).float().cuda().view(p.shape)

                # Accuracy                
                acc = get_accuracy()

                # # MSE Trust Region Penalty
                # delta = params_flat - original_solution
                # trust_region = 0.0
                # for t in range(self._cur_task + 1):
                #     s, e = head_param_slices[t]
                #     dt = delta[s:e]
                #     n_t = max((e - s), 1)
                #     penalty = float(np.dot(dt, dt) / n_t)
                #     trust_region += penalty

                # print(f"Acc {acc}, TR {trust_region}")
                # loss = -acc + 1000 * trust_region

                loss = -acc

                for task_idx in range(self._cur_task + 1):
                    head = self._network.fc.heads[task_idx]
                    for p, w in zip(head.parameters(), original_params_snap[task_idx]):
                        p.data = w

                return float(loss)

            except Exception as e:
                logging.warning(f"[Alignment] Error in objective function: {e}")
                return 1.0

        # Build per task sigma_vec with adaptive scheduling -------------------
        base_sigma_init = float(self.args.get("train_ca_nes_sigma_init", 1e-3))
        base_sigma_final = float(self.args.get("train_ca_nes_sigma_final", 1e-4))
        sigma_min  = float(self.args.get("train_ca_nes_sigma_min", 1e-5))
        sigma_max  = float(self.args.get("train_ca_nes_sigma_max", 5e-2))

        def lambda_to_scale(lmb):
            return 1.0 / np.sqrt(max(lmb, 1e-12))

        sigma_vec_base = np.empty_like(theta, dtype=np.float32)
        for t in range(self._cur_task + 1):
            s, e = head_param_slices[t]
            scale_t = lambda_to_scale(lambda_task[t])
            sigma_vec_base[s:e] = scale_t

        def get_sigma_vec(iteration, total_iters):
            decay_factor = (base_sigma_final / base_sigma_init) ** (iteration / max(total_iters - 1, 1))
            current_base_sigma = base_sigma_init * decay_factor
            sigma_vec = current_base_sigma * sigma_vec_base
            return np.clip(sigma_vec, sigma_min, sigma_max).astype(np.float32)

        # NES loop ------------------------------------------------------------
        lr    = float(self.args.get("train_ca_nes_lr", 2e-2))
        iters = int(self.args.get("train_ca_nes_iterations", 200))
        pop   = int(self.args.get("train_ca_nes_popsize", 50))

        logging.info(f"[Alignment][NES-diag] iters={iters}, pop={pop}, lr={lr}")

        best_theta = theta.copy()
        best_loss  = objective_function(best_theta)
        no_improve_steps = 0
        patience = int(self.args.get("train_ca_nes_patience", 20))

        for it in range(iters):
            sigma_vec = get_sigma_vec(it, iters)
            
            eps = np.random.randn(pop, theta.size).astype(np.float32)

            losses_pos = np.empty(pop, dtype=np.float32)
            losses_neg = np.empty(pop, dtype=np.float32)
            for i in range(pop):
                step = sigma_vec * eps[i] 
                losses_pos[i] = objective_function(theta + step)
                losses_neg[i] = objective_function(theta - step)

            deltaL = (losses_pos - losses_neg)[:, None]
            grad = (deltaL * (eps / np.maximum(sigma_vec, 1e-12))).mean(axis=0) / 2.0
            theta = theta - lr * grad.astype(np.float32)

            cur_loss = objective_function(theta)
            if cur_loss < best_loss - 1e-6:
                best_loss = cur_loss
                best_theta = theta.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            if it % patience == 0 or it == iters - 1:
                param_idx = 0
                snap = {}
                for t in range(self._cur_task + 1):
                    head = self._network.fc.heads[t]
                    snap[t] = [p.data.clone() for p in head.parameters()]
                    for p in head.parameters():
                        sz = p.numel()
                        new_data = best_theta[param_idx:param_idx + sz]
                        param_idx += sz
                        p.data = torch.from_numpy(new_data).float().cuda().view(p.shape)

                acc = get_accuracy()

                for t in range(self._cur_task + 1):
                    head = self._network.fc.heads[t]
                    for p, w in zip(head.parameters(), snap[t]):
                        p.data = w
                
                current_sigma_range = f"[{sigma_vec.min():.3e}, {sigma_vec.max():.3e}]"
                logging.info(f"[Alignment][NES-diag] iter {it}: best_loss={best_loss:.6f}, acc={acc:.4f}, sigma_range={current_sigma_range}")

            if no_improve_steps >= patience:
                logging.info(f"[Alignment][NES-diag] Early stop at iter {it} (no improvement for {patience}).")
                break

        # Apply best solution -------------------------------------------------
        param_idx = 0
        for t in range(self._cur_task + 1):
            head = self._network.fc.heads[t]
            for p in head.parameters():
                sz = p.numel()
                new_data = best_theta[param_idx:param_idx + sz]
                param_idx += sz
                p.data = torch.from_numpy(new_data).float().cuda().view(p.shape)

    def _align_projection(self):
        if self._cur_task == 0: return

        def get_fc(head):
            fc = None
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    fc = m
            if fc is None:
                raise RuntimeError("No nn.Linear found in head")
            return fc

        def stitch_classifier():
            feature_dim = self._network.feature_dim             
            total_classes = self._total_classes
            known_classes = self._known_classes

            W = torch.zeros((feature_dim, total_classes))    
            b = torch.zeros((total_classes,)) if self._network.fc.with_bias else None

            for i in range(self._cur_task):
                head_i = self._network.fc.heads[i]         
                fc = get_fc(head_i)

                s, e = self._class_increments[i]

                W[:, s:e+1] = fc.weight.data.detach().cpu().t()

                if b is not None:
                    if fc.bias is None:
                        raise RuntimeError("with_bias=True but fc.bias is None")
                    b[s:e+1] = fc.bias.data.detach().cpu()

            W_prev = W[:, :known_classes]
            W_curr = W[:, known_classes:total_classes]
            if b is None:
                b_prev = b_curr = None
            else:
                b_prev = b[:known_classes]
                b_curr = b[known_classes:total_classes]

            return (W, b), (W_prev, b_prev), (W_curr, b_curr)

        @torch.no_grad()
        def collect_features(backbone, loader):
            features = []
            for _, x, _ in loader:
                z = backbone(x.cuda())
                z = self._network.norm(z)
                features.append(z.cpu())
            return torch.cat(features, dim=0)
    
        def procrustes(Z_src, Z_tgt):
            M = Z_src.T @ Z_tgt
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            return U @ Vh
        

        def _auto_lambda_simple(X, Y):
            X = X.detach().to(dtype=torch.float32)
            Y = Y.detach().to(dtype=torch.float32)
            device = X.device
            n = X.shape[0]

            # # handle tiny datasets
            # if n < 3:
            #     return 1e-1

            # shuffle & split
            idx = torch.randperm(n, device=device)
            X = X[idx]; Y = Y[idx]
            n_tr = max(1, int(0.8 * n))
            X_tr, X_val = X[:n_tr], X[n_tr:]
            Y_tr, Y_val = Y[:n_tr], Y[n_tr:]
            if X_val.shape[0] == 0:  # ensure non-empty val set
                X_tr, X_val = X[:-1], X[-1:].contiguous()
                Y_tr, Y_val = Y[:-1], Y[-1:].contiguous()

            G = X_tr.T @ X_tr                 # (d,d)
            Q = X_tr.T @ Y_tr                 # (d,k)
            I = torch.eye(G.size(0), dtype=G.dtype, device=device)
            lambdas = (torch.tensor(10.0, device=device) ** torch.arange(-3, 6, device=device)).float()

            best_lam, best_loss = float(lambdas[0].item()), float("inf")
            for lam in lambdas:
                W = torch.linalg.solve(G + lam * I, Q)     # (d,k)
                Y_hat = X_val @ W                           # (m,k)
                loss = F.mse_loss(Y_hat, Y_val).item()
                if loss < best_loss:
                    best_loss, best_lam = loss, float(lam.item())
            return best_lam

        def fit_ridge_map(ZM, Zi, identity_prior = False):
            ZM = ZM.float().detach()
            Zi = Zi.float().detach()
            device = ZM.device
            n, dM = ZM.shape
            dS = Zi.shape[1]

            lam = _auto_lambda_simple(ZM, Zi)

            G = ZM.T @ ZM                        # (dM,dM)
            H = ZM.T @ Zi                        # (dM,dS)
            I = torch.eye(dM, dtype=G.dtype, device=device)

            # identity prior nudges A toward I when dM==dS
            if identity_prior and dM == dS:
                H = H + lam * I

            A = torch.linalg.solve(G + lam * I, H)       # (dM,dS)
            c = Zi.mean(0, keepdim=True) - ZM.mean(0, keepdim=True) @ A
            return A, c.squeeze(0)
        

        (W_old, _), (_, _), (_, _) = stitch_classifier()


        if not hasattr(self, "_proxy_loader"):
            logging.info("[Alignment] Creating proxy data loader for feature collection")
            proxy_data_manager = DataManager("tinyimagenet", True, self.args["seed"], 200, 0, self.args)
            proxy_set = proxy_data_manager.get_dataset(np.arange(0, 200), source="train", mode="test")
            proxy_subset = self.get_subset_per_class(proxy_set, percent=0.1, seed=self.args.get("seed", 42))
            self._proxy_loader = DataLoader(proxy_subset, batch_size=512, shuffle=False, num_workers=4)

        data_loader = self._proxy_loader

        if self.args.get("train_merge_incremental", False):
            prev_backbone = copy.deepcopy(self._network.backbone).cuda().eval()
            curr_backbone = copy.deepcopy(self._network.backbone).cuda().eval()

            prev_backbone.load_state_dict(torch.load(self.merge_checkpoint(self._cur_task - 1)), strict=False)
            curr_backbone.load_state_dict(torch.load(self.backbone_checkpoint(self._cur_task)), strict=False)

            (_, _), (W_prev, b_prev), (W_curr, b_curr) = stitch_classifier()
            W_prev = W_prev.float().contiguous()
            W_curr = W_curr.float().contiguous()
            if b_prev is not None: b_prev = b_prev.float().contiguous()
            if b_curr is not None: b_curr = b_curr.float().contiguous()

            ZM = collect_features(self._network.backbone, data_loader) 
            ZP = collect_features(prev_backbone, data_loader) 
            ZC = collect_features(curr_backbone, data_loader)


            # R_MP = procrustes(ZM, ZP)
            # R_MC = procrustes(ZM, ZC) 

            # W_prev_rot = R_MP @ W_prev
            # W_curr_rot = R_MC @ W_curr
            # b_prev_rot = b_prev # bias not rotated
            # b_curr_rot = b_curr # bias not rotated


            R_MP, c = fit_ridge_map(ZM, ZP)   
            W_prev_rot = R_MP @ W_prev                                 
            b_prev_rot = None if b_prev is None else (c @ W_prev) + b_prev
            R_MC, c = fit_ridge_map(ZM, ZC)
            W_curr_rot = R_MC @ W_curr
            b_curr_rot = None if b_curr is None else (c @ W_curr) + b_curr

            offset_prev = 0
            for i in range(self._cur_task):
                head = self._network.fc.heads[i]
                fc = get_fc(head)

                s, e = self._class_increments[i]
                C_slice = e - s + 1

                if i < self._cur_task:
                    W_block = W_prev_rot[:, offset_prev:offset_prev + C_slice]
                    b_block = None if b_prev_rot is None else b_prev_rot[offset_prev:offset_prev + C_slice]
                    offset_prev += C_slice
                else:
                    W_block = W_curr_rot[:, :C_slice]
                    b_block = None if b_curr_rot is None else b_curr_rot[:C_slice]
                    W_curr_rot = W_curr_rot[:, C_slice:]
                    if b_curr_rot is not None:
                        b_curr_rot = b_curr_rot[C_slice:]

                fc.weight.data.copy_(W_block.t().to(fc.weight.device))

                if (fc.bias is not None) and (b_block is not None):
                    fc.bias.data.copy_(b_block.to(fc.bias.device))
        else:
            ZM = collect_features(self._network.backbone, data_loader)
            backbone = copy.deepcopy(self._network.backbone).cuda().eval()

            for i in range(self._cur_task + 1):
                sd_i = torch.load(self.backbone_checkpoint(i))
                backbone.load_state_dict(sd_i, strict=False)
                head = self._network.fc.heads[i]
                fc = get_fc(head)

                Zi   = collect_features(backbone, data_loader)
                Wi = fc.weight.data.detach().cpu().t().float()
                bi = None if fc.bias is None else fc.bias.data.detach().cpu().float()

                R_Mi = procrustes(ZM, Zi).cpu()
                Wi_rot = R_Mi @ Wi                                   # (dM, C_i)
                bi_rot = bi
                
                # A, c = fit_ridge_map(ZM, Zi)           # robust, no SVDs
                # Wi_rot = A @ Wi                                  # (dM, C_i)
                # bi_rot = None if bi is None else (c @ Wi) + bi   # (C_i,)

                fc.weight.data.copy_(Wi_rot.t().to(fc.weight.device))
                if (fc.bias is not None) and (bi_rot is not None):
                    fc.bias.data.copy_(bi_rot.to(fc.bias.device))
        

        (W_new, b_new), (_, _), (_, _) = stitch_classifier()

        # Calculate Stats -----------------------------------------------------
        def fro_rel(W_new, W_old, eps=1e-12):
            return (W_new - W_old).norm('fro') / (W_old.norm('fro') + eps)

        def colwise_cos(W_new, W_old, reduce='mean', eps=1e-12):
            a = torch.nn.functional.normalize(W_new, dim=0, eps=eps)
            b = torch.nn.functional.normalize(W_old, dim=0, eps=eps)
            cos = (a * b).sum(0)  # (C,)
            return cos.mean().item() if reduce=='mean' else cos  # return vector if reduce=None

        stats = {}
        stats['W_relF'] = fro_rel(W_old, W_new) # < 0.1
        stats['W_cos'] = colwise_cos(W_old, W_new) # ~ 0.98 - 1.00

        logging.info(f"[Alignment] Stats {stats}")

    def head_alignment_checkpoint(self, task):
        filename = f"{self.prefix()}_head_alignment_{task}.pt"
        return os.path.join(self.CHECKPOINT_DIR, filename)
