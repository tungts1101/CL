import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import SLCANet
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm
import os

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SLCANet(args, pretrained=True)
        self.batch_size = args['batch_size']
        self.epochs = args['epochs']
        self.lrate = args['lrate']
        self.lrate_decay = args['lrate_decay']
        self.weight_decay = args['weight_decay']
        self.milestones = args['milestones']
        if 'bcb_lrscale' in args.keys():
            self.bcb_lrscale = args['bcb_lrscale']
        else:
            self.bcb_lrscale = 1.0/100
        if self.bcb_lrscale == 0:
            self.fix_bcb = True
        else:
            self.fix_bcb = False
        
        self.ca_epochs = args['ca_epochs']
        
        if 'ca_with_logit_norm' in args.keys() and args['ca_with_logit_norm']>0:
            self.logit_norm = args['ca_with_logit_norm']
        else:
            self.logit_norm = None
        
        if 'save_before_ca' in args.keys() and args['save_before_ca']:
            self.save_before_ca = True
        else:
            self.save_before_ca = False
        
        self.args = args
        self.seed = args['seed']
        self.task_sizes = []
        self.cls2task = {}

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        # self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}'.format(self.seed), head_only=self.fix_bcb)
        self._network.fc.recall()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        for i in range(self._known_classes, self._total_classes):
            self.cls2task[i] = self._cur_task
        self.topk = self._total_classes if self._total_classes<5 else 5
        self._network.update_fc(data_manager.get_task_size(self._cur_task))
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        self._network.to(self._device)

        train_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                  source='train', mode='train',
                                                  appendent=[])
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        dset_name = data_manager.dataset_name.lower()

        self.train_loader = DataLoader(train_dset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        filename = self.checkpoint_path(self._cur_task)

        if os.path.exists(filename) and not self.args['reset']:
            saved = torch.load(filename)
            assert saved["tasks"] == self._cur_task
            self._network.cpu()
            self._network.load_state_dict(saved["model_state_dict"])
        else:
            self._stage1_training(self.train_loader, self.test_loader)
            self.save_checkpoint(filename)

        self._network.to(self._device)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # CA
        self._network.fc.backup()
        # if self.save_before_ca:
            # self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}_before_ca'.format(self.seed), head_only=self.fix_bcb)
        
        self._compute_class_mean(data_manager)
        if self._cur_task > 0 and self.ca_epochs > 0:
            self._stage2_compact_classifier(task_size, data_manager)
        

    def _run(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._network(inputs, bcb_no_grad=self.fix_bcb)['logits']
                cur_targets = torch.where(targets-self._known_classes>=0, targets-self._known_classes, -100)
                loss = F.cross_entropy(logits[:, self._known_classes:], cur_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()

            train_acc = self._compute_accuracy(self._network, train_loader)
            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def _stage1_training(self, train_loader, test_loader):
        '''
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        '''
        base_params = self._network.backbone.parameters()
        base_fc_params = [p for p in self._network.fc.parameters() if p.requires_grad==True]
        head_scale = 1.
        if not self.fix_bcb:
            base_params = {'params': base_params, 'lr': self.lrate*self.bcb_lrscale, 'weight_decay': self.weight_decay}
            base_fc_params = {'params': base_fc_params, 'lr': self.lrate*head_scale, 'weight_decay': self.weight_decay}
            network_params = [base_params, base_fc_params]
        else:
            for p in base_params:
                p.requires_grad = False
            network_params = [{'params': base_fc_params, 'lr': self.lrate*head_scale, 'weight_decay': self.weight_decay}]
        optimizer = optim.SGD(network_params, lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.milestones, gamma=self.lrate_decay)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._run(train_loader, test_loader, optimizer, scheduler)


    def _stage2_compact_classifier(self, task_size, data_manager):
        for p in self._network.fc.parameters():
            p.requires_grad=True

        if self.args.get("use_ori", False):
            run_epochs = self.ca_epochs
            crct_num = self._total_classes    
            param_list = [p for p in self._network.fc.parameters() if p.requires_grad]
            network_params = [{'params': param_list, 'lr': self.lrate,
                            'weight_decay': self.weight_decay}]
            optimizer = optim.SGD(network_params, lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay)
            # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[4], gamma=lrate_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

            self._network.to(self._device)
            if len(self._multiple_gpus) > 1:
                self._network = nn.DataParallel(self._network, self._multiple_gpus)

            self._network.eval()
            for epoch in range(run_epochs):
                losses = 0.

                sampled_data = []
                sampled_label = []
                num_sampled_pcls = 256
            
                for c_id in range(crct_num):
                    t_id = c_id//task_size
                    decay = (t_id+1)/(self._cur_task+1)*0.1
                    cls_mean = torch.tensor(self._class_means_slca[c_id], dtype=torch.float64).to(self._device)*(0.9+decay) # torch.from_numpy(self._class_means_slca[c_id]).to(self._device)
                    cls_cov = self._class_covs_slca[c_id].to(self._device)
                    
                    m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                    sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                    sampled_data.append(sampled_data_single)                
                    sampled_label.extend([c_id]*num_sampled_pcls)

                sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
                sampled_label = torch.tensor(sampled_label).long().to(self._device)

                inputs = sampled_data
                targets= sampled_label

                sf_indexes = torch.randperm(inputs.size(0))
                inputs = inputs[sf_indexes]
                targets = targets[sf_indexes]

                
                for _iter in range(crct_num):
                    inp = inputs[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                    tgt = targets[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                    outputs = self._network(inp, bcb_no_grad=True, fc_only=True)
                    logits = outputs['logits']

                    if self.logit_norm is not None:
                        per_task_norm = []
                        prev_t_size = 0
                        cur_t_size = 0
                        for _ti in range(self._cur_task+1):
                            cur_t_size += self.task_sizes[_ti]
                            temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                            per_task_norm.append(temp_norm)
                            prev_t_size += self.task_sizes[_ti]
                        per_task_norm = torch.cat(per_task_norm, dim=-1)
                        norms = per_task_norm.mean(dim=-1, keepdim=True)
                            
                        norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                        decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                        loss = F.cross_entropy(decoupled_logits, tgt)

                    else:
                        loss = F.cross_entropy(logits[:, :crct_num], tgt)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()

                scheduler.step()
                test_acc = self._compute_accuracy(self._network, self.test_loader)
                info = 'CA Task {} => Loss {:.3f}, Test_accy {:.3f}'.format(
                    self._cur_task, losses/self._total_classes, test_acc)
                logging.info(info)
        
        else:
            self.classifier_alignment(data_manager=data_manager)
            # # sample data
            # sampled_data = []
            # sampled_label = []
            # num_sampled_pcls = self.args.get("ca_sample_per_cls", 256)
            # batch_size = self.args.get("ca_batch_size", 64)

            # for class_idx in range(self._total_classes):
            #     mean = torch.tensor(self._class_means_slca[class_idx], dtype=torch.float64).to(self._device)
            #     cov = self._class_covs_slca[class_idx].to(self._device)

            #     m = MultivariateNormal(mean.float(), cov.float())
            #     sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))

            #     sampled_data.append(sampled_data_single)
            #     sampled_label.extend([class_idx] * num_sampled_pcls)

            # sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            # sampled_label = torch.tensor(sampled_label).long().to(self._device)

            # inputs = sampled_data
            # targets = sampled_label

            # # create optimizer
            # ca_epochs = self.args.get("crct_epochs", 10)
            # ca_lr = self.args.get("ca_lr", 0.005)

            # param_list = [p for p in self._network.fc.parameters() if p.requires_grad]
            # network_params = [{'params': param_list, 'lr': ca_lr, 'weight_decay': self.weight_decay}]
            # optimizer = optim.SGD(network_params, lr=ca_lr, momentum=0.9, weight_decay=self.weight_decay)
            # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=ca_epochs)


            # robust_weight_base = self.args.get("ca_robust_weight", 0.0)
            # entropy_weight = self.args.get("ca_entropy_weight", 0.0)
            # logit_norm = self.args.get("ca_logit_norm", 0.0)

            # self._network.to(self._device)
            # if len(self._multiple_gpus) > 1:
            #     self._network = nn.DataParallel(self._network, self._multiple_gpus)
            # self._network.eval()

            # cls_mean = torch.zeros((self._total_classes, self._network.feature_dim), device=self._device)
            # for class_i in range(self._total_classes):
            #     cls_mean[class_i] = torch.tensor(self._class_means_slca[class_i], dtype=torch.float64).to(self._device)

            # prog_bar = tqdm(range(ca_epochs))
            # for _ in prog_bar:
            #     sf_indexes = torch.randperm(inputs.size(0))
            #     inputs = inputs[sf_indexes]
            #     targets = targets[sf_indexes]

            #     total_loss = total = 0
            #     total_ce_loss = total_rb_loss = total_entropy_loss = 0
            #     total_acc = 0

            #     for i in range(0, len(sampled_data), batch_size):
            #         x = sampled_data[i : i + batch_size]
            #         y = sampled_label[i : i + batch_size]

            #         outputs = self._network(x, bcb_no_grad=True, fc_only=True)
            #         logits = outputs['logits']
                    
            #         if logit_norm != 0:
            #             batch_size = logits.size(0)
            #             num_tasks = self._cur_task + 1
                        
            #             # Compute per-task norms for averaging
            #             task_norms = torch.zeros(batch_size, num_tasks, device=logits.device)
                        
            #             for task in range(num_tasks):
            #                 # Get class indices for this task
            #                 cls_indices = [clz for clz in self.cls2task if self.cls2task[clz] == task]
            #                 if cls_indices:
            #                     # Compute L2 norm for this task's logits
            #                     task_logits = logits[:, cls_indices]  # (batch_size, num_classes_in_task)
            #                     task_norms[:, task] = torch.norm(task_logits, p=2, dim=-1) + 1e-7
                        
            #             # Average norms across all tasks
            #             avg_norms = task_norms.sum(dim=-1) / num_tasks  # Average across all tasks
            #             avg_norms = avg_norms.unsqueeze(-1)  # (batch_size, 1)
                        
            #             # Apply normalization: logits / avg_norm / logit_norm_factor
            #             normalized_logits = logits / (avg_norms + 1e-7) / logit_norm
            #             loss_vec = F.cross_entropy(normalized_logits, y, reduction="none")
            #         else:
            #             loss_vec = F.cross_entropy(logits, y, reduction="none")

            #         if robust_weight_base == 0 and entropy_weight == 0:
            #             loss = loss_vec.mean()
            #             optimizer.zero_grad()
            #             loss.backward()
            #             optimizer.step()
                        
            #             bs = len(y)
            #             total_loss += loss.item() * bs
            #             total_ce_loss += loss.item() * bs
            #             total_rb_loss += 0
            #             total_entropy_loss += 0
            #             total_acc += (logits.argmax(dim=1) == y).sum().item()
            #             total += bs
            #         else:
            #             L_total = torch.tensor(0.0, device=x.device)  # L = Σ Li
            #             total_term1 = torch.tensor(0.0, device=x.device)  # For logging: sum of all term1
            #             total_term2 = torch.tensor(0.0, device=x.device)  # For logging: sum of all term2
            #             total_term3 = torch.tensor(0.0, device=x.device)  # For logging: sum of all term3 (entropy)
                        
            #             unique_classes = torch.unique(y)
            #             class_dist = torch.cdist(x, cls_mean)
            #             class_indices = torch.argmin(class_dist, dim=1)
            #             for class_i in unique_classes:
            #                 label_mask = (y == class_i)
            #                 distance_mask = (class_indices == class_i)
            #                 class_mask = distance_mask & label_mask
                            
            #                 # Get the samples that belong to this class
            #                 class_samples = torch.where(class_mask)[0]
                            
            #                 # If no samples meet the conditions, fall back to label-only (term1 only)
            #                 if len(class_samples) == 0:
            #                     # Fall back to using only label condition for term1
            #                     label_only_samples = torch.where(label_mask)[0]
            #                     if len(label_only_samples) == 0:
            #                         continue  # Skip if no samples with this label at all
                                
            #                     label_losses = loss_vec[label_mask]
            #                     term1 = label_losses.mean()
            #                     term2 = torch.tensor(0.0).cuda()
            #                     term3 = torch.tensor(0.0).cuda()
            #                 else:
            #                     class_losses = loss_vec[class_mask]
            #                     term1 = class_losses.mean()
                                
            #                     # Second term: E_{x,x'~Ni}[|ℓ(yi, ht+1(x)) - ℓ(yi, ht+1(x'))|] where x,x' ∈ Ai
            #                     if len(class_samples) >= 2:
            #                         pairwise_diffs = torch.abs(
            #                             class_losses.unsqueeze(1) - class_losses.unsqueeze(0)
            #                         )
            #                         # Remove diagonal (self-comparisons)
            #                         mask = ~torch.eye(len(class_losses), dtype=torch.bool, device=x.device)
            #                         pairwise_diffs = pairwise_diffs[mask]
            #                         term2 = pairwise_diffs.mean()
            #                     else:
            #                         term2 = torch.tensor(0.0, device=x.device)
                                
            #                     # Third term: Cluster entropy minimization
            #                     if len(class_samples) >= 1 and entropy_weight != 0:
            #                         cluster_logits = logits[class_mask]  # Shape: (n_cluster_samples, n_classes)
            #                         cluster_probs = F.softmax(cluster_logits, dim=1)  # Shape: (n_cluster_samples, n_classes)
                                    
            #                         # Compute entropy for each sample: -Σ p_i * log(p_i)
            #                         # Add small epsilon to prevent log(0)
            #                         cluster_entropy = -torch.sum(cluster_probs * torch.log(cluster_probs + 1e-8), dim=1)
            #                         term3 = cluster_entropy.mean()  # Average entropy across cluster samples
            #                     else:
            #                         term3 = torch.tensor(0.0, device=x.device)
                            
            #                 Li = term1 + robust_weight_base * term2 + entropy_weight * term3
            #                 L_total += Li
            #                 total_term1 += term1
            #                 total_term2 += robust_weight_base * term2
            #                 total_term3 += entropy_weight * term3

            #             num_classes_in_batch = len(unique_classes)
            #             if num_classes_in_batch > 0:
            #                 loss = L_total / num_classes_in_batch
            #             else:
            #                 loss = loss_vec.mean()  # fallback
                        
            #             optimizer.zero_grad()
            #             loss.backward()
            #             optimizer.step()
                        
            #             bs = len(y)
                        
            #             # Average the terms by number of classes to get per-sample equivalent
            #             if num_classes_in_batch > 0:
            #                 avg_term1 = total_term1 / num_classes_in_batch
            #                 avg_term2 = total_term2 / num_classes_in_batch
            #                 avg_term3 = total_term3 / num_classes_in_batch
            #                 avg_loss = L_total / num_classes_in_batch
            #             else:
            #                 avg_term1 = torch.tensor(0.0, device=x.device)
            #                 avg_term2 = torch.tensor(0.0, device=x.device)
            #                 avg_term3 = torch.tensor(0.0, device=x.device)
            #                 avg_loss = loss_vec.mean()
                        
            #             total_loss += avg_loss.item() * bs
            #             total_ce_loss += avg_term1.item() * bs
            #             total_rb_loss += avg_term2.item() * bs
            #             total_entropy_loss += avg_term3.item() * bs
            #             total_acc += (logits.argmax(dim=1) == y).sum().item()
            #             total += bs

            #     scheduler.step()

            #     info = f"[Alignment] "
            #     info += f"Base Loss: {total_ce_loss/total:.4f}, "
            #     info += f"Robust Term: {total_rb_loss/total:.4f}, "
            #     info += f"Entropy Term: {total_entropy_loss/total:.4f}, "
            #     info += f"Total LCA Loss: {total_loss/total:.4f}, "
            #     info += f"Accuracy: {total_acc/total:.4f}"

            #     prog_bar.set_description(info)

            # logging.info(info)

    def _compute_class_mean(self, data_manager):
        if hasattr(self, '_class_means_slca') and self._class_means_slca is not None:
            ori_classes = self._class_means_slca.shape[0]
            assert ori_classes==self._known_classes
            new_class_means_slca = np.zeros((self._total_classes, self.feature_dim))
            new_class_means_slca[:self._known_classes] = self._class_means_slca
            self._class_means_slca = new_class_means_slca
            new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[:self._known_classes] = self._class_covs_slca
            self._class_covs_slca = new_class_cov
        else:
            self._class_means_slca = np.zeros((self._total_classes, self.feature_dim))
            self._class_covs_slca = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
        
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                  mode='test', ret_data=True)
            idx_loader = DataLoader(idx_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            vectors, _ = self._extract_vectors(idx_loader)

            # vectors = np.concatenate([vectors_aug, vectors])

            class_mean = np.mean(vectors, axis=0)
            # class_cov = np.cov(vectors.T)
            class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)+torch.eye(class_mean.shape[-1])*1e-4
            self._class_means_slca[class_idx, :] = class_mean
            self._class_covs_slca[class_idx, ...] = class_cov