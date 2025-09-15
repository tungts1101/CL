import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
import os
from .utils import merge
from utils.net import SimpleNet


num_workers = 8
EPSILON = 1e-8


class BaseMergingLearner(BaseLearner):
    CHECKPOINT_DIR = "checkpoints"
    def __init__(self, args):
        super().__init__(args)

        self._network = SimpleNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 5e-4
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args

        self._network.to(self._device)
        self.task_sizes = []
        self._class_increments = []

        # Ensure checkpoint directory exists before saving
        import os
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)

        torch.save(
            self._network.get_backbone_trainable_params(),
            self.backbone_checkpoint(self._cur_task),
        )
        
    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._class_increments.append((self._known_classes, self._total_classes - 1))

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes-1))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", )
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        prototype_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test", )
        self.prototype_loader = DataLoader(prototype_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        self._train(self.train_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        
        self.post_training()
    
    def post_training(self):
        if self.args["train_merge_method"] != "none":
            self.merge()

    def _train(self, train_loader):
        freeze_old = self.args["train_freeze_old"]
        self._network.update_fc(
            self._total_classes - self._known_classes,
            freeze_old=freeze_old,
        )
        self._network.to(self._device)
        
        if (
            not os.path.exists(self.backbone_checkpoint(self._cur_task))
            or not os.path.exists(self.head_checkpoint(self._cur_task))
            or self.args["reset"]
        ):
            self._network.backbone.load_state_dict(
                torch.load(self.backbone_checkpoint(self._cur_task - 1)), strict=False
            )

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
                        for p in self._network.fc.heads[self._cur_task].parameters()
                        if p.requires_grad
                    ],
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                },
            ]
            if self._network.norm is not None:
                parameters.append(
                    {
                        "params": [
                            p for p in self._network.norm.parameters() if p.requires_grad
                        ],
                        "lr": base_lr,
                        "weight_decay": weight_decay,
                    }
                )
            if not freeze_old:
                parameters.append(
                    {
                        "params": [
                            p
                            for i, head in enumerate(self._network.fc.heads)
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
            logging.info(f"[Training] {self._network}")

            for epoch in range(epochs):
                total_loss, total_acc, total = 0, 0, 0

                for _, (_, x, y) in enumerate(train_loader):
                    x, y = x.to(self._device), y.to(self._device)

                    features = self._network.get_features(x)

                    if freeze_old:
                        y = torch.where(
                            y - self._known_classes >= 0, y - self._known_classes, -100
                        )
                        logits = self._network.fc.heads[self._cur_task](features)
                    else:
                        logits = self._network.fc(features)["logits"]

                    loss = F.cross_entropy(logits, y)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * len(y)
                    total_acc += (logits.argmax(dim=1) == y).sum().item()
                    total += len(y)

                scheduler.step()

                if epoch % 5 == 4 or epoch == epochs - 1:
                    logging.info(
                        f"[Training] Epoch {epoch + 1}/{epochs}, "
                        f"Total Loss: {total_loss / total:.4f}, "
                        f"Acc: {total_acc / total:.4f}"
                    )

            torch.save(
                self._network.get_backbone_trainable_params(),
                self.backbone_checkpoint(self._cur_task),
            )
            torch.save(
                self._network.fc.heads[-1].state_dict(),
                self.head_checkpoint(self._cur_task),
            )
        else:
            logging.info(f"[Training] Load existing model for task {self._cur_task}")
            self._network.backbone.load_state_dict(
                torch.load(self.backbone_checkpoint(self._cur_task)), strict=False
            )
            self._network.fc.heads[-1].load_state_dict(
                torch.load(self.head_checkpoint(self._cur_task)), strict=True
            )
    
    def load_backbone(self, backbone_params):
        peft_params = {}
        norm_params = {}
        for name, param in backbone_params.items():
            if name.startswith("norm."):
                norm_name = name[5:]
                norm_params[norm_name] = param
            else:
                peft_params[name] = param
        self._network.backbone.load_state_dict(peft_params, strict=False)
        if norm_params:
            self._network.norm.load_state_dict(norm_params, strict=True)

    def merge(self):
        if os.path.exists(self.merged_checkpoint(self._cur_task)) and not self.args.get("reset", False):
            logging.info(f"[Merging] Load merged checkpoint for task {self._cur_task}")
            backbone_params = torch.load(self.merged_checkpoint(self._cur_task))
            self.load_backbone(backbone_params)
            return

        if self._cur_task == 0:
            logging.info(
                f"[Merging] Save merged backbone checkpoint for task {self._cur_task}"
            )
            torch.save(
                self._network.get_backbone_trainable_params(),
                self.merged_checkpoint(self._cur_task),
            )
            return

        logging.info(f"[Merging] Method {self.args['train_merge']}")
        base_params = torch.load(self.backbone_checkpoint(-1))
        num_merged_params = sum(param.numel() for param in base_params.values())
        logging.info(f"[Merging] Merging with {num_merged_params:,} total parameters")

        if self.args.get("train_merge_incremental", False):
            task_params = []
            task_params.append(torch.load(self.merged_checkpoint(self._cur_task - 1)))
            task_params.append(torch.load(self.backbone_checkpoint(self._cur_task)))
        else:
            task_params = [
                torch.load(self.backbone_checkpoint(task))
                for task in range(self._cur_task + 1)
            ]
        logging.info(f"[Merging] Loaded {len(task_params)} tasks for merging")

        # logging.info("[Merging] Norm layer values BEFORE merging:")
        # logging.info(f"  norm.weight: mean={self._network.norm.weight.data.mean():.6f}, std={self._network.norm.weight.data.std():.6f}")
        # logging.info(f"  norm.bias: mean={self._network.norm.bias.data.mean():.6f}, std={self._network.norm.bias.data.std():.6f}")

        backbone_params = merge(
            base_params,
            task_params,
            method=self.args["train_merge"],
            lamb=self.args["train_merge_coef"],
            topk=self.args["train_merge_topk"],
        )
        self.load_backbone(backbone_params)

        # logging.info("[Merging] Norm layer values AFTER merging:")
        # logging.info(f"  norm.weight: mean={self._network.norm.weight.data.mean():.6f}, std={self._network.norm.weight.data.std():.6f}")
        # logging.info(f"  norm.bias: mean={self._network.norm.bias.data.mean():.6f}, std={self._network.norm.bias.data.std():.6f}")

        logging.info(
            f"[Merging] Save merged backbone checkpoint for task {self._cur_task}"
        )
        torch.save(
            self._network.get_backbone_trainable_params(),
            self.merged_checkpoint(self._cur_task),
        )
    
    def prefix(self):
        prefix_parts = [
            str(self.args['seed']),
            self.args['dataset'], 
            str(self.args['init_cls']),
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

    def merged_checkpoint(self, task):
        filename = f"{self.prefix()}_merged_{task}.pt"
        return os.path.join(self.CHECKPOINT_DIR, filename)