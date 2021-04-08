import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from typing import List, Dict, Tuple


class PairWiseMatrixFactorization(nn.Module):

    def __init__(self,
                 user_num: int,
                 item_num: int,
                 factors: int = 32,
                 epochs: int = 20,
                 lr: float = 0.01,
                 reg_1: float = 0.001,
                 reg_2: float = 0.001,
                 gpuid: str = '0',
                 early_stop: bool = True):
        """
        Pair-wise Matrix Factorization Recommender Class
        Parameters
        ----------
        user_num : int, the number of users
        item_num : int, the number of items
        factors : int, the number of latent factor
        epochs : int, number of training epochs
        lr : float, learning rate
        reg_1 : float, first-order regularization term
        reg_2 : float, second-order regularization term
        gpuid : str, GPU ID
        early_stop : bool, whether to activate early stop mechanism
        """
        super(PairWiseMatrixFactorization, self).__init__()

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.epochs = epochs
        self.lr = lr
        self.reg_1 = reg_1
        self.reg_2 = reg_2

        self.user_embeddings = nn.Embedding(user_num, factors)
        self.item_embeddings = nn.Embedding(item_num, factors)

        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

        self.early_stop = early_stop

    def forward(self,
                users: torch.Tensor,
                positive_items: torch.Tensor,
                negative_items: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        user_embeddings = self.user_embeddings(users)
        item_positive_embeddings = self.item_embeddings(positive_items)
        item_negative_embeddings = self.item_embeddings(negative_items)

        positive_preds = (user_embeddings * item_positive_embeddings).sum(dim=-1)
        negative_preds = (user_embeddings * item_negative_embeddings).sum(dim=-1)

        return positive_preds, negative_preds

    def loss(self, positive_preds: torch.Tensor, negative_preds: torch.Tensor):
        raise NotImplementedError(f'Implement loss in {self.__class__.__name__}')

    def fit(self, train_loader, optimizer: torch.optim.Optimizer):

        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

        last_loss = 0.
        for epoch in range(1, self.epochs + 1):
            self.train()

            current_loss = 0.
            # set process bar display
            pbar = tqdm(train_loader)
            pbar.set_description(f'[Epoch {epoch:03d}]')
            for users, positive_items, negative_items, label in pbar:
                if torch.cuda.is_available():
                    users = users.cuda()
                    positive_items = positive_items.cuda()
                    negative_items = negative_items.cuda()
                else:
                    users = users.cpu()
                    positive_items = positive_items.cpu()
                    negative_items = negative_items.cpu()

                self.zero_grad()
                positive_preds, negative_preds = self.forward(users, positive_items, negative_items)

                loss = self.loss(positive_preds, negative_preds)

                loss += self.reg_1 * (self.item_embeddings.weight.norm(p=1) + self.user_embeddings.weight.norm(p=1))
                loss += self.reg_2 * (self.item_embeddings.weight.norm() + self.user_embeddings.weight.norm())

                if torch.isnan(loss):
                    raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the recommender')

                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
                current_loss += loss.item()

            self.eval()
            delta_loss = float(current_loss - last_loss)
            if (abs(delta_loss) < 1e-5) and self.early_stop:
                print('Satisfy early stop mechanism')
                break
            else:
                last_loss = current_loss

    def predict(self, users: torch.Tensor, k: int = 10) -> Dict[int, List[Tuple[int, float]]]:

        user_embeddings = self.user_embeddings(users)
        scores = user_embeddings.dot(self.item_embeddings.weight.T)
        item_lists = scores.argsort(dim=1, descending=True)[:, :k].detach().numpy()

        n_users = users.shape[0]
        weight_lists = scores[np.repeat(np.arange(n_users), k), item_lists.ravel()].detach().numpy()
        users = users.detach().numpy()

        recommend_lists = {}
        for i in range(n_users):
            items = item_lists[i]
            weights = weight_lists[i]
            user = users[i]
            recommend_lists[int(user)] = [(int(item), float(weight)) for item, weight in zip(items, weights)]

        return recommend_lists


class BPRMatrixFactorization(PairWiseMatrixFactorization):

    def __init__(self,
                 user_num,
                 item_num,
                 factors=32,
                 epochs=20,
                 lr=0.01,
                 reg_1=0.001,
                 reg_2=0.001,
                 gpuid='0',
                 early_stop=True):

        super(BPRMatrixFactorization, self).__init__(user_num,
                                                     item_num,
                                                     factors,
                                                     epochs,
                                                     lr,
                                                     reg_1,
                                                     reg_2,
                                                     gpuid,
                                                     early_stop)

    def loss(self, positive_preds: torch.Tensor, negative_preds: torch.Tensor):

        loss = -(positive_preds - negative_preds).sigmoid().log().sum()
        return loss
