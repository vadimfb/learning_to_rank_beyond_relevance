import torch.utils.data as data

from typing import List, Tuple, Dict


class PairwiseRankingData(data.Dataset):

    def __init__(self, dataset: Dict[int, Dict[str, List[int]]], is_training: bool = True):
        """
        Dataset formatter adapted pair-wise algorithms
        Parameters
        ----------
        dataset : List,
        is_training : bool,
        """
        super(PairwiseRankingData, self).__init__()

        self.dataset_ = []
        for user, item_lists in dataset.items():
            if is_training:
                positive_items = item_lists['positive']
                negative_items = item_lists['negative']
                assert len(positive_items) == len(negative_items)
                for i in range(len(positive_items)):
                    self.dataset_.append((user, positive_items[i], negative_items[i]))
            else:
                positive_items = item_lists['positive']
                for i in range(len(positive_items)):
                    self.dataset_.append((user, positive_items[i], positive_items[i]))

    def __len__(self) -> int:
        return len(self.dataset_)

    def __getitem__(self, idx: int) -> Tuple[int, int, int]:
        user = self.dataset_[idx][0]
        positive_item = self.dataset_[idx][1]
        negative_item = self.dataset_[idx][2]

        return user, positive_item, negative_item
