import random
from torch.utils.data import IterableDataset

class BufferedShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size=10000, seed=None):
        """
        Args:
            dataset (IterableDataset): The input dataset to shuffle.
            buffer_size (int): The size of the shuffle buffer.
            seed (int, optional): Random seed for reproducibility.
        """
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        buffer = []
        for item in iter(self.dataset):
            buffer.append(item)
            if len(buffer) >= self.buffer_size:
                idx = rng.randint(0, len(buffer) - 1)
                yield buffer.pop(idx)
        # Yield the rest
        while buffer:
            idx = rng.randint(0, len(buffer) - 1)
            yield buffer.pop(idx)