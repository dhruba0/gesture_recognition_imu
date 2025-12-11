class CustomDataset(Dataset):
    def __init__(
        self, config,df: pd.DataFrame, X: list[np.ndarray], y = None,
        transforms = None, mode: str = "train"
    ): 
        self.config = config
        self.df = df
        self.X = X
        self.y = y
        self.indexes = self.df.sequence_id.unique()
        self.alpha = 0.3
        self.mode = mode
        self.transforms = transforms
        self.num_classes = 18
        
    def __len__(self):
        """
        Length of dataset.
        """
        return len(self.indexes)
        
    def __getitem__(self, i):
        """
        Get one item.
        """
        sequence_id = self.indexes[i]
        p = np.random.rand()
        if p <= 0.0:
            if self.mode == "train":
                X, y = self.get_data(i)
                X = self.transforms(X)
                X = random_padding(X)
                output = {
                    "X": torch.tensor(X, dtype=torch.float32),
                    "y": torch.tensor(y, dtype=torch.float32),
                }
            elif self.mode == "test":
                X = self.get_data(i)
                X = random_padding(X)
                output = {
                    "X": torch.tensor(X, dtype=torch.float32)
                }
            elif self.mode == "val":
                X ,y= self.get_data(i)
                X = random_padding(X)
                output = {
                    "X": torch.tensor(X, dtype=torch.float32),
                    "y": torch.tensor(y, dtype=torch.float32)
                }
        else:
            if self.mode == "train":
                lam = np.random.beta(self.alpha, self.alpha)
                j = np.random.randint(0, len(self.indexes))
                X1, y1 = self.get_data(i)
                X2, y2 = self.get_data(j)
                X1, X2 = random_padding(X1), random_padding(X2) 
                X = lam * X1 + (1 - lam) * X2
                y = lam * y1 + (1 - lam) * y2
                output = {
                    "X": torch.tensor(X, dtype=torch.float32),
                    "y": torch.tensor(y, dtype=torch.float32),
                }
            elif self.mode == "test":
                X = self.get_data(i)
                X = random_padding(X)
                output = {
                    "X": torch.tensor(X, dtype=torch.float32)
                }
            elif self.mode == "val":
                X ,y= self.get_data(i)
                X = random_padding(X)
                output = {
                    "X": torch.tensor(X, dtype=torch.float32),
                    "y": torch.tensor(y, dtype=torch.float32)
                }
                  
        return output

    def get_data(self, index):
        X = self.X[index]
        if self.mode in ["train","val"]:
            y = self.y[index]
            y = np.eye(self.num_classes, dtype=int)[y]
            return X, y
        elif self.mode == "test":
            return X
