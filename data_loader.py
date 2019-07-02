import torch
from torch.utils.data import Dataset


class ExplicitDataset(Dataset):
    """Recommender dataset with explicit ratings"""

    def __init__(self, csv_file, users='user_id', items='item_id', ratings='rating'):
        """
        Args:
        csv_file (string):
            Path to the csv file with user-item interactions

        users (string):
            column name from long_df for user IDs

        items (string):
            column name from long_df for item IDs

        ratings (string):
            column name from long_df for ratings
        """

        self.df = pd.read_csv(csv_file)

        # get column numbers
        self.user_loc = self.df.columns.get_loc(users)
        self.item_loc = self.df.columns.get_loc(items)
        self.rating_loc = self.df.columns.get_loc(ratings)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user = torch.tensor(int(self.df.iloc[idx, self.user_loc])).cuda()
        item = torch.tensor(int(self.df.iloc[idx, self.item_loc])).cuda()
        rating = torch.tensor(self.df.iloc[idx, self.rating_loc]).cuda()

        return (user, item, rating)

        # inputs = torch.tensor(self.df.iloc[idx, :self.item_loc+1].values.astype('int64')).cuda()
        #
        # return (inputs, rating)