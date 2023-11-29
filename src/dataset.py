import pdb

import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class MicrosoftDataset(Dataset):
    def __init__(self, data_file):
        self.raw_data = pd.read_csv(data_file)
        self.processed_data, self.processed_label = self.preprocess(self.raw_data)

    def preprocess(self, raw_data):
        # down sample the data to 1 sample per 15 hour
        data_copy = raw_data.copy()
        data_copy["model"] = data_copy["model"].apply(lambda x: float(x[5:]))
        data_copy["failure"] = data_copy["failure"].apply(lambda x: float(x))
        data_copy.drop(columns=["datetime"], inplace=True)
        # pdb.set_trace()
        data_array = data_copy.to_numpy(dtype=float)  # (58300, 17)
        data_array = data_array.reshape((100, 583, 17))

        # find each sequence that leads to failure
        all_seqs = []
        for i in range(0, 100):
            one_seq = []
            for j in range(0, 583):
                assert data_array[i][j][0] == i + 1  # check machine id

                if data_array[i][j][-1] == 0.0:
                    # no failure, add to sequence
                    one_seq.append(data_array[i][j])
                elif data_array[i][j][-1] == 1.0:
                    # failure occurs, add if the sequence is no empty
                    if len(one_seq) > 0:
                        one_seq.append(data_array[i][j])
                        all_seqs.append(one_seq)
                        one_seq = []
                else:
                    assert False

        # for each sequence, correct all possible training data and labels
        # Note: max look back is 128 steps, encoder is unrolled 64 steps
        look_back_steps = 128
        unrolled_steps = 64
        features = []
        labels = []
        for one_seq in all_seqs:
            length = len(one_seq)
            for i in range(0, length - 1):
                lower_bound = max(0, i - look_back_steps)
                upper_bound = min(length, i + unrolled_steps)

                one_seq_array = np.array(one_seq)
                x = one_seq_array[lower_bound : i + 1][:, :-1]
                y = one_seq_array[i + 1 : upper_bound][:, -1]

                features.append(x)
                labels.append(y)

        return features, labels

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        x = self.processed_data[idx]
        y = self.processed_label[idx]
        return x, y


if __name__ == "__main__":
    MicrosoftDataset("../AMLWorkshop/Data/features_15h.csv")
    # MicrosoftDataset("/Users/yuningwang/Desktop/CS535_final_project/AMLWorkshop/Data/features_15h.csv")