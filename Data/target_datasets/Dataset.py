class Dataset:
    def __init__(self, name, num_classes, res):
        self.name = name
        self.num_classes = num_classes
        self.resolution = res

    def get_name(self): return self.name

    def get_num_classes(self): return self.num_classes

    def get_resolution(self): return self.resolution

    def get_labels(self):
        raise NotImplementedError("Must override method")

    def load_data_real(self, batch_size = None, return_raw = False, categorical = True, shuffle = True, seed = 1234):
        raise NotImplementedError("Must override method")

    def load_data_syn(self, exp, batch_size = None, shuffle=True, seed=1234, take=None):
        raise NotImplementedError("Must override method")

    def load_data(self, batch_size, exp = None, synthetic: bool = False, return_raw=False, categorical=True,
                  shuffle = False, seed = 1234, take=None):
        """
            Loads and returns the dataset, either synthetically generated or real.

            This method acts as a dispatcher, delegating the data loading process based on the `synthetic` flag.
            If `synthetic` is True, it calls `load_data_syn` to generate a synthetic dataset. Otherwise, it calls
            `load_data_real` to load a real dataset.

            Args:
                batch_size (int): The batch size for loading data.
                exp (str): Experiment identifier for synthetic data generation. Only used if `synthetic` is True.
                synthetic (bool, optional): If True, loads a synthetic dataset. If False, loads a real dataset. Defaults to False.
                return_raw (bool, optional): If True, returns raw (unprocessed) data when loading real data. Only used if `synthetic` is False. Defaults to False.
                categorical (bool, optional): If True, converts labels to one-hot encoding when loading real data. Only used if `synthetic` is False. Defaults to True.
                shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
                seed (int, optional): Random seed for shuffling and synthetic data generation. Defaults to 1234.
                take (int, optional): Number of samples per class to take (if specified). Defaults to None (take all samples).

            Returns:
                The return value depends on whether `synthetic` is True or False:

                * If `synthetic` is True: Returns the output of `self.load_data_syn`.
                * If `synthetic` is False: Returns the output of `self.load_data_real`.

        """
        return self.load_data_syn(batch_size, exp, shuffle=shuffle, seed=seed, take=take) if synthetic \
            else self.load_data_real(batch_size=batch_size, return_raw=return_raw, categorical=categorical,
                                     shuffle=shuffle, seed=seed)
