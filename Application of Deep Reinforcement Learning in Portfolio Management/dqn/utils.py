import numpy as np

class History:
    def __init__(self, max_size = 10000):
        self.height = max_size
    def set(self, **kwargs):
        # Flattening the inputs to put it in np.array
        self.columns = []
        for name, value in kwargs.items():
            if isinstance(value, list):
                self.columns.extend([f"{name}_{i}" for i in range(len(value))])
            elif isinstance(value, dict):
                self.columns.extend([f"{name}_{key}" for key in value.keys()])
            else:
                self.columns.append(name)
        
        self.width = len(self.columns)
        self.history_storage = np.zeros(shape=(self.height, self.width), dtype= 'O')
        self.size = 0
        self.add(**kwargs)

    def add(self, **kwargs):
        values = []
        columns = []
        for name, value in kwargs.items():
            if isinstance(value, list):
                columns.extend([f"{name}_{i}" for i in range(len(value))])
                values.extend(value[:])
            elif isinstance(value, dict):
                columns.extend([f"{name}_{key}" for key in value.keys()])
                values.extend(list(value.values()))
            else:
                columns.append(name)
                values.append(value)

        if columns == self.columns:
            self.history_storage[self.size, :] = values
            self.size = min(self.size+1, self.height)
        else:
            raise ValueError(f"Make sure that your inputs match the initial ones... Initial ones : {self.columns}. New ones {columns}")
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, arg):
        if isinstance(arg, tuple):
            column, t = arg
            try:
                column_index = self.columns.index(column)
            except ValueError as e:
                raise ValueError(f"Feature {column} does not exist ... Check the available features : {self.columns}")
            return self.history_storage[:self.size][t, column_index]
        if isinstance(arg, int):
            t = arg
            return dict(zip(self.columns, self.history_storage[:self.size][t]))
        if isinstance(arg, str):
            column = arg
            try:
                column_index = self.columns.index(column)
            except ValueError as e:
                raise ValueError(f"Feature {column} does not exist ... Check the available features : {self.columns}")
            return self.history_storage[:self.size][:, column_index]
        if isinstance(arg, list):
            columns = arg
            column_indexes = []
            for column in columns:
                try:
                    column_indexes.append(self.columns.index(column))
                except ValueError as e:
                    raise ValueError(f"Feature {column} does not exist ... Check the available features : {self.columns}")
            return self.history_storage[:self.size][:, column_indexes]

    def __setitem__(self, arg, value):
        column, t = arg
        try:
            column_index = self.columns.index(column)
        except ValueError as e:
            raise ValueError(f"Feature {column} does not exist ... Check the available features : {self.columns}")
        self.history_storage[:self.size][t, column_index] = value


class Portfolio:
    def __init__(self, asset, fiat):
        self.asset =asset
        self.fiat =fiat
    def valorisation(self, price):
        return sum([
            self.asset * price,
            self.fiat
        ])
    def real_position(self, price):
        return self.asset * price / self.valorisation(price)
    def position(self, price):
        return self.asset * price / self.valorisation(price)
    def trade_to_position(self, position, price, trading_fees):
        # Proceed to trade
        asset_trade = (position * self.valorisation(price) / price - self.asset)
        if asset_trade > 0:
            asset_trade = asset_trade / (1 - trading_fees + trading_fees * position)
            asset_fiat = - asset_trade * price
            self.asset = self.asset + asset_trade * (1 - trading_fees)
            self.fiat = self.fiat + asset_fiat
        else:
            asset_trade = asset_trade / (1 - trading_fees * position)
            asset_fiat = - asset_trade * price
            self.asset = self.asset + asset_trade 
            self.fiat = self.fiat + asset_fiat * (1 - trading_fees)
    def __str__(self): return f"{self.__class__.__name__}({self.__dict__})"
    def describe(self, price): print("Value : ", self.valorisation(price), "Position : ", self.position(price))
    def get_portfolio_distribution(self):
        return {
            "asset":max(0, self.asset),
            "fiat":max(0, self.fiat),
        }

# TargetPortfolio from .utils.portfolio
class TargetPortfolio(Portfolio):
    def __init__(self, position ,value, price):
        super().__init__(
            asset = position * value / price,
            fiat = (1-position) * value,
        )