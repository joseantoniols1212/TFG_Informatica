import numpy as np
import pandas as pd
from utils import get_data_by_index
import sys


def imprimir_barra_progreso(iteracion, total, longitud=50):
    progreso = iteracion / total
    bloque = int(longitud * progreso)
    barra = '█' * bloque + '-' * (longitud - bloque)
    porcentaje = progreso * 100
    sys.stdout.write(f'\r|{barra}| {porcentaje:.2f}%')
    sys.stdout.flush()


class Portfolio:
    def __init__(self, w_p, w_s):
        self.w_p = w_p
        self.w_s = w_s


class Assets:
    def __init__(self, sigma, mu, esg):
        self.sigma = sigma
        self.mu = mu
        self.esg = esg


class Metrics:
    def __init__(self, portfolio, assets):
        self.portfolio = portfolio
        self.assets = assets

    def sharpe_ratio(self):
        w_p = self.portfolio.w_p
        sigma = self.assets.sigma
        mu = self.assets.mu
        return (w_p@mu)/(w_p@sigma@w_p)**0.5

    def returns(self):
        w_p = self.portfolio.w_p
        mu = self.assets.mu
        return w_p@mu

    def risk(self):
        w_p = self.portfolio.w_p
        sigma = self.assets.sigma
        return w_p@sigma@w_p

    def esg_score(self):
        w_p = self.portfolio.w_p
        w_s = self.portfolio.w_s
        esg = self.assets.esg
        return w_p@esg@w_s

    def wp_norm(self):
        return np.linalg.norm(self.portfolio.w_p, 1)

    def ws_norm(self):
        return np.linalg.norm(self.portfolio.w_s, 1)

    def report(self):
        print("Sharpe ratio: ", self.sharpe_ratio())
        print("Return: ", self.returns())
        print("Risk: ", self.risk())
        print("ESG Score: ", self.esg_score())
        print("w_p 1-norm: ", np.linalg.norm(self.portfolio.w_p, 1))
        print("w_s 1-norm: ", np.linalg.norm(self.portfolio.w_s, 1))

    def to_dict(self):
        return {
            'Sharpe ratio': self.sharpe_ratio(),
            'Return': self.returns(),
            'Risk': self.risk(),
            'ESG Score': self.esg_score(),
        }


class PortfolioOptimizer:
    def __init__(self, assets, strategy, **strategy_params):
        self.assets = assets
        self.strategy = strategy
        self.strategy_params = strategy_params
        self.portfolio = self.optimize()

    def optimize(self):
        if not self.strategy_params:
            w_p, w_s = self.strategy(
                self.assets.sigma, self.assets.mu, self.assets.esg)
        else:
            w_p, w_s = self.strategy(
                self.assets.sigma, self.assets.mu, self.assets.esg, **self.strategy_params)
        return Portfolio(w_p, w_s)

    def metrics(self):
        return Metrics(self.portfolio, self.assets)


class PortfolioComparison:
    def __init__(self, assets_list, strategies):
        self.assets_list = assets_list
        self.strategies = strategies

    def compare(self):
        results = []
        for i, assets in enumerate(self.assets_list):
            for j, (strategy, params) in enumerate(self.strategies):
                optimizer = PortfolioOptimizer(assets, strategy, **params)
                metrics = optimizer.metrics()
                result = metrics.to_dict()
                result['Grupo'] = i
                result['Estrategia'] = j
                results.append(result)
                imprimir_barra_progreso(
                    j+i*len(assets), len(self.assets) * len(self.strategies))
        return pd.DataFrame(results)


if __name__ == "__main__":

    def dummy_strategy(sigma, mu, esg, **params):
        # Esta es una estrategia ficticia para ilustrar el ejemplo
        (n, m) = esg.shape
        w_p = np.ones(n) / n
        w_s = np.ones(m) / m
        return w_p, w_s

    assets0 = Assets(*get_data_by_index(0))
    assets1 = Assets(*get_data_by_index(1))

    strategies = [
        (dummy_strategy, {'param1': 1}),
        (dummy_strategy, {'param2': 2}),
    ]

    comparison = PortfolioComparison([assets0, assets1], strategies)
    comparison.to_csv('comparison_results.csv', index=False)
