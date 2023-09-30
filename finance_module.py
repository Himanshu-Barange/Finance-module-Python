import numpy as np


class FinanceModule:

    @staticmethod
    def future_value(PV, r, n):
        """
        Calculates the future value of an investment.

        Parameters:
        - PV (float): Present value or initial investment.
        - r (float): Annual interest rate (as a decimal from 0 to 1).
        - n (int): Number of years the money is invested for.

        Returns:
        - float: Future value of the investment.
        """
        return PV * (1 + r) ** n

    @staticmethod
    def present_value(FV, r, n):
        """
        Calculate the present value of a sum to be received in the future.

        Parameters:
        - FV (float): Future value or final amount.
        - r (float): Annual discount rate (0 to 1).
        - n (int): Number of years until the future value is realized.

        Returns:
        - float: Present value of the investment.
        """
        return FV / (1 + r) ** n

    @staticmethod
    def npv(rate, cash_flows):
        """
        Calculate the net present value of a series of cash flows.

        Parameters:
        - rate (float): Discount rate (0 to 1).
        - cash_flows (numpy array): Array of cash flows. The first value is assumed to be the initial investment.

        Returns:
        - float: Net present value.
        """
        years = np.arange(len(cash_flows))
        return np.sum(cash_flows / (1 + rate) ** years)

    @staticmethod
    def irr(cash_flows, guess=0.1, max_iter=100, tol=1e-6):
        """
        Calculate the internal rate of return for a series of cash flows using the Newton-Raphson method.

        Parameters:
        - cash_flows (numpy array): Array of cash flows. The first value is typically the initial investment.
        - guess (float, optional): Initial guess for the IRR.
        - max_iter (int, optional): Maximum number of iterations for the Newton-Raphson method.
        - tol (float, optional): Tolerance for stopping criterion. The method will stop if the absolute value of NPV is less than tol.

        Returns:
        - float: Internal rate of return.
        """
        rate = guess
        for _ in range(max_iter):
            npv = FinanceModule.npv(rate, cash_flows)
            d_npv = np.sum(-cash_flows * np.arange(len(cash_flows)) / (1 + rate) ** (np.arange(len(cash_flows)) + 1))
            rate -= npv / d_npv
            if abs(npv) < tol:
                break
        return rate

    @staticmethod
    def mean_return(returns):
        """
        Calculates the mean return of a given series.

        Parameters:
        - returns (numpy array): Series of returns.

        Returns:
        - float: Mean return of the series.
        """
        return np.mean(returns)

    @staticmethod
    def variance(returns):
        """
        Calculates the variance of a given series.

        Parameters:
        - returns (numpy array): Series of returns.

        Returns:
        - float: Variance of the series.
        """
        return np.var(returns)

    @staticmethod
    def standard_deviation(returns):
        """
        Calculates the standard deviation of a given series.

        Parameters:
        - returns (numpy array): Series of returns.

        Returns:
        - float: Standard deviation of the series.
        """
        return np.std(returns)

    @staticmethod
    def covariance(returns1, returns2):
        """
        Calculates the covariance between two series of returns.

        Parameters:
        - returns1 (numpy array): First series of returns.
        - returns2 (numpy array): Second series of returns.

        Returns:
        - float: Covariance between the two series.
        """
        return np.cov(returns1, returns2)[0][1]

    @staticmethod
    def correlation(returns1, returns2):
        """
        Calculates the correlation coefficient between two series of returns.

        Parameters:
        - returns1 (numpy array): First series of returns.
        - returns2 (numpy array): Second series of returns.

        Returns:
        - float: Correlation coefficient between the two series.
        """
        return np.corrcoef(returns1, returns2)[0][1]

    @staticmethod
    def portfolio_expected_return(w1, returns1, w2, returns2):
        """
        Calculates the expected return of a two-asset portfolio based on weights and individual asset returns.

        Parameters:
        - w1 (float): Weight of the first asset.
        - returns1 (numpy array): Series of returns for the first asset.
        - w2 (float): Weight of the second asset.
        - returns2 (numpy array): Series of returns for the second asset.

        Returns:
        - float: Expected return of the portfolio.
        """
        mean_return1 = FinanceModule.mean_return(returns1)
        mean_return2 = FinanceModule.mean_return(returns2)
        return w1 * mean_return1 + w2 * mean_return2

    @staticmethod
    def portfolio_variance(w1, returns1, w2, returns2):
        """
        Calculates the variance of a two-asset portfolio based on weights and individual asset returns.

        Parameters:
        - w1 (float): Weight of the first asset.
        - returns1 (numpy array): Returns of the first asset.
        - w2 (float): Weight of the second asset.
        - returns2 (numpy array): Returns of the second asset.

        Returns:
        - float: Variance of the portfolio.
        """
        var1 = FinanceModule.variance(returns1)
        var2 = FinanceModule.variance(returns2)
        cov12 = FinanceModule.covariance(returns1, returns2)
        return w1 ** 2 * var1 + w2 ** 2 * var2 + 2 * w1 * w2 * cov12

    @staticmethod
    def sharpe_ratio(returns, risk_free_rate):
        """
        Calculate the Sharpe Ratio.

        Parameters:
        - returns (numpy array): Series of returns for the portfolio.
        - risk_free_rate (float): Risk-free rate.

        Returns:
        - float: Sharpe Ratio.
        """
        expected_return = FinanceModule.mean_return(returns)
        std_dev = FinanceModule.standard_deviation(returns)
        return (expected_return - risk_free_rate) / std_dev

    @staticmethod
    def beta(returns_asset, returns_market):
        """
        Calculate the beta of an asset.

        Parameters:
        - returns_asset (numpy array): Series of returns for the asset.
        - returns_market (numpy array): Series of returns for the market.

        Returns:
        - float: Beta of the asset.
        """
        covariance = FinanceModule.covariance(returns_asset, returns_market)
        market_variance = FinanceModule.variance(returns_market)
        return covariance / market_variance

    @staticmethod
    def expected_return_capm(risk_free_rate, returns_asset, returns_market):
        """
        Calculate the expected return of an asset using CAPM.

        Parameters:
        - risk_free_rate (float): Risk-free rate.
        - returns_asset (numpy array): Series of returns for the asset.
        - returns_market (numpy array): Series of returns for the market.

        Returns:
        - float: Expected return of the asset.
        """
        beta_value = FinanceModule.beta(returns_asset, returns_market)
        market_return = FinanceModule.mean_return(returns_market)
        return risk_free_rate + beta_value * (market_return - risk_free_rate)
