# Finance-module-Python
Implementation of basic and intermediate Finance functions


1. **Time Value of Money (TVM) Calculations**:
    - `future_value(PV, r, n)`
    - `present_value(FV, r, n)`
    - `npv(rate, cash_flows)`
    - `irr(cash_flows)`

2. **Basic Statistics for Portfolio Analysis**:
    - `mean_return(returns)`
    - `variance(returns)`
    - `standard_deviation(returns)`
    - `covariance(returns1, returns2)`
    - `correlation(returns1, returns2)`
    
3. **Basic Portfolio Theory**:
    - `portfolio_expected_return(w1, returns1, w2, returns2)`
    - `portfolio_variance(w1, returns1, w2, returns2)`

4. **Additional Finance Metrics**:
    - `sharpe_ratio(returns, risk_free_rate)`
    - `beta(returns_asset, returns_market)`
    - `expected_return_capm(risk_free_rate, returns_asset, returns_market)`
