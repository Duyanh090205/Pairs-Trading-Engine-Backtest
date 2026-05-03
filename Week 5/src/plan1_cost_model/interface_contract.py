import pandas as pd
import numpy as np

TRADE_LOG_SCHEMA = {
    'trade_id': 'string', 'fold_id': 'int', 'pair_id': 'string',
    'ticker_A': 'string', 'ticker_B': 'string', 
    'side_A': 'int', 'side_B': 'int', 
    'entry_ts': 'datetime64[ns, US/Eastern]', 
    'exit_ts': 'datetime64[ns, US/Eastern]', 
    'notional_A_entry': 'float', 'notional_B_entry': 'float', 
    'notional_A_exit': 'float', 'notional_B_exit': 'float', 
    'gross_pnl_dollars': 'float', 'allocated_capital': 'float'
}

REBALANCE_LOG_SCHEMA = {
    'trade_id': 'string', 'fold_id': 'int', 'pair_id': 'string',
    'ticker': 'string', 'rebalance_ts': 'datetime64[ns, US/Eastern]', 
    'delta_shares': 'float', 'price_at_rebalance': 'float', 
    'notional_rebalanced': 'float'
}

COST_LOG_SCHEMA = {
    'trade_id': 'string', 'spread_cost_dollars': 'float', 
    'impact_cost_dollars': 'float', 'borrow_cost_dollars': 'float', 
    'rebalance_cost_dollars': 'float', 'total_cost_dollars': 'float', 
    'net_pnl_dollars': 'float', 'net_return': 'float'
}

def _validate_schema(df: pd.DataFrame, schema: dict, schema_name: str) -> bool:
    """Helper to validate if a dataframe matches a schema."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{schema_name} must be a pandas DataFrame.")
        
    missing_cols = set(schema.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{schema_name} is missing required columns: {missing_cols}")
        
    return True

def validate_trade_log(df: pd.DataFrame) -> bool:
    """Validates the input trade_log matches the Week 4 interface contract."""
    return _validate_schema(df, TRADE_LOG_SCHEMA, "Trade Log")

def validate_rebalance_log(df: pd.DataFrame) -> bool:
    """Validates the input rebalance_log matches the Week 4 interface contract."""
    return _validate_schema(df, REBALANCE_LOG_SCHEMA, "Rebalance Log")

def validate_cost_log(df: pd.DataFrame) -> bool:
    """Validates the output cost_log matches the Week 5 interface contract."""
    return _validate_schema(df, COST_LOG_SCHEMA, "Cost Log")
