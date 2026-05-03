import pandas as pd


def load_data(filepath):
    """
    Load and clean 1987 crash market data from CSV.

    Removes:
    - Weekend data (Saturdays and Sundays have no real trading)
    - Trading halts (null SP500_Futures values on 10/19 and 10/20)
    - Non-trading hours (outside 9:30-16:00 EST)

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame with Timestamp column parsed, weekends removed, halts removed,
        non-trading hours removed, and sorted chronologically
    """
    df = pd.read_csv(filepath)

    # Parse timestamp as datetime with format %m/%d/%Y %H:%M
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y %H:%M')

    # Convert price columns to float
    df['SP500_Futures'] = pd.to_numeric(df['SP500_Futures'], errors='coerce')

    # Remove weekend data (Saturday=5, Sunday=6)
    df = df[df['Timestamp'].dt.dayofweek < 5].copy()

    # Remove trading halts (null SP500_Futures values)
    df = df[df['SP500_Futures'].notna()].copy()

    # Remove non-trading hours (keep only 9:30 - 16:00, which is hours 9-15 inclusive)
    # 9:30 = hour 9, 16:00 = hour 16, so keep hours 9-15 (9:XX to 15:XX)
    df = df[df['Timestamp'].dt.hour.isin(range(9, 16))].copy()

    # Sort by timestamp ascending to ensure chronological order (no look-ahead bias)
    df = df.sort_values('Timestamp').reset_index(drop=True)

    return df


def get_halt_periods(filepath):
    """
    Detect trading halt periods from raw data (before any filtering).
    Returns list of (start_timestamp, end_timestamp) tuples.

    Args:
        filepath: Path to the CSV file

    Returns:
        List of tuples: [(start_ts, end_ts), ...]
    """
    df = pd.read_csv(filepath)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y %H:%M')
    df['SP500_Futures'] = pd.to_numeric(df['SP500_Futures'], errors='coerce')

    # Remove weekends
    df = df[df['Timestamp'].dt.dayofweek < 5].copy()

    halts = []
    in_halt = False
    halt_start = None

    for idx, row in df.iterrows():
        is_null = pd.isna(row['SP500_Futures'])

        if is_null and not in_halt:
            # Start of halt
            in_halt = True
            halt_start = row['Timestamp']
        elif not is_null and in_halt:
            # End of halt
            in_halt = False
            halts.append((halt_start, row['Timestamp']))

    return halts
