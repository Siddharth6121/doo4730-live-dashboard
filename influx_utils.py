"""Shared InfluxDB connection helper. Credentials load from .env (not hardcoded)."""
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from influxdb_client_3 import InfluxDBClient3


def get_client():
    return InfluxDBClient3(
        host=os.environ["INFLUX_HOST"],
        token=os.environ["INFLUX_TOKEN"],
        database=os.environ["INFLUX_DATABASE"],
    )


def query_chunked(client, table, columns, device_id, start, end, chunk_days, extra_where=""):
    """Pull a table in time chunks to stay under the server's 432-Parquet-file scan limit.
    Returns a single concatenated DataFrame.
    """
    import pandas as pd

    cols = ", ".join(columns)
    where_extra = f" AND {extra_where}" if extra_where else ""
    parts = []
    t = pd.Timestamp(start)
    end = pd.Timestamp(end)
    chunk = pd.Timedelta(days=chunk_days)
    while t < end:
        t2 = min(t + chunk, end)
        q = (
            f"SELECT {cols} FROM {table} "
            f"WHERE device_id = '{device_id}' "
            f"AND time >= TIMESTAMP '{t.strftime('%Y-%m-%d %H:%M:%S')}' "
            f"AND time < TIMESTAMP '{t2.strftime('%Y-%m-%d %H:%M:%S')}'"
            f"{where_extra}"
        )
        df = client.query(q, language="sql").to_pandas()
        if len(df):
            parts.append(df)
        t = t2
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
