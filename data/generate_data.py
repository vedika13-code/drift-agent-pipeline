"""
Synthetic NYC-taxi-like trip data generator.

We simulate ~N_DAYS of daily trip logs with a stable generative process,
then hand the frame to drift.injector to corrupt it at known points in time.
Keeping this synthetic (rather than downloading the real NYC TLC dataset)
means the whole pipeline is reproducible offline and drift ground-truth
is exactly known, which the real dataset can't give us.
"""
import numpy as np
import pandas as pd

N_DAYS = 60
TRIPS_PER_DAY = 400
SEED = 7


def generate_clean_taxi_data(n_days: int = N_DAYS, trips_per_day: int = TRIPS_PER_DAY,
                              seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2023-01-01")

    for day in range(n_days):
        date = start + pd.Timedelta(days=day)
        n = trips_per_day
        trip_distance = rng.lognormal(mean=1.0, sigma=0.5, size=n).clip(0.1, 40)  # miles
        passenger_count = rng.poisson(lam=1.4, size=n).clip(1, 6)
        base_fare = 2.5 + 1.75 * trip_distance
        fare_amount = (base_fare * rng.normal(1.0, 0.08, size=n)).clip(2.5, 250)
        payment_type = rng.choice(["card", "cash"], size=n, p=[0.72, 0.28])
        pickup_hour = rng.integers(0, 24, size=n)
        tip_amount = (fare_amount * rng.uniform(0.05, 0.2, size=n)).clip(0, 60)

        for i in range(n):
            rows.append({
                "day": day,
                "pickup_datetime": date + pd.Timedelta(hours=int(pickup_hour[i])),
                "trip_distance": round(float(trip_distance[i]), 2),
                "passenger_count": int(passenger_count[i]),
                "fare_amount": round(float(fare_amount[i]), 2),
                "tip_amount": round(float(tip_amount[i]), 2),
                "payment_type": payment_type[i],
            })

    df = pd.DataFrame(rows)
    return df


EXPECTED_SCHEMA = {
    "day": "int64",
    "pickup_datetime": "datetime64[ns]",
    "trip_distance": "float64",
    "passenger_count": "int64",
    "fare_amount": "float64",
    "tip_amount": "float64",
    "payment_type": "object",
}

BUSINESS_RULES = {
    "trip_distance": {"min": 0.0, "max": 200.0},
    "fare_amount": {"min": 0.0, "max": 1000.0},
    "tip_amount": {"min": 0.0, "max": 300.0},
    "passenger_count": {"min": 1, "max": 8},
}

if __name__ == "__main__":
    df = generate_clean_taxi_data()
    print(df.head())
    print(df.dtypes)
    print(f"rows={len(df)}, days={df['day'].nunique()}")
