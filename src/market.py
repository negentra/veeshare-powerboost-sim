"""Layer 3 — Synthetic intra-day MIBEL-style electricity prices.

Generates a deterministic hourly EUR/MWh profile for one representative
week, with morning shoulder, midday solar dip, evening peak, and
overnight trough. NOT fetched from OMIE/MIBEL; synthetic by construction.
"""

from __future__ import annotations

import numpy as np

from src.utils.seed import module_rng

# A-04: 24-hour anchor profile (EUR/MWh), shape calibrated against typical
# Iberian day-ahead patterns. Order of magnitude only.
HOURLY_ANCHOR_EUR_PER_MWH = np.array(
    [
        45, 42, 40, 40, 42, 50,        # 00-05  overnight trough
        70, 95, 110, 95, 80, 65,       # 06-11  morning shoulder + cooling
        50, 35, 25, 30, 50, 75,        # 12-17  midday solar dip + rebound
        110, 130, 140, 120, 90, 65,    # 18-23  evening peak
    ],
    dtype=float,
)


def generate_weekly_prices(seed_root: int, week_hours: int = 168) -> np.ndarray:
    """Return a deterministic EUR/MWh profile of length `week_hours`.

    Uses the 24-hour anchor profile repeated, with small deterministic noise
    (~5%) seeded from seed_root. Saturday/Sunday have a slight discount.
    """
    rng = module_rng(seed_root, "market")
    hours = np.arange(week_hours)
    base = HOURLY_ANCHOR_EUR_PER_MWH[hours % 24]
    # weekend discount (Sat=5, Sun=6 in our DoW convention)
    dow = (hours // 24) % 7
    weekend = ((dow == 5) | (dow == 6)).astype(float)
    multiplier = 1.0 - 0.08 * weekend
    noise = rng.normal(0, 0.05, size=week_hours)
    return base * multiplier * (1.0 + noise)
