"""Generate SYNTHETIC monthly CSV reports (one file per month), French-style: ';' separator, ',' decimals."""
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(7)
AS_OF = pd.Timestamp("2024-07-31")
PROJECTS = [  # name, length_km, start offset (days before AS_OF), planned duration, performance
    ("Atlantic Link A", 5200, 230, 300, 0.97), ("Med Ring North", 1800, 190, 200, 0.88),
    ("Gulf Express", 2600, 150, 180, 1.02), ("Pacific Spur 1", 3900, 220, 280, 0.82),
    ("Baltic Connect", 900, 100, 120, 1.05), ("Indian Ocean East", 4700, 210, 290, 0.91),
    ("North Sea Feeder", 650, 75, 90, 0.99), ("Caribbean Loop", 2100, 180, 210, 0.79),
    ("West Africa Coast", 3300, 225, 270, 0.94), ("Arctic Pilot", 1200, 130, 160, 0.86),
    ("Red Sea Bridge", 1500, 120, 140, 1.00),
]
ROUTES = {  # name: (port A, lat, lon, port B, lat, lon) - fictional projects, real ports
    "Atlantic Link A": ("Brest", 48.39, -4.49, "Virginia Beach", 36.85, -75.98),
    "Med Ring North": ("Marseille", 43.30, 5.37, "Heraklion", 35.34, 25.14),
    "Gulf Express": ("Fujairah", 25.12, 56.34, "Mumbai", 19.08, 72.88),
    "Pacific Spur 1": ("Los Angeles", 33.74, -118.27, "Honolulu", 21.31, -157.86),
    "Baltic Connect": ("Kiel", 54.32, 10.14, "Helsinki", 60.17, 24.94),
    "Indian Ocean East": ("Mombasa", -4.04, 39.67, "Chennai", 13.08, 80.27),
    "North Sea Feeder": ("Aberdeen", 57.14, -2.09, "Stavanger", 58.97, 5.73),
    "Caribbean Loop": ("Miami", 25.76, -80.19, "San Juan", 18.47, -66.11),
    "West Africa Coast": ("Lisbon", 38.72, -9.14, "Dakar", 14.69, -17.44),
    "Arctic Pilot": ("Tromso", 69.65, 18.96, "Longyearbyen", 78.22, 15.63),
    "Red Sea Bridge": ("Djibouti", 11.59, 43.15, "Jeddah", 21.54, 39.17),
}
months = pd.date_range("2024-02-29", AS_OF, freq="ME")
rows = []
for name, length, off, dur, perf in PROJECTS:
    start = AS_OF - pd.Timedelta(days=off)
    end = start + pd.Timedelta(days=dur)
    noise = rng.normal(1, 0.06, len(months))
    prev = 0.0
    for m, n in zip(months, noise):
        if m < start:
            continue
        deployed = min(length, max(prev, length * ((m - start).days / dur) * perf * n))
        prev = deployed
        rows.append({"Projet": name, "Date_rapport": m.strftime("%d/%m/%Y"),
                     "Longueur_prevue_km": length, "Longueur_deployee_km": round(deployed, 1),
                     "Longueur_restante_km": round(length - deployed, 1),
                     "Date_fin_prevue": end.strftime("%d/%m/%Y"),
                     **dict(zip(["Port_depart", "Lat_depart", "Lon_depart", "Port_arrivee", "Lat_arrivee", "Lon_arrivee"],
                                ROUTES[name]))})
df = pd.DataFrame(rows)
out = Path(__file__).parent / "sample_data"
out.mkdir(exist_ok=True)
for f in out.glob("*.csv"):
    f.unlink()
for m in months:
    part = df[df.Date_rapport == m.strftime("%d/%m/%Y")]
    part.to_csv(out / f"rapport_{m:%Y-%m}.csv", sep=";", decimal=",", index=False)
print(f"{len(df)} rows across {len(months)} monthly files")
