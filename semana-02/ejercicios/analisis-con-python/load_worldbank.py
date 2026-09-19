import os

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

## CARGA DE DATOS
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "worldbank-life-expectancy.csv")

df = pd.read_csv(DATA, skiprows=4)
table = df.to_numpy()

## EXPLORACION
print(f"table: {table.shape} (rows x columns), dtype {table.dtype}")
print(f"columns: {df.columns[0]!r} .. {df.columns[-1]!r}")
print(f"missing cells (NaN): {int(np.isnan(table[:, 4:].astype(float)).sum())} of {table.size}")
print(f"sample row: {table[0, 0]} ({table[0, 1]}), {df.columns[4]} = {table[0, 4]}")

# ignore first 4 columns from the table, take only the life expectancy data through the years
grid = table[:, 4:-1].astype(float)
# years casted to ints from strings so they dont overlap when plotting
years = df.columns[4:-1].to_numpy(dtype=int)

## ANALISIS
recorded = ~np.isnan(grid).all(axis=0)
first, last = np.flatnonzero(recorded)[[0, -1]]
print(f"earliest recorded year {years[first]}: "
      f"mean life expectancy {np.nanmean(grid[:, first]):.1f} "
      f"({np.isfinite(grid[:, first]).sum()} countries)")
print(f"latest recorded year {years[last]}: "
      f"mean life expectancy {np.nanmean(grid[:, last]):.1f} "
      f"({np.isfinite(grid[:, last]).sum()} countries)")

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

## VISUALIZACION
mean_per_year = np.nanmean(grid[:, recorded], axis=0)
plt.figure(figsize=(8, 4.5))
plt.plot(years[recorded], mean_per_year)
plt.xlabel("year")
plt.ylabel("mean life expectancy (years)")
plt.title("Mean life expectancy across countries, by year")
plt.savefig(os.path.join(PLOTS_DIR, "life-expectancy-mean.png"),
            dpi=150, bbox_inches="tight")

## INTERPRETACION
"""
La esperanza de vida ha aumentado en 19.6 años desde 1960 hasta 2024,
este aumento ha sido constante con la excepción de 2020-2021, donde
se ha puede observar una caida respecto a 2019, seguramente debido al
COVID-19
"""
