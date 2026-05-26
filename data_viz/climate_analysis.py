"""
Climate Analysis
==========================
Description: Analyzes climate and energy data from 1990–2023 by visualizing trends in CO2 emissions and
renewable energy adoption across countries, and exploring relationships between emissions, temperature, and
GDP to assess environmental and economic patterns.

Author : Nathaniel Morrison
Date   : 05/22/2026
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)

data = pd.read_csv("data/climate_energy.csv")
print(data)

print(data["country"].unique())
print(data.isna().count())



# 1. Trends over time — How have CO2 emissions and renewable energy percentage changed from 1990-2023 for each country?
# 2. Temperature vs CO2 — Is there a correlation between CO2 emissions and average temperature across countries?
# 3. GDP vs Emissions — Do wealthier countries emit more CO2, or are they transitioning to cleaner energy?

sns.lineplot(data=data, x="year", y="renewable_pct", hue="country")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Year")
plt.ylabel("Renewable Percent")
plt.title("Renewable Percent by Year since 1990")
plt.tight_layout()

sns.lineplot(data=data, x="year", y="co2_emissions_mt", hue="country")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel("Year")
plt.ylabel("CO2 Emissions (Metric Tons)")
plt.title("CO2 by Year since 1990")
plt.tight_layout()


# Note: total CO2 emissions are influenced by population size.
# Per capita emissions would provide a more accurate comparison across countries.

sns.scatterplot(data=data, x="co2_emissions_mt", y="avg_temp_c", hue="country")

#The chart does not show a big change with CO2 emission and temperature change.
#10 countries over a 33-year time period is not enough to determine a strong correlation.

sns.lmplot(data=data, x="co2_emissions_mt", y="avg_temp_c", hue="country")

#The trend lines shows a correlation with temperature increase with an increase of CO2 emissions.

sns.lmplot(data=data, x="gdp_billion_usd", y="co2_emissions_mt", hue="country",
           legend=False, height=6, aspect=1.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Country")
plt.tight_layout()
plt.show()

#Wealthy countries invest more in renewables but still emit high CO2 due to
#those countries tend to have more manufacturing, heavy industry, oil and gas, and long distance freight which drives CO2 emissions and GDP.
