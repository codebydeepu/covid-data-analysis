import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd

#Loading the dataset using Pandas.
df = pd.read_csv("D:\Seaborn\project\owid-covid-data.csv")

#Displaing first and last 5 rows.
print(f"First 5 rows {df.head()}")
print(f"Last 5 rows {df.tail()}")

#Showing dataset shape (rows, columns)
print(f"\nRows and Columns : {df.shape}")

#Showing column names.
print(f"\nColumns  : {df.columns}")

#Checking data types.
print(f"\n Data Types : \n{df.dtypes}")

#Finding missing values.
print(f"\nMissing values \n{df.isnull().sum()}")

#Handling missing values
cols_needed = [
    "continent","location","date",
    "total_cases","new_cases",
    "total_deaths","new_deaths",
    "total_vaccinations","people_vaccinated",
    "population"
]

df["continent"] = df["continent"].fillna(0)
df["new_cases"] = df["new_cases"].fillna(0)
df["new_deaths"] = df["new_deaths"].fillna(0)
df["total_cases"] = df["total_cases"].fillna(0)
df["total_deaths"] = df["total_deaths"].fillna(0)
df["total_vaccinations"] = df["total_vaccinations"].fillna(0)
df["people_vaccinated"] = df["people_vaccinated"].fillna(0)
df = df.dropna(subset=["location","date"])
print(f"\nAfter Handling Missing values \n{df.isnull().sum()}")


#Converting the date column into datetime format
df["date"] = pd.to_datetime(df["date"])


#Total global cases and deaths
latest = df.sort_values("date").groupby("location").last()

Total_confirmed_cases = latest["total_cases"].sum()
Total_deaths = latest["total_deaths"].sum()
Total_vaccinations = latest["total_vaccinations"].sum()

print(f"\nTotal confirmed cases : {Total_confirmed_cases/1e6:.2f} Million")
print(f"Total deaths : {Total_deaths/1e6:.2f} Million")
print(f"Total vaccinations : {Total_vaccinations/1e9:.2f} Billion")

#Finding the top 10 Location with the highest total cases.
highest_cases = df.groupby("location")["total_cases"].max().sort_values(ascending=False).head(10)
print(f"\nTop 10 Location with the highest total cases :\n{highest_cases.round(0)}")

#Locations with highest death rate
df["Deaths rate"] = (df["total_deaths"] / df["total_cases"])*100
dt_rate_cntry = df.groupby("location")["Deaths rate"].last().sort_values(ascending=False).head(5)
print(f"\nLocation with highest death rate : \n{dt_rate_cntry}")

#Global cases over time
global_cases = df.groupby("date")["new_cases"].sum()
plt.figure(figsize=(10,5))
plt.plot(global_cases , color="r")
plt.title("Global COVID-19 Cases Over Time")
plt.xlabel("Date")
plt.ylabel("Daily Cases")
plt.grid()
plt.tight_layout()
plt.savefig("covid_global_cases_time.png",dpi=300)
plt.show()

#COVID trend for a specific country
#Total Death
india = df[df["location"] == "India"]
india = india[india["total_deaths"].notna()]
india = india[india["total_deaths"] > 0]
india["date"] = pd.to_datetime(india["date"])
plt.figure(figsize=(12,6))
plt.plot(india["date"], india["total_deaths"], linewidth=2)
plt.title("COVID-19 Total Deaths in India")
plt.xlabel("Date")
plt.ylabel("Total Deaths")
plt.grid()
plt.savefig("Covid_Total_deaths_India.png",dpi=300)
plt.show()

#New Deaths in India
plt.figure(figsize=(12,6))
plt.plot(india["date"], india["new_deaths"])
plt.title("Daily COVID-19 Deaths in India")
plt.xlabel("Date")
plt.ylabel("Daily Deaths")
plt.grid(True)
plt.savefig("covid_new_death_India.png",dpi=300)
plt.show()

#Total Vaccinations in India
plt.figure(figsize=(10,5))
plt.plot(india["date"] , india["total_vaccinations"])
plt.title("COVID-19 Total Vaccinations in India")
plt.xlabel("Date")
plt.ylabel("Total Vaccinations")
plt.tight_layout()
plt.savefig("Total_covid_vaccination_india.png",dpi=300)
plt.show()

#Finding the top 10 locations with highest vaccination numbers.
df["continent"] = df["continent"].astype(str).str.strip()
df = df[df["continent"] != "0"]
df = df[df["continent"] != ""]
df = df[df["continent"] != "nan"]

high_vaccination = df.groupby("location")["total_vaccinations"].max().sort_values(ascending=False).head(10)
print(f"\nTop 10 locations with highest vaccination numbers: \n{high_vaccination}")
plt.figure(figsize=(22,8))
sns.barplot(x=high_vaccination.index , y = high_vaccination.values, palette="tab20")
plt.title("Top 10 locations with highest vaccination numbers")
plt.ylabel("Vaccination number")
plt.grid("lightgrid")
plt.tight_layout()
plt.savefig("Top_loc_high_vaccination_covid.png",dpi=300)
plt.show()

#Relationship between cases and deaths
sns.scatterplot(x=df["new_cases"],y=df["new_deaths"],hue=df["continent"] ,alpha=0.5)
plt.xlabel("New Cases")
plt.ylabel("New Deaths")
plt.legend()
plt.grid()
plt.title("Relationship Between Cases and Deaths")
plt.savefig("covid_cases&deaths.png",dpi=300)
plt.show()

#Find total cases for each continent.
continent_cases = (df.groupby(["continent","location"])["total_cases"].max().groupby("continent").sum())
print(f"\nThe total cases of each continent : \n{continent_cases}")
plt.figure(figsize=(10,5))
sns.barplot(x=continent_cases.index, y=continent_cases.values)
plt.title("Total COVID-19 Cases by Continent")
plt.xlabel("Continent")
plt.ylabel("Total Cases")
plt.tight_layout()
plt.savefig("Total_covide_cases_continent.png",dpi=300)
plt.show()

#Identify the worst COVID wave
df["Month"]=pd.to_datetime(df["date"]).dt.to_period("M")
monthly_cases = df.groupby("Month")["new_cases"].sum()
worst_month = monthly_cases.idxmax()
max_cases = monthly_cases.max()

print("\nWorst COVID month:", worst_month)
print("Total cases:", max_cases)

#Final Insight 
print("\nFINAL INSIGHT")
print("The country with the highest COVID-19 cases was United States," \
" indicating a significant outbreak compared to other nations.")
print("Asia was the most affected continent, contributing the largest share of global cases.")
print("The biggest wave of COVID-19 occurred around 2022-01, where global daily cases peaked, " \
"showing the most severe phase of the pandemic.")
print("Countries like China,India and United States vaccinated the highest number of people," \
"reflecting strong vaccination efforts.")