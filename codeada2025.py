import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

col1, col2 = st.columns([1, 8])  

with col1:
    st.write("")
    st.image("djun2.png", width=50)

with col2:
    st.title("Health Expenditure Forecast")

st.markdown(
    """
    <style>
    /* Make the sidebar background dark */
    section[data-testid="stSidebar"] {
        background-color: #2c3e50;  /* dark navy */
        color: white;
    }

    /* Force all text inside the sidebar to white */
    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Optional: change font if needed */
    section[data-testid="stSidebar"] {
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)


df = pd.read_csv("dataset_updated.csv")
df = df.drop(columns=["Indicator Name"], errors='ignore')
cols = ['Country Name', 'Country Code'] + [str(y) for y in range(2000, 2024) if str(y) in df.columns]
df = df[cols]

df_long = df.melt(id_vars=['Country Name', 'Country Code'], var_name='Year', value_name='Health_Expenditure_%GDP')
df_long['Year'] = df_long['Year'].astype(int)
df_long = df_long.dropna()

st.sidebar.header("Forecast Settings")
st.sidebar.markdown("""Choose a country from the dropdown and use the slider to choose a year to forecast health expenditure (% of GDP).  """)

country = st.sidebar.selectbox("Select Country", sorted(df_long['Country Name'].unique()))
forecast_year = st.sidebar.slider("Select Year to Predict", 2021, 2050, 2030)

country_data = df_long[df_long['Country Name'] == country].copy()
country_data = country_data.dropna(subset=['Year', 'Health_Expenditure_%GDP'])

train_data = country_data[country_data['Year'] <= 2020]
X_train = train_data[['Year']]
y_train = train_data['Health_Expenditure_%GDP']

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)

rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)

short_term_years = np.arange(2021, 2026).reshape(-1, 1)
long_term_years = np.arange(2026, 2051).reshape(-1, 1)

short_term_preds = rf.predict(short_term_years)
long_term_preds = lin_model.predict(long_term_years)

full_years = np.vstack((short_term_years, long_term_years))
full_preds = np.concatenate((short_term_preds, long_term_preds))

y_train_pred = lin_model.predict(X_train)
residuals = y_train - y_train_pred
std_error = np.std(residuals)
ci_upper = full_preds + 1.96 * std_error
ci_lower = full_preds - 1.96 * std_error

if forecast_year <= 2025:
    pred_value = rf.predict(np.array([[forecast_year]]))[0]
else:
    pred_value = lin_model.predict(np.array([[forecast_year]]))[0]

sns.set_theme(style="whitegrid")
sns.set_context("talk")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(X_train['Year'], y_train, 'o-', label='Train (≤2020)', color='tab:blue')
ax.plot(full_years, full_preds, '-', color='tab:orange', linewidth=2, label='Hybrid Forecast (2021–2050)')
ax.fill_between(full_years.flatten(), ci_upper, ci_lower, color='orange', alpha=0.2, label='95% Confidence Interval')
ax.scatter(forecast_year, pred_value, color='red', s=80, zorder=5, label=f'Predicted {forecast_year}')
ax.annotate(f"{forecast_year}: {pred_value:.2f}%", xy=(forecast_year, pred_value), xytext=(forecast_year + 2, pred_value + 0.5),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')
ax.set_title(f"{country} Health Expenditure (% GDP): Hybrid Forecast", fontsize=16, fontweight='bold')
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Health Expenditure (% of GDP)", fontsize=12)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

st.pyplot(fig)
st.metric(label=f"Predicted Health Expenditure for {forecast_year}", value=f"{pred_value:.2f}%")
st.markdown(f"""> In **{forecast_year}**, the predicted health expenditure for **{country}** is **{pred_value:.2f}% of GDP**.
""")
