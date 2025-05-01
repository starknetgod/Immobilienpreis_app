
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Titel
st.title("🏠 Web-App zur Vorhersage des House Price Index (HPI)")
st.markdown("Basierend auf Inflation, Arbeitslosenrate und Leitzins")
st.markdown("**Hinweis:** Der House Price Index (HPI) ist ein normierter Index mit Basisjahr 2015 = 100. Ein Wert von 130 bedeutet: Immobilienpreise sind um 30 % gegenüber 2015 gestiegen.")

# Daten laden
df = pd.read_csv("data_final_ökonometrie_projekt.csv")

# Modell vorbereiten
X = df[['inflation', 'arbeitslosenrate', 'leitzinsen']]
y = df['hpi']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Benutzereingaben
st.sidebar.header("🔢 Eingabewerte für Vorhersage")
inflation = st.sidebar.slider("Inflation (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
arbeitslosigkeit = st.sidebar.slider("Arbeitslosenrate (%)", min_value=0.0, max_value=30.0, value=6.0, step=0.1)
leitzins = st.sidebar.slider("Leitzins (%)", min_value=-1.0, max_value=20.0, value=3.0, step=0.1)

# DataFrame mit Eingaben
input_data = pd.DataFrame({
    'inflation': [inflation],
    'arbeitslosenrate': [arbeitslosigkeit],
    'leitzinsen': [leitzins]
})

# Vorhersage berechnen
prediction = model.predict(input_data)[0]

# Ausgabe
st.subheader("📈 Vorhergesagter HPI")
st.success(f"Der prognostizierte HPI beträgt: **{round(prediction, 2)}**")

# Hinweis
st.markdown("---")
st.markdown("📊 Modell: Random Forest Regression")
# Quellenangabe
st.markdown("📚 Datenquelle: Eigene Berechnungen basierend auf öffentlich verfügbaren Wirtschaftsdaten.")
st.markdown("💡 Hinweis: Dieses Modell dient nur zu Demonstrationszwecken und sollte nicht für tatsächliche Investitionsentscheidungen verwendet werden.")
