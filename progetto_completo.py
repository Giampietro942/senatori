import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor            # Domanda 1
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Esempio di dati ipotetici
n = 500
np.random.seed(0)
casual_utente = np.random.rand(n)
casual_capo = np.random.rand(n)
eco_utente = np.random.rand(n)
sostenibilita = np.random.rand(n) * 100
budget = np.random.rand(n) * 300
prezzo = np.random.rand(n) * 300

gradimento = (
    3.0 +
    2.5 * (1 - np.abs(casual_utente - casual_capo)) +          # Domanda 2
    1.5 * eco_utente * (sostenibilita / 100) +
    1.5 * (1 - np.abs(budget - prezzo) / 300) +
    np.random.normal(0, 0.5, n)
).clip(1, 10)                                                  # Domanda 3

df = pd.DataFrame({
    'casual_utente': casual_utente,
    'casual_capo': casual_capo,
    'eco_utente': eco_utente,
    'sostenibilita': sostenibilita,
    'budget': budget,
    'prezzo': prezzo,
    'gradimento': gradimento
})

df['allineamento'] = 1 - np.abs(df['casual_utente'] - df['casual_capo'])  # Domanda 4
df['fascia'] = pd.cut(df['allineamento'], bins=4,                         # Domanda 5
                      labels=['Molto diverso', 'Diverso', 'Simile', 'Identico'])

# Modello di regressione
X = df[['casual_utente', 'casual_capo', 'eco_utente', 'sostenibilita', 'budget', 'prezzo']]
y = df['gradimento']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

modello = RandomForestRegressor(n_estimators=100, random_state=42)
modello.fit(X_train, y_train)

y_pred = modello.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)                      # Domanda 6
r2  = r2_score(y_test, y_pred)

# Visualizzazione
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(y_test, y_pred, alpha=0.4, s=25, color='#8e44ad')
ax.plot([1,10], [1,10], 'r--', linewidth=1.5, label='Predizione perfetta')  # Domanda 7
ax.text(1.3, 9.0, f'MAE={mae:.2f}  R2={r2:.2f}',              # Domanda 8
        fontsize=10, color='darkblue', fontweight='bold')
ax.set_xlabel('Reale')
ax.set_ylabel('Predetto')
ax.legend()
plt.show()

# Predizione su nuovi capi
nomi_capi = ['Giacca Lino Bio', 'Sneakers Hemp', 'Vestito Tencel']

capi_da_valutare = pd.DataFrame({
    'casual_utente': [0.8, 0.4, 0.2],
    'casual_capo': [0.7, 0.3, 0.2],
    'eco_utente': [0.9, 0.6, 0.8],
    'sostenibilita': [80, 60, 90],
    'budget': [150, 120, 180],
    'prezzo': [140, 110, 175]
})

pred = modello.predict(capi_da_valutare)                      # Domanda 9

for nome, voto in zip(nomi_capi, pred):
    print(f"  {nome:20s}  →  gradimento previsto: {voto:.1f}/10")  # Domanda 10
