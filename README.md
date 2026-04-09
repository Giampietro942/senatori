# 👗 Fashion Gradimento Predictor

> **Progetto:** RISEVET — Corso di Intelligenza Artificiale
>
> **Gruppo:**
> Alfonso Palillo · Giuseppe Vella · Calogero Iacono · Giampietro Vizzini

Modello di machine learning che predice il **gradimento previsto** di un capo d'abbigliamento per un utente, in base alla compatibilità stilistica, alle preferenze eco-sostenibili e al budget.

---

## 📋 Descrizione

Il progetto simula un sistema di raccomandazione per la moda sostenibile. Dati alcuni parametri di un capo e del profilo utente, il modello stima un voto di gradimento da 1 a 10.

La variabile target `gradimento` è costruita come combinazione lineare di:

- **Allineamento stilistico** tra utente e capo (`casual_utente` vs `casual_capo`)
- **Sensibilità eco** dell'utente × sostenibilità del capo
- **Compatibilità di budget** (distanza tra budget utente e prezzo del capo)

---

## 🗂️ Struttura del progetto

```
.
├── model.py          # Script principale
└── README.md
```

---

## ⚙️ Requisiti

- Python 3.8+
- numpy
- pandas
- matplotlib
- scikit-learn

Installazione dipendenze:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## 🚀 Utilizzo

```bash
python model.py
```

Lo script:
1. Genera un dataset sintetico di 500 osservazioni
2. Addestra un `RandomForestRegressor` (100 alberi)
3. Valuta il modello su test set (25%)
4. Mostra un grafico **Reale vs Predetto**
5. Stampa le predizioni per 3 capi di esempio

---

## 📊 Feature utilizzate

| Feature | Descrizione |
|---|---|
| `casual_utente` | Stile casual dell'utente (0=formale, 1=casual) |
| `casual_capo` | Stile casual del capo (0=formale, 1=casual) |
| `eco_utente` | Sensibilità ecologica dell'utente (0–1) |
| `sostenibilita` | Indice di sostenibilità del capo (0–100) |
| `budget` | Budget dell'utente in € |
| `prezzo` | Prezzo del capo in € |

Feature derivate calcolate internamente:

- `allineamento` = `1 - |casual_utente - casual_capo|`
- `fascia` = categorizzazione dell'allineamento in 4 classi

---

## 📈 Metriche di valutazione

Il modello viene valutato con:

- **MAE** (Mean Absolute Error): errore medio assoluto in punti di voto
- **R²** (coefficiente di determinazione): proporzione di varianza spiegata

---

## 👔 Esempio di predizione

```
Giacca Lino Bio      →  gradimento previsto: 8.3/10
Sneakers Hemp        →  gradimento previsto: 7.1/10
Vestito Tencel       →  gradimento previsto: 8.7/10
```

---

## 🌱 Note

Il dataset è completamente sintetico e generato con `numpy.random.seed(0)` per riproducibilità. Per un utilizzo reale, sostituire la generazione dei dati con un dataset autentico mantenendo le stesse colonne.