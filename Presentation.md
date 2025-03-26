# DSC PJATK Car Price Prediction — Prezentacja końcowa

**Cyprian Szewczak i Jakub Graliński**  
- Grupa 7


## 1. Wprowadzenie
Witamy w naszej prezentacji rozwiązania konkursowego organizowanego przez **DSC PJATK**. Naszym zadaniem było przygotowanie **modelu predykcyjnego** do przewidywania cen pojazdów na podstawie danych z ofert sprzedaży.

**Kluczowe etapy pracy:**
1. Eksploracyjna analiza danych (EDA)
2. Czyszczenie i inżynieria cech (feature engineering)
3. Budowa i ocena modelu
4. Wnioski i rekomendacje

# A. Analiza i Wizualizacje (EDA + Wnioski)

## 2. Eksploracyjna Analiza Danych
W tej części koncentrujemy się na:
- Wczytaniu danych
- Podstawowych statystykach opisowych
- Identyfikacji braków w danych i wartości odstających
- Prostej wizualizacji rozkładów i zależności


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

sns.set(style="whitegrid")

# Wczytanie danych
df_train = pd.read_csv("Cleaned_train.csv", index_col="ID")  # przykładowa nazwa
df_test  = pd.read_csv("Cleaned_test.csv", index_col="ID")   # przykładowa nazwa
print("Rozmiary zbioru treningowego:", df_train.shape)
df_train.head()
```

    Rozmiary zbioru treningowego: (135397, 116)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cena</th>
      <th>Waluta</th>
      <th>Marka_pojazdu</th>
      <th>Model_pojazdu</th>
      <th>Wersja_pojazdu</th>
      <th>Rok_produkcji</th>
      <th>Przebieg_km</th>
      <th>Moc_KM</th>
      <th>Pojemnosc_cm3</th>
      <th>Kraj_pochodzenia</th>
      <th>...</th>
      <th>SUV</th>
      <th>city_cars</th>
      <th>compact</th>
      <th>convertible</th>
      <th>coupe</th>
      <th>minivan</th>
      <th>sedan</th>
      <th>small_cars</th>
      <th>station_wagon</th>
      <th>Kierownica_strona</th>
    </tr>
    <tr>
      <th>ID</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>13900.0</td>
      <td>1</td>
      <td>Renault</td>
      <td>Grand Espace</td>
      <td>Gr 2.0T 16V Expression</td>
      <td>2005.0</td>
      <td>213000.0</td>
      <td>170.0</td>
      <td>1998.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25900.0</td>
      <td>1</td>
      <td>Renault</td>
      <td>Megane</td>
      <td>1.6 16V 110</td>
      <td>2010.0</td>
      <td>117089.0</td>
      <td>110.0</td>
      <td>1598.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35900.0</td>
      <td>1</td>
      <td>Opel</td>
      <td>Zafira</td>
      <td>Tourer 1.6 CDTI ecoFLEX Start/Stop</td>
      <td>2015.0</td>
      <td>115600.0</td>
      <td>136.0</td>
      <td>1598.0</td>
      <td>Denmark</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5999.0</td>
      <td>1</td>
      <td>Ford</td>
      <td>Focus</td>
      <td>1.6 TDCi FX Silver / Silver X</td>
      <td>2007.0</td>
      <td>218000.0</td>
      <td>90.0</td>
      <td>1560.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>44800.0</td>
      <td>1</td>
      <td>Toyota</td>
      <td>Avensis</td>
      <td>1.8</td>
      <td>2013.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1798.0</td>
      <td>Poland</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 116 columns</p>
</div>



### 2.1 Statystyki opisowe


```python
# Podstawowe statystyki opisowe
df_train.describe(include="all")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Cena</th>
      <th>Waluta</th>
      <th>Marka_pojazdu</th>
      <th>Model_pojazdu</th>
      <th>Wersja_pojazdu</th>
      <th>Rok_produkcji</th>
      <th>Przebieg_km</th>
      <th>Moc_KM</th>
      <th>Pojemnosc_cm3</th>
      <th>Kraj_pochodzenia</th>
      <th>...</th>
      <th>SUV</th>
      <th>city_cars</th>
      <th>compact</th>
      <th>convertible</th>
      <th>coupe</th>
      <th>minivan</th>
      <th>sedan</th>
      <th>small_cars</th>
      <th>station_wagon</th>
      <th>Kierownica_strona</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.353970e+05</td>
      <td>135397.000000</td>
      <td>132046</td>
      <td>132088</td>
      <td>87336</td>
      <td>125572.000000</td>
      <td>1.313940e+05</td>
      <td>131664.000000</td>
      <td>130711.000000</td>
      <td>74977</td>
      <td>...</td>
      <td>135397.000000</td>
      <td>135397.000000</td>
      <td>135397.000000</td>
      <td>135397.000000</td>
      <td>135397.000000</td>
      <td>135397.000000</td>
      <td>135397.000000</td>
      <td>135397.000000</td>
      <td>135397.000000</td>
      <td>135397.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>102</td>
      <td>1139</td>
      <td>16014</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>34</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Volkswagen</td>
      <td>Astra</td>
      <td>2.0 TDI</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Poland</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>11692</td>
      <td>3331</td>
      <td>596</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>36122</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.306938e+04</td>
      <td>0.948913</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2012.090251</td>
      <td>1.531563e+05</td>
      <td>151.716696</td>
      <td>1881.811753</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.195122</td>
      <td>0.112853</td>
      <td>0.149848</td>
      <td>0.011692</td>
      <td>0.029927</td>
      <td>0.103577</td>
      <td>0.155173</td>
      <td>0.028132</td>
      <td>0.188867</td>
      <td>0.001086</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.807748e+04</td>
      <td>0.313601</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.996779</td>
      <td>3.421378e+06</td>
      <td>77.386471</td>
      <td>727.605417</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.396296</td>
      <td>0.316415</td>
      <td>0.356924</td>
      <td>0.107494</td>
      <td>0.170386</td>
      <td>0.304712</td>
      <td>0.362071</td>
      <td>0.165351</td>
      <td>0.391404</td>
      <td>0.032932</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.850000e+02</td>
      <td>-1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1923.000000</td>
      <td>1.000000e+00</td>
      <td>1.000000</td>
      <td>400.000000</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.780000e+04</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2008.000000</td>
      <td>5.300000e+04</td>
      <td>105.000000</td>
      <td>1461.000000</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.580000e+04</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2013.000000</td>
      <td>1.448635e+05</td>
      <td>136.000000</td>
      <td>1798.000000</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.599000e+04</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2017.000000</td>
      <td>2.060000e+05</td>
      <td>173.000000</td>
      <td>1997.000000</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.999000e+06</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2021.000000</td>
      <td>1.111111e+09</td>
      <td>1398.000000</td>
      <td>8400.000000</td>
      <td>NaN</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 116 columns</p>
</div>



### 2.2 Rozkład wybranych zmiennych
Poniżej przykład prostych wykresów przedstawiających rozkład rocznika produkcji i zależność między przebiegiem a ceną.


```python
# Rozkład roku produkcji
plt.figure(figsize=(6,4))
df_train["Rok_produkcji"].hist(bins=20, color='skyblue')
plt.title("Rozkład Roku Produkcji")
plt.xlabel("Rok Produkcji")
plt.ylabel("Liczba ofert")
plt.show()
```


    
![png](Presentation_files/Presentation_7_0.png)
    


#### Interpretacja:
- Większość pojazdów pochodzi z lat 2000–2020.
- Możliwe, że dane dotyczą nowszych aut.


```python
# Cena vs. Przebieg
plt.figure(figsize=(6,4))
sns.scatterplot(data=df_train, x="Przebieg_km", y="Cena", alpha=0.4)
plt.title("Cena vs. Przebieg")
plt.xlabel("Przebieg (km)")
plt.ylabel("Cena (PLN)")
plt.show()
```


    
![png](Presentation_files/Presentation_9_0.png)
    


#### Interpretacja:
- Widzimy dość **ujemną korelację** między przebiegiem a ceną.
- Wyższy przebieg zwykle oznacza niższą cenę.

### 2.3 Braki w Danych i Wartości Odstające
- Sprawdzaliśmy uzupełnianie braków (np. średnia, mediana, tryb).
- Ewentualne wartości odstające (np. bardzo wysokie ceny) były rozważane pod kątem obcięcia (1–99 percentile).

## 3. Wnioski z EDA
1. **Marka** ma duże znaczenie – marki premium (np. Audi, BMW) osiągają wyższe ceny.
2. **Rok produkcji** dodatnio koreluje z ceną (nowsze auta są droższe).
3. **Przebieg** jest jedną z kluczowych cech obniżających wartość.

Z tymi wnioskami przechodzimy do części dotyczącej modelowania.

# B. Modelowanie + Wnioski

## 4. Przygotowanie Zbioru Treningowego
- Zakładamy, że plik `Cleaned_train.csv` zawiera już dane po wstępnym czyszczeniu i inżynierii cech.
- Dzielimy dane na cechy (X) i etykietę (Cena).


```python
df_train_cleaned = df_train.copy()

# Wybieramy wszystkie kolumny poza "Cena" jako cechy
X_all = df_train_cleaned.drop(columns=["Cena"], errors='ignore')
y_all = df_train_cleaned["Cena"]

X_train_l, X_val_l, y_train_l, y_val_l = train_test_split(
    X_all, 
    y_all, 
    test_size=0.2,
    random_state=42
)

print("Rozmiar zbioru treningowego:", X_train_l.shape)
print("Rozmiar zbioru walidacyjnego:", X_val_l.shape)
```

    Rozmiar zbioru treningowego: (108317, 115)
    Rozmiar zbioru walidacyjnego: (27080, 115)


## 5. Budowa i Ocena Modelu
Testowaliśmy kilka algorytmów, m.in. Random Forest, XGBoost.  
Poniżej przykład z **XGBoost** 


```python
# Parametry specyficzne dla GPU
param_grid_restricted = {
    # 21 min
    "n_estimators": [200],  # Zamiast [50, 100, 200, 300]
    "max_depth": [7],  # Zamiast [3, 5, 7, 9]
    "learning_rate": [0.05],  # Zamiast [0.01, 0.05, 0.1, 0.2]
    "subsample": [1],  # Zamiast [0.6, 0.8, 1.0]
    "colsample_bytree": [0.8],  # Zamiast [0.6, 0.8, 1.0]
    "gamma": [0],  # Zamiast [0, 0.1, 0.2]
    "reg_alpha": [0.1],  # Zamiast [0, 0.1, 1]
    "reg_lambda": [0.1],  # Zamiast [0, 0.1, 1]
}
params = {
    "tree_method": "hist",  # Użyj GPU do budowy drzew
    "objective": "reg:squarederror",  # Zadanie regresji
    "eval_metric": "rmse",  # Metryka RMSE
    "gpu_id": 0,  # Użyj GPU o indeksie 0
    "predictor": "gpu_predictor",  # Użyj GPU do predykcji
}

# Inicjalizacja modelu XGBRegressor z parametrami GPU
xgb = XGBRegressor(**params, enable_categorical=True, random_state=42)

# Konfiguracja Grid Search
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid_restricted,
    scoring="neg_mean_squared_error",  # Metryka do optymalizacji (RMSE)
    cv=5,  # 5-krotna walidacja krzyżowa
    n_jobs=1,  # Użyj wszystkich dostępnych rdzeni CPU
    verbose=2,  # Wyświetl postęp
)

# random_search = RandomizedSearchCV(
#     estimator=xgb,
#     param_distributions=param_grid,
#     n_iter=50,  # Przetestuj tylko 50 losowych kombinacji
#     scoring="neg_mean_squared_error",
#     cv=5,
#     n_jobs=-1,
#     verbose=2,
#     random_state=42,
# )

# Trenowanie modelu z Grid Search
grid_search.fit(X_train, y_train)

# Najlepsze parametry i wynik
print("Najlepsze parametry:", grid_search.best_params_)
print("Najlepszy wynik (Negative MSE):", grid_search.best_score_)

# Ocena modelu na zbiorze testowym
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE na zbiorze testowym: {rmse}")
print(f"R² na zbiorze testowym: {r2}")
```

**Uwagi:**
- Dodatkowe strojenie hiperparametrów (GridSearchCV, RandomizedSearchCV, Optuna) pozwoliłoby zredukować błąd.
- Uzyskane ~19k RMSE (np. 19107.78) jest naszym punktem odniesienia.

### 5.1 Zapis Ostatecznych Wyników
Po wybraniu najlepszego modelu używamy go do przewidzenia cen w zbiorze testowym.


```python
# Przygotowanie zbioru testowego (analogiczne preprocessing)
df_test_cleaned = df_test.copy()

X_test_final = df_test_cleaned.drop(columns=["Cena"], errors='ignore')

# Predykcja
y_pred_test = xgb_model.predict(X_test_final)

# Tworzenie pliku submission
submission = pd.DataFrame({
    "ID": df_test_cleaned.index,
    "Cena": y_pred_test
})
submission.to_csv("submission.csv", index=False)
print("Zapisano plik submission.csv")
```

## 6. Największe Wyzwania i Jak Sobie z Nimi Poradziliśmy
1. **Braki w danych** – Imputacja (średnia/mediana/najczęstsza kategoria).
2. **Różne waluty** – Konwersja na PLN (jeśli występowało EUR, itp.).
3. **Wartości odstające** – Rozważenie usunięcia lub obcięcia (1–99 percentyl).
4. **Wybór cech** – Marka, rok produkcji i przebieg okazały się kluczowe.

## 7. Podsumowanie i Wnioski
- **Model:** Użyliśmy algorytmu gradient boosting (XGBoost) z parametrami dopasowanymi do danych.
- **RMSE:** Wartość na walidacji ~19k PLN. Możliwe dalsze usprawnienia przez bardziej rozbudowaną inżynierię cech.
- **Przydatność:** Tego typu model może pomagać dealerom w szybszym i bardziej trafnym szacowaniu cen.

### Możliwe dalsze kroki
- Łączenie modeli w stacking/ensemble.

## Dziękujemy za uwagę!


```python

```
