![wissal foto](https://github.com/user-attachments/assets/7bed0080-2c6c-448b-a4de-323f0ee1e5de)
---
wissal el bouj
# 📘 PRÉDICTION DE MOUVEMENTS BOURSIERS PAR MACHINE LEARNING

Ce document décortique chaque étape du cycle de vie d'un projet de Machine Learning appliqué à la finance. Il est conçu pour passer du niveau "débutant qui copie du code" au niveau "ingénieur qui comprend les mécanismes internes et les pièges du trading algorithmique".

---

## 1. Le Contexte Métier et la Mission

### Le Problème (Business Case)
Dans le domaine financier, la prédiction des mouvements de marché est le Saint Graal des investisseurs. La volatilité et la complexité des marchés rendent cette tâche extrêmement difficile.
*   **Objectif :** Développer un workflow de Machine Learning pour un dataset financier personnalisé afin de **prédire les mouvements de prix** (hausse vs baisse) à partir d'indicateurs économiques et financiers.
*   **L'Enjeu critique :** La matrice des coûts d'erreur est asymétrique.
    *   Prédire une hausse alors que le prix baisse (Faux Positif) génère des pertes financières directes.
    *   Prédire une baisse alors que le prix monte (Faux Négatif) crée un coût d'opportunité (gains manqués).
    *   **L'IA doit donc équilibrer précision et rappel, avec une attention particulière à la robustesse contre l'overfitting.**

### Les Données (L'Input)
Nous utilisons un *Dataset Financier & Économique personnalisé*.
*   **Période :** 2000-01-01 → 2008-03-18 (8 ans, incluant la crise des subprimes)
*   **X (Features) :** 22 colonnes. Ce ne sont pas des pixels de graphiques, mais des **indicateurs économiques et financiers** (PIB, inflation, taux de change, volume de trading, prix de l'or, etc.).
*   **y (Target) :** Binaire créé manuellement. `1` = Hausse (Close > Open), `0` = Baisse.

### 📊 Composition du Dataset
*   **3,000 observations** réparties sur 3 indices boursiers : Dow Jones, NASDAQ, S&P 500
*   **Distribution de la cible :**
    *   Baisse (0) : 1,545 observations (51.5%)
    *   Hausse (1) : 1,455 observations (48.5%)
    *   ✅ Classes relativement équilibrées (pas de déséquilibre majeur)

---

## 2. Le Code Python (Laboratoire Financier)

Ce script est votre salle de marché algorithmique. Il contient toutes les étapes de la prédiction.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configuration
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')

# --- PHASE 1 : ACQUISITION DES DONNÉES FINANCIÈRES ---
df = pd.read_csv('finance_economics_dataset.csv')

# --- PHASE 2 : FEATURE ENGINEERING (CRÉATION DE LA CIBLE) ---
# La cible n'existe pas : on la crée !
df['Price_Movement'] = (df['Close Price'] > df['Open Price']).astype(int)
# 1 = Hausse intraday, 0 = Baisse intraday

# --- PHASE 3 : DATA WRANGLING (NETTOYAGE) ---
# Encodage des variables catégorielles
le = LabelEncoder()
df['Stock Index'] = le.fit_transform(df['Stock Index'])

# Séparation Features / Target
X = df.drop(['Price_Movement', 'Date'], axis=1)
y = df['Price_Movement']

# Stratégie d'imputation robuste (médiane pour résister aux outliers)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X_clean = pd.DataFrame(X_imputed, columns=X.columns)

# Normalisation (crucial pour des variables d'échelles différentes)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)
X_final = pd.DataFrame(X_scaled, columns=X.columns)

# --- PHASE 4 : ANALYSE EXPLORATOIRE (EDA) ---
print("--- Distribution de la Cible ---")
print(y.value_counts())
print(f"\nÉquilibre : {y.value_counts(normalize=True)*100}")

# --- PHASE 5 : PROTOCOLE EXPÉRIMENTAL (SPLIT) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

# --- PHASE 6 : INTELLIGENCE ARTIFICIELLE (RANDOM FOREST) ---
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# --- PHASE 7 : AUDIT DE PERFORMANCE ---
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print(f"\n--- Accuracy Entraînement : {accuracy_score(y_train, y_pred_train)*100:.2f}% ---")
print(f"--- Accuracy Test : {accuracy_score(y_test, y_pred_test)*100:.2f}% ---")

print("\n--- Rapport Détaillé (Test Set) ---")
print(classification_report(y_test, y_pred_test, target_names=['Baisse', 'Hausse']))

# --- PHASE 8 : ANALYSE DES FEATURES ---
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n--- Top 10 Features les Plus Importantes ---")
print(feature_importance.head(10))

# Visualisation
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt='d', cmap='RdYlGn')
plt.title('Matrice de Confusion : Réalité vs IA')
plt.ylabel('Vraie Classe')
plt.xlabel('Classe Prédite')
plt.show()
```

---

## 3. Analyse Approfondie : Feature Engineering (La Création de la Cible)

### Le Problème de la Variable Cible Manquante
Contrairement aux datasets classiques où la cible est fournie, ici nous devons **créer notre propre définition du succès**.

### La Mécanique de la Création
Nous utilisons une logique simple mais puissante :
```python
df['Price_Movement'] = (df['Close Price'] > df['Open Price']).astype(int)
```

1.  **La Comparaison Booléenne :** Python évalue `Close > Open` et retourne `True` (hausse) ou `False` (baisse).
2.  **La Conversion Numérique (`.astype(int)`) :** Transforme `True` en `1` et `False` en `0`. C'est le format requis par les algorithmes de classification.

### 💡 Le Coin de l'Expert (Choix de la Fenêtre Temporelle)
*Attention :* Cette définition prédit un mouvement **intraday** (dans la même journée). Dans un système de trading réel, vous pourriez vouloir :
*   *Prédire le lendemain :* `df['Target'] = (df['Close Price'].shift(-1) > df['Close Price']).astype(int)`
*   *Prédire une tendance sur 5 jours :* Utiliser les moyennes mobiles ou le rendement cumulé
*   *La différence critique :* Plus l'horizon est long, plus c'est facile à prédire (tendance lourde), mais moins c'est exploitable pour du day-trading.

---

## 4. Analyse Approfondie : Nettoyage & Prétraitement

### A. L'Encodage des Variables Catégorielles
Le dataset contient `Stock Index` (texte : "Dow Jones", "NASDAQ", "S&P 500"). Les algorithmes ne comprennent que les nombres.

**La transformation :**
```python
le = LabelEncoder()
df['Stock Index'] = le.fit_transform(df['Stock Index'])
# "Dow Jones" → 0, "NASDAQ" → 1, "S&P 500" → 2
```

*   *Pourquoi ?* L'algèbre linéaire ne peut pas calculer la "distance" entre deux mots.
*   *Limitation :* Cela impose un ordre artificiel (0 < 1 < 2) qui n'a pas de sens économique. Pour 3 catégories, c'est acceptable. Pour plus, on utiliserait le **One-Hot Encoding**.

### B. Imputation : Médiane vs Moyenne
Nous utilisons `SimpleImputer(strategy='median')` au lieu de `'mean'`.

**Pourquoi la Médiane ?**
*   *Robustesse :* En finance, il y a des événements extrêmes (krachs, bulles). Si le PIB a explosé à +15% une fois (outlier), la moyenne serait tirée vers le haut. La médiane (50ème percentile) est immunisée contre ces valeurs aberrantes.
*   *Exemple :* Données de taux d'intérêt : [2%, 2.1%, 2.2%, 15% (erreur)]. Moyenne = 5.3%, Médiane = 2.15%.

### C. La Normalisation (StandardScaler)
Variables comme `Trading Volume` (milliards) et `Inflation` (pourcentages) ont des échelles radicalement différentes.

**L'impact sans normalisation :**
*   L'algorithme penserait que le Volume est 1000x plus important que l'inflation, juste à cause de l'unité de mesure.

**La transformation :**
$$z = \frac{x - \mu}{\sigma}$$

*   Chaque variable est centrée (moyenne = 0) et mise à l'échelle (écart-type = 1).
*   *Résultat :* Toutes les variables sont comparables sur le même "terrain de jeu".

---

## 5. Analyse Approfondie : L'Algorithme Random Forest 🌲 (Version Financière)

### A. Pourquoi Random Forest pour la Finance ?
Les marchés financiers sont **non-linéaires** et **bruités**.

*   *Non-linéaire :* La relation entre inflation et prix des actions n'est pas une droite. C'est un système complexe avec des seuils, des interactions.
*   *Bruité :* Des événements aléatoires (tweets de PDG, guerres) créent du chaos.

**Random Forest** est résilient car :
1.  Il capture les interactions complexes sans formule mathématique rigide.
2.  Le vote de 200 arbres "lisse" le bruit aléatoire.

### B. Les Hyperparamètres Cruciaux
`n_estimators=200` signifie 200 arbres indépendants.

*   *Trop peu (ex: 10) :* Le vote n'est pas assez diversifié, le modèle est instable.
*   *Trop (ex: 1000) :* Coût de calcul élevé, rendements décroissants (après ~200, les gains de précision sont marginaux).

### C. Le Piège de l'Overfitting en Finance
**Résultat observé :**
*   Accuracy Train : **99.96%** (quasi-parfait)
*   Accuracy Test : **49.00%** (pire qu'une pièce de monnaie)

**Diagnostic : Surapprentissage Massif**

*Explication :* Le modèle a "mémorisé" les 2,400 observations d'entraînement (motifs spécifiques de 2000-2006) mais ne peut pas généraliser sur 2007-2008 (nouvelles conditions de marché).

*Analogie :* C'est comme un étudiant qui mémorise les réponses du sujet d'examen de l'année dernière, mais ne comprend pas les concepts. Devant un nouveau sujet, il échoue.

---

## 6. FOCUS THÉORIQUE : Les Features en Finance 📊

### A. Le Top 10 des Variables Prédictives

| Rang | Feature | Importance | Interprétation Économique |
|------|---------|------------|---------------------------|
| 1️⃣ | **Forex USD/JPY** | 5.64% | Indicateur de risque global (yen = valeur refuge) |
| 2️⃣ | **Bankruptcy Rate** | 5.40% | Santé des entreprises, anticipation de crises |
| 3️⃣ | **Gold Price** | 5.19% | Peur des investisseurs (or = valeur refuge) |
| 4️⃣ | **GDP Growth** | 5.13% | Moteur fondamental de l'économie |
| 5️⃣ | **Trading Volume** | 5.01% | Liquidité du marché, conviction des traders |

### B. La Logique Économique des Features
*   **USD/JPY élevé :** Les investisseurs recherchent du rendement (risk-on), marchés haussiers.
*   **Taux de faillite en hausse :** Signal de récession imminente, marchés baissiers.
*   **Or en hausse :** Peur systémique, fuite vers la sécurité.

### C. La Multicollinéarité (Le Problème en Finance)
Des variables comme `GDP Growth`, `Corporate Profits`, et `Consumer Spending` sont fortement corrélées (>0.8).

*   *Pourquoi ?* Elles mesurent toutes la "santé économique" sous des angles différents.
*   *Impact :* Pour Random Forest, ce n'est pas grave (l'arbre choisit l'une ou l'autre). Mais pour une Régression Logistique, cela créerait de l'instabilité dans les coefficients.

---

## 7. Analyse Approfondie : Évaluation (L'Heure de Vérité)

### A. La Matrice de Confusion
Pour un modèle prédisant ~49% correctement (aléatoire), la matrice révèle :

*   **Vrais Positifs (TP) :** *Prédit Hausse | Réel Hausse.* → Gains captés.
*   **Vrais Négatifs (TN) :** *Prédit Baisse | Réel Baisse.* → Pertes évitées.
*   **Faux Positifs (FP) :** *Prédit Hausse | Réel Baisse.* → **Perte financière directe.**
*   **Faux Négatifs (FN) :** *Prédit Baisse | Réel Hausse.* → **Coût d'opportunité** (gains manqués).

### B. Pourquoi 49% ≈ Hasard ?
Lancer une pièce de monnaie (50%) aurait le même résultat. Le modèle n'a **aucun pouvoir prédictif réel** sur les données de test.

**Cause :** L'overfitting total. Le modèle a appris des corrélations spurieuses (fausses) qui n'existent que dans le jeu d'entraînement.

### C. Les Métriques Avancées (Pour un Bon Modèle)
Si le modèle était performant, on regarderait :

1.  **Sharpe Ratio Algorithmique :**
    $$\text{Sharpe} = \frac{\text{Rendement Moyen} - \text{Taux sans risque}}{\text{Volatilité des Rendements}}$$
    *   Mesure le rendement ajusté au risque.

2.  **Maximum Drawdown :**
    *   La pire perte consécutive. En finance, survivre aux pertes est plus important que maximiser les gains.

3.  **Profit Factor :**
    $$\frac{\text{Somme des gains}}{\text{Somme des pertes}}$$
    *   Doit être >1.5 pour un système viable.

---

## 8. 💡 Solutions au Problème d'Overfitting

### A. Réduire la Complexité du Modèle
```python
model = RandomForestClassifier(
    n_estimators=100,        # Réduire de 200 → 100
    max_depth=5,             # Limiter la profondeur des arbres
    min_samples_leaf=20,     # Forcer au moins 20 exemples par feuille
    random_state=42
)
```

*   *Logique :* Un arbre moins profond ne peut pas mémoriser les détails.

### B. Feature Engineering Avancé
Créer des variables techniques utilisées par les vrais traders :

```python
# Moyennes mobiles
df['MA_5'] = df['Close Price'].rolling(5).mean()
df['MA_20'] = df['Close Price'].rolling(20).mean()

# Volatilité
df['Volatility'] = df['Close Price'].rolling(10).std()

# RSI (Relative Strength Index)
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = compute_rsi(df['Close Price'])
```

### C. Validation Croisée Temporelle
En finance, on ne peut pas mélanger passé et futur (fuite d'information).

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    # Entraîner et évaluer
```

*   *Principe :* Le modèle est toujours testé sur des données **postérieures** à l'entraînement.

### D. Algorithmes Alternatifs
*   **XGBoost :** Régularisation intégrée contre l'overfitting.
*   **LSTM (Réseaux Récurrents) :** Capturent les dépendances temporelles (aujourd'hui dépend d'hier).
*   **Ensemble Stacking :** Combiner Random Forest + XGBoost + Régression Logistique.

---

## 9. Différences avec les Projets Classiques de ML

| Aspect | ML Classique | Finance |
|--------|---------------|---------|
| **Nature du problème** | Classification statique | Prédiction de série temporelle (dynamique) |
| **Coût de l'erreur** | Souvent symétrique | Asymétrique (FP = pertes, FN = gains manqués) |
| **Stabilité des patterns** | Généralement stable | Instable (marchés évoluent) |
| **Overfitting** | Modéré | **Très fréquent** (bruit élevé) |
| **Métrique clé** | Accuracy, F1-Score | Profit Factor, Sharpe Ratio |
| **Validation** | Train/Test classique | **Time Series Split obligatoire** |

---

## 10. Conclusion & Leçons Stratégiques

Ce projet illustre la **différence cruciale** entre ML académique et ML appliqué à la finance :

✅ **Les marchés financiers présentent des défis uniques.**
*   Les marchés sont adversariaux (quelqu'un perd quand vous gagnez).
*   Les patterns changent constamment (non-stationnarité).

✅ **L'overfitting est l'ennemi #1 en finance.**
*   99.96% en train et 49% en test est un **signal d'alarme rouge**.
*   La complexité doit être contrôlée de manière agressive.

✅ **Le Feature Engineering est ROI.**
*   Les indicateurs techniques (RSI, MACD, Bollinger) encodent la "sagesse" de 50 ans de trading.
*   Ils battent souvent les features brutes.

✅ **La validation temporelle est non-négociable.**
*   Tester sur le futur est la seule façon honnête de mesurer la performance.


*"En finance, il ne suffit pas de prédire juste. Il faut prédire mieux que le consensus du marché."*
https://drive.google.com/file/d/1yqnI0RPkmyFXcJuD6R9wM37MzvRjeTSY/view?usp=drive_link
