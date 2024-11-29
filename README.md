# Deep Learning Experiment Series for Bitcoin Prediction

## Projektübersicht
Dieses Projekt konzentriert sich auf die Entwicklung und Verfeinerung von Deep-Learning-Modellen zur Vorhersage von Kryptowährungskursen. Durch eine Serie von Experimenten wird die Verarbeitung von Finanzdaten, die Implementierung technischer Indikatoren und die Modellarchitektur kontinuierlich verbessert. Ziel ist es, robuste Vorhersagen zu treffen und Trends in Kryptowährungsmärkten vorherzusagen.

---

## Python-Version
Das Projekt wurde mit **Python 3.10.11** entwickelt und getestet.
CUDA-Version 12.4.

---

## Abhängigkeiten
Installiere die erforderlichen Python-Bibliotheken mit:
```bash
pip install -r requirements.txt
```

---
<br>
<br>

# Experimente

## Experiment 1: Grundlegende Datenverarbeitung
- **Ziel**: Erste Schritte mit der Verarbeitung von Kryptowährungsdaten.
- **Hauptfokus**: Implementierung grundlegender technischer Indikatoren wie RSI, MACD und CCI.

## Experiment 2: Erweiterung der technischen Indikatoren
- **Ziel**: Einführung zusätzlicher Indikatoren, um die Datenvielfalt zu erhöhen.
- **Hauptfokus**: Nutzung von TA-Lib und sklearn für erweiterte Features.

## Experiment 3: Einführung von Feature-Reduktion
- **Ziel**: Reduktion der Anzahl an Features durch Korrelationsanalyse.
- **Hauptfokus**: Beibehaltung nur der hochkorrelierten Features mit dem Zielwert.

## Experiment 4: Nutzung eines Autoencoders
- **Ziel**: Nicht-lineare Reduktion der Feature-Menge.
- **Hauptfokus**: Einsatz eines Autoencoders zur Transformation und Kompression der Features.

## Experiment 5: Sequenzbasierte Datenaufbereitung
- **Ziel**: Umwandlung der Daten in Sequenzen für zeitliche Modelle.
- **Hauptfokus**: Generierung von Eingabe- und Zielsequenzen basierend auf historischen Kursdaten.

## Experiment 6: Dynamische Modellarchitektur
- **Ziel**: Einführung eines dynamischen Modells mit LSTMs und einem CNN zur Feature-Fusion.
- **Hauptfokus**: Parallelverarbeitung verschiedener Indikatorgruppen durch spezialisierte LSTM-Schichten.

## Experiment 7: Optimierung der Gruppierung und Architektur
- **Ziel**: Verfeinerung der Indikatorgruppen und Einführung von automatischem Padding für kürzere Sequenzen.
- **Hauptfokus**: Stabilere Vorhersageleistung durch modularisierte und erweiterbare Modellstruktur.

---

# Verwendung

## Datenverarbeitung
Für jedes Experiment wird ein eigenes Datenverarbeitungsskript bereitgestellt. Dieses führt folgende Schritte aus:
1. Laden der Kryptowährungsdaten von **Yahoo Finance**.
2. Berechnung technischer Indikatoren.
3. Feature-Reduktion durch Korrelationsanalyse oder Autoencoder.
4. Generierung von Sequenzen zur Modellierung.

## Training
Das Training erfolgt in den **experiment_X_model.py**-Dateien:
- Modelle basieren auf **LSTMs**, **Attention-Mechanismen** und **CNNs**.
- Das Training berücksichtigt Validierungsdaten, um die Performance zu messen.

## Vorhersage
Die Testdaten werden durch die **experiment_X.py**-Dateien verarbeitet:
1. Skalierung und Sequenzierung der Daten.
2. Nutzung des trainierten Modells zur Vorhersage zukünftiger Kursbewegungen.

---

# Visualisierungen
1. **Kursverlauf**: Darstellung der tatsächlichen und vorhergesagten Preise.
2. **Trainingsmetriken**: Plots für Training- und Validation-Loss sowie MSE und RMSE.
---
# Ergebnisse
---
Experiment 1
- Kaufpreis: 23723.76953125
- Tatsächlicher Preis: 23475.466796875
- Vorhergesagter Preis: 24873.352011097533
- RMSE: 1397.8852142225332
- R²: 0.9965
- Prozentualer Fehler: 5.9547%
---
Experiment 2
- Kaufpreis: 23723.76953125
- Tatsächlicher Preis: 23475.466796875
- Vorhergesagter Preis: 22962.032153532404
- RMSE: 513.4346433425962
- R²: 0.9995
- Prozentualer Fehler: 2.1871%
---
Experiment 3
- Kaufpreis: 23723.76953125
- Tatsächlicher Preis: 23475.466796875
- Vorhergesagter Preis: 23139.476272299886
- RMSE: 335.99052457511425
- R²: 0.9998
- Prozentualer Fehler: 1.431241%
---
Experiment 4
- Kaufpreis: 23723.76953125
- Tatsächlicher Preis: 23475.466796875
- Vorhergesagter Preis: 23139.482546299696
- RMSE: 335.98425057530403
- R²: 0.9998
- Prozentualer Fehler: 1.431214%
---
Experiment 5
- Kaufpreis: 23723.76953125
- Tatsächlicher Preis: 23475.466796875
- Vorhergesagter Preis: 19100.634425390042
- RMSE: 4374.832371484958
- R²: 0.9660
- Prozentualer Fehler: 18.635763%
---
Experiment 6
- Kaufpreis: 23723.76953125
- Tatsächlicher Preis: 23475.466796875
- Vorhergesagter Preis: 18871.89256790196
- RMSE: 4603.574228973041
- R²: 0.9623
- Prozentualer Fehler: 19.610150%
---
Experiment 7
- Kaufpreis: 23723.76953125
- Tatsächlicher Preis: 23475.466796875
- Vorhergesagter Preis: 23139.147860452533
- RMSE: 336.31893642246723
- R²: 0.9998
- Prozentualer Fehler: 1.432640%
