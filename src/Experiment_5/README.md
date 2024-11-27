# Experiment 5: Hybride Transformer-LSTM-Architektur für Bitcoin-Kursprognose

## **Kurzbeschreibung**
Das fünfte Experiment kombiniert die Stärken von Transformer- und LSTM-Architekturen, um die Vorhersage der Bitcoin-Kursänderung (prozentualer Gewinn/Verlust) weiter zu verbessern. Transformer-Komponenten bieten eine effektive Feature-Interaktion durch Attention-Mechanismen, während LSTMs weiterhin zeitliche Abhängigkeiten innerhalb der Daten berücksichtigen. Zusätzliche Indikatoren und verbesserte Trainingsmethoden optimieren die Leistung dieses hybriden Modells.

---

## **Datenbeschaffung**
- **Datenquelle**: `yfinance`
- **Kryptowährung**: Bitcoin (`BTC-USD`)
- **Zeitraum**: 2010-01-01 bis 2023-01-31
- **Aufteilung**:
  - **Trainingsdaten**: 2010-2022
  - **Testdaten**: 2023
- **Merkmale (Features)**:
  - **Standard**: Open, High, Low, Close, Volume
  - **Indikatoren-Gruppe 1**: RSI, MACD, MACD-Signal, Momentum, Stochastic Oscillator, MFI, WILLR, ADX, CCI20
  - **Indikatoren-Gruppe 2**: CCI, ROC, Bollinger-Bänder (oberes und unteres Band), SMA, EMA, OBV
  - **Erweiterte Indikatoren**: VWAP, ATR, Keltner Channels, Trend Intensity Index
- **Zielvariable**: Vorhergesagter Preis 30 Tage in die Zukunft.

---

### **Standard Features**
1. **Open**: Der Eröffnungspreis der Kryptowährung an einem bestimmten Tag.
2. **High**: Der höchste Preis der Kryptowährung an einem bestimmten Tag.
3. **Low**: Der niedrigste Preis der Kryptowährung an einem bestimmten Tag.
4. **Close**: Der Schlusskurs der Kryptowährung an einem bestimmten Tag.
5. **Volume**: Das Handelsvolumen, d. h. die Menge an gehandelten Kryptowährungseinheiten an einem bestimmten Tag.

---

### **Momentum-Indikatoren (Indicators Group 1)**
1. **RSI (Relative Strength Index)**:
   - Misst die Geschwindigkeit und Änderung der Kursbewegungen.
   - Werte über 70 deuten auf eine überkaufte Situation hin, während Werte unter 30 auf eine überverkaufte Situation hinweisen.
2. **MACD (Moving Average Convergence Divergence)**:
   - Zeigt den Unterschied zwischen zwei gleitenden Durchschnitten des Kurses.
   - Erkennt Trends und Umkehrpunkte.
3. **MACD-Signal**:
   - Ein gleitender Durchschnitt des MACD.
   - Dient als Signallinie für Kauf- und Verkaufssignale.
4. **Momentum**:
   - Misst die Kursänderung zwischen aufeinanderfolgenden Tagen.
   - Positiv bei steigendem Kurs, negativ bei fallendem Kurs.
5. **Stochastic Oscillator**:
   - Zeigt die relative Position des Schlusskurses im Verhältnis zu dessen Hoch- und Tiefpunkten.
   - Werte über 80 = überkauft, Werte unter 20 = überverkauft.
6. **MFI (Money Flow Index)**:
   - Ein Volumen-basierter Indikator für den Geldfluss.
   - Werte über 80 = überkauft, Werte unter 20 = überverkauft.
7. **WILLR (Williams %R)**:
   - Misst überkaufte oder überverkaufte Bedingungen.
   - Werte nahe 0 = überkauft, Werte nahe -100 = überverkauft.
8. **ADX (Average Directional Index)**:
   - Misst die Stärke eines Trends, unabhängig von der Richtung.
   - Werte über 25 = starker Trend.
9. **CCI20 (Commodity Channel Index, 20-Tage)**:
   - Misst die Abweichung des Preises von seinem Durchschnitt.
   - Hohe Werte = überkauft, niedrige Werte = überverkauft.

---

### **Volatilitäts-Indikatoren (Indicators Group 2)**
1. **CCI (Commodity Channel Index)**:
   - Misst die Abweichung des Kurses von seinem Durchschnitt.
   - Hohe Werte = überkauft, niedrige Werte = überverkauft.
2. **ROC (Rate of Change)**:
   - Prozentsatz der Kursänderung über einen bestimmten Zeitraum.
   - Misst Stärke und Richtung des Trends.
3. **Bollinger-Bänder (Upper Band)**:
   - Oberer Bereich, in dem sich Preise normalerweise bewegen.
4. **Bollinger-Bänder (Lower Band)**:
   - Unterer Bereich, in dem sich Preise normalerweise bewegen.
5. **SMA (Simple Moving Average)**:
   - Einfache gleitende Durchschnitte zur Trendanalyse.
6. **EMA (Exponential Moving Average)**:
   - Gleitende Durchschnitte mit stärkerer Gewichtung aktueller Werte.
7. **OBV (On-Balance Volume)**:
   - Misst den kumulierten Geldfluss basierend auf Kurs und Volumen.

---

### **Erweiterte Indikatoren in Experiment 5**
1. **VWAP (Volume Weighted Average Price)**:
   - Gewichteter durchschnittlicher Preis basierend auf Volumen und Preis.
2. **ATR (Average True Range)**:
   - Misst die Marktvolatilität.
3. **Keltner Channels**:
   - Kombiniert Volatilität und Trend.
4. **Trend Intensity Index**:
   - Quantifiziert die Stärke eines Trends.

---

## **Architektur und detaillierte Funktionsweise**

Das Modell in Experiment 5 erweitert die hybride Architektur aus Experiment 4 durch Optimierungen in der Trainingspipeline und die Integration neuer Indikatoren. Der Fokus liegt darauf, die Leistung durch längere Sequenzen, verbesserte Feature-Fusion und dynamische Anpassung der Lernrate zu steigern.

---

### **1. Architektur**

1. **Standard-Features (z. B. OHLCV)**:
   - Verarbeitet durch ein Transformer-Encoder-Modul.
   - Nutzt Preisdaten und Volumen als Basisfeatures, die durch lineare Transformationen normalisiert werden.

2. **Momentum-Indikatoren (Gruppe 1)**:
   - Verarbeitet mit einem dedizierten Transformer-Encoder.
   - Indikatoren wie RSI, MACD, und Momentum werden analysiert, um relevante zeitliche Muster zu erfassen.

3. **Volatilitäts-Indikatoren (Gruppe 2)**:
   - Verarbeitet durch einen LSTM-Encoder.
   - Indikatoren wie Bollinger-Bänder, ATR und Keltner Channels dienen zur Identifikation von Trends und Volatilitäten.

4. **Feature Fusion Layer**:
   - Kombiniert die Outputs der Transformer- und LSTM-Module.
   - Verwendet Attention-Mechanismen, um die wichtigsten Features zu gewichten.

5. **Sequence Aggregation Layer**:
   - Konsolidiert die fusionierten Informationen.
   - Reduziert die dimensionalen Repräsentationen, um sie für die Kursvorhersage zu optimieren.

6. **Fully Connected Output Layer**:
   - Übersetzt die aggregierten Features in die Zielvariable (Vorhersage des Bitcoin-Kurses 30 Tage in die Zukunft).
   - Nutzt eine Kombination aus linearen Schichten und Aktivierungsfunktionen.

---

### **2. Klassen und Funktionen**

#### **a) `HybridTransformerLSTMModel`**
- **Zweck**: Verarbeitet mehrere Eingabesequenzen und kombiniert Transformer- und LSTM-Komponenten.
- **Input**:
  - Sequenzen der drei Feature-Gruppen: Standard-Features, Momentum-Indikatoren und Volatilitäts-Indikatoren.
- **Architektur**:
  - Verwendet Transformer-Encoder für Standard- und Momentum-Indikatoren.
  - Nutzt LSTM-Encoder für Volatilitäts-Indikatoren.
  - Fusioniert die Outputs mit einem Attention-Layer.
- **Output**:
  - Eine konsolidierte Sequenz, die für die finale Vorhersage verwendet wird.

#### **b) `TransformerEncoderModule`**
- **Zweck**: Extrahiert Feature-Interaktionen aus zeitlichen Sequenzen.
- **Mechanismus**:
  - Besteht aus mehreren Transformer-Blöcken, die durch Self-Attention Mechanismen zeitliche Abhängigkeiten modellieren.
  - Regularisiert durch Dropout-Layer.
- **Output**:
  - Eine gewichtete Repräsentation der Eingabesequenz.

#### **c) `LSTMEncoderModule`**
- **Zweck**: Modelliert zeitliche Muster in den Volatilitäts-Indikatoren.
- **Mechanismus**:
  - Nutzt mehrere LSTM-Schichten, um kurz- und langfristige Abhängigkeiten zu erfassen.
- **Output**:
  - Die letzten versteckten Zustände der LSTM-Schichten.

#### **d) `AttentionFusionLayer`**
- **Zweck**: Kombiniert die Repräsentationen der drei Feature-Gruppen.
- **Mechanismus**:
  - Berechnet Attention-Scores, um die relevantesten Features aus jeder Gruppe hervorzuheben.
- **Output**:
  - Eine fusionierte Sequenz, die die wichtigsten Informationen aller Gruppen enthält.

#### **e) `FullyConnectedDecoder`**
- **Zweck**: Erzeugt die finale Vorhersage.
- **Mechanismus**:
  - Nutzt lineare Schichten, um die aggregierten Features in die Zielvariable zu übersetzen.
  - Verwendet Aktivierungsfunktionen wie ReLU für eine bessere Modellanpassung.
- **Output**:
  - Die vorhergesagte Kursänderung.

---

### **3. Trainingsfunktion**

#### **`train_hybrid_model`**
- **Zweck**: Trainiert das hybride Modell, um zukünftige Kursbewegungen vorherzusagen.
- **Input**:
  - Trainingsdaten: Sequenzen aus drei Feature-Gruppen.
  - Zielwerte: Tatsächliche Kursdaten 30 Tage in die Zukunft.
  - Modellparameter:
    - `hidden_dim`: Dimension der versteckten Schichten.
    - `num_heads`: Anzahl der Attention-Heads im Transformer.
    - `num_layers`: Anzahl der Transformer-Blöcke.
    - `batch_size`: Anzahl der Sequenzen pro Batch.
    - `epochs`: Anzahl der Iterationen.
    - `learning_rate`: Schrittweite für die Optimierung.
- **Ablauf**:
  1. **Datenaufbereitung**:
     - Skaliert die Eingabedaten und generiert Sequenzen.
     - Padding für Sequenzen mit unterschiedlichen Längen.
  2. **Modellinitialisierung**:
     - Erstellt das hybride Modell und initialisiert die Parameter.
  3. **Training**:
     - Berechnet Vorhersagen und Verluste (z. B. MSE) für jeden Batch.
     - Optimiert die Modellparameter mit AdamW.
  4. **Validierung**:
     - Bewertet die Leistung auf Validierungsdaten.
     - Berechnet Metriken wie MSE, RMSE und R².
  5. **Speicherung**:
     - Speichert das trainierte Modell in `.pth`.
  6. **Visualisierung**:
     - Plottet Trainings- und Validierungsverluste sowie Metriken.

---

### **4. Testskript**

#### **Zweck**: Evaluierung des trainierten Modells auf unsichtbaren Testdaten und Analyse der Vorhersagegenauigkeit.

---

1. **Daten laden und vorbereiten**:
   - **Quell-Daten**: Historische Kursdaten von Bitcoin (`BTC-USD`) werden mit der `yfinance`-Bibliothek heruntergeladen.
   - **Indikatoren berechnen**: Technische Indikatoren wie RSI, MACD, Bollinger-Bänder und weitere werden auf Basis der historischen Kursdaten berechnet.
   - **Sequenzgenerierung**:
     - Die Testdaten werden in Eingabesequenzen für Standard-Features, Momentum-Indikatoren und Volatilitäts-Indikatoren umgewandelt.
     - Ein gespeicherter `StandardScaler` wird geladen, um die Testsequenzen entsprechend zu normalisieren.

2. **Modell laden**:
   - Das trainierte Modell (`HybridTransformerLSTMModel`) wird geladen und in den Evaluierungsmodus (`eval()`) versetzt.
   - Auf GPU oder CPU basierende Berechnungen werden automatisch konfiguriert.

3. **Vorhersage durchführen**:
   - Die vorbereiteten Eingabesequenzen (drei Gruppen) werden in das Modell eingespeist.
   - Das Modell gibt die vorhergesagten Bitcoin-Preise (30 Tage in die Zukunft) zurück.
   - Die Vorhersagen werden aus dem skalierten Wertebereich zurücktransformiert.

4. **Leistung bewerten**:
   - **Berechnete Metriken**:
     - **MSE (Mean Squared Error)**: Durchschnitt der quadrierten Abweichungen zwischen vorhergesagten und tatsächlichen Werten.
     - **RMSE (Root Mean Squared Error)**: Quadratwurzel des MSE.
     - **R² (Bestimmtheitsmaß)**: Maß der Anpassungsgüte, das die Übereinstimmung zwischen Vorhersagen und tatsächlichen Werten angibt.
     - **Prozentualer Fehler**: Durchschnittlicher Fehler relativ zum tatsächlichen Wert.
     - **Absoluter Fehler**: Durchschnittliche absolute Abweichung zwischen tatsächlichen und vorhergesagten Preisen.
   - **Weitere Berechnungen**:
     - Tatsächlicher Gewinn/Verlust über den Zeitraum.
     - Vorhergesagter Gewinn/Verlust über den Zeitraum.

5. **Visualisierung**:
   - Ein Plot zeigt:
     - Tatsächliche Kursverläufe.
     - Vorhergesagte Kursverläufe.
     - Differenzen zwischen den tatsächlichen und vorhergesagten Werten.
   - Die Ergebnisse werden zusätzlich in tabellarischer Form ausgegeben (z. B. Kaufpreis, Endpreis, Fehlermaße).

6. **Ergebnisse ausgeben**:
   - **Textausgabe**:
     - Kaufpreis zu Beginn des Testzeitraums.
     - Tatsächlicher und vorhergesagter Endpreis.
     - Fehlermaße (MSE, RMSE, R², Absoluter und Prozentualer Fehler).
   - **Dateiausgabe**:
     - Ergebnisse werden in einer `.csv`-Datei gespeichert.

---

### **5. Parameter und Feature-Bedeutung**

#### **Parameter**:
- `hidden_dim`: Dimension der versteckten Schichten.
- `seq_length`: Länge der Eingabesequenzen.
- `batch_size`: Anzahl der Sequenzen pro Batch.
- `learning_rate`: Dynamische Anpassung mit einem Cyclic Learning Rate Scheduler.
- `epochs`: Optimiert für längere Trainingszyklen.

---

### **6. Verlust- und Metrikenberechnung**

#### **a) Trainingsverlust**
- Der Trainingsverlust misst, wie gut das Modell die Vorhersagen auf den Trainingsdaten macht. In **Experiment 5** wurde der Verlust durch optimierte Layer-Normalisierung und längere Sequenzen weiter reduziert.

#### **b) Validierungsverlust**
- Der Validierungsverlust misst die Modellleistung auf nicht genutzten Daten. Die Kombination aus Transformer-Mechanismen und LSTM-Schichten führte zu einer besseren Generalisierung.

#### **c) Mean Squared Error (MSE)**
- Der MSE wurde durch die effizientere Integration von Attention-Mechanismen und verbesserten Feature-Aggregationen weiter gesenkt.

#### **d) Root Mean Squared Error (RMSE)**
- Der RMSE sank durch die erweiterte Hybrid-Architektur, die eine präzisere Modellierung komplexer Beziehungen zwischen Features ermöglicht.

#### **Zusammenfassung**
- **Trainingsverlust**: Weiter reduziert durch optimierte Transformer-LSTM-Integration.
- **Validierungsverlust**: Verbesserte Generalisierungsleistung durch zusätzliche Indikatoren und längere Eingabesequenzen.
- **MSE und RMSE**: Diese Metriken verbesserten sich deutlich durch die effizientere Modellarchitektur und optimierte Trainingsmethoden.

---

### **7. Plots und Visualisierung**

#### **a) Trainings- und Validierungsverlust**
- Visualisiert den Verlust für Training und Validierung über die Epochen.
- **Experiment 5** zeigt eine stabilere Konvergenz durch die verbesserten Trainingsmechanismen.

#### **b) Validierungsmetriken (MSE, RMSE, R²)**
- Zeigt die Entwicklung der Metriken für die Validierungsdaten über die Epochen.
- **R²-Wert**: In Experiment 5 erreichte dieser Wert nahezu perfekte Anpassungswerte, was die Effektivität der Transformer-LSTM-Architektur verdeutlicht.

---

## **Testskript - Funktionsweise und Ablauf**

Das Testskript in **Experiment 5** umfasst die folgenden Schritte und Verbesserungen:

---

### **1. Initialisierung und Vorbereitung**

#### **Zufallszahlen-Seed**
- Zusätzlich zu den Seeds aus den vorherigen Experimenten wurden spezielle Seeds für die erweiterten Attention-Mechanismen gesetzt, um reproduzierbare Ergebnisse zu gewährleisten.

#### **Gerätekonfiguration**
- Optimiert für die parallele Verarbeitung von Feature-Gruppen durch Transformer- und LSTM-Layer.
- Automatische Erkennung und Nutzung verfügbarer GPUs zur Beschleunigung der Berechnungen.

---

### **2. Funktionen**

#### **a) Berechnung technischer Indikatoren**
- Die Funktion zur Berechnung technischer Indikatoren wurde erweitert, um zusätzliche Features wie:
  - **Trend Intensity Index (TII)**,
  - **VWAP (Volume Weighted Average Price)** und
  - **Keltner Channels** zu integrieren.

#### **b) Laden von Testdaten**
- Historische Kursdaten werden geladen und bereinigt.
- Berechnete Indikatoren werden mit den ursprünglichen Kursdaten kombiniert, um vollständige Datensätze zu erstellen.

#### **c) Sequenzierung der Daten**
- Die Funktion zur Sequenzierung wurde angepasst, um längere Sequenzen (bis zu 60 Tage) und die erweiterten Feature-Gruppen zu berücksichtigen.
- Jede Feature-Gruppe wird getrennt skaliert und in Sequenzen konvertiert.

---

### **3. Hauptablauf**

#### **a) Datenvorbereitung**
1. **Scaler laden**:
   - Skalierungsparameter, die während des Trainings gespeichert wurden, werden geladen.
   - Zusätzliche Scaler für neue technische Indikatoren wurden implementiert.
2. **Testdaten laden**:
   - Historische Kursdaten werden heruntergeladen und bereinigt.
   - Erweiterte technische Indikatoren wie TII und ATR werden berechnet und integriert.
3. **Sequenzgenerierung**:
   - Eingabesequenzen für die drei Gruppen (Standard, Momentum, Volatilität) werden erstellt.

#### **b) Modell laden**
- Das trainierte Modell (`HybridTransformerLSTMModel`) wird geladen und in den Evaluierungsmodus (`eval()`) versetzt.
- Das Modell wird automatisch auf das verfügbare Gerät (CPU oder GPU) übertragen.

#### **c) Vorhersage**
1. **Daten an das Modell übergeben**:
   - Die Sequenzen für die drei Gruppen werden verarbeitet.
2. **Skalierung zurücksetzen**:
   - Die vorhergesagten Werte werden aus dem skalierten Bereich zurücktransformiert, um die tatsächlichen Kurswerte wiederherzustellen.

#### **d) Visualisierung**
- Zusätzliche Plots zeigen die Gewichtung der Features durch die Transformer-Mechanismen und die zeitlichen Abhängigkeiten durch die LSTM-Komponenten.
- Diagramme umfassen:
  - Tatsächliche Kursverläufe,
  - Vorhergesagte Kursverläufe und
  - Differenzen zwischen tatsächlichen und vorhergesagten Werten.

---

### **4. Ausgabe**

#### **Textausgabe**
- Kaufpreis zu Beginn des Testzeitraums.
- Tatsächlicher und vorhergesagter Endpreis.
- Berechnete Fehlermaße:
  - **MSE (Mean Squared Error)**,
  - **RMSE (Root Mean Squared Error)**,
  - **R² (Bestimmtheitsmaß)**,
  - Absoluter und prozentualer Fehler.

#### **Dateiausgabe**
- Ergebnisse werden in einer `.csv`-Datei gespeichert, die Vorhersagen und tatsächliche Werte sowie alle berechneten Fehlermaße enthält.

#### **Diagramme**
- Vergleich der tatsächlichen und vorhergesagten Kursverläufe.
- Visualisierung der Aufmerksamkeitsschwerpunkte innerhalb der Transformer-Architektur.

---

### **Wichtige Hinweise**

#### **Skalierung**
- Erweiterte Indikatoren wie ATR und Keltner Channels erfordern eine separate Skalierung, um Verzerrungen durch unterschiedliche Wertebereiche zu vermeiden.

#### **Fehlerbehandlung**
- Zusätzliche Validierungen wurden implementiert, um fehlende Werte in Indikatoren oder Kursdaten zu erkennen und zu beheben.

---

## **Baseline**

### **Definition**
- **MSE-Baseline**: Durchschnittliche Kursänderung aller vorherigen Tage als naive Vorhersage.
- **Erweiterung in Experiment 5**:
  - Eine zusätzliche Baseline berücksichtigt die aggregierten technischen Indikatoren, um eine fairere Vergleichsbasis zu schaffen.
  - Der gleitende Durchschnitt der letzten 60 Tage wird als Vorhersage verwendet.
---

## **Ausführung**

### **Dateien**
- **`experiment_5_data_processing.py`**:
  - Bereitet die erweiterten Daten vor, berechnet zusätzliche technische Indikatoren und erstellt Sequenzen.
  - **Eingabe**: Historische Kursdaten (`yfinance`).
  - **Ausgabe**: Sequenzen (`Standard_X.pkl`, `Indicators_Group_1_X.pkl`, `Indicators_Group_2_X.pkl`).

- **`experiment_5_model_layer.py`**:
  - Definiert die Bausteine des Modells:
    - Transformer-Layer mit Multi-Head Attention.
    - Advanced Dual Attention Layer.
  - **Erweiterung in Experiment 5**:
    - Optimierte Fusion von Transformer- und LSTM-Komponenten für verbesserte zeitliche und feature-spezifische Verarbeitung.

- **`experiment_5_model.py`**:
  - Enthält die Modelllogik für Training und Evaluation unter Nutzung der hybriden Transformer-LSTM-Architektur.
  - Trainiert das Modell und speichert die Ergebnisse in `fusion_model_v5.pth`.

- **`experiment_5.py`**:
  - Führt Vorhersagen auf Testdaten aus und visualisiert die Ergebnisse.
  - **Eingabe**: `fusion_model_v5.pth`, Testdaten von `yfinance`.
  - **Ausgabe**: Vorhergesagte Preise und Performance-Metriken.

---

### **Schritte zur Ausführung**

```bash
# Verzeichnis wechseln
cd src/Experiment_5

# Datenverarbeitung
python experiment_5_data_processing.py
# Ergebnis: Generiert die vorbereiteten Sequenzen in ./Data/Samples/

# Training des Modells
python experiment_5_model.py
# Ergebnis: Trainiert das Modell und speichert es unter ./Data/Models/fusion_model_v5.pth

# Vorhersagen und Evaluation
python experiment_5.py
# Ergebnis: Zeigt die vorhergesagten Preise, Fehlermaße und einen Plot des Kursverlaufs

# Visualisierung des Modells
# TensorBoard: Visualisiert die Trainings- und Validierungsmetriken in Echtzeit
python experiment_5_tensorboard.py

# Netron: Zeigt die Modellarchitektur an
python experiment_5_netron.py

# TorchViz: Visualisiert den Datenfluss durch das Modell
python experiment_5_torchviz.py

---

### **Ergebnisse**
- Kaufpreis am 2023-02-01: 23723.76953125
- Tatsächlicher Preis am 2023-03-02: 23475.466796875
- Vorhergesagter Preis: 24512.87645192847
- Tatsächlicher Gewinn: -248.302734375
- Vorhergesagter Gewinn: 789.1069203781717
- MSE im Preis: 875000.3821479321
- RMSE im Preis: 935.5551980437621
- Absoluter Fehler: 937.4096550531245
- Prozentualer Fehler: 3.9902%
- **R² (Bestimmtheitsmaß)**: 0.9991

---

### **Änderungen im Vergleich zu Experiment 4**
- **Integration zusätzlicher technischer Indikatoren**:
  - Einführung des **Trend Intensity Index**, um die Stärke von Trends in den Daten besser zu erfassen.
- **Erweiterung der Architektur**:
  - Hybridisierung der Modellstruktur durch die Kombination von **Transformer-Layern** mit **LSTM-Komponenten**, um zeitliche und feature-spezifische Abhängigkeiten effektiver zu modellieren.
  - Verbesserung des **Feature Fusion Layers**, um die Outputs der Transformer- und LSTM-Komponenten effizient zu integrieren.
- **Training und Hyperparameter**:
  - Einführung eines **Cyclic Learning Rate Schedulers**, um die Lernrate dynamisch anzupassen.
  - Erhöhung der **Eingabesequenzen auf 60 Tage**, um längerfristige Muster zu erfassen.
  - Einsatz eines **größeren Batch-Sizes**, um die Trainingseffizienz zu verbessern.
- **Erweiterte Visualisierung**:
  - Visualisierung der **Interaktion zwischen Transformer- und LSTM-Komponenten**.
  - Darstellung der Gewichtungen innerhalb des hybriden Attention-Layers.
- **Zusätzliche Baseline**:
  - Ergänzung einer Baseline, die den **durchschnittlichen Kurs der letzten 60 Tage** als Vorhersage heranzieht.
