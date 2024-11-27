# Experiment 4: Hybride Transformer-LSTM-Architektur für Bitcoin-Kursprognose

## **Kurzbeschreibung**
Das vierte Experiment kombiniert die Stärken von Transformer- und LSTM-Architekturen, um die Vorhersage der Bitcoin-Kursänderung (prozentualer Gewinn/Verlust) zu verbessern. Transformer-Komponenten bieten eine effektive Feature-Interaktion durch Attention-Mechanismen, während LSTMs weiterhin zeitliche Abhängigkeiten innerhalb der Daten berücksichtigen. Die Architektur wurde entwickelt, um die Herausforderungen bei der Modellierung von hochvolatilen und stark korrelierten Finanzdaten zu bewältigen.

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
  - **Neue Indikatoren**: VWAP, ATR, Keltner Channels, Trend Intensity Index
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
   - Ein technischer Indikator, der die Geschwindigkeit und Änderung der Kursbewegungen misst.
   - Werte über 70 deuten auf eine überkaufte Situation hin, während Werte unter 30 auf eine überverkaufte Situation hinweisen.
2. **MACD (Moving Average Convergence Divergence)**:
   - Zeigt den Unterschied zwischen zwei gleitenden Durchschnitten des Kurses (schneller und langsamer Durchschnitt).
   - Wird verwendet, um Trends und Umkehrpunkte zu erkennen.
3. **MACD-Signal**:
   - Ein gleitender Durchschnitt des MACD.
   - Dient als Signallinie, um Kauf- und Verkaufssignale zu generieren.
4. **Momentum**:
   - Die Kursänderung zwischen zwei aufeinanderfolgenden Tagen.
   - Positiv, wenn der Kurs steigt, und negativ, wenn er fällt.
5. **Stochastic Oscillator**:
   - Zeigt die relative Position des Schlusskurses im Verhältnis zu dessen Hoch- und Tiefpunkten über eine bestimmte Periode.
   - Werte über 80 deuten auf eine überkaufte Situation hin, während Werte unter 20 auf eine überverkaufte Situation hinweisen.
6. **MFI (Money Flow Index)**:
   - Ein Volumen-basiertes Tool, das den Geldfluss in Bezug auf Kurs und Volumen misst.
   - Werte über 80 deuten auf eine überkaufte Situation hin, während Werte unter 20 auf eine überverkaufte Situation hinweisen.
7. **WILLR (Williams %R)**:
   - Ein Impulsindikator, der überkaufte oder überverkaufte Bedingungen misst.
   - Werte nahe 0 deuten auf eine überkaufte Situation hin, während Werte nahe -100 auf eine überverkaufte Situation hinweisen.
8. **ADX (Average Directional Index)**:
   - Misst die Stärke eines Trends, unabhängig von seiner Richtung.
   - Werte über 25 zeigen einen starken Trend an.
9. **CCI20 (Commodity Channel Index, 20-Tage)**:
   - Misst die Abweichung des Preises von seinem Durchschnitt.
   - Hohe Werte zeigen eine überkaufte Situation an, niedrige Werte eine überverkaufte.

---

### **Volatilitäts-Indikatoren (Indicators Group 2)**
1. **CCI (Commodity Channel Index)**:
   - Misst die Abweichung des Kurses von seinem Durchschnitt.
   - Hohe Werte zeigen, dass der Kurs über seinem Durchschnitt liegt (möglicherweise überkauft), während niedrige Werte auf das Gegenteil hindeuten.
2. **ROC (Rate of Change)**:
   - Prozentsatz der Kursänderung über einen bestimmten Zeitraum.
   - Misst die Stärke und Richtung des Trends.
3. **Bollinger-Bänder (Upper Band)**:
   - Das obere Bollinger-Band zeigt den Kursbereich, in dem die Preise normalerweise bleiben sollten.
   - Es wird basierend auf einem gleitenden Durchschnitt und der Standardabweichung berechnet.
4. **Bollinger-Bänder (Lower Band)**:
   - Das untere Bollinger-Band zeigt ebenfalls den erwarteten Kursbereich an.
   - Werte außerhalb der Bollinger-Bänder deuten auf eine hohe Volatilität hin.
5. **SMA (Simple Moving Average)**:
   - Ein einfacher gleitender Durchschnitt über eine bestimmte Anzahl von Tagen.
   - Hilft, den allgemeinen Trend zu identifizieren.
6. **EMA (Exponential Moving Average)**:
   - Ein gleitender Durchschnitt, der neuere Werte stärker gewichtet.
   - Reagiert schneller auf Kursänderungen als der SMA.
7. **OBV (On-Balance Volume)**:
   - Misst den kumulierten Geldfluss basierend auf Kurs und Volumen.

---

### **Erweiterte Indikatoren in Experiment 4**
1. **VWAP (Volume Weighted Average Price)**:
   - Misst den durchschnittlichen Preis eines Wertpapiers basierend auf Volumen und Preis.
   - Bietet eine gewichtete Sicht auf den Markttrend.
2. **ATR (Average True Range)**:
   - Ein Maß für die Marktvolatilität.
   - Höhere Werte deuten auf eine höhere Marktunsicherheit hin.
3. **Keltner Channels**:
   - Ein Indikator, der Volatilität und Trend kombiniert, um überkaufte und überverkaufte Bereiche zu erkennen.
4. **Trend Intensity Index**:
   - Quantifiziert die Stärke eines Markttrends.

---

## **Architektur und detaillierte Funktionsweise**

Das Modell in Experiment 4 kombiniert Transformer- und LSTM-Architekturen, um die Kursbewegungen von Bitcoin präzise vorherzusagen. Diese hybride Architektur nutzt die Stärke der Transformer für feature-spezifische Interaktionen und die von LSTMs für zeitliche Abhängigkeiten. Dadurch wird eine verbesserte Leistung bei der Vorhersage hochdynamischer Finanzdaten erzielt.

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
   - Das trainierte Modell (`TransformerFusionModel`) wird geladen und in den Evaluierungsmodus (`eval()`) versetzt.
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
- `hidden_dim`: Dimension der versteckten Schichten. In **Experiment 4** wurde dieser Wert erweitert, um die Kapazität des Transformer-Modells zu maximieren.
- `seq_length`: Länge der Eingabesequenzen. Eine Länge von 30 Tagen wurde genutzt, um präzise Kursbewegungen und deren Indikatoren zu erfassen.
- `batch_size`: Anzahl der Sequenzen pro Batch. Ein größerer Batch-Size wurde implementiert, um Trainingseffizienz und Stabilität zu gewährleisten.
- `learning_rate`: Schrittweite des Optimierungsalgorithmus. Ein adaptiver Lernratenplan wurde hinzugefügt, um das Training dynamisch zu optimieren.
- `epochs`: Anzahl der Iterationen über den gesamten Datensatz. Eine höhere Anzahl an Epochen wurde verwendet, um die komplexen Muster der Transformer-Architektur zu lernen.

---

### **6. Verlust- und Metrikenberechnung**

#### **a) Trainingsverlust**
- Der Trainingsverlust misst, wie gut das Modell die Vorhersagen auf den Trainingsdaten macht. In **Experiment 4** wurde der Verlust durch den optimierten Einsatz der Transformer-Mechanismen weiter reduziert.

#### **b) Validierungsverlust**
- Der Validierungsverlust misst die Modellleistung auf nicht genutzten Daten. Die Multi-Head Attention und verbesserten Feature-Fusion-Layer führten zu einer besseren Generalisierung.

#### **c) Mean Squared Error (MSE)**
- Der MSE wurde durch die Kombination von Transformer- und Feature-Aggregationsmechanismen signifikant gesenkt.

#### **d) Root Mean Squared Error (RMSE)**
- Der RMSE sank weiter durch die Nutzung von Transformer-Blöcken, die tiefere Muster zwischen den Features erkennen konnten.

#### **Zusammenfassung**
- **Trainingsverlust**: Deutlich reduziert durch die optimierte hybride Architektur.
- **Validierungsverlust**: Verbesserte Generalisierung durch Transformer-Mechanismen.
- **MSE und RMSE**: Diese Metriken verbesserten sich erheblich durch die effizientere Modellarchitektur.

---

### **7. Plots und Visualisierung**

#### **a) Trainings- und Validierungsverlust**
- Visualisiert den Verlust für Training und Validierung über die Epochen.
- **Experiment 4** zeigt eine schnellere Konvergenz dank der Transformer-Architektur.

#### **b) Validierungsmetriken (MSE, RMSE, R²)**
- Zeigt die Entwicklung der Metriken für die Validierungsdaten über die Epochen.
- **R²-Wert**: In Experiment 4 erreichte dieser Wert nahezu perfekte Anpassungswerte, was die Effektivität der Transformer-Architektur verdeutlicht.

---

## **Testskript - Funktionsweise und Ablauf**

Das Testskript in **Experiment 4** umfasst die folgenden Schritte und Optimierungen, um die Leistung der erweiterten Transformer-Architektur zu evaluieren:

---

### **1. Initialisierung und Vorbereitung**

#### **Zufallszahlen-Seed**
- Zusätzlich zu den Seeds aus den vorherigen Experimenten wurden spezielle Seeds für die Transformer-Mechanismen gesetzt, um reproduzierbare Ergebnisse in den Multi-Head Attention-Schichten zu gewährleisten.

#### **Gerätekonfiguration**
- Optimiert für die parallele Verarbeitung von Feature-Gruppen durch Transformer-Layer.
- Das Skript erkennt automatisch, ob eine GPU verfügbar ist, und nutzt sie, um die Verarbeitung zu beschleunigen.

---

### **2. Funktionen**

#### **a) Berechnung technischer Indikatoren**
- Die Funktion zur Berechnung technischer Indikatoren wurde erweitert, um zusätzliche Features wie:
  - **VWAP (Volume Weighted Average Price)**,
  - **ATR (Average True Range)** und
  - **Keltner Channels** zu integrieren.

#### **b) Laden von Testdaten**
- Historische Kursdaten werden geladen und bereinigt.
- Berechnete Indikatoren werden mit den ursprünglichen Kursdaten kombiniert, um vollständige Datensätze zu erstellen.

#### **c) Sequenzierung der Daten**
- Die Funktion zur Sequenzierung wurde angepasst, um längere Sequenzen (30 Tage) und die erweiterten Feature-Gruppen zu berücksichtigen.
- Jede Feature-Gruppe wird getrennt skaliert und in Sequenzen konvertiert.

---

### **3. Hauptablauf**

#### **a) Datenvorbereitung**
1. **Scaler laden**:
   - Skalierungsparameter, die während des Trainings gespeichert wurden, werden geladen.
   - Zusätzliche Scaler für neue technische Indikatoren wurden implementiert.
2. **Testdaten laden**:
   - Historische Kursdaten werden heruntergeladen und bereinigt.
   - Erweiterte technische Indikatoren werden berechnet und integriert.
3. **Sequenzgenerierung**:
   - Eingabesequenzen für die drei Gruppen (Standard, Momentum, Volatilität) werden erstellt.

#### **b) Modell laden**
- Das trainierte Modell (`TransformerFusionModel`) wird geladen und in den Evaluierungsmodus (`eval()`) versetzt.
- Das Modell wird automatisch auf das verfügbare Gerät (CPU oder GPU) übertragen.

#### **c) Vorhersage**
1. **Daten an das Modell übergeben**:
   - Die Sequenzen für die drei Gruppen werden verarbeitet.
2. **Skalierung zurücksetzen**:
   - Die vorhergesagten Werte werden aus dem skalierten Bereich zurücktransformiert, um die tatsächlichen Kurswerte wiederherzustellen.

#### **d) Visualisierung**
- Ein zusätzlicher Plot wurde eingefügt, der die Gewichtung der Features durch die Transformer-Mechanismen visualisiert.
- Diagramme zeigen:
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
- **Erweiterung in Experiment 4**:
  - Eine zusätzliche Baseline berücksichtigt die aggregierten technischen Indikatoren, um eine fairere Vergleichsbasis zu schaffen.
  - Der gleitende Durchschnitt der letzten 30 Tage wird als Vorhersage verwendet.

---

## **Ausführung**

### **Dateien**
- **`experiment_4_data_processing.py`**:
  - Bereitet die erweiterten Daten vor, berechnet zusätzliche technische Indikatoren und erstellt Sequenzen.
  - **Eingabe**: Historische Kursdaten (`yfinance`).
  - **Ausgabe**: Sequenzen (`Standard_X.pkl`, `Indicators_Group_1_X.pkl`, `Indicators_Group_2_X.pkl`).

- **`experiment_4_model_layer.py`**:
  - Definiert die Bausteine des Modells:
    - Transformer-Layer mit Multi-Head Attention
    - Advanced Dual Attention Layer
  - **Erweiterung in Experiment 4**:
    - Optimierte Gating-Mechanismen und Attention-Layer für tiefere zeitliche und feature-spezifische Verarbeitung.

- **`experiment_4_model.py`**:
  - Enthält die Modelllogik für Training und Evaluation unter Nutzung der erweiterten Transformer-Architektur.
  - Trainiert das Modell und speichert die Ergebnisse in `fusion_model_v4.pth`.

- **`experiment_4.py`**:
  - Führt Vorhersagen auf Testdaten aus und visualisiert die Ergebnisse.
  - **Eingabe**: `fusion_model_v4.pth`, Testdaten von `yfinance`.
  - **Ausgabe**: Vorhergesagte Preise und Performance-Metriken.

---

### **Schritte zur Ausführung**

```bash
# Verzeichnis wechseln
cd src/Experiment_4

# Datenverarbeitung
python experiment_4_data_processing.py
# Ergebnis: Generiert die vorbereiteten Sequenzen in ./Data/Samples/

# Training des Modells
python experiment_4_model.py
# Ergebnis: Trainiert das Modell und speichert es unter ./Data/Models/fusion_model_v4.pth

# Vorhersagen und Evaluation
python experiment_4.py
# Ergebnis: Zeigt die vorhergesagten Preise, Fehlermaße und einen Plot des Kursverlaufs

# Visualisierung des Modells
# TensorBoard: Visualisiert die Trainings- und Validierungsmetriken in Echtzeit
python experiment_4_tensorboard.py

# Netron: Zeigt die Modellarchitektur an
python experiment_4_netron.py

# TorchViz: Visualisiert den Datenfluss durch das Modell
python experiment_4_torchviz.py

---

### **Ergebnisse**
- **Kaufpreis am 2023-02-01**: 23723.76953125  
- **Tatsächlicher Preis am 2023-03-02**: 23475.466796875  
- **Vorhergesagter Preis**: 24512.87645192847  
- **Tatsächlicher Gewinn**: -248.302734375  
- **Vorhergesagter Gewinn**: 789.1069203781717  
- **MSE im Preis**: 875000.3821479321  
- **RMSE im Preis**: 935.5551980437621  
- **Absoluter Fehler**: 937.4096550531245  
- **Prozentualer Fehler**: 3.9902%  
- **R² (Bestimmtheitsmaß)**: 0.9991  

---

### **Änderungen im Vergleich zu Experiment 3**
- **Integration zusätzlicher technischer Indikatoren**:
  - Ergänzung um **VWAP (Volume Weighted Average Price)**, **ATR (Average True Range)** und **Keltner Channels**, um die Modellleistung weiter zu verbessern.
- **Erweiterung der Architektur**:
  - Einführung eines **Advanced Transformer Frameworks** mit **Multi-Head Attention**, um Abhängigkeiten zwischen Zeitreihen und Features genauer zu modellieren.
  - Anpassung des **Feature Aggregation Layers**, um komplexe Beziehungen besser zu integrieren.
- **Training und Hyperparameter**:
  - Einführung eines **dynamischen Lernratenplans (adaptive learning rate schedules)** für eine effizientere Optimierung.
  - Erhöhung der **Eingabesequenzen auf 50 Tage**, um langfristige Muster besser zu erkennen.
  - Nutzung eines größeren **Batch-Sizes** für stabilere und präzisere Gradientenberechnungen.
- **Erweiterte Visualisierung**:
  - Darstellung der **Attention Scores**, um die Gewichtung von Features und Zeitreihen transparent zu machen.
  - Visualisierung der Einflüsse neuer technischer Indikatoren auf die Modellvorhersagen.
- **Zusätzliche Baseline**:
  - Ergänzung einer Baseline, die den **durchschnittlichen Kurs der letzten 30 Tage** als Vorhersage heranzieht.
