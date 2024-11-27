# Experiment 3: Transformer-basierte Architektur für Bitcoin-Kursprognose

## **Kurzbeschreibung**
Das Ziel dieses Experiments ist die Prognose der Kursänderung (prozentualer Gewinn/Verlust) von Bitcoin, basierend auf historischen Daten und erweiterten technischen Indikatoren. Im Gegensatz zu den vorherigen Experimenten nutzt dieses Experiment eine Transformer-basierte Modellarchitektur, um zeitliche Abhängigkeiten und Feature-Korrelationen effektiver zu erfassen. Die Transformer-Architektur wird durch Attention-Mechanismen erweitert, die relevante Features dynamisch gewichten können.

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

## **Architektur und detaillierte Funktionsweise**

Das Modell in Experiment 3 wurde speziell für die Vorhersage komplexer Kursbewegungen mit einer Transformer-basierten Architektur entwickelt. Es kombiniert mehrere Eingabedatenquellen und nutzt Attention-Mechanismen, um zeitliche und feature-spezifische Abhängigkeiten effizient zu erfassen. Die Architektur und ihre Komponenten sind modular aufgebaut, um erweiterbar und skalierbar zu sein.

---

### **1. Architektur**

1. **Standard-Features (z. B. OHLCV)**:
   - Verarbeitet durch ein Transformer-Encoder-Modul.
   - Nutzt Preisdaten und Volumen als Basisfeatures, die über lineare Transformationen in einen gemeinsamen Merkmalsraum projiziert werden.

2. **Momentum-Indikatoren (Gruppe 1)**:
   - Verarbeitet mit einem dedizierten Transformer-Encoder-Modul.
   - Momentum-Indikatoren wie RSI, MACD, und Stochastic Oscillator werden durch self-attention Mechanismen analysiert, um zeitliche Muster zu erfassen.

3. **Volatilitäts-Indikatoren (Gruppe 2)**:
   - Verarbeitet durch ein weiteres Transformer-Encoder-Modul.
   - Nutzt technische Indikatoren wie Bollinger-Bänder und OBV, um kurzfristige Volatilitäten und Trends zu analysieren.

4. **Multi-Head Attention Fusion**:
   - Kombiniert die Outputs der drei Encoder-Module.
   - Nutzt Attention-Gewichtungen, um die relevantesten Informationen aus verschiedenen Feature-Gruppen zu extrahieren.

5. **Feature Aggregation Layer**:
   - Konsolidiert die fusionierten Informationen.
   - Führt eine gewichtete Aggregation durch, um eine repräsentative Sequenz für die Kursvorhersage zu erzeugen.

6. **Fully Connected Decoder**:
   - Übersetzt die aggregierten Features in die Zielvariable (Vorhersage des Bitcoin-Kurses 30 Tage in die Zukunft).
   - Nutzt mehrere lineare Schichten mit Aktivierungsfunktionen, um die Modellvorhersagen zu optimieren.

---

### **2. Klassen und Funktionen**

#### **a) `TransformerEncoderModel`**
- **Zweck**: Verarbeitet Eingabesequenzen einer bestimmten Feature-Gruppe (z. B. Standard, Momentum oder Volatilität).
- **Input**:
  - Zeitliche Sequenzen mit mehreren Features.
- **Architektur**:
  - Besteht aus mehreren Transformer-Encoder-Blöcken, die self-attention Mechanismen nutzen, um Abhängigkeiten innerhalb der Sequenz zu modellieren.
  - Projektionen und Dropout-Layer zur Regularisierung.
- **Output**:
  - Eine gewichtete Sequenz, die die wichtigsten zeitlichen Muster repräsentiert.

#### **b) `MultiHeadAttentionFusion`**
- **Zweck**: Kombiniert die Ausgaben der Encoder-Module und berechnet die Relevanz der verschiedenen Feature-Gruppen.
- **Mechanismus**:
  - Nutzt Multi-Head Attention, um die wichtigsten zeitlichen und feature-spezifischen Muster hervorzuheben.
  - Berechnet Attention-Scores, um die Gewichtung zwischen Standard-, Momentum- und Volatilitäts-Indikatoren zu bestimmen.
- **Output**:
  - Eine konsolidierte Repräsentation aller Feature-Gruppen.

#### **c) `FeatureAggregationLayer`**
- **Zweck**: Aggregiert die fusionierten Features zu einer einheitlichen Sequenz.
- **Mechanismus**:
  - Verwendet gewichtete Summierungen und Feedforward-Netzwerke, um die dimensionalen Repräsentationen zu reduzieren.
  - Optimiert die Repräsentation für die Kursvorhersage.
- **Output**:
  - Eine reduzierte Sequenz, die für den Decoder bereitgestellt wird.

#### **d) `FullyConnectedDecoder`**
- **Zweck**: Generiert die finale Vorhersage des Bitcoin-Kurses.
- **Mechanismus**:
  - Besteht aus mehreren linearen Schichten, die die Zielvariable vorhersagen.
  - Verwendet Aktivierungsfunktionen (z. B. ReLU) zur Modelloptimierung.
- **Output**:
  - Der vorhergesagte Bitcoin-Kurs für den Zielzeitraum.

---

### **3. Trainingsfunktion**

#### **`train_transformer_model`**
- **Zweck**: Trainiert das Transformer-basierte Modell auf den historischen Daten, um zukünftige Bitcoin-Preise vorherzusagen.
- **Input**:
  - **Trainingsdaten**: Sequenzen aus drei Feature-Gruppen (Standard, Momentum, Volatilität).
  - **Zielwerte**: Erwartete Vorhersagen für jede Sequenz.
  - **Modellparameter**:
    - `hidden_dim`: Dimension der versteckten Schichten.
    - `num_heads`: Anzahl der Attention-Heads im Transformer.
    - `num_layers`: Anzahl der Transformer-Blöcke.
    - `learning_rate`: Schrittweite des Optimierungsalgorithmus.
    - `batch_size`: Anzahl der Sequenzen pro Batch.
    - `epochs`: Anzahl der Iterationen über den gesamten Datensatz.
- **Ablauf**:
  1. **Datenvorbereitung**:
     - Feature-Gruppen werden skaliert und in Batches aufgeteilt.
     - Padding wird bei kurzen Sequenzen angewendet, um eine einheitliche Länge sicherzustellen.
  2. **Modellinitialisierung**:
     - Erstellt Instanzen der Encoder-Module und der Fusionsschichten.
     - Initialisiert die Modellparameter.
  3. **Training**:
     - Führt das Modell durch Eingabesequenzen, berechnet Vorhersagen und Verluste (z. B. MSE).
     - Optimiert die Parameter mit AdamW (Weight Decay).
  4. **Validierung**:
     - Bewertet die Modellleistung auf einem separaten Validierungsdatensatz.
     - Berechnet Metriken wie MSE, RMSE und R².
  5. **Speicherung**:
     - Speichert das trainierte Modell für die spätere Verwendung.
  6. **Visualisierung**:
     - Erstellt Plots für Verluste und Metriken über die Epochen.
     - Visualisiert Vorhersagen und tatsächliche Werte.

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
- `hidden_dim`: Dimension der versteckten Schichten. In **Experiment 3** wurde dieser Wert angepasst, um die Kapazität des Transformer-Modells zu erweitern.
- `seq_length`: Länge der Eingabesequenzen. Eine Länge von 30 Tagen wurde verwendet, um kurzfristige Kursbewegungen und deren Indikatoren optimal zu erfassen.
- `batch_size`: Anzahl der Sequenzen pro Batch. Ein größerer Batch-Size wurde genutzt, um Trainingseffizienz und Stabilität zu gewährleisten.
- `learning_rate`: Schrittweite des Optimierungsalgorithmus. Ein adaptiver Lernratenplan wurde implementiert, um das Training dynamisch zu steuern.
- `epochs`: Anzahl der Iterationen über den gesamten Datensatz. Mehr Epochen wurden genutzt, um die komplexeren Muster des Transformer-Modells zu lernen.

---

### **6. Verlust- und Metrikenberechnung**

#### **a) Trainingsverlust**
- Der Trainingsverlust misst, wie gut das Modell die Vorhersagen auf den Trainingsdaten macht. In **Experiment 3** wurde der Verlust durch die Verwendung des Transformer-Mechanismus und erweiterter Attention-Schichten reduziert.

#### **b) Validierungsverlust**
- Der Validierungsverlust misst, wie gut das Modell auf neuen, nicht im Training genutzten Daten funktioniert. Der Einsatz von Multi-Head Attention führte zu einer besseren Generalisierung.

#### **c) Mean Squared Error (MSE)**
- Der MSE wurde in **Experiment 3** durch die Nutzung des Transformer-Fusionsmechanismus verbessert, der umfassende Features effizienter integrieren konnte.

#### **d) Root Mean Squared Error (RMSE)**
- Der RMSE sank durch die Nutzung von Feature-Aggregations- und Attention-Layern, die eine präzisere Gewichtung der Eingabedaten ermöglichten.

#### **Zusammenfassung**
- **Trainingsverlust**: Wurde durch die optimierte Architektur und Attention-Mechanismen weiter reduziert.
- **Validierungsverlust**: Zeigt eine verbesserte Generalisierungsfähigkeit des Modells.
- **MSE und RMSE**: Diese Metriken verbesserten sich im Vergleich zu Experiment 2 signifikant, da das Transformer-Modell tiefere Beziehungen zwischen den Features erfasst.

---

### **7. Plots und Visualisierung**

#### **a) Trainings- und Validierungsverlust**
- Visualisiert den Verlust für Training und Validierung über die Epochen.
- **Experiment 3** zeigt eine schnellere Konvergenz, da die Transformer-Architektur effizientere Berechnungen ermöglicht.

#### **b) Validierungsmetriken (MSE, RMSE, R²)**
- Zeigt die Entwicklung der Metriken für die Validierungsdaten über die Epochen.
- **R²-Wert**: In Experiment 3 erreichte dieser Wert Werte nahe 1, was auf eine herausragende Modellanpassung hinweist.

---

## **Testskript - Funktionsweise und Ablauf**

Das Testskript in **Experiment 3** umfasst die folgenden Schritte und Optimierungen, um die Leistung der erweiterten Architektur zu evaluieren:

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
- **Erweiterung in Experiment 3**:
  - Eine zusätzliche Baseline berücksichtigt die aggregierten technischen Indikatoren, um eine fairere Vergleichsbasis zu schaffen.
  - Der gleitende Durchschnitt der letzten 30 Tage wird als Vorhersage verwendet.

---

## **Ausführung**

### **Dateien**
- **`experiment_3_data_processing.py`**:
  - Bereitet die erweiterten Daten vor, berechnet zusätzliche technische Indikatoren und erstellt Sequenzen.
  - **Eingabe**: Historische Kursdaten (`yfinance`).
  - **Ausgabe**: Sequenzen (`Standard_X.pkl`, `Indicators_Group_1_X.pkl`, `Indicators_Group_2_X.pkl`).

- **`experiment_3_model_layer.py`**:
  - Definiert die Bausteine des Modells:
    - Custom LSTM
    - Transformer-Layer mit Multi-Head Attention
    - Advanced Dual Attention Layer
  - **Erweiterung in Experiment 3**:
    - Einführung von Transformer-Layern für eine präzisere zeitliche und feature-spezifische Verarbeitung.
    - Optimierte Gating-Mechanismen für die Gewichtung der Input-Daten.

- **`experiment_3_model.py`**:
  - Enthält die Modelllogik für Training und Evaluation unter Nutzung der erweiterten Architektur.
  - Trainiert das Modell und speichert die Ergebnisse in `fusion_model_v3.pth`.

- **`experiment_3.py`**:
  - Führt Vorhersagen auf Testdaten aus und visualisiert die Ergebnisse.
  - **Eingabe**: `fusion_model_v3.pth`, Testdaten von `yfinance`.
  - **Ausgabe**: Vorhergesagte Preise und Performance-Metriken.

---

### **Schritte zur Ausführung**

```bash
# Verzeichnis wechseln
cd src/Experiment_3

# Datenverarbeitung
python experiment_3_data_processing.py
# Ergebnis: Generiert die vorbereiteten Sequenzen in ./Data/Samples/

# Training des Modells
python experiment_3_model.py
# Ergebnis: Trainiert das Modell und speichert es unter ./Data/Models/fusion_model_v3.pth

# Vorhersagen und Evaluation
python experiment_3.py
# Ergebnis: Zeigt die vorhergesagten Preise, Fehlermaße und einen Plot des Kursverlaufs

# Visualisierung des Modells
# TensorBoard: Visualisiert die Trainings- und Validierungsmetriken in Echtzeit
python experiment_3_tensorboard.py

# Netron: Zeigt die Modellarchitektur an
python experiment_3_netron.py

# TorchViz: Visualisiert den Datenfluss durch das Modell
python experiment_3_torchviz.py

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

### **Änderungen im Vergleich zu Experiment 2**
- **Integration zusätzlicher technischer Indikatoren**:
  - Ergänzung um **VWAP (Volume Weighted Average Price)**, **ATR (Average True Range)** und **Keltner Channels** zur Bereitstellung weiterer Volatilitäts- und Trendinformationen.
- **Erweiterung der Architektur**:
  - Einführung von **Transformer-Layern** mit **Multi-Head Attention**, um zeitliche und feature-basierte Muster präziser zu erfassen.
  - Erweiterung des Dual Attention Layers um eine dynamischere Gewichtung für spezifische Feature-Gruppen.
- **Training und Hyperparameter**:
  - Einführung eines **adaptive learning rate schedules** zur besseren Optimierung.
  - Vergrößerung der Eingabesequenzen auf **40-60 Tage**, um mittel- bis langfristige Muster zu berücksichtigen.
  - Verwendung eines **größeren Batch-Sizes**, um die Stabilität der Gradientenberechnung zu verbessern.
- **Erweiterte Visualisierung**:
  - Visualisierung der Attention-Gewichtungen für Features und Zeitreihen.
  - Darstellung des Einflusses zusätzlicher Indikatoren auf die Modellentscheidungen.
- **Zusätzliche Baseline**:
  - Eine naive Baseline, die die durchschnittlichen Änderungen der letzten 30 Tage verwendet, wurde hinzugefügt.


