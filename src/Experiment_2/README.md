# Experiment 2: Erweiterte Multi-Input LSTM für Bitcoin-Kursprognose

## Kurzbeschreibung
Das Ziel dieses Projekts ist die Vorhersage der Kursänderung (prozentualer Gewinn/Verlust) von Bitcoin, basierend auf historischen Daten und erweiterten technischen Indikatoren. Das Modell verwendet eine verbesserte Multi-Input LSTM-Architektur, die zusätzliche technische Indikatoren integriert und Modifikationen in der Modellstruktur nutzt, um die Vorhersagegenauigkeit zu steigern.

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

Das Modell ist modular aufgebaut, um mehrere Eingabedatenquellen zu kombinieren und mithilfe von LSTM-, Gating- und Attention-Mechanismen präzise Vorhersagen zu treffen. Es wurden verschiedene Komponenten integriert, die zusammenarbeiten, um die komplexen Kursbewegungen von Bitcoin vorherzusagen. Die wichtigsten Elemente der Architektur und deren Funktionsweise werden nachfolgend beschrieben.

---

### **1. Architektur**
1. **Standard-Features (z. B. OHLCV)**:
   - Verarbeitet durch einen separaten `CustomLSTM`-Layer.
   - Nutzt Preisdaten und Volumen als Basisfeatures.
2. **Momentum-Indikatoren (Gruppe 1)**:
   - Verarbeitet von einem weiteren `CustomLSTM`.
   - Momentum-Indikatoren wie RSI, MACD, WILLR dienen als Eingangsdaten.
3. **Volatilitäts-Indikatoren (Gruppe 2)**:
   - Verarbeitet von einem separaten `CustomLSTM`.
   - Nutzt Indikatoren wie Bollinger-Bänder, OBV und SMA.
4. **Multi-Input-LSTM mit Gating**:
   - Kombiniert die Ausgaben der drei LSTMs.
   - Verwendet ein Gating-Mechanismus zur Gewichtung der Inputs.
5. **Dual Attention Layer**:
   - Wendet zeitbasierte und feature-basierte Gewichtungen an.
   - Verstärkt relevante Informationen zur Verbesserung der Vorhersage.
6. **Fusion Fully Connected Layer**:
   - Kombiniert alle Eingaben durch eine Feedforward-Schicht.
   - Liefert die finale Ausgabe in Form der Vorhersage.

---
### **2. Klassen und Funktionen**

#### **a) `ThreeGroupLSTMModel`**
- **Zweck**: Verarbeitet drei verschiedene Eingabedatenquellen (Standard-Features, Momentum-Indikatoren und Volatilitäts-Indikatoren) und kombiniert deren Ausgaben für die Vorhersage.
- **Input**:
  - `Y`: Sequenzen der Standard-Features (z. B. Open, High, Low, Close, Volume).
  - `X1`: Sequenzen der Momentum-Indikatoren (z. B. RSI, MACD, Stochastic Oscillator).
  - `X2`: Sequenzen der Volatilitäts-Indikatoren (z. B. Bollinger-Bänder, CCI, OBV).
- **Architektur**:
  - Verwendet separate `CustomLSTM`-Layer, um jede Eingabedatenquelle unabhängig zu verarbeiten.
  - Kombiniert die Ausgaben dieser Layer mithilfe eines `MultiInputLSTMWithGates`, das dynamische Gewichtungen basierend auf der Relevanz der Inputs berechnet.
  - Wendet eine `DualAttention`-Schicht an, um zeitlich und feature-basierte Relevanz zu priorisieren.
  - Gibt die verarbeiteten Daten über eine `Fully Connected`-Schicht aus, um die finale Sequenz zu generieren.

#### **b) `FusionModel`**
- **Zweck**: Integriert die Ausgaben mehrerer `ThreeGroupLSTMModel`-Instanzen, um umfassende Vorhersagen basierend auf allen Feature-Gruppen zu erstellen.
- **Input**:
  - Kombinierte Sequenzen aus den drei Feature-Gruppen (Standard, Momentum, Volatilität).
- **Architektur**:
  - Erstellt drei Instanzen von `ThreeGroupLSTMModel` (jeweils eine für jede Feature-Gruppe).
  - Kombiniert die Ausgaben dieser Modelle entlang der Feature-Dimension.
  - Verwendet eine `Fully Connected`-Schicht, um die finale Vorhersage zu generieren.

#### **c) `CustomLSTM`**
- **Zweck**: Dient als Basismodul, um zeitliche Muster innerhalb einzelner Feature-Gruppen zu extrahieren.
- **Input**:
  - Zeitliche Sequenzen mit mehreren Features (z. B. historische Preisdaten oder technische Indikatoren).
- **Architektur**:
  - Besteht aus mehreren LSTM-Schichten, um die komplexen zeitlichen Beziehungen innerhalb der Sequenzen zu modellieren.
  - Verwendet Dropout, um Überanpassung (Overfitting) zu vermeiden.
  - Gibt die vollständige Sequenz sowie die letzten versteckten Zustände aus.

#### **d) `MultiInputLSTMWithGates`**
- **Zweck**: Kombiniert mehrere Eingabesequenzen (z. B. Standard-Features, Momentum-Indikatoren und Volatilitäts-Indikatoren) in einer einzigen modellierten Sequenz.
- **Mechanismus**:
  - Jedes Eingabesignal wird dynamisch angepasst, basierend auf der Relevanz, die durch einen Gating-Mechanismus berechnet wird.
  - Verarbeitet die kombinierten Sequenzen mithilfe eines LSTM, um die zeitlichen Beziehungen zwischen den Datenquellen zu erfassen.
- **Output**:
  - Gibt die finale Sequenz sowie die letzten versteckten Zustände aus.

#### **e) `DualAttention`**
- **Zweck**: Verstärkt relevante Informationen aus den kombinierten Sequenzen, indem sowohl zeitliche als auch feature-basierte Relevanzen berücksichtigt werden.
- **Mechanismus**:
  - **Zeitliche Aufmerksamkeit**: Gewichtung einzelner Schritte in der Sequenz basierend auf deren Relevanz für die Vorhersage.
  - **Feature-Aufmerksamkeit**: Gewichtung der Eingabefeatures, um diejenigen hervorzuheben, die für die Kursvorhersage am wichtigsten sind.
- **Output**:
  - Eine gewichtetete und verstärkte Sequenz, die die Vorhersagegenauigkeit erhöht.

---

### **3. Trainingsfunktion**

#### **`train_fusion_model`**
- **Zweck**: Trainiert das `FusionModel` mit den kombinierten Eingaben aus den drei Feature-Gruppen, um zukünftige Kursbewegungen vorherzusagen.
- **Input**:
  - **Trainingsdaten**: Sequenzen aus drei Feature-Gruppen (Standard, Momentum, Volatilität).
  - **Zielwerte**: Erwartete Vorhersagen für jede Sequenz (Bitcoin-Kurs 30 Tage in die Zukunft).
  - **Modellparameter**: 
    - `hidden_dim`: Dimension der versteckten Schichten.
    - `batch_size`: Anzahl der Sequenzen pro Batch.
    - `learning_rate`: Schrittweite des Optimierungsalgorithmus.
    - `epochs`: Anzahl der Iterationen über den gesamten Datensatz.
    - `seq_length`: Länge der Eingabesequenzen.
- **Ablauf**:
  1. **Datenaufteilung**:
     - Der Datensatz wird in Trainings- und Validierungsdaten aufgeteilt (z. B. 80% Training, 20% Validierung).
     - Die Feature-Gruppen (Standard, Momentum, Volatilität) werden separat verarbeitet und skaliert.
  2. **Modellinitialisierung**:
     - Erstellt eine Instanz von `FusionModel` mit den spezifischen Architekturparametern.
  3. **Datenverarbeitung**:
     - Nutzt `DataLoader`, um Daten in Batches zu laden und effiziente Verarbeitung zu ermöglichen.
     - Skaliert Eingabesequenzen mit einem `StandardScaler`.
  4. **Training und Validierung**:
     - **Training**:
       - Eingabedaten aus den drei Feature-Gruppen werden durch das Modell verarbeitet.
       - Der Verlust wird mit dem **Mean Squared Error (MSE)** berechnet, um die Vorhersageabweichung zu messen.
       - Optimierung der Modellparameter erfolgt mit dem Adam-Optimierer.
     - **Validierung**:
       - Die Modellleistung wird auf unsichtbaren Validierungsdaten gemessen.
       - Validierungsmetriken wie **MSE**, **RMSE** und **R²** werden berechnet, um die Generalisierungsfähigkeit zu evaluieren.
  5. **Speicherung**:
     - Speichert das trainierte Modell als `.pth` zur späteren Verwendung.
  6. **Visualisierung**:
     - Erstellt Plots für Trainings- und Validierungsverlust sowie die Entwicklung von Metriken (MSE, RMSE, R²) über die Epochen.
     - Visualisiert die Kursverläufe (tatsächliche vs. vorhergesagte Werte) für Validierungsdaten.

---

### **4. Testskript**

#### **Zweck**: Evaluierung des trainierten Modells auf unsichtbaren Testdaten und Analyse der Vorhersagegenauigkeit.

1. **Daten laden und vorbereiten**:
   - **Quell-Daten**: Historische Kursdaten von Bitcoin (`BTC-USD`) werden mit der `yfinance`-Bibliothek heruntergeladen.
   - **Indikatoren berechnen**: Technische Indikatoren wie RSI, MACD, Bollinger-Bänder und mehr werden auf Basis der historischen Kursdaten berechnet.
   - **Sequenzgenerierung**:
     - Die Testdaten werden in Eingabesequenzen für Standard-Features, Momentum-Indikatoren und Volatilitäts-Indikatoren umgewandelt.
     - Ein gespeicherter `StandardScaler` wird geladen, um die Testsequenzen entsprechend zu normalisieren.

2. **Modell laden**:
   - Das trainierte Modell (`FusionModel`) wird geladen und in den Evaluierungsmodus (`eval()`) versetzt.
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
- `hidden_dim`: Dimension der versteckten Schichten. In **Experiment 2** wurde dieser Wert erhöht, um die Kapazität des Modells zu erweitern.
- `seq_length`: Länge der Eingabesequenzen. Hier wurde eine optimale Länge von 30 Tagen ermittelt, um kurzfristige Kursbewegungen zu erfassen.
- `batch_size`: Anzahl der Sequenzen pro Batch. Ein größerer Batch-Size wurde verwendet, um eine stabilere Gradientenberechnung zu ermöglichen.
- `learning_rate`: Schrittweite des Optimierungsalgorithmus. Ein dynamischer Lernratenplan wurde implementiert, um das Training effizienter zu gestalten.
- `epochs`: Anzahl der Iterationen über den gesamten Datensatz. Mehr Epochen wurden genutzt, um die komplexeren Muster der erweiterten Indikatoren zu lernen.

---

### **6. Verlust- und Metrikenberechnung**

#### **a) Trainingsverlust**
- Der Trainingsverlust misst, wie gut das Modell die Vorhersagen auf den Trainingsdaten macht. In **Experiment 2** wurde der Verlust während des Trainings durch erweiterte Indikatoren und optimierte LSTM-Schichten deutlich reduziert.
- Overfitting wurde durch regelmäßiges Monitoring und den Einsatz von Dropout minimiert.

#### **b) Validierungsverlust**
- Der Validierungsverlust misst, wie gut das Modell auf neuen, nicht im Training genutzten Daten funktioniert. Ein optimierter Gating-Mechanismus in **Experiment 2** half, die Diskrepanz zwischen Trainings- und Validierungsverlust zu verringern.

#### **c) Mean Squared Error (MSE)**
- Der MSE wurde in **Experiment 2** durch die Verwendung zusätzlicher technischer Indikatoren reduziert, da mehr Informationen für die Vorhersage bereitgestellt wurden.

#### **d) Root Mean Squared Error (RMSE)**
- Der RMSE sank durch die verbesserte Architektur und den Einsatz von Dual-Attention-Layern, die relevantere Feature-Gewichtungen ermöglichten.

#### **Zusammenfassung**
- **Trainingsverlust**: Wurde durch die optimierte Architektur und ein besseres Batch-Management weiter reduziert.
- **Validierungsverlust**: Zeigt die verbesserte Generalisierungsfähigkeit des Modells.
- **MSE und RMSE**: Diese Metriken haben sich im Vergleich zu Experiment 1 verbessert, da das Modell nun mehr Informationen und tiefere Schichten verwendet.

---

### **7. Plots und Visualisierung**

#### **a) Trainings- und Validierungsverlust**
- Visualisiert den Verlust für Training und Validierung über die Epochen.
- **Experiment 2** zeigt eine stabilere Konvergenz, da die zusätzlichen Indikatoren und Schichten besser integrierte Vorhersagen ermöglichen.

#### **b) Validierungsmetriken (MSE, RMSE, R²)**
- Zeigt die Entwicklung der Metriken für die Validierungsdaten über die Epochen.
- **R²-Wert**: In Experiment 2 näherte sich dieser Wert weiter an 1, was auf eine verbesserte Modellanpassung hinweist.

---

## **Testskript - Funktionsweise und Ablauf**

Das Testskript in **Experiment 2** enthält folgende Anpassungen und Verbesserungen:

### **1. Initialisierung und Vorbereitung**
- **Zufallszahlen-Seed**:
  - Zusätzlich zu den Seeds aus Experiment 1 wurden spezielle Seeds für die Dual-Attention-Mechanismen gesetzt, um eine konsistente Gewichtung der Features sicherzustellen.
- **Gerätekonfiguration**:
  - Optimiert für die parallele Verarbeitung mehrerer Feature-Gruppen.

---

### **2. Funktionen**

#### **a) Berechnung technischer Indikatoren**
- Zusätzliche Indikatoren wie MFI, ADX und OBV wurden implementiert, um erweiterte Marktinformationen zu integrieren.

#### **b) Laden von Testdaten**
- Die neuen technischen Indikatoren wurden in die Testdatenberechnung integriert.

#### **c) Sequenzierung der Daten**
- Sequenzierungen wurden für die drei Gruppen (Standard, Momentum, Volatilität) angepasst, um die erweiterten Eingabegrößen zu verarbeiten.

---

### **3. Hauptablauf**

#### **a) Datenvorbereitung**
1. **Scaler laden**:
   - Zusätzliche Scaler für die neuen Indikatorgruppen wurden implementiert.
2. **Testdaten laden**:
   - Die erweiterten technischen Indikatoren werden berechnet und mit den Kursdaten kombiniert.
3. **Sequenzgenerierung**:
   - Längere Sequenzen (30 Tage) wurden verwendet, um die erweiterten Features vollständig zu integrieren.

#### **b) Modell laden**
- Das trainierte Modell aus **Experiment 2** wird geladen und evaluiert.

#### **c) Vorhersage**
1. **Daten an das Modell übergeben**:
   - Die erweiterten Eingabesequenzen (drei Gruppen) werden verarbeitet.
2. **Skalierung zurücksetzen**:
   - Die Ergebnisse werden aus dem skalierten Bereich zurücktransformiert.

#### **d) Visualisierung**
- Zusätzliche Diagramme wurden eingefügt, die den Einfluss der neuen Indikatoren auf die Vorhersagen visualisieren.

---

### **4. Ausgabe**
Die erweiterte Ausgabe aus **Experiment 2** enthält:
- Zusätzliche Metriken, die den Einfluss der erweiterten Feature-Sets zeigen.
- Diagramme, die die neue Architektur visualisieren.

---

### **Wichtige Hinweise**
- **Skalierungsprobleme**:
  - Die neuen Indikatoren erfordern eine sorgfältige Skalierung, um keine dominierenden Features zu erzeugen.
- **Fehlerbehandlung**:
  - Das Skript wurde erweitert, um spezielle Probleme wie fehlende Werte in erweiterten Indikatoren zu behandeln.

---

## Baseline
- **MSE-Baseline**: Durchschnittliche Kursänderung aller vorherigen Tage als Vorhersage.
- **Erweiterung in Experiment 2**:
  - Die Baseline berücksichtigt zusätzlich die neuen Indikatoren, um eine realistischere Einschätzung der Modellleistung zu gewährleisten.
  - Eine naive Vorhersage, die die durchschnittlichen prozentualen Änderungen über die letzten 30 Tage verwendet, wurde als zusätzliche Baseline hinzugefügt.

---

## Ausführung
### Dateien
- **`experiment_2_data_processing.py`**: 
  - Bereitet die Daten vor, berechnet erweiterte technische Indikatoren und erstellt Sequenzen.
  - **Eingabe**: Historische Kursdaten (`yfinance`).
  - **Ausgabe**: Sequenzen (`Standard_X.pkl`, `Indicators_Group_1_X.pkl`, `Indicators_Group_2_X.pkl`).

- **`experiment_2_model_layer.py`**: 
  - Definiert die Bausteine des Modells:
    - Custom LSTM
    - Multi-Input LSTM mit Gating
    - Dual Attention Layer
  - **Erweiterung in Experiment 2**:
    - Implementiert erweiterte Gating-Mechanismen und Feature-spezifische Attention-Layer.

- **`experiment_2_model.py`**: 
  - Enthält die erweiterte Modelllogik für Training und Evaluation.
  - Trainiert das Modell und speichert die Ergebnisse in `fusion_model_v2.pth`.

- **`experiment_2.py`**: 
  - Führt Vorhersagen auf Testdaten aus und visualisiert die Ergebnisse.
  - **Eingabe**: `fusion_model_v2.pth`, Testdaten von `yfinance`.
  - **Ausgabe**: Vorhergesagte Preise und Performance-Metriken.

---

### Schritte zur Ausführung

```bash
# Verzeichnis wechseln
cd src/Experiment_2

# Datenverarbeitung
python experiment_2_data_processing.py
# Ergebnis: Generiert die vorbereiteten Sequenzen in ./Data/Samples/

# Training des Modells
python experiment_2_model.py
# Ergebnis: Trainiert das Modell und speichert es unter ./Data/Models/fusion_model_v2.pth

# Vorhersagen und Evaluation
python experiment_2.py
# Ergebnis: Zeigt die vorhergesagten Preise, Fehlermaße und einen Plot des Kursverlaufs

# Visualisierung des Modells
# TensorBoard: Visualisiert die Trainings- und Validierungsmetriken in Echtzeit
python experiment_2_tensorboard.py

# Netron: Zeigt die Modellarchitektur an
python experiment_2_netron.py

# TorchViz: Visualisiert den Datenfluss durch das Modell
python experiment_2_torchviz.py

---

### **Ergebnisse**
- Kaufpreis am 2023-02-01: 23723.76953125
- Tatsächlicher Preis am 2023-03-02: 23475.466796875
- Vorhergesagter Preis: 24441.06708843768
- Tatsächlicher Gewinn: -248.302734375
- Vorhergesagter Gewinn: 717.2975571876814
- MSE im Preis: 932383.9230659353
- RMSE im Preis: 965.6002915626814
- Absoluter Fehler: 965.6002915626814
- Prozentualer Fehler: 4.1132%
- **R² (Bestimmtheitsmaß)**: 0.9983

---

### **Änderungen im Vergleich zu Experiment 1**
- **Integration zusätzlicher technischer Indikatoren**:
  - Momentum-Indikatoren: MFI, WILLR, ADX, CCI20.
  - Volatilitäts-Indikatoren: SMA, EMA, OBV.
- **Erweiterung der Architektur**:
  - Die Anzahl der LSTM-Schichten wurde erhöht, um eine tiefere Analyse zu ermöglichen.
  - Anpassungen im Gating-Mechanismus und in den Attention-Schichten.
- **Training und Hyperparameter**:
  - Erhöhte Batchgröße und feinere Abstimmung der Lernrate.
  - Verbesserte Optimierung durch Gradienten-Skalierung.
- **Erweiterte Visualisierung**:
  - Plots zeigen detaillierte Metriken wie MSE und RMSE über die Epochen.
