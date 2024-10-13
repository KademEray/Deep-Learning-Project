from tensorflow.keras.models import load_model

# Lade das trainierte Modell
model_path = "./models/trained/lstm_standard_model"
model = load_model(model_path)

# Zeige die Modellzusammenfassung an, um die Input-Form zu sehen
model.summary()

# Überprüfe die Input-Shape explizit
input_shape = model.input_shape
print(f"Das Modell erwartet die Input-Shape: {input_shape}")
