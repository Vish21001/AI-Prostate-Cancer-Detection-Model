from tensorflow.keras.models import load_model
from preprocess import load_images
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

model = load_model("prostate_cancer_model.h5")
X, y_true = load_images()
y_pred = np.argmax(model.predict(X), axis=1)
y_true_labels = np.argmax(y_true, axis=1)

print("Classification Report:\n", classification_report(y_true_labels, y_pred, target_names=["Healthy","Cancerous"]))
print("Confusion Matrix:\n", confusion_matrix(y_true_labels, y_pred))
