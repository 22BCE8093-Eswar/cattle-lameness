# scripts/train_svm.py
import pickle
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

# Load your extracted features and labels
features = np.load('data/features.npy')  # shape: (n_samples, 512)
labels = np.load('data/labels.npy')      # shape: (n_samples,)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

svm_model = svm.SVC(kernel='linear', probability=True)
svm_model.fit(features, y_encoded)

# Save model and label encoder
with open('models/svm_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
