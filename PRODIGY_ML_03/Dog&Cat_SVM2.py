import pickle
import random
import os
import numpy as np
import cv2
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import time

start_time = time.time()

# Load the data
with open('data1.pickle', 'rb') as pick_in:
    data = pickle.load(pick_in)

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# FIRST EXECUTE THE TRAINING MODEL PART 

# # Standardize the data
# scaler = StandardScaler().fit(X_train)
# X_train_scaled = scaler.transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Determine the number of components to retain 95% of the variance
# pca = PCA().fit(X_train_scaled)
# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# n_components = np.argmax(cumulative_variance >= 0.95) + 1

# print(f'Number of components to retain 95% variance: {n_components}')


# # Apply PCA with the determined number of components
# pca = PCA(n_components=n_components)
# X_train_pca = pca.fit_transform(X_train_scaled)
# X_test_pca = pca.transform(X_test_scaled)

# # Use GridSearchCV to find the best hyperparameters for the SVM
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': ['scale', 'auto'],
#     'kernel': ['rbf', 'poly', 'sigmoid']
# }

# grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=5)
# grid_search.fit(X_train_pca, y_train)
# best_model = grid_search.best_estimator_

# print(f"Best parameters found: {grid_search.best_params_}")
# print(f"Training model time: {time.time() - start_time} seconds")

# # Save the best model
# with open('best_model.sav', 'wb') as pick:
#     pickle.dump(best_model, pick)
# print(f"Saving model time: {time.time() - start_time} seconds")

# THIS IS THE AFTER TRAINING THE MODEL, UNCOMMENT THIS PART AND EXECUTE FOR PREDICTION

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# The above calculated number of components for PCA is (n_components = 980)
# Apply PCA with the determined number of components
pca = PCA(n_components=980)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


# Load the best model and evaluate
with open('best_model.sav', 'rb') as pick:
    model = pickle.load(pick)

model_score = model.score(X_test_pca, y_test)

print('Model score: ', model_score)

predictions = model.predict(X_test_pca)

# Print classification report
categories = ['Cat', 'Dog']
print(classification_report(y_test, predictions, target_names=categories))

# Select 5 random images from the test set
random_indices = random.sample(range(len(X_test_pca)), 6)

plt.figure(figsize=(10, 5))
for i, random_index in enumerate(random_indices):
    random_image = X_test[random_index]
    random_image_pca = X_test_pca[random_index].reshape(1, -1)

    # Predict the label for the random image
    predicted_label = model.predict(random_image_pca)[0]

    predicted_category = categories[predicted_label]
    actual_category = categories[y_test[random_index]]

    # Reshape the image back to the original shape
    reshaped_image = np.array(random_image).reshape(50, 50, 3)

    # Display the image with the prediction and actual label
    plt.subplot(2, 3, i + 1)
    plt.imshow(reshaped_image)
    plt.title(f"Prediction: {predicted_category}\nActual: {actual_category}")
    plt.axis('off')

plt.tight_layout()
plt.show()
