import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier

# Load UCI HAR Dataset
def load_data():
    base_path = "C:/Users/sabth/Downloads/human+activity+recognition+using+smartphones/UCI HAR Dataset/UCI HAR Dataset"
    
    # Load training and testing data
    X_train = pd.read_csv(f"{base_path}/train/X_train.txt", delim_whitespace=True, header=None)
    X_test = pd.read_csv(f"{base_path}/test/X_test.txt", delim_whitespace=True, header=None)
    y_train = pd.read_csv(f"{base_path}/train/y_train.txt", delim_whitespace=True, header=None).squeeze()
    y_test = pd.read_csv(f"{base_path}/test/y_test.txt", delim_whitespace=True, header=None).squeeze()

    # Load activity labels
    activity_labels = pd.read_csv(f"{base_path}/activity_labels.txt", delim_whitespace=True, header=None, index_col=0)
    activity_labels = activity_labels[1].to_dict()
    
    return X_train, X_test, y_train, y_test, activity_labels

# Preprocess the data
def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA for dimensionality reduction (optional)
    pca = PCA(n_components=50)  # Retain 50 components
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return X_train_pca, X_test_pca

# Plot ROC Curve
def plot_roc_curve(y_test, y_score, n_classes, activity_labels):
    plt.figure(figsize=(10, 8))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f"Activity {activity_labels[i+1]} (AUC = {roc_auc[i]:.2f})")

    # Plot micro-average ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"], label=f"Micro-average (AUC = {roc_auc['micro']:.2f})", linestyle="--", color="black")

    # Plot the diagonal
    plt.plot([0, 1], [0, 1], "k--", lw=2)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Multi-Class Classification")
    plt.legend(loc="lower right")
    plt.show()

# Train and evaluate the model
def train_and_evaluate(X_train, X_test, y_train, y_test, activity_labels):
    # Binarize labels for multi-class ROC Curve
    n_classes = len(activity_labels)
    y_train_bin = label_binarize(y_train, classes=range(1, n_classes+1))
    y_test_bin = label_binarize(y_test, classes=range(1, n_classes+1))

    # Train-test split for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Train Random Forest Classifier with One-vs-Rest
    model = OneVsRestClassifier(RandomForestClassifier(random_state=42, n_estimators=100))
    model.fit(X_train_split, y_train_split)

    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    y_test_score = model.predict_proba(X_test)

    print("\nTest Performance:")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Precision:", precision_score(y_test, y_test_pred, average="macro"))
    print("Recall:", recall_score(y_test, y_test_pred, average="macro"))
    print("F1-Score:", f1_score(y_test, y_test_pred, average="macro"))
    print("\nClassification Report:\n", classification_report(y_test, y_test_pred, target_names=list(activity_labels.values())))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(activity_labels))
    plt.xticks(tick_marks, activity_labels.values(), rotation=45)
    plt.yticks(tick_marks, activity_labels.values())
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    # Plot ROC Curve
    plot_roc_curve(y_test_bin, y_test_score, n_classes, activity_labels)

if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test, activity_labels = load_data()

    # Preprocess data
    X_train_pca, X_test_pca = preprocess_data(X_train, X_test)

    # Train and evaluate the model
    train_and_evaluate(X_train_pca, X_test_pca, y_train, y_test, activity_labels)
