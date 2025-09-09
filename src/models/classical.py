from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

def train_logistic(X, y, random_state=42):
    """Train logistic regression on X,y and return fitted model + metrics dict."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)
    
    model = LogisticRegression(random_state=random_state)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test)
    }
    
    return model, metrics

def train_svm_rbf(X, y, random_state=42):
    """Train SVM with RBF kernel; do a simple grid search on C and gamma and return best model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)
    
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }
    
    svm = SVC(kernel='rbf', random_state=random_state)
    grid_search = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    metrics = {
        'best_params': grid_search.best_params_,
        'train_accuracy': accuracy_score(y_train, y_pred_train),
        'test_accuracy': accuracy_score(y_test, y_pred_test)
    }
    
    return best_model, metrics

if __name__ == '__main__':
    # This is a placeholder for demonstration. 
    # In the final notebook, we will load the actual data.
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=100, n_features=6, n_informative=4, n_redundant=2, random_state=42)
    
    print("Training Logistic Regression...")
    lr_model, lr_metrics = train_logistic(X, y)
    print(f"Logistic Regression Metrics: {lr_metrics}")

    print("\nTraining SVM with RBF Kernel...")
    svm_model, svm_metrics = train_svm_rbf(X, y)
    print(f"SVM RBF Metrics: {svm_metrics}")
