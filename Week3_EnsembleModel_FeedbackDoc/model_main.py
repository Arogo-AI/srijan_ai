import numpy as np
import lightgbm as lgb
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# 1. Define the PyTorch model architecture
# ------------------------------------------------------------------------------
class SimpleNet(nn.Module):
    def __init__(self, input_size=25, hidden1=256, hidden2=512, hidden3=256, hidden4=128, num_classes=1203):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden3, hidden4)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(hidden4, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.fc5(x)  # Raw logits (will apply softmax later)
        return x

# ------------------------------------------------------------------------------
# 2. Load the pre-trained models
# ------------------------------------------------------------------------------

# Load LightGBM model (assumed saved as a text file)
lgb_model_path = 'lgb_model.txt'  # Update with your actual file path
loaded_lgb = lgb.Booster(model_file=lgb_model_path)

# Load XGBoost model from a pickle file
xgb_model_path = 'xgb_model.pkl'  # Update with your actual file path
loaded_xgb = joblib.load(xgb_model_path)

# Load PyTorch model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytorch_model_path = 'simple_net_state_dict.pth'  # Update with your actual file path
pytorch_model = SimpleNet(input_size=25, num_classes=1203).to(device)
pytorch_model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
pytorch_model.eval()

# ------------------------------------------------------------------------------
# 3. Define the ensemble prediction functions
# ------------------------------------------------------------------------------

def ensemble_predict(X, weights=None):
    """
    Returns the ensemble's top-1 prediction for each sample.
    This is useful for computing overall accuracy.
    
    Args:
        X (np.array): Input feature array of shape (n_samples, n_features)
        weights (list or tuple): Weights for the three models [w_lgb, w_xgb, w_pt].
                                 Defaults to equal weighting if None.
    
    Returns:
        final_preds (np.array): Array of shape (n_samples,) with the predicted class labels.
    """
    if weights is None:
        weights = [1/3, 1/3, 1/3]
    w_lgb, w_xgb, w_pt = weights

    # LightGBM: Get probability distribution
    lgb_probs = loaded_lgb.predict(X)  # Shape: (n_samples, num_classes)
    
    # XGBoost: Get probability distribution
    xgb_probs = loaded_xgb.predict_proba(X)  # Shape: (n_samples, num_classes)
    
    # PyTorch: Get probability distribution (apply softmax to logits)
    X_tensor = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        outputs = pytorch_model(X_tensor)
        pt_probs = F.softmax(outputs, dim=1).cpu().numpy()  # Shape: (n_samples, num_classes)
    
    # Compute weighted sum of probabilities
    ensemble_probs = w_lgb * lgb_probs + w_xgb * xgb_probs + w_pt * pt_probs

    # Return the class with the highest probability for each sample
    final_preds = np.argmax(ensemble_probs, axis=1)
    return final_preds

def ensemble_topk(X, top_k=3, weights=None):
    """
    Returns the top-k predictions (class indices and their probabilities) for each sample.
    
    Args:
        X (np.array): Input feature array of shape (n_samples, n_features)
        top_k (int): Number of top predictions to return.
        weights (list or tuple): Weights for the three models [w_lgb, w_xgb, w_pt].
                                 Defaults to equal weighting if None.
    
    Returns:
        top_k_indices (np.array): Array of shape (n_samples, top_k) with top-k class indices.
        top_k_probs (np.array): Array of shape (n_samples, top_k) with corresponding probabilities.
    """
    if weights is None:
        weights = [1/3, 1/3, 1/3]
    w_lgb, w_xgb, w_pt = weights

    # Get probability distributions from each model
    lgb_probs = loaded_lgb.predict(X)
    xgb_probs = loaded_xgb.predict_proba(X)
    X_tensor = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        outputs = pytorch_model(X_tensor)
        pt_probs = F.softmax(outputs, dim=1).cpu().numpy()

    # Compute weighted sum of probabilities
    ensemble_probs = w_lgb * lgb_probs + w_xgb * xgb_probs + w_pt * pt_probs

    # For each sample, retrieve the indices of the top k probabilities.
    top_k_indices = np.argsort(ensemble_probs, axis=1)[:, -top_k:][:, ::-1]
    top_k_probs = np.take_along_axis(ensemble_probs, top_k_indices, axis=1)
    
    return top_k_indices, top_k_probs

# ------------------------------------------------------------------------------
# 4. Example usage
# ------------------------------------------------------------------------------

# Assuming X_scaled is your NumPy array of input features:
# For instance:
# X_scaled = np.load('X_scaled.npy')

# # Get top-1 predictions (for accuracy computation)
# final_predictions = ensemble_predict(X_scaled, weights=[0.4, 0.3, 0.3])
# print("Final predictions (top-1) for each sample:\n", final_predictions)

# # # Get top-3 predictions (with probabilities)
# top_k_indices, top_k_probs = ensemble_topk(X_scaled, top_k=3, weights=[0.4, 0.3, 0.3])
# print("Top-3 class indices for each sample:\n", top_k_indices)
# print("Corresponding probabilities for top-3 predictions:\n", top_k_probs)