import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# LSTM model architecture
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, (hn, cn) = self.lstm(x)
        # We take the output of the last time step
        last_step_out = out[:, -1, :]
        logits = self.fc(last_step_out)
        return logits

def load_and_preprocess_sequences(file_path, seq_len=10):
    """
    Loads the master training dataset, segments it by stock (detecting date jumps),
    scales the features, and builds sequences for LSTM training.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Run training first.")
        
    df = pd.read_csv(file_path)
    
    # 16 standard features
    feature_cols = [
        'RSI', 'MACD_Hist', 'MACD_Crossover', 
        'Price_Above_MA50', 'Price_Above_MA200', 'Golden_Cross', 
        'Volume_Ratio', 'Sentiment_Score', 'Positive_Headlines', 'Negative_Headlines',
        'FII_10d_Net', 'DII_10d_Net', 'FII_Trend', 'DII_Trend', 
        'Divergence_Flag', 'Multi_Timeframe_Alignment',
        'SP500_Return', 'Crude_Return', 'USD_INR_Return'
    ]
    
    # Fill any NaNs just in case
    df[feature_cols] = df[feature_cols].fillna(0.0)
    
    # Detect boundaries between different stocks (when Date decreases chronologically)
    dates = pd.to_datetime(df['Date'])
    split_indices = [0]
    for i in range(1, len(dates)):
        if dates.iloc[i] < dates.iloc[i-1]:
            split_indices.append(i)
    split_indices.append(len(df))
    
    # Segment data by stock
    stock_dfs = []
    for k in range(len(split_indices) - 1):
        stock_dfs.append(df.iloc[split_indices[k]:split_indices[k+1]])
    
    print(f"[+] Segmented master dataset into {len(stock_dfs)} individual stock series.")
    
    # Fit scaler on the entire feature set
    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    
    # Build sequences for each stock independently to avoid mixing across stocks
    X_seq = []
    y_seq = []
    
    for s_df in stock_dfs:
        if len(s_df) < seq_len:
            continue
        # Scale the features for this stock
        scaled_features = scaler.transform(s_df[feature_cols])
        labels = s_df['Label'].values
        
        # Slide window
        for i in range(len(s_df) - seq_len + 1):
            X_seq.append(scaled_features[i : i + seq_len])
            y_seq.append(labels[i + seq_len - 1])
            
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.int64)
    
    # Save the scaler for inference use if needed
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/lstm_scaler.pkl")
    
    return X_seq, y_seq, feature_cols

def run_lstm_training():
    print("=========================================================================")
    print("[*] Starting LSTM Deep Learning Comparison Pipeline")
    print("=========================================================================")
    
    dataset_path = "data/processed/master_training_dataset.csv"
    seq_len = 10
    
    try:
        X, y, feature_cols = load_and_preprocess_sequences(dataset_path, seq_len=seq_len)
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        return
        
    print(f"[+] Prepared {len(X)} sequences of length {seq_len}.")
    
    # Stratified Train-Test Split (80% train, 20% validation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[+] Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    
    # Create PyTorch DataLoaders
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset  = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Model parameters
    input_dim  = len(feature_cols)
    hidden_dim = 64
    num_layers = 2
    output_dim = 3  # SELL, HOLD, BUY
    
    model = LSTMClassifier(input_dim, hidden_dim, num_layers, output_dim, dropout=0.2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\n[*] Training LSTM model on CPU...")
    epochs = 10
    t0 = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        train_acc = correct / total
        train_loss = epoch_loss / total
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                
        val_acc = val_correct / val_total
        val_loss = val_loss / val_total
        
        print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
    lstm_time = time.time() - t0
    print(f"[+] LSTM training complete in {lstm_time:.2f}s.")
    
    # Save LSTM model
    torch.save(model.state_dict(), "models/lstm_model.pth")
    print("[+] Saved LSTM weights to models/lstm_model.pth")
    
    # Get final predictions for classification report
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_targets.extend(batch_y.numpy())
            
    lstm_acc = accuracy_score(all_targets, all_preds)
    
    print("\nLSTM Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=["SELL", "HOLD", "BUY"]))
    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))
    
    # ── Load and Evaluate Random Forest & XGBoost accuracies on same validation data for comparison ──
    # Since RF and XGBoost are tabular, we evaluate them on the *last day* of each validation sequence.
    # The last day of sequence X_test corresponds to row X_test[:, -1, :].
    # But wait, X_test is standardized using our scaler. Tree models don't care about scaling,
    # but we can invert scaling or just pass the unscaled values.
    # To keep it exact, we can load the raw tabular test set or run them on the unscaled last step.
    # Let's load the saved models and test them on the unscaled test features to get the exact accuracies.
    
    rf_acc = 0.0
    xgb_acc = 0.0
    
    # Load scaler to unscale validation last-step features
    scaler = joblib.load("models/lstm_scaler.pkl")
    # shape of X_test: (N, 10, 16). Last step: X_test[:, -1, :]
    X_test_last_step = X_test[:, -1, :]
    X_test_unscaled = scaler.inverse_transform(X_test_last_step)
    
    # Convert back to dataframe matching feature names
    X_test_df = pd.DataFrame(X_test_unscaled, columns=feature_cols)
    
    # Load XGBoost model
    xgb_path = "models/xgboost_model.pkl"
    if os.path.exists(xgb_path):
        try:
            xgb_model = joblib.load(xgb_path)
            xgb_preds = xgb_model.predict(X_test_df)
            xgb_acc = accuracy_score(y_test, xgb_preds)
        except Exception as e:
            print(f"[Warning] Could not evaluate XGBoost: {e}")
            
    # Load Random Forest model if it exists
    # Wait, did we keep RF baseline anywhere? Let's check models/ folder
    # In train_model.py, we trained RF but we didn't save it because we removed old RF files
    # wait! The verify script says: "Clean - only xgboost_model.pkl present".
    # So rf_model.pkl was deleted. We can quickly retrain a quick Random Forest classifier on the training set
    # of the sequence's last step to get a true, fresh RF accuracy on the exact same data split!
    # This is a very clean and mathematically honest way to compare them.
    
    print("\n[*] Training comparison Random Forest baseline on sequence last-step...")
    X_train_last_step = X_train[:, -1, :]
    X_train_unscaled = scaler.inverse_transform(X_train_last_step)
    X_train_df = pd.DataFrame(X_train_unscaled, columns=feature_cols)
    
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf_model.fit(X_train_df, y_train)
    rf_preds = rf_model.predict(X_test_df)
    rf_acc = accuracy_score(y_test, rf_preds)
    
    # Print the Comparison Table
    print("\n" + "="*70)
    print("                FINAL MULTI-MODEL COMPARISON TABLE")
    print("="*70)
    print(f"  {'Model Type':<25} {'Architecture / Spec':<22} {'Val Accuracy':>15}")
    print(f"  {'-'*25} {'-'*22} {'-'*15}")
    print(f"  {'Random Forest':<25} {'Tabular (Baseline)':<22} {rf_acc*100:>13.2f}%")
    print(f"  {'XGBoost':<25} {'Tabular (Tuned)':<22} {xgb_acc*100:>13.2f}%")
    print(f"  {'LSTM Neural Network':<25} {'Sequence 10-day (CPU)':<22} {lstm_acc*100:>13.2f}%")
    print("="*70)
    print("Note: LSTM models temporal patterns using 10 days of historical sequential data,")
    print("while Random Forest and XGBoost make decisions based on single-day snapshot features.")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_lstm_training()
