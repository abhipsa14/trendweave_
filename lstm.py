import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Load the data
data = pd.read_csv('sentiment_results.csv')

print("Dataset Overview:")
print(f"Shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"Sentiment distribution: {data['sentiment'].value_counts()}")
print(f"Unique sentiments: {data['sentiment'].unique()}")

# Custom Dataset class for PyTorch
class FashionTrendDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.LongTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# LSTM Model using PyTorch
class FashionTrendLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, num_classes=2, dropout=0.2):
        super(FashionTrendLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size//2, batch_first=True, dropout=dropout)
        self.lstm3 = nn.LSTM(hidden_size//2, hidden_size//4, batch_first=True, dropout=dropout)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        self.bn3 = nn.BatchNorm1d(hidden_size//4)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size//4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
        # Final dropout
        self.dropout_final = nn.Dropout(0.3)
        
    def forward(self, x):
        # LSTM layers with batch normalization
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.bn1(lstm_out1.transpose(1, 2)).transpose(1, 2)
        lstm_out1 = self.dropout1(lstm_out1)
        
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.bn2(lstm_out2.transpose(1, 2)).transpose(1, 2)
        lstm_out2 = self.dropout2(lstm_out2)
        
        lstm_out3, _ = self.lstm3(lstm_out2)
        lstm_out3 = self.bn3(lstm_out3.transpose(1, 2)).transpose(1, 2)
        lstm_out3 = self.dropout3(lstm_out3)
        
        # Take the last output
        last_output = lstm_out3[:, -1, :]
        
        # Dense layers
        x = F.relu(self.fc1(last_output))
        x = self.dropout_final(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# Main predictor class
class FashionTrendPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.sequence_length = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def preprocess_data(self, data):
        """Preprocess the fashion trend data"""
        # Create category features from filename
        data['category'] = data['filename'].apply(self.extract_category)
        data['season'] = data['filename'].apply(self.extract_season)
        data['gender'] = data['filename'].apply(self.extract_gender)
        data['city'] = data['filename'].apply(self.extract_city)
        
        # Encode categorical variables
        categorical_cols = ['category', 'season', 'gender', 'city']
        for col in categorical_cols:
            le = LabelEncoder()
            data[f'{col}_encoded'] = le.fit_transform(data[col].fillna('unknown'))
        
        # Create trend momentum feature
        data['trend_momentum'] = data['positive_prob'] - data['negative_prob']
        
        # Create confidence-weighted sentiment score
        data['weighted_sentiment'] = data['confidence'] * data['positive_prob']
        
        # Create sentiment volatility (measure of uncertainty)
        data['sentiment_volatility'] = 1 - data['confidence']
        
        # Encode sentiment for target variable (binary: 0=negative, 1=positive)
        data['sentiment_encoded'] = (data['sentiment'] == 'positive').astype(int)
        
        return data
    
    def extract_category(self, filename):
        """Extract category from filename"""
        filename_lower = filename.lower()
        categories = {
            'denim': ['denim'],
            'knitwear': ['knitwear', 'knit'],
            'footwear': ['footwear', 'shoes'],
            'bags': ['bags', 'bag'],
            'eyewear': ['eyewear', 'glasses'],
            'outerwear': ['outerwear', 'jacket'],
            'accessories': ['accessories', 'soft_accessories'],
            'intimates': ['intimates', 'loungewear'],
            'suits': ['suits', 'sets'],
            'shirts': ['shirts', 'woven'],
            'trousers': ['trousers', 'shorts'],
            'evening': ['evening', 'special_occasion'],
            'color': ['colour', 'color'],
            'prints': ['prints', 'graphics'],
            'textiles': ['textiles', 'materials']
        }
        
        for category, keywords in categories.items():
            if any(keyword in filename_lower for keyword in keywords):
                return category
        return 'general'
    
    def extract_season(self, filename):
        """Extract season from filename"""
        if 'a_w' in filename.lower() or 'aw' in filename.lower():
            return 'autumn_winter'
        elif 's_s' in filename.lower() or 'ss' in filename.lower():
            return 'spring_summer'
        return 'unknown'
    
    def extract_gender(self, filename):
        """Extract gender from filename"""
        filename_lower = filename.lower()
        if 'men' in filename_lower and 'women' not in filename_lower:
            return 'men'
        elif 'women' in filename_lower:
            return 'women'
        return 'unisex'
    
    def extract_city(self, filename):
        """Extract city from filename"""
        cities = ['london', 'paris', 'milan', 'new_york', 'copenhagen']
        filename_lower = filename.lower()
        for city in cities:
            if city in filename_lower:
                return city
        return 'unknown'
    
    def create_sequences(self, data, target_col):
        """Create sequences for LSTM training"""
        features = ['confidence', 'negative_prob', 'positive_prob', 'trend_momentum', 
                   'weighted_sentiment', 'sentiment_volatility', 'category_encoded', 
                   'gender_encoded', 'city_encoded']
        
        # Scale features
        scaled_features = self.scaler.fit_transform(data[features])
        
        X, y = [], []
        for i in range(len(scaled_features) - self.sequence_length):
            X.append(scaled_features[i:(i + self.sequence_length)])
            y.append(data[target_col].iloc[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def train_model(self, data, epochs=100, batch_size=32, learning_rate=0.001):
        """Train the LSTM model"""
        # Preprocess data
        processed_data = self.preprocess_data(data.copy())
        
        # Sort by a logical order (e.g., by filename to maintain some temporal consistency)
        processed_data = processed_data.sort_values('filename').reset_index(drop=True)
        
        # Create sequences
        X, y = self.create_sequences(processed_data, 'sentiment_encoded')
        
        print(f"Created {len(X)} sequences of length {self.sequence_length}")
        print(f"Feature dimension: {X.shape[2]}")
        print(f"Target distribution: {np.bincount(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create datasets and dataloaders
        train_dataset = FashionTrendDataset(X_train, y_train)
        test_dataset = FashionTrendDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_size = X.shape[2]
        self.model = FashionTrendLSTM(input_size=input_size, num_classes=2).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        # Training history
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15
        
        print(f"Starting training on {self.device}...")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate averages
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # Print progress
            if epoch % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, '
                      f'Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, '
                      f'Val Acc: {val_acc:.2f}%')
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered at epoch {epoch+1}')
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        # Final evaluation
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        # Print final results
        final_accuracy = accuracy_score(all_targets, all_predictions)
        print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_targets, all_predictions, 
                                  target_names=['Negative', 'Positive']))
        
        # Return training history
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_accuracy': train_accuracies,
            'val_accuracy': val_accuracies
        }
        
        return history, X_test, y_test, all_predictions
    
    def predict_future_trends(self, data, num_predictions=5):
        """Predict future trends based on recent patterns"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        processed_data = self.preprocess_data(data.copy())
        processed_data = processed_data.sort_values('filename').reset_index(drop=True)
        
        # Get the last sequence
        features = ['confidence', 'negative_prob', 'positive_prob', 'trend_momentum', 
                   'weighted_sentiment', 'sentiment_volatility', 'category_encoded', 
                   'gender_encoded', 'city_encoded']
        
        last_sequence = self.scaler.transform(processed_data[features])[-self.sequence_length:]
        
        predictions = []
        current_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for step in range(num_predictions):
                # Predict next sentiment
                outputs = self.model(current_sequence)
                probabilities = F.softmax(outputs, dim=1)
                pred_class = torch.argmax(probabilities, dim=1).item()
                
                # Get probabilities
                neg_prob = probabilities[0][0].item()
                pos_prob = probabilities[0][1].item()
                confidence = torch.max(probabilities).item()
                
                # Store prediction
                predictions.append({
                    'step': step + 1,
                    'predicted_sentiment': 'positive' if pred_class == 1 else 'negative',
                    'confidence': confidence,
                    'positive_prob': pos_prob,
                    'negative_prob': neg_prob,
                    'trend_momentum': pos_prob - neg_prob
                })
                
                # Update sequence for next prediction (rolling window)
                new_features = np.array([
                    confidence,  # confidence
                    neg_prob,    # negative_prob
                    pos_prob,    # positive_prob
                    pos_prob - neg_prob,  # trend_momentum
                    confidence * pos_prob,  # weighted_sentiment
                    1 - confidence,  # sentiment_volatility
                    current_sequence[0, -1, -3].item(),  # category_encoded (keep last)
                    current_sequence[0, -1, -2].item(),  # gender_encoded (keep last)
                    current_sequence[0, -1, -1].item()   # city_encoded (keep last)
                ])
                
                # Roll the sequence
                current_sequence = torch.roll(current_sequence, -1, dims=1)
                current_sequence[0, -1, :] = torch.FloatTensor(new_features).to(self.device)
        
        return predictions
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Plot training & validation loss
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot training & validation accuracy
        ax2.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot loss difference
        loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
        ax3.plot(epochs, loss_diff, 'g-', linewidth=2)
        ax3.set_title('Overfitting Monitor (Val Loss - Train Loss)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss Difference')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Plot accuracy difference
        acc_diff = np.array(history['val_accuracy']) - np.array(history['train_accuracy'])
        ax4.plot(epochs, acc_diff, 'orange', linewidth=2)
        ax4.set_title('Generalization Monitor (Val Acc - Train Acc)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy Difference (%)')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_trends(self, data):
        """Analyze current trends in the data"""
        processed_data = self.preprocess_data(data.copy())
        
        # Trend analysis by category
        category_trends = processed_data.groupby('category').agg({
            'positive_prob': ['mean', 'std', 'count'],
            'negative_prob': ['mean', 'std'],
            'confidence': ['mean', 'std'],
            'trend_momentum': ['mean', 'std']
        }).round(3)
        
        # Flatten column names
        category_trends.columns = ['_'.join(col).strip() for col in category_trends.columns]
        
        print("Trend Analysis by Category:")
        print(category_trends.sort_values('trend_momentum_mean', ascending=False))
        
        # Trend analysis by gender
        gender_trends = processed_data.groupby('gender').agg({
            'positive_prob': ['mean', 'std', 'count'],
            'negative_prob': ['mean', 'std'],
            'confidence': ['mean', 'std'],
            'trend_momentum': ['mean', 'std']
        }).round(3)
        
        gender_trends.columns = ['_'.join(col).strip() for col in gender_trends.columns]
        
        print("\nTrend Analysis by Gender:")
        print(gender_trends.sort_values('trend_momentum_mean', ascending=False))
        
        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Sentiment distribution
        sentiment_counts = data['sentiment'].value_counts()
        colors = ['#ff6b6b', '#4ecdc4']
        axes[0,0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
                     colors=colors, startangle=90)
        axes[0,0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        
        # Trend momentum by category
        cat_momentum = processed_data.groupby('category')['trend_momentum'].mean().sort_values(ascending=True)
        colors_cat = ['#ff6b6b' if x < 0 else '#4ecdc4' for x in cat_momentum.values]
        axes[0,1].barh(range(len(cat_momentum)), cat_momentum.values, color=colors_cat)
        axes[0,1].set_yticks(range(len(cat_momentum)))
        axes[0,1].set_yticklabels(cat_momentum.index, fontsize=10)
        axes[0,1].set_title('Trend Momentum by Category', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Trend Momentum')
        axes[0,1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Confidence distribution
        axes[0,2].hist(data['confidence'], bins=20, alpha=0.7, color='#4ecdc4', edgecolor='black')
        axes[0,2].set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        axes[0,2].set_xlabel('Confidence Score')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].axvline(x=data['confidence'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {data["confidence"].mean():.3f}')
        axes[0,2].legend()
        
        # Positive vs Negative probability scatter
        scatter = axes[1,0].scatter(data['positive_prob'], data['negative_prob'], 
                                  c=data['confidence'], cmap='viridis', alpha=0.6, s=50)
        axes[1,0].set_title('Positive vs Negative Probability', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Positive Probability')
        axes[1,0].set_ylabel('Negative Probability')
        axes[1,0].plot([0, 1], [1, 0], 'r--', alpha=0.5, label='Decision Boundary')
        axes[1,0].legend()
        plt.colorbar(scatter, ax=axes[1,0], label='Confidence')
        
        # Trend momentum distribution
        axes[1,1].hist(processed_data['trend_momentum'], bins=20, alpha=0.7, 
                      color='#ffa07a', edgecolor='black')
        axes[1,1].set_title('Trend Momentum Distribution', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Trend Momentum')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral')
        axes[1,1].axvline(x=processed_data['trend_momentum'].mean(), color='blue', 
                         linestyle='--', label=f'Mean: {processed_data["trend_momentum"].mean():.3f}')
        axes[1,1].legend()
        
        # Gender trend comparison
        gender_data = processed_data.groupby('gender')['trend_momentum'].mean()
        colors_gender = ['#ff6b6b' if x < 0 else '#4ecdc4' for x in gender_data.values]
        axes[1,2].bar(gender_data.index, gender_data.values, color=colors_gender, 
                     alpha=0.8, edgecolor='black')
        axes[1,2].set_title('Trend Momentum by Gender', fontsize=14, fontweight='bold')
        axes[1,2].set_xlabel('Gender')
        axes[1,2].set_ylabel('Average Trend Momentum')
        axes[1,2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for i, v in enumerate(gender_data.values):
            axes[1,2].text(i, v + 0.01 if v > 0 else v - 0.01, f'{v:.3f}', 
                          ha='center', va='bottom' if v > 0 else 'top', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

# Initialize and train the model
predictor = FashionTrendPredictor()

# Analyze current trends
print("=== CURRENT TREND ANALYSIS ===")
predictor.analyze_trends(data)

# Train the model
print("\n=== TRAINING PYTORCH LSTM MODEL ===")
history, X_test, y_test, y_pred = predictor.train_model(data, epochs=100, batch_size=16)

# Plot training history
predictor.plot_training_history(history)

# Predict future trends
print("\n=== FUTURE TREND PREDICTIONS ===")
future_predictions = predictor.predict_future_trends(data, num_predictions=10)

print("Predicted Future Trends:")
for i, pred in enumerate(future_predictions, 1):
    momentum_emoji = "📈" if pred['trend_momentum'] > 0 else "📉"
    confidence_emoji = "🔥" if pred['confidence'] > 0.8 else "⚡" if pred['confidence'] > 0.6 else "💫"
    
    print(f"Step {pred['step']}: {momentum_emoji} {pred['predicted_sentiment'].upper()} {confidence_emoji}")
    print(f"  Confidence: {pred['confidence']:.3f} | Momentum: {pred['trend_momentum']:+.3f}")
    print(f"  Positive: {pred['positive_prob']:.3f} | Negative: {pred['negative_prob']:.3f}")
    print()

# Create a summary report
print("=== TREND PREDICTION SUMMARY ===")
positive_trends = sum(1 for p in future_predictions if p['predicted_sentiment'] == 'positive')
negative_trends = sum(1 for p in future_predictions if p['predicted_sentiment'] == 'negative')
avg_confidence = np.mean([p['confidence'] for p in future_predictions])
avg_momentum = np.mean([p['trend_momentum'] for p in future_predictions])

print(f"Prediction Overview:")
print(f"  • Positive trends: {positive_trends}/{len(future_predictions)} ({positive_trends/len(future_predictions)*100:.1f}%)")
print(f"  • Negative trends: {negative_trends}/{len(future_predictions)} ({negative_trends/len(future_predictions)*100:.1f}%)")
print(f"  • Average confidence: {avg_confidence:.3f}")
print(f"  • Average momentum: {avg_momentum:+.3f}")

if positive_trends > negative_trends:
    print(f"\n Overall Outlook: BULLISH")
    print("Fashion trends are expected to be predominantly positive!")
elif negative_trends > positive_trends:
    print(f"\nOverall Outlook: BEARISH") 
    print("Fashion trends may face challenges ahead.")
else:
    print(f"\nOverall Outlook: NEUTRAL")
    print("Mixed expectations for fashion trends.")

print(f"\nModel Performance:")
print(f"  • Trained on {len(data)} fashion trend reports")
print(f"  • Using {predictor.sequence_length}-step sequences")
print(f"  • Binary classification (Positive/Negative)")
print(f"  • PyTorch LSTM with {predictor.device} acceleration")

print("\n Ready for real-time trend prediction and analysis!")