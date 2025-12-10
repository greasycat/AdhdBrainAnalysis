import torch
import torch.nn as nn
from cnn.model import CNN
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
from tqdm import tqdm

def train_model(model, train_dataset, val_dataset, epochs=50, batch_size=4, lr=0.0001):
    # Data loaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    # Training tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    f1_scores, precision_scores, recall_scores, roc_auc_scores = [], [], [], []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, total=len(train_loader))
        for batch_idx, (data, labels, subject_ids) in enumerate(pbar):
            data, labels = data.to(device), labels.to(device)

            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            pbar.set_description(f"Training: Subject ID: {subject_ids}")
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_preds, val_true = [], []
        
        with torch.no_grad():
            tbar = tqdm(val_loader, total=len(val_loader))
            for data, labels, subject_ids in tbar:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                tbar.set_description(f"Validation: Subject ID: {subject_ids}")
                
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        f1_scores.append(f1_score(val_true, val_preds))
        precision_scores.append(precision_score(val_true, val_preds))
        recall_scores.append(recall_score(val_true, val_preds))
        roc_auc_scores.append(roc_auc_score(val_true, val_preds))
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.4f}')
        print('-' * 50)
    
    # Final evaluation
    print("\nClassification Report:")
    print(classification_report(val_true, val_preds, target_names=['False', 'True'])) # type: ignore
    print("\nConfusion Matrix:")
    print(confusion_matrix(val_true, val_preds)) # type: ignore
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses, 
        'train_accs': train_accs,
        'val_accs': val_accs,
        'f1_scores': f1_scores,
        'precision_scores': precision_scores,
        'recall_scores': recall_scores,
        'roc_auc_scores': roc_auc_scores,
    }