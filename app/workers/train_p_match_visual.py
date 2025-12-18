# app/workers/train_p_match_visual.py

"""
🎯 TRAINING p_match MODEL WITH VISUALIZATION
Tạo biểu đồ đẹp với chữ to, rõ ràng cho presentation và luận văn.

Cách chạy:
    python -m app.workers.train_p_match_visual

Output:
    - Classification report trên terminal
    - File ảnh: p_match_training_results.png
"""

import asyncio
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Import from existing training script
from app.workers.train_p_match import build_dataset_df
from app.models.ml_models import P_FREELANCER_MODEL_PATH
import joblib

# 🎨 ENHANCED STYLING FOR PRESENTATION
plt.rcParams.update({
    'font.size': 14,           # Base font size - BIGGER
    'axes.titlesize': 18,      # Title size - BIGGER  
    'axes.labelsize': 16,      # Axis label size - BIGGER
    'xtick.labelsize': 14,     # X tick size - BIGGER
    'ytick.labelsize': 14,     # Y tick size - BIGGER
    'legend.fontsize': 14,     # Legend size - BIGGER
    'figure.titlesize': 22,    # Figure title - BIGGER
    'font.weight': 'bold',     # Make text bold
    'axes.titleweight': 'bold',
    'figure.figsize': (20, 16), # MUCH BIGGER figure
    'figure.dpi': 100,         # Good resolution
    'savefig.dpi': 300,        # High quality save
    'savefig.bbox': 'tight',   # No clipping
    'axes.grid': True,         # Add grid
    'grid.alpha': 0.3,         # Light grid
})

# Create output directory
OUTPUT_DIR = Path("visualization_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_p_match_visualization(model, X_test, y_test, y_pred, feature_cols, dataset_info):
    """Create presentation-ready visualization for p_match model"""
    
    # Create figure with more space
    fig = plt.figure(figsize=(24, 18))
    
    # Main title
    fig.suptitle('🎯 p_match MODEL TRAINING RESULTS', 
                 fontsize=28, fontweight='bold', y=0.95)
    
    # Create grid layout with more space
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1],
                         hspace=0.4, wspace=0.3)
    
    # 1. CONFUSION MATRIX (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(y_test, y_pred)
    
    # Create heatmap manually for better control
    im = ax1.imshow(cm, interpolation='nearest', cmap='Greens')
    ax1.set_title('📊 CONFUSION MATRIX', fontsize=20, fontweight='bold', pad=20)
    
    # Add large text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > thresh else "black"
            ax1.text(j, i, f'{cm[i, j]}', ha="center", va="center",
                    color=text_color, fontsize=24, fontweight='bold')
    
    # Labels
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['NO MATCH', 'MATCH SUCCESS'], fontsize=16, fontweight='bold')
    ax1.set_yticklabels(['NO MATCH', 'MATCH SUCCESS'], fontsize=16, fontweight='bold')
    ax1.set_xlabel('PREDICTED', fontsize=18, fontweight='bold')
    ax1.set_ylabel('ACTUAL', fontsize=18, fontweight='bold')
    
    # Add accuracy below
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    ax1.text(0.5, -0.2, f'ACCURACY: {accuracy:.1%}', 
             transform=ax1.transAxes, ha='center', 
             fontsize=18, fontweight='bold', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    # 2. PERFORMANCE METRICS (Top Middle)
    ax2 = fig.add_subplot(gs[0, 1])
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'ACCURACY': accuracy_score(y_test, y_pred),
        'PRECISION': precision_score(y_test, y_pred),
        'RECALL': recall_score(y_test, y_pred),
        'F1-SCORE': f1_score(y_test, y_pred)
    }
    
    colors = ['#2ECC71', '#3498DB', '#E74C3C', '#F39C12']
    bars = ax2.bar(range(len(metrics)), list(metrics.values()), color=colors, alpha=0.8)
    
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels(list(metrics.keys()), rotation=0, fontsize=14, fontweight='bold')
    ax2.set_title('📈 PERFORMANCE METRICS', fontsize=20, fontweight='bold', pad=20)
    ax2.set_ylabel('SCORE', fontsize=18, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    
    # Add large values on bars
    for i, (bar, value) in enumerate(zip(bars, metrics.values())):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{value:.3f}', ha='center', va='bottom', 
                fontsize=16, fontweight='bold')
    
    # 3. DATASET INFO (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    dataset_text = f"""📊 DATASET INFORMATION

🔢 SAMPLES:
   • Total: {dataset_info['total']:,}
   • Training: {dataset_info['train']:,}
   • Testing: {dataset_info['test']:,}

📋 DISTRIBUTION:
   • Match Success: {dataset_info['positive']:,} ({dataset_info['pos_rate']:.1%})
   • No Match: {dataset_info['negative']:,} ({dataset_info['neg_rate']:.1%})

⚙️ MODEL CONFIG:
   • Algorithm: Logistic Regression
   • Features: {len(feature_cols)}
   • Preprocessing: StandardScaler
   • Class Weight: Balanced

🎯 PURPOSE:
   Predict if job-freelancer pair
   will result in successful contract"""
    
    ax3.text(0.05, 0.95, dataset_text, transform=ax3.transAxes, 
             fontsize=14, verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.8", facecolor="lightcyan", alpha=0.8))
    
    # 4. TOP FEATURE IMPORTANCE (Bottom - spans 2 columns)
    ax4 = fig.add_subplot(gs[1:, :])
    
    coef = model.named_steps['logreg'].coef_[0]
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(coef),
        'coefficient': coef
    }).sort_values('importance', ascending=True)  # Ascending for horizontal bar
    
    # Take top 12 for better visibility
    top_features = feature_importance.tail(12)
    
    # Create color based on positive/negative coefficient
    colors = ['#E74C3C' if x < 0 else '#2ECC71' for x in top_features['coefficient']]
    
    bars = ax4.barh(range(len(top_features)), top_features['importance'], 
                    color=colors, alpha=0.8, height=0.7)
    
    # Clean feature names
    clean_names = []
    for name in top_features['feature']:
        clean_name = name.replace('_', ' ').title()
        # Shorten long names
        if len(clean_name) > 25:
            clean_name = clean_name[:22] + '...'
        clean_names.append(clean_name)
    
    ax4.set_yticks(range(len(top_features)))
    ax4.set_yticklabels(clean_names, fontsize=16, fontweight='bold')
    ax4.set_title('🔍 TOP FEATURE IMPORTANCE (Absolute Coefficient Values)', 
                  fontsize=22, fontweight='bold', pad=30)
    ax4.set_xlabel('IMPORTANCE SCORE', fontsize=18, fontweight='bold')
    
    # Add values on bars
    for i, (bar, val, coef) in enumerate(zip(bars, top_features['importance'], top_features['coefficient'])):
        # Add importance value
        ax4.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=14, fontweight='bold')
        
        # Add coefficient sign
        sign = '+' if coef > 0 else '-'
        ax4.text(val + 0.05, i, f'({sign})', va='center', fontsize=12, 
                color='green' if coef > 0 else 'red', fontweight='bold')
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ECC71', alpha=0.8, label='Positive Impact (↑ Match Success)'),
        Patch(facecolor='#E74C3C', alpha=0.8, label='Negative Impact (↓ Match Success)')
    ]
    ax4.legend(handles=legend_elements, loc='lower right', fontsize=14)
    
    # Add grid for better readability
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Save with descriptive filename
    output_file = OUTPUT_DIR / "p_match_training_results.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n📊 Visualization saved to: {output_file}")
    print(f"📁 Full path: {output_file.absolute()}")
    
    return top_features


async def main():
    """Main training function for p_match with enhanced visualization"""
    print("🚀 TRAINING p_match MODEL WITH VISUALIZATION")
    print("=" * 60)
    
    # Build dataset
    print("\n📊 Building dataset...")
    df = await build_dataset_df()
    if df.empty:
        print("❌ Dataset is empty!")
        return
    
    print(f"✅ Dataset loaded: {len(df)} samples")
    
    # Prepare features (same as p_freelancer_accept)
    feature_cols = [
        "similarity_score", "budget_gap", "timezone_gap_hours", "level_gap",
        "job_experience_level_num", "job_required_skill_count", "job_screening_question_count",
        "job_stats_applies", "job_stats_offers", "job_stats_accepts",
        "freelancer_skill_count", "freelancer_stats_applies", "freelancer_stats_offers",
        "freelancer_stats_accepts", "freelancer_invite_accept_rate",
        "skill_overlap_count", "skill_overlap_ratio", "has_past_collaboration",
        "past_collaboration_count", "has_viewed_job",
    ]
    
    X = df[feature_cols].values
    y = df["label"].values
    
    # Dataset info for visualization
    dataset_info = {
        'total': len(df),
        'positive': sum(y == 1),
        'negative': sum(y == 0),
        'pos_rate': sum(y == 1) / len(y),
        'neg_rate': sum(y == 0) / len(y),
    }
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    dataset_info.update({
        'train': len(X_train),
        'test': len(X_test)
    })
    
    # Train model
    print("\n🤖 Training model...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Print classification report (original terminal output)
    print("\n📊 CLASSIFICATION REPORT:")
    print("=" * 50)
    print(classification_report(y_test, y_pred))
    
    # Create enhanced visualization
    print("\n🎨 Creating presentation-ready visualization...")
    top_features = create_p_match_visualization(
        model, X_test, y_test, y_pred, feature_cols, dataset_info
    )
    
    # Print top features
    print("\n🔍 TOP 5 MOST IMPORTANT FEATURES:")
    print("-" * 40)
    for i, (_, row) in enumerate(top_features.tail(5).iterrows(), 1):
        impact = "↑ Increases" if row['coefficient'] > 0 else "↓ Decreases"
        print(f"{i}. {row['feature'].replace('_', ' ').title():<30} {impact} Match Success")
    
    # Save model (use different path for p_match)
    model_path = P_FREELANCER_MODEL_PATH.parent / "p_match_logreg.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("📊 Check the visualization file for presentation-ready charts.")


if __name__ == "__main__":
    asyncio.run(main())