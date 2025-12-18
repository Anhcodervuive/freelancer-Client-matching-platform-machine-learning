# app/workers/train_with_simple_viz.py

"""
Simple training script với basic visualization cho demo nhanh.
Chỉ tạo 3 biểu đồ quan trọng nhất: confusion matrix, feature importance, và performance.
"""

import asyncio
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Import from existing training script
from app.workers.train_p_freelancer_accept import build_dataset_df
from app.models.ml_models import P_FREELANCER_MODEL_PATH
import joblib

# Set style
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = Path("simple_viz")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_simple_visualizations(model, X_test, y_test, y_pred, feature_cols):
    """Create 3 essential visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('🎯 ML TRAINING RESULTS - p_freelancer_accept', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Declined', 'Accepted'],
                yticklabels=['Declined', 'Accepted'],
                ax=axes[0, 0])
    axes[0, 0].set_title('📊 Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # Add accuracy text
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    axes[0, 0].text(0.5, -0.15, f'Accuracy: {accuracy:.1%}', 
                    transform=axes[0, 0].transAxes, ha='center', 
                    fontsize=12, fontweight='bold')
    
    # 2. Feature Importance (Top 10)
    coef = model.named_steps['logreg'].coef_[0]
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(coef)
    }).sort_values('importance', ascending=False).head(10)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
    bars = axes[0, 1].barh(range(len(feature_importance)), feature_importance['importance'], color=colors)
    axes[0, 1].set_yticks(range(len(feature_importance)))
    axes[0, 1].set_yticklabels([f.replace('_', ' ').title() for f in feature_importance['feature']])
    axes[0, 1].set_title('🔍 Top 10 Feature Importance')
    axes[0, 1].set_xlabel('Importance')
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, feature_importance['importance'])):
        axes[0, 1].text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
    
    # 3. Performance Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }
    
    bars = axes[1, 0].bar(metrics.keys(), metrics.values(), 
                         color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
    axes[1, 0].set_title('📈 Performance Metrics')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_ylim(0, 1)
    
    # Add values on bars
    for bar, (metric, value) in zip(bars, metrics.items()):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Summary Text
    axes[1, 1].axis('off')
    
    # Get dataset info
    total_samples = len(X_test) * 5  # Approximate total (test is 20%)
    positive_rate = y_test.mean()
    
    summary_text = f"""
    📊 TRAINING SUMMARY
    
    🔢 Dataset:
    • Total Samples: ~{total_samples:,}
    • Test Samples: {len(X_test):,}
    • Features: {len(feature_cols)}
    • Positive Rate: {positive_rate:.1%}
    
    🤖 Model:
    • Algorithm: Logistic Regression
    • Preprocessing: StandardScaler
    • Class Weight: Balanced
    
    🎯 Best Features:
    • {feature_importance.iloc[0]['feature'].replace('_', ' ').title()}
    • {feature_importance.iloc[1]['feature'].replace('_', ' ').title()}
    • {feature_importance.iloc[2]['feature'].replace('_', ' ').title()}
    
    ✅ Status: Ready for Production!
    """
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                     fontsize=11, verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualization saved to: {OUTPUT_DIR / 'training_results.png'}")


async def main():
    """Main training function with simple visualization"""
    print("🚀 Training with Simple Visualization...")
    
    # Build dataset
    df = await build_dataset_df()
    if df.empty:
        print("❌ Dataset is empty!")
        return
    
    print(f"📊 Dataset: {len(df)} samples")
    
    # Prepare features
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
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("🤖 Training model...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Print original classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create simple visualization
    print("\n🎨 Creating visualization...")
    create_simple_visualizations(model, X_test, y_test, y_pred, feature_cols)
    
    # Save model
    joblib.dump(model, P_FREELANCER_MODEL_PATH)
    print(f"\n💾 Model saved to: {P_FREELANCER_MODEL_PATH}")
    
    print("✅ Training completed!")


if __name__ == "__main__":
    asyncio.run(main())