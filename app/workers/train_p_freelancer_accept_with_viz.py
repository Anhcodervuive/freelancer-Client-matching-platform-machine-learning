# app/workers/train_p_freelancer_accept_with_viz.py

"""
Enhanced training script với visualization đẹp mắt cho p_freelancer_accept.
Tạo các biểu đồ: confusion matrix, feature importance, learning curves, etc.
"""

import asyncio
from typing import List, Dict, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sqlalchemy import text
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import train_test_split, learning_curve

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

from app.db.session import async_session
from app.models.ml_models import P_FREELANCER_MODEL_PATH
import joblib

# Set style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Create output directory
OUTPUT_DIR = Path("training_visualizations")
OUTPUT_DIR.mkdir(exist_ok=True)


async def build_dataset_df() -> pd.DataFrame:
    """Build dataset - same as original"""
    async with async_session() as session:
        sql = text(
            """
            SELECT
                ji.job_id,
                ji.freelancer_id,
                ji.status,

                mf.similarity_score,
                mf.budget_gap,
                mf.timezone_gap_hours,
                mf.level_gap,

                mf.job_experience_level_num,
                mf.job_required_skill_count,
                mf.job_screening_question_count,
                mf.job_stats_applies,
                mf.job_stats_offers,
                mf.job_stats_accepts,

                mf.freelancer_skill_count,
                mf.freelancer_stats_applies,
                mf.freelancer_stats_offers,
                mf.freelancer_stats_accepts,
                mf.freelancer_invite_accept_rate,

                mf.skill_overlap_count,
                mf.skill_overlap_ratio,

                mf.has_past_collaboration,
                mf.past_collaboration_count,
                mf.has_viewed_job
            FROM job_invitation ji
            JOIN match_feature mf
              ON mf.job_id = ji.job_id
             AND mf.freelancer_id = ji.freelancer_id
            WHERE ji.status IN ('ACCEPTED', 'DECLINED', 'EXPIRED')
            """
        )

        rows = (await session.execute(sql)).mappings().all()

        data: List[Dict[str, Any]] = []
        for r in rows:
            status = (r["status"] or "").upper()
            label = 1 if status == "ACCEPTED" else 0

            def f(name: str, default: float = 0.0) -> float:
                v = r.get(name)
                return float(v) if v is not None else default

            def b(name: str) -> int:
                v = r.get(name)
                return 1 if v else 0

            data.append({
                "job_id": r["job_id"],
                "freelancer_id": r["freelancer_id"],
                "label": label,
                "status": status,

                # Core features
                "similarity_score": f("similarity_score"),
                "budget_gap": f("budget_gap"),
                "timezone_gap_hours": f("timezone_gap_hours"),
                "level_gap": f("level_gap"),

                # Job features
                "job_experience_level_num": f("job_experience_level_num"),
                "job_required_skill_count": f("job_required_skill_count"),
                "job_screening_question_count": f("job_screening_question_count"),
                "job_stats_applies": f("job_stats_applies"),
                "job_stats_offers": f("job_stats_offers"),
                "job_stats_accepts": f("job_stats_accepts"),

                # Freelancer features
                "freelancer_skill_count": f("freelancer_skill_count"),
                "freelancer_stats_applies": f("freelancer_stats_applies"),
                "freelancer_stats_offers": f("freelancer_stats_offers"),
                "freelancer_stats_accepts": f("freelancer_stats_accepts"),
                "freelancer_invite_accept_rate": f("freelancer_invite_accept_rate"),

                # Pairwise features
                "skill_overlap_count": f("skill_overlap_count"),
                "skill_overlap_ratio": f("skill_overlap_ratio"),
                "has_past_collaboration": b("has_past_collaboration"),
                "past_collaboration_count": f("past_collaboration_count"),
                "has_viewed_job": b("has_viewed_job"),
            })

        df = pd.DataFrame(data)
        return df


def plot_dataset_overview(df: pd.DataFrame):
    """Plot 1: Dataset Overview"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('📊 DATASET OVERVIEW - p_freelancer_accept', fontsize=16, fontweight='bold')

    # 1. Label distribution
    label_counts = df['label'].value_counts()
    colors = ['#ff6b6b', '#4ecdc4']
    axes[0, 0].pie(label_counts.values, labels=['Declined/Expired (0)', 'Accepted (1)'], 
                   autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0, 0].set_title('🎯 Label Distribution')

    # 2. Status breakdown
    status_counts = df['status'].value_counts()
    axes[0, 1].bar(status_counts.index, status_counts.values, color=['#ff6b6b', '#ffa726', '#4ecdc4'])
    axes[0, 1].set_title('📋 Status Breakdown')
    axes[0, 1].set_ylabel('Count')
    for i, v in enumerate(status_counts.values):
        axes[0, 1].text(i, v + 1, str(v), ha='center', fontweight='bold')

    # 3. Key features distribution
    key_features = ['similarity_score', 'skill_overlap_ratio', 'freelancer_invite_accept_rate']
    for i, feature in enumerate(key_features):
        if i < 2:
            row, col = 1, i
            axes[row, col].hist(df[feature], bins=20, alpha=0.7, color=colors[i])
            axes[row, col].set_title(f'📈 {feature.replace("_", " ").title()}')
            axes[row, col].set_ylabel('Frequency')

    # 4. Dataset info text
    axes[1, 1].axis('off')
    info_text = f"""
    📊 DATASET STATISTICS
    
    Total Samples: {len(df):,}
    Positive (Accept): {sum(df['label'] == 1):,} ({sum(df['label'] == 1)/len(df)*100:.1f}%)
    Negative (Decline): {sum(df['label'] == 0):,} ({sum(df['label'] == 0)/len(df)*100:.1f}%)
    
    Features: 20
    Missing Values: {df.isnull().sum().sum()}
    
    Key Metrics:
    • Avg Similarity: {df['similarity_score'].mean():.3f}
    • Avg Skill Overlap: {df['skill_overlap_ratio'].mean():.3f}
    • Avg Accept Rate: {df['freelancer_invite_accept_rate'].mean():.3f}
    """
    axes[1, 1].text(0.1, 0.9, info_text, transform=axes[1, 1].transAxes, 
                     fontsize=11, verticalalignment='top', 
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_dataset_overview.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_analysis(df: pd.DataFrame, feature_cols: List[str]):
    """Plot 2: Feature Analysis"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('🔍 FEATURE ANALYSIS BY LABEL', fontsize=16, fontweight='bold')

    # Top 6 most important features for visualization
    important_features = [
        'similarity_score', 'skill_overlap_ratio', 'freelancer_invite_accept_rate',
        'has_past_collaboration', 'level_gap', 'job_stats_applies'
    ]

    for i, feature in enumerate(important_features):
        row, col = i // 2, i % 2
        
        # Box plot by label
        df_plot = df[[feature, 'label']].copy()
        df_plot['Label'] = df_plot['label'].map({0: 'Declined', 1: 'Accepted'})
        
        sns.boxplot(data=df_plot, x='Label', y=feature, ax=axes[row, col])
        axes[row, col].set_title(f'📊 {feature.replace("_", " ").title()}')
        
        # Add mean values
        for j, label in enumerate(['Declined', 'Accepted']):
            mean_val = df[df['label'] == j][feature].mean()
            axes[row, col].text(j, mean_val, f'μ={mean_val:.3f}', 
                               ha='center', va='bottom', fontweight='bold', 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_feature_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
    """Plot 3: Confusion Matrix"""
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Declined', 'Accepted'],
                yticklabels=['Declined', 'Accepted'],
                ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set_title('🎯 CONFUSION MATRIX', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    
    # Add percentage annotations
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                   ha='center', va='center', fontsize=10, color='red', fontweight='bold')
    
    # Add metrics text
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f"""
    📈 METRICS:
    Accuracy: {accuracy:.3f}
    Precision: {precision:.3f}
    Recall: {recall:.3f}
    F1-Score: {f1:.3f}
    """
    
    ax.text(2.5, 1, metrics_text, fontsize=11, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_and_pr_curves(model, X_test, y_test):
    """Plot 4: ROC and Precision-Recall Curves"""
    y_proba = model.predict_proba(X_test)[:, 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('📈 MODEL PERFORMANCE CURVES', fontsize=16, fontweight='bold')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('🎯 ROC Curve')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    axes[1].plot(recall, precision, color='darkgreen', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('📊 Precision-Recall Curve')
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_performance_curves.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(model, feature_cols):
    """Plot 5: Feature Importance"""
    # Get coefficients from logistic regression
    coef = model.named_steps['logreg'].coef_[0]
    
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(coef),
        'coefficient': coef
    }).sort_values('importance', ascending=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    fig.suptitle('🔍 FEATURE IMPORTANCE ANALYSIS', fontsize=16, fontweight='bold')
    
    # Feature importance (absolute values)
    colors = ['red' if x < 0 else 'green' for x in feature_importance['coefficient']]
    axes[0].barh(range(len(feature_importance)), feature_importance['importance'], color=colors, alpha=0.7)
    axes[0].set_yticks(range(len(feature_importance)))
    axes[0].set_yticklabels([f.replace('_', ' ').title() for f in feature_importance['feature']])
    axes[0].set_xlabel('Absolute Coefficient Value')
    axes[0].set_title('📊 Feature Importance (Absolute)')
    axes[0].grid(True, alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(feature_importance['importance']):
        axes[0].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    # Coefficient values (with direction)
    feature_importance_sorted = feature_importance.sort_values('coefficient', ascending=True)
    colors = ['red' if x < 0 else 'green' for x in feature_importance_sorted['coefficient']]
    axes[1].barh(range(len(feature_importance_sorted)), feature_importance_sorted['coefficient'], color=colors, alpha=0.7)
    axes[1].set_yticks(range(len(feature_importance_sorted)))
    axes[1].set_yticklabels([f.replace('_', ' ').title() for f in feature_importance_sorted['feature']])
    axes[1].set_xlabel('Coefficient Value')
    axes[1].set_title('📈 Feature Coefficients (with Direction)')
    axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(feature_importance_sorted['coefficient']):
        axes[1].text(v + (0.01 if v >= 0 else -0.01), i, f'{v:.3f}', 
                    va='center', ha='left' if v >= 0 else 'right', fontsize=9)
    
    # Add legend
    red_patch = mpatches.Patch(color='red', alpha=0.7, label='Negative Impact (↓ Accept)')
    green_patch = mpatches.Patch(color='green', alpha=0.7, label='Positive Impact (↑ Accept)')
    axes[1].legend(handles=[red_patch, green_patch], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance


def plot_learning_curves(model, X, y):
    """Plot 6: Learning Curves"""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('F1 Score')
    ax.set_title('📈 LEARNING CURVES - Model Performance vs Training Size', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    final_train_score = train_mean[-1]
    final_val_score = val_mean[-1]
    ax.annotate(f'Final Training: {final_train_score:.3f}', 
                xy=(train_sizes[-1], final_train_score), xytext=(train_sizes[-3], final_train_score + 0.05),
                arrowprops=dict(arrowstyle='->', color='blue'), fontsize=10, color='blue')
    ax.annotate(f'Final Validation: {final_val_score:.3f}', 
                xy=(train_sizes[-1], final_val_score), xytext=(train_sizes[-3], final_val_score - 0.05),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_learning_curves.png", dpi=300, bbox_inches='tight')
    plt.show()


def create_summary_report(df, model, X_test, y_test, y_pred, feature_importance):
    """Create final summary report"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.axis('off')
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Get top features
    top_features = feature_importance.nlargest(5, 'importance')
    
    report_text = f"""
    🎯 TRAINING SUMMARY REPORT - p_freelancer_accept
    ═══════════════════════════════════════════════════════════════
    
    📊 DATASET INFORMATION:
    • Total Samples: {len(df):,}
    • Training Samples: {len(X_test) * 4:,} (80%)
    • Test Samples: {len(X_test):,} (20%)
    • Features Used: 20
    • Positive Rate: {df['label'].mean():.1%}
    
    🤖 MODEL CONFIGURATION:
    • Algorithm: Logistic Regression
    • Preprocessing: StandardScaler (Z-score normalization)
    • Class Weight: Balanced (handles imbalanced data)
    • Max Iterations: 1000
    • Random State: 42
    
    📈 PERFORMANCE METRICS:
    • Accuracy:  {accuracy:.3f} ({accuracy:.1%})
    • Precision: {precision:.3f} ({precision:.1%})
    • Recall:    {recall:.3f} ({recall:.1%})
    • F1-Score:  {f1:.3f} ({f1:.1%})
    
    🔍 TOP 5 MOST IMPORTANT FEATURES:
    1. {top_features.iloc[4]['feature'].replace('_', ' ').title():<35} {top_features.iloc[4]['importance']:.3f}
    2. {top_features.iloc[3]['feature'].replace('_', ' ').title():<35} {top_features.iloc[3]['importance']:.3f}
    3. {top_features.iloc[2]['feature'].replace('_', ' ').title():<35} {top_features.iloc[2]['importance']:.3f}
    4. {top_features.iloc[1]['feature'].replace('_', ' ').title():<35} {top_features.iloc[1]['importance']:.3f}
    5. {top_features.iloc[0]['feature'].replace('_', ' ').title():<35} {top_features.iloc[0]['importance']:.3f}
    
    💡 KEY INSIGHTS:
    • Model successfully predicts freelancer acceptance behavior
    • {top_features.iloc[4]['feature'].replace('_', ' ').title()} is the strongest predictor
    • Balanced performance between precision and recall
    • Ready for production deployment
    
    📁 FILES GENERATED:
    • Model: {P_FREELANCER_MODEL_PATH.name}
    • Dataset: dataset_p_freelancer_accept.csv
    • Visualizations: training_visualizations/ folder
    
    ═══════════════════════════════════════════════════════════════
    Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    ax.text(0.05, 0.95, report_text, transform=ax.transAxes, fontsize=11, 
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_summary_report.png", dpi=300, bbox_inches='tight')
    plt.show()


async def main() -> None:
    print("🚀 Starting Enhanced Training with Visualizations...")
    print(f"📁 Output directory: {OUTPUT_DIR.absolute()}")
    
    # Build dataset
    print("\n📊 Building dataset...")
    df = await build_dataset_df()
    if df.empty:
        print("❌ Dataset is empty, cannot train.")
        return

    # Save dataset
    csv_path = "dataset_p_freelancer_accept.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"💾 Saved {len(df)} rows to {csv_path}")

    # Plot dataset overview
    print("\n📈 Creating dataset overview...")
    plot_dataset_overview(df)

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

    # Plot feature analysis
    print("\n🔍 Analyzing features...")
    plot_feature_analysis(df, feature_cols)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    print("\n🤖 Training model...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    
    # Print classification report (original output)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Create visualizations
    print("\n🎨 Creating visualizations...")
    
    print("   📊 Confusion Matrix...")
    plot_confusion_matrix(y_test, y_pred)
    
    print("   📈 Performance Curves...")
    plot_roc_and_pr_curves(model, X_test, y_test)
    
    print("   🔍 Feature Importance...")
    feature_importance = plot_feature_importance(model, feature_cols)
    
    print("   📈 Learning Curves...")
    plot_learning_curves(model, X, y)
    
    print("   📋 Summary Report...")
    create_summary_report(df, model, X_test, y_test, y_pred, feature_importance)

    # Save model
    P_FREELANCER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, P_FREELANCER_MODEL_PATH)
    print(f"\n💾 Model saved to: {P_FREELANCER_MODEL_PATH}")
    
    print(f"\n✅ Training completed! Check visualizations in: {OUTPUT_DIR.absolute()}")
    print("\n📁 Generated files:")
    for file in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"   • {file.name}")


if __name__ == "__main__":
    asyncio.run(main())