# app/workers/train_p_freelancer_accept_separate_charts.py

"""
🎯 TRAINING p_freelancer_accept MODEL - SEPARATE CHARTS
Tạo từng biểu đồ riêng biệt để tránh bị đè chữ, dễ nhìn hơn.

Cách chạy:
    python -m app.workers.train_p_freelancer_accept_separate_charts

Output:
    - Classification report trên terminal
    - 4 files ảnh riêng biệt trong folder separate_charts/
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

from app.workers.train_p_freelancer_accept import build_dataset_df
from app.models.ml_models import P_FREELANCER_MODEL_PATH
import joblib

# Create output directory
OUTPUT_DIR = Path("separate_charts")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_confusion_matrix_chart(y_test, y_pred):
    """Chart 1: Confusion Matrix - Font lớn cho in A4"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    cm = confusion_matrix(y_test, y_pred)
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    # Title - font 32
    ax.set_title('CONFUSION MATRIX\np_freelancer_accept Model', 
                 fontsize=32, fontweight='bold', pad=25)
    
    # Số trong ô - font 48
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f'{cm[i, j]}', ha="center", va="center",
                   color=text_color, fontsize=48, fontweight='bold')
    
    # Labels - font 24
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['DECLINED', 'ACCEPTED'], fontsize=24, fontweight='bold')
    ax.set_yticklabels(['DECLINED', 'ACCEPTED'], fontsize=24, fontweight='bold')
    ax.set_xlabel('DU DOAN (PREDICTED)', fontsize=26, fontweight='bold', labelpad=20)
    ax.set_ylabel('THUC TE (ACTUAL)', fontsize=26, fontweight='bold', labelpad=20)
    
    # Accuracy - font 28
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    fig.text(0.5, 0.02, f'ACCURACY: {accuracy:.1%}', 
             ha='center', fontsize=28, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.8", facecolor="yellow", alpha=0.9))
    
    plt.subplots_adjust(bottom=0.12)
    
    output_file = OUTPUT_DIR / "01_confusion_matrix.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Confusion Matrix saved: {output_file}")


def create_performance_metrics_chart(y_test, y_pred):
    """Chart 2: Performance Metrics - Font lớn cho in A4"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'ACCURACY': accuracy_score(y_test, y_pred),
        'PRECISION': precision_score(y_test, y_pred),
        'RECALL': recall_score(y_test, y_pred),
        'F1-SCORE': f1_score(y_test, y_pred)
    }
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    x_pos = range(len(metrics))
    bars = ax.bar(x_pos, list(metrics.values()), color=colors, alpha=0.85, width=0.55)
    
    # Title - font 32
    ax.set_title('PERFORMANCE METRICS\np_freelancer_accept Model', 
                 fontsize=32, fontweight='bold', pad=25)
    
    # Labels - font 22
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(metrics.keys()), fontsize=22, fontweight='bold')
    ax.set_ylabel('DIEM SO (SCORE)', fontsize=24, fontweight='bold')
    ax.set_ylim(0, 1.3)
    ax.tick_params(axis='y', labelsize=18)
    
    # Giá trị trên bar - font 22
    for bar, value in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, 
                f'{value:.3f} ({value:.1%})', 
                ha='center', va='bottom', 
                fontsize=22, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    
    # Explanation box - font 14
    explanation = (
        "ACCURACY: Ty le du doan dung tong the\n"
        "PRECISION: Trong so du doan ACCEPT, bao nhieu % dung\n"
        "RECALL: Trong so thuc te ACCEPT, model phat hien duoc bao nhieu %\n"
        "F1-SCORE: Trung binh dieu hoa cua Precision va Recall"
    )
    ax.text(0.98, 0.98, explanation, transform=ax.transAxes, 
            fontsize=14, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "02_performance_metrics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Performance Metrics saved: {output_file}")


def create_feature_importance_chart(model, feature_cols):
    """Chart 3: Feature Importance - Font CỰC LỚN cho in A4"""
    
    # Figsize rất cao để mỗi bar có nhiều không gian
    fig, ax = plt.subplots(1, 1, figsize=(20, 18))
    
    coef = model.named_steps['logreg'].coef_[0]
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(coef),
        'coefficient': coef
    }).sort_values('importance', ascending=True)
    
    # Chỉ lấy top 8 để có nhiều không gian hơn
    top_features = feature_importance.tail(8)
    
    colors = ['#FF6B6B' if x < 0 else '#4ECDC4' for x in top_features['coefficient']]
    
    # Height 0.6 để có khoảng cách giữa các bar
    bars = ax.barh(range(len(top_features)), top_features['importance'], 
                   color=colors, alpha=0.85, height=0.6)
    
    translations = {
        'similarity_score': 'Similarity Score',
        'skill_overlap_ratio': 'Skill Match Ratio',
        'freelancer_invite_accept_rate': 'Invite Accept Rate',
        'has_past_collaboration': 'Past Collaboration',
        'job_stats_offers': 'Job Offers Sent',
        'job_experience_level_num': 'Job Experience Level',
        'skill_overlap_count': 'Matching Skills Count',
        'level_gap': 'Experience Level Gap',
        'freelancer_skill_count': 'Freelancer Skills',
        'job_required_skill_count': 'Required Skills',
        'budget_gap': 'Budget Gap',
        'timezone_gap_hours': 'Timezone Gap',
        'job_stats_accepts': 'Job Accepts',
        'freelancer_stats_accepts': 'FL Accepts',
        'freelancer_stats_offers': 'Freelancer Offers',
    }
    
    clean_names = [translations.get(name, name.replace('_', ' ').title()) 
                   for name in top_features['feature']]
    
    # Y labels - font 28
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(clean_names, fontsize=28, fontweight='bold')
    
    # Title - font 36
    ax.set_title('TOP 8 FEATURE IMPORTANCE\np_freelancer_accept Model', 
                 fontsize=36, fontweight='bold', pad=30)
    
    # X label - font 28
    ax.set_xlabel('MUC DO QUAN TRONG (IMPORTANCE SCORE)', fontsize=28, fontweight='bold', labelpad=15)
    ax.tick_params(axis='x', labelsize=22)
    
    max_importance = top_features['importance'].max()
    
    # Giá trị trên bar - font 26
    for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
        ax.text(val + max_importance * 0.02, i, f'{val:.3f}', 
                va='center', fontsize=26, fontweight='bold')
    
    # Legend - font 24
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4ECDC4', alpha=0.85, label='Increases Accept Probability'),
        Patch(facecolor='#FF6B6B', alpha=0.85, label='Decreases Accept Probability')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=24)
    
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max_importance * 1.3)
    
    # Thêm padding
    plt.subplots_adjust(left=0.35, right=0.95, top=0.92, bottom=0.1)
    
    output_file = OUTPUT_DIR / "03_feature_importance.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Feature Importance saved: {output_file}")
    
    return top_features


def create_dataset_overview_chart(dataset_info, feature_cols):
    """Chart 4: Dataset Overview - Font CỰC LỚN cho in A4"""
    
    # Figsize lớn hơn và dùng GridSpec để kiểm soát khoảng cách
    fig = plt.figure(figsize=(22, 18))
    
    # GridSpec với khoảng cách lớn giữa các subplot
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Title - font 34
    fig.suptitle('DATASET OVERVIEW\np_freelancer_accept Model', 
                 fontsize=34, fontweight='bold', y=0.98)
    
    # ===== 1. Label Distribution (Pie Chart) =====
    sizes = [dataset_info['negative'], dataset_info['positive']]
    colors = ['#FF6B6B', '#4ECDC4']
    explode = (0.03, 0.03)
    
    wedges, texts, autotexts = ax1.pie(
        sizes, 
        explode=explode,
        autopct='%1.1f%%', 
        colors=colors, 
        startangle=90,
        textprops={'fontsize': 24}
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(28)
        autotext.set_fontweight('bold')
    
    ax1.legend(
        [f'DECLINED: {dataset_info["negative"]:,}', 
         f'ACCEPTED: {dataset_info["positive"]:,}'],
        loc='upper left',
        fontsize=20
    )
    ax1.set_title('PHAN BO NHAN', fontsize=28, fontweight='bold', pad=15)
    
    # ===== 2. Sample Counts (Bar Chart) =====
    categories = ['Train', 'Test', 'Total']
    counts = [dataset_info['train'], dataset_info['test'], dataset_info['total']]
    colors_bar = ['#45B7D1', '#96CEB4', '#FFA726']
    
    bars = ax2.bar(categories, counts, color=colors_bar, alpha=0.85, width=0.5)
    ax2.set_title('SO LUONG SAMPLES', fontsize=28, fontweight='bold', pad=15)
    ax2.set_ylabel('So Luong', fontsize=22, fontweight='bold')
    ax2.set_ylim(0, max(counts) * 1.3)
    ax2.tick_params(axis='both', labelsize=20)
    
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.03, 
                f'{count:,}', ha='center', va='bottom', 
                fontsize=26, fontweight='bold')
    
    # ===== 3. Model Info - Table =====
    ax3.axis('off')
    ax3.set_title('THONG TIN MO HINH', fontsize=28, fontweight='bold', pad=15)
    
    table_data = [
        ['Algorithm', 'Logistic Regression'],
        ['Features', f'{len(feature_cols)} dac trung'],
        ['Preprocessing', 'StandardScaler'],
        ['Class Weight', 'Balanced'],
        ['', ''],
        ['Tong samples', f'{dataset_info["total"]:,}'],
        ['Ty le Accept', f'{dataset_info["pos_rate"]:.1%}'],
        ['Ty le Decline', f'{dataset_info["neg_rate"]:.1%}'],
    ]
    
    table = ax3.table(
        cellText=table_data,
        colLabels=['THONG SO', 'GIA TRI'],
        loc='center',
        cellLoc='left',
        colWidths=[0.5, 0.5]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(20)
    table.scale(1.4, 2.8)
    
    for i in range(2):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(fontweight='bold', color='white', fontsize=22)
    
    # ===== 4. Feature Categories (Bar Chart) =====
    feature_categories = {'Core': 4, 'Job': 6, 'FL': 5, 'Pair': 5}
    
    colors_feat = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars_feat = ax4.bar(feature_categories.keys(), feature_categories.values(), 
                        color=colors_feat, alpha=0.85, width=0.5)
    ax4.set_title('PHAN LOAI FEATURES', fontsize=28, fontweight='bold', pad=15)
    ax4.set_ylabel('So Luong', fontsize=22, fontweight='bold')
    ax4.set_ylim(0, 10)
    ax4.tick_params(axis='both', labelsize=20)
    
    for bar, count in zip(bars_feat, feature_categories.values()):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{count}', ha='center', va='bottom', 
                fontsize=26, fontweight='bold')
    
    ax4.text(0.5, 0.88, f'Tong: {len(feature_cols)} features', 
             transform=ax4.transAxes, ha='center',
             fontsize=22, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))
    
    output_file = OUTPUT_DIR / "04_dataset_overview.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Dataset Overview saved: {output_file}")


async def main():
    """Main training function with separate charts"""
    print("=" * 60)
    print("TRAINING p_freelancer_accept MODEL - SEPARATE CHARTS")
    print("=" * 60)
    
    print("\n[1/5] Building dataset...")
    df = await build_dataset_df()
    if df.empty:
        print("ERROR: Dataset is empty!")
        return
    
    print(f"      Dataset loaded: {len(df)} samples")
    
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
    
    dataset_info = {
        'total': len(df),
        'positive': int(sum(y == 1)),
        'negative': int(sum(y == 0)),
        'pos_rate': sum(y == 1) / len(y),
        'neg_rate': sum(y == 0) / len(y),
    }
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    dataset_info.update({
        'train': len(X_train),
        'test': len(X_test)
    })
    
    print("\n[2/5] Training model...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\n[3/5] Classification Report:")
    print("-" * 50)
    print(classification_report(y_test, y_pred))
    
    print(f"\n[4/5] Creating charts in: {OUTPUT_DIR}/")
    print("-" * 50)
    
    create_confusion_matrix_chart(y_test, y_pred)
    create_performance_metrics_chart(y_test, y_pred)
    top_features = create_feature_importance_chart(model, feature_cols)
    create_dataset_overview_chart(dataset_info, feature_cols)
    
    print(f"\n[5/5] Saving model...")
    P_FREELANCER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, P_FREELANCER_MODEL_PATH)
    print(f"      Model saved to: {P_FREELANCER_MODEL_PATH}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"\nOutput files in: {OUTPUT_DIR.absolute()}")
    print("  1. 01_confusion_matrix.png")
    print("  2. 02_performance_metrics.png")
    print("  3. 03_feature_importance.png")
    print("  4. 04_dataset_overview.png")


if __name__ == "__main__":
    asyncio.run(main())