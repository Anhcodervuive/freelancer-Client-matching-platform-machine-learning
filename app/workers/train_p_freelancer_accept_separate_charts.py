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
    """Chart 1: Confusion Matrix"""
    
    # Tăng figsize để có nhiều không gian hơn
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    # Title - đặt ở trên cùng với pad lớn
    ax.set_title('CONFUSION MATRIX\np_freelancer_accept Model', 
                 fontsize=24, fontweight='bold', pad=20)
    
    # Add numbers in cells
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f'{cm[i, j]}', ha="center", va="center",
                   color=text_color, fontsize=36, fontweight='bold')
    
    # Labels - đơn giản hóa để tránh đè
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['DECLINED', 'ACCEPTED'], fontsize=18, fontweight='bold')
    ax.set_yticklabels(['DECLINED', 'ACCEPTED'], fontsize=18, fontweight='bold')
    ax.set_xlabel('DU DOAN (PREDICTED)', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_ylabel('THUC TE (ACTUAL)', fontsize=18, fontweight='bold', labelpad=15)
    
    # Accuracy - đặt bên dưới với khoảng cách lớn hơn
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    
    # Thêm text box riêng biệt bên dưới figure
    fig.text(0.5, 0.02, f'ACCURACY: {accuracy:.1%}', 
             ha='center', fontsize=22, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.8", facecolor="yellow", alpha=0.9))
    
    # Điều chỉnh layout để có không gian cho text bên dưới
    plt.subplots_adjust(bottom=0.15)
    
    output_file = OUTPUT_DIR / "01_confusion_matrix.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Confusion Matrix saved: {output_file}")


def create_performance_metrics_chart(y_test, y_pred):
    """Chart 2: Performance Metrics"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Tên ngắn gọn hơn để tránh đè
    metrics = {
        'ACCURACY': accuracy_score(y_test, y_pred),
        'PRECISION': precision_score(y_test, y_pred),
        'RECALL': recall_score(y_test, y_pred),
        'F1-SCORE': f1_score(y_test, y_pred)
    }
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    x_pos = range(len(metrics))
    bars = ax.bar(x_pos, list(metrics.values()), color=colors, alpha=0.85, width=0.6)
    
    # Title
    ax.set_title('PERFORMANCE METRICS\np_freelancer_accept Model', 
                 fontsize=24, fontweight='bold', pad=20)
    
    # Labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(metrics.keys()), fontsize=16, fontweight='bold')
    ax.set_ylabel('DIEM SO (SCORE)', fontsize=18, fontweight='bold')
    ax.set_ylim(0, 1.25)  # Tăng ylim để có không gian cho text
    
    # Chỉ hiển thị 1 giá trị trên mỗi bar - kết hợp số và %
    for bar, value in zip(bars, metrics.values()):
        # Hiển thị cả số thập phân và % trên cùng 1 dòng
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, 
                f'{value:.3f} ({value:.1%})', 
                ha='center', va='bottom', 
                fontsize=16, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    
    # Explanation box - đặt ở góc trên bên phải
    explanation = (
        "ACCURACY: Ty le du doan dung tong the\n"
        "PRECISION: Trong so du doan ACCEPT, bao nhieu % dung\n"
        "RECALL: Trong so thuc te ACCEPT, model phat hien duoc bao nhieu %\n"
        "F1-SCORE: Trung binh dieu hoa cua Precision va Recall"
    )
    ax.text(0.98, 0.98, explanation, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "02_performance_metrics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Performance Metrics saved: {output_file}")


def create_feature_importance_chart(model, feature_cols):
    """Chart 3: Feature Importance"""
    
    # Tăng figsize width để có không gian cho text bên phải
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    coef = model.named_steps['logreg'].coef_[0]
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(coef),
        'coefficient': coef
    }).sort_values('importance', ascending=True)
    
    top_features = feature_importance.tail(10)
    
    colors = ['#FF6B6B' if x < 0 else '#4ECDC4' for x in top_features['coefficient']]
    
    bars = ax.barh(range(len(top_features)), top_features['importance'], 
                   color=colors, alpha=0.85, height=0.7)
    
    # Vietnamese translations
    translations = {
        'similarity_score': 'Diem Tuong Dong',
        'skill_overlap_ratio': 'Ty Le Skill Trung',
        'freelancer_invite_accept_rate': 'Ty Le Accept Loi Moi',
        'has_past_collaboration': 'Da Tung Hop Tac',
        'job_stats_offers': 'So Offer Job Da Gui',
        'job_experience_level_num': 'Level Kinh Nghiem Job',
        'skill_overlap_count': 'So Skill Trung Khop',
        'level_gap': 'Chenh Lech Level',
        'freelancer_skill_count': 'So Skill Freelancer',
        'job_required_skill_count': 'So Skill Job Yeu Cau',
        'budget_gap': 'Chenh Lech Ngan Sach',
        'timezone_gap_hours': 'Chenh Lech Mui Gio',
        'job_stats_accepts': 'Job Stats Accepts',
        'freelancer_stats_accepts': 'FL Stats Accepts',
    }
    
    clean_names = [translations.get(name, name.replace('_', ' ').title()) 
                   for name in top_features['feature']]
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(clean_names, fontsize=14, fontweight='bold')
    
    ax.set_title('TOP 10 FEATURE IMPORTANCE\np_freelancer_accept Model', 
                 fontsize=24, fontweight='bold', pad=20)
    ax.set_xlabel('MUC DO QUAN TRONG (IMPORTANCE SCORE)', fontsize=16, fontweight='bold')
    
    # Tính max importance để đặt text position hợp lý
    max_importance = top_features['importance'].max()
    
    # Chỉ hiển thị giá trị số, không hiển thị "Tăng/Giảm" để tránh đè
    for i, (bar, val) in enumerate(zip(bars, top_features['importance'])):
        ax.text(val + max_importance * 0.02, i, f'{val:.3f}', 
                va='center', fontsize=13, fontweight='bold')
    
    # Legend thay vì text "Tăng/Giảm" trên mỗi bar
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4ECDC4', alpha=0.85, label='Tang kha nang Accept'),
        Patch(facecolor='#FF6B6B', alpha=0.85, label='Giam kha nang Accept')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=14)
    
    ax.grid(True, alpha=0.3, axis='x')
    
    # Mở rộng xlim để có không gian cho text
    ax.set_xlim(0, max_importance * 1.2)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "03_feature_importance.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Feature Importance saved: {output_file}")
    
    return top_features


def create_dataset_overview_chart(dataset_info, feature_cols):
    """Chart 4: Dataset Overview - Chia thành 4 phần rõ ràng"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    
    fig.suptitle('DATASET OVERVIEW - p_freelancer_accept Model', 
                 fontsize=22, fontweight='bold', y=1.02)
    
    # ===== 1. Label Distribution (Pie Chart) =====
    sizes = [dataset_info['negative'], dataset_info['positive']]
    colors = ['#FF6B6B', '#4ECDC4']
    explode = (0.02, 0.02)
    
    wedges, texts, autotexts = ax1.pie(
        sizes, 
        explode=explode,
        autopct='%1.1f%%', 
        colors=colors, 
        startangle=90,
        textprops={'fontsize': 14}
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(16)
        autotext.set_fontweight('bold')
    
    ax1.legend(
        [f'DECLINED: {dataset_info["negative"]:,}', 
         f'ACCEPTED: {dataset_info["positive"]:,}'],
        loc='upper left',
        fontsize=12
    )
    ax1.set_title('Phan Bo Nhan', fontsize=16, fontweight='bold')
    
    # ===== 2. Sample Counts (Bar Chart) =====
    categories = ['Train', 'Test', 'Total']
    counts = [dataset_info['train'], dataset_info['test'], dataset_info['total']]
    colors_bar = ['#45B7D1', '#96CEB4', '#FFA726']
    
    bars = ax2.bar(categories, counts, color=colors_bar, alpha=0.85, width=0.5)
    ax2.set_title('So Luong Samples', fontsize=16, fontweight='bold')
    ax2.set_ylabel('So Luong', fontsize=12)
    ax2.set_ylim(0, max(counts) * 1.25)
    
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.03, 
                f'{count:,}', ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
    
    # ===== 3. Model Info - Dùng table thay vì text =====
    ax3.axis('off')
    ax3.set_title('Thong Tin Mo Hinh', fontsize=16, fontweight='bold')
    
    # Tạo table data
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
        colLabels=['Thong so', 'Gia tri'],
        loc='center',
        cellLoc='left',
        colWidths=[0.4, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(fontweight='bold', color='white')
    
    # ===== 4. Feature Categories (Bar Chart) =====
    feature_categories = {'Core': 4, 'Job': 6, 'Freelancer': 5, 'Pairwise': 5}
    
    colors_feat = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars_feat = ax4.bar(feature_categories.keys(), feature_categories.values(), 
                        color=colors_feat, alpha=0.85, width=0.5)
    ax4.set_title('Phan Loai Features', fontsize=16, fontweight='bold')
    ax4.set_ylabel('So Luong', fontsize=12)
    ax4.set_ylim(0, 9)
    
    for bar, count in zip(bars_feat, feature_categories.values()):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{count}', ha='center', va='bottom', 
                fontsize=14, fontweight='bold')
    
    # Total annotation
    ax4.text(0.5, 0.92, f'Tong: {len(feature_cols)} features', 
             transform=ax4.transAxes, ha='center',
             fontsize=12, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))
    
    plt.tight_layout()
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