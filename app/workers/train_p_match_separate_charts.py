# app/workers/train_p_match_separate_charts.py

"""
🎯 TRAINING p_match MODEL - SEPARATE CHARTS
Tạo từng biểu đồ riêng biệt để tránh bị đè chữ, dễ nhìn hơn.

Cách chạy:
    python -m app.workers.train_p_match_separate_charts

Output:
    - Classification report trên terminal
    - 4 files ảnh riêng biệt trong folder p_match_separate_charts/
"""

import asyncio
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from app.workers.train_p_match import load_dataset, preprocess
import joblib

# Create output directory
OUTPUT_DIR = Path("p_match_separate_charts")
OUTPUT_DIR.mkdir(exist_ok=True)


def create_confusion_matrix_chart(y_test, y_pred):
    """Chart 1: Confusion Matrix - Font lớn cho in A4"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    cm = confusion_matrix(y_test, y_pred)
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    # Title - font 32
    ax.set_title('CONFUSION MATRIX\np_match Model', 
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
    ax.set_xticklabels(['NO MATCH', 'MATCH'], fontsize=24, fontweight='bold')
    ax.set_yticklabels(['NO MATCH', 'MATCH'], fontsize=24, fontweight='bold')
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


def create_performance_metrics_chart(y_test, y_pred, y_proba):
    """Chart 2: Performance Metrics - Font lớn cho in A4"""
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'ACCURACY': accuracy_score(y_test, y_pred),
        'PRECISION': precision_score(y_test, y_pred),
        'RECALL': recall_score(y_test, y_pred),
        'F1-SCORE': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba)
    }
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA726']
    x_pos = range(len(metrics))
    bars = ax.bar(x_pos, list(metrics.values()), color=colors, alpha=0.85, width=0.5)
    
    # Title - font 32
    ax.set_title('PERFORMANCE METRICS\np_match Model', 
                 fontsize=32, fontweight='bold', pad=25)
    
    # Labels - font 20
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(metrics.keys()), fontsize=20, fontweight='bold')
    ax.set_ylabel('DIEM SO (SCORE)', fontsize=24, fontweight='bold')
    ax.set_ylim(0, 1.3)
    ax.tick_params(axis='y', labelsize=18)
    
    # Giá trị trên bar - font 20
    for bar, value in zip(bars, metrics.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, 
                f'{value:.3f} ({value:.1%})', 
                ha='center', va='bottom', 
                fontsize=20, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    
    # Explanation - font 14
    explanation = (
        "ACCURACY: Ty le du doan dung tong the\n"
        "PRECISION: Trong so du doan MATCH, bao nhieu % dung\n"
        "RECALL: Trong so thuc te MATCH, model phat hien duoc bao nhieu %\n"
        "F1-SCORE: Trung binh dieu hoa cua Precision va Recall\n"
        "AUC: Kha nang phan biet giua MATCH va NO MATCH"
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
        'freelancer_stats_accepts': 'FL Stats Accepts',
        'job_stats_accepts': 'Job Stats Accepts',
        'past_collaboration_count': 'So Lan Hop Tac',
        'has_viewed_job': 'Da Xem Job'
    }
    
    clean_names = []
    for name in top_features['feature']:
        if name.startswith('region_'):
            region_name = name.replace('region_', '').replace('_', ' ').title()
            clean_name = f'Region: {region_name}'
        else:
            clean_name = translations.get(name, name.replace('_', ' ').title())
        clean_names.append(clean_name)
    
    # Y labels - font 28
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(clean_names, fontsize=28, fontweight='bold')
    
    # Title - font 36
    ax.set_title('TOP 8 FEATURE IMPORTANCE\np_match Model', 
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
        Patch(facecolor='#4ECDC4', alpha=0.85, label='Tang kha nang Match'),
        Patch(facecolor='#FF6B6B', alpha=0.85, label='Giam kha nang Match')
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
    fig.suptitle('DATASET OVERVIEW\np_match Model', 
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
        [f'NO MATCH: {dataset_info["negative"]:,}', 
         f'MATCH: {dataset_info["positive"]:,}'],
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
        ['Ty le Match', f'{dataset_info["pos_rate"]:.1%}'],
        ['Ty le No Match', f'{dataset_info["neg_rate"]:.1%}'],
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
    core_count = 4
    job_count = 6
    fl_count = 6
    pair_count = 5
    region_count = len([c for c in feature_cols if c.startswith('region_')])
    
    feature_categories = {'Core': core_count, 'Job': job_count, 
                          'FL': fl_count, 'Pair': pair_count}
    if region_count > 0:
        feature_categories['Region'] = region_count
    
    colors_feat = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFA726'][:len(feature_categories)]
    bars_feat = ax4.bar(feature_categories.keys(), feature_categories.values(), 
                        color=colors_feat, alpha=0.85, width=0.45)
    ax4.set_title('PHAN LOAI FEATURES', fontsize=28, fontweight='bold', pad=15)
    ax4.set_ylabel('So Luong', fontsize=22, fontweight='bold')
    ax4.set_ylim(0, max(feature_categories.values()) + 5)
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
    print("=" * 55)
    print("TRAINING p_match MODEL - SEPARATE CHARTS")
    print("=" * 55)
    
    print("\n[1/5] Loading dataset...")
    df = await load_dataset()
    if df.empty:
        print("ERROR: Dataset is empty!")
        return
    
    print(f"      Dataset loaded: {len(df)} samples")
    
    print("\n[2/5] Preprocessing...")
    X, y, feature_cols = preprocess(df)
    
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
    
    print("\n[3/5] Training model...")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            max_iter=1000, 
            class_weight="balanced", 
            solver="lbfgs",
            random_state=42
        )),
    ])
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n[4/5] Classification Report:")
    print("-" * 50)
    print(classification_report(y_test, y_pred))
    print(f"AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    print(f"\n[5/5] Creating charts in: {OUTPUT_DIR}/")
    print("-" * 50)
    
    create_confusion_matrix_chart(y_test, y_pred)
    create_performance_metrics_chart(y_test, y_pred, y_proba)
    top_features = create_feature_importance_chart(model, feature_cols)
    create_dataset_overview_chart(dataset_info, feature_cols)
    
    print("\nSaving model...")
    os.makedirs("app/models", exist_ok=True)
    joblib.dump(model, "app/models/p_match_logreg.joblib")
    
    with open("app/models/p_match_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(list(feature_cols), f, ensure_ascii=False, indent=2)
    
    print(f"      Model saved to: app/models/p_match_logreg.joblib")
    
    print("\n" + "=" * 55)
    print("TRAINING COMPLETED!")
    print("=" * 55)
    print(f"\nOutput files in: {OUTPUT_DIR.absolute()}")
    print("  1. 01_confusion_matrix.png")
    print("  2. 02_performance_metrics.png")
    print("  3. 03_feature_importance.png")
    print("  4. 04_dataset_overview.png")


if __name__ == "__main__":
    asyncio.run(main())