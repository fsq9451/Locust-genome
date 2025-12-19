import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('default')
sns.set_palette("husl")

# Set larger fonts for publication
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

def load_and_prepare_data():
    """Load and prepare gene feature data"""
    df = pd.read_csv('insect_orders_genome_stats_with_size.csv', index_col=0)
    return df

def create_gene_feature_main_figure(df):
    """Create main gene feature comparison figure"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Average Gene Number Comparison
    ax1 = axes[0, 0]
    df_sorted_genes = df.sort_values('avg_genes_per_genome', ascending=False)
    colors = ['#E74C3C' if idx == 'Orthoptera' else '#3498DB' for idx in df_sorted_genes.index]
    
    bars = ax1.bar(range(len(df_sorted_genes)), df_sorted_genes['avg_genes_per_genome'], 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_title('A. Average Gene Number per Genome', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Number of Genes', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Insect Orders', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(df_sorted_genes)))
    ax1.set_xticklabels(df_sorted_genes.index, rotation=45, ha='right', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 100,
                 f'{height:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Average Gene Length Comparison
    ax2 = axes[0, 1]
    df_sorted_length = df.sort_values('avg_gene_length', ascending=False)
    colors2 = ['#E74C3C' if idx == 'Orthoptera' else '#2ECC71' for idx in df_sorted_length.index]
    
    bars2 = ax2.bar(range(len(df_sorted_length)), df_sorted_length['avg_gene_length']/1000, 
                    color=colors2, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax2.set_title('B. Average Gene Length', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Gene Length (kb)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Insect Orders', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(df_sorted_length)))
    ax2.set_xticklabels(df_sorted_length.index, rotation=45, ha='right', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f} kb', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. Gene Structure Complexity
    ax3 = axes[1, 0]
    complexity_metrics = [
        df['total_exons'] / df['total_genes'],  # Exons per gene
        df['total_mrna'] / df['total_genes']    # Transcripts per gene
    ]
    metric_names = ['Exons per Gene', 'Transcripts per Gene']
    
    x_pos = np.arange(len(df))
    width = 0.35
    
    bars3a = ax3.bar(x_pos - width/2, complexity_metrics[0], width, 
                    label=metric_names[0], color='#9B59B6', alpha=0.8, edgecolor='black')
    bars3b = ax3.bar(x_pos + width/2, complexity_metrics[1], width, 
                    label=metric_names[1], color='#F39C12', alpha=0.8, edgecolor='black')
    
    ax3.set_title('C. Gene Structure Complexity', fontsize=16, fontweight='bold', pad=20)
    ax3.set_ylabel('Ratio', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Insect Orders', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(df.index, rotation=45, ha='right', fontsize=12)
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Highlight Orthoptera
    orthoptera_idx = df.index.get_loc('Orthoptera')
    bars3a[orthoptera_idx].set_edgecolor('red')
    bars3a[orthoptera_idx].set_linewidth(2)
    bars3b[orthoptera_idx].set_edgecolor('red')
    bars3b[orthoptera_idx].set_linewidth(2)
    
    # 4. CDS to Gene Ratio
    ax4 = axes[1, 1]
    cds_ratio = (df['total_cds'] / df['total_genes']).sort_values(ascending=False)
    colors4 = ['#E74C3C' if idx == 'Orthoptera' else '#E67E22' for idx in cds_ratio.index]
    
    bars4 = ax4.bar(range(len(cds_ratio)), cds_ratio, 
                    color=colors4, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax4.set_title('D. CDS to Gene Ratio', fontsize=16, fontweight='bold', pad=20)
    ax4.set_ylabel('CDS per Gene', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Insect Orders', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(cds_ratio)))
    ax4.set_xticklabels(cds_ratio.index, rotation=45, ha='right', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('gene_feature_analysis_main.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('gene_feature_analysis_main.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

def create_orthoptera_gene_analysis_figure(df):
    """Create specialized Orthoptera gene feature analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    orthoptera = df.loc['Orthoptera']
    others = df[df.index != 'Orthoptera']
    
    # 1. Gene Feature Expansion
    ax1 = axes[0]
    features = ['avg_genes_per_genome', 'avg_gene_length', 'total_exons', 'total_mrna']
    feature_names = ['Gene Number', 'Gene Length', 'Total Exons', 'Total Transcripts']
    
    orthoptera_values = [orthoptera[feat] for feat in features]
    other_avg_values = [others[feat].mean() for feat in features]
    
    # Normalize for display
    orthoptera_norm = [val / 1000 if i == 1 else val / 1000 if i >= 2 else val for i, val in enumerate(orthoptera_values)]
    other_norm = [val / 1000 if i == 1 else val / 1000 if i >= 2 else val for i, val in enumerate(other_avg_values)]
    
    x_pos = np.arange(len(features))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, orthoptera_norm, width,
                   label='Orthoptera', color='#E74C3C', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x_pos + width/2, other_norm, width,
                   label='Other Orders Average', color='#3498DB', alpha=0.8, edgecolor='black')
    
    ax1.set_title('A. Orthoptera Gene Feature Expansion', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Values (Count or kb)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(feature_names, fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add expansion ratios
    for i, (orth, other) in enumerate(zip(orthoptera_values, other_avg_values)):
        ratio = orth / other
        ax1.text(x_pos[i], max(orthoptera_norm[i], other_norm[i]) * 1.1, 
                f'{ratio:.1f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 2. Feature Ratios Comparison
    ax2 = axes[1]
    ratios = [
        orthoptera['total_cds'] / orthoptera['total_genes'],
        orthoptera['total_exons'] / orthoptera['total_genes'],
        orthoptera['total_mrna'] / orthoptera['total_genes']
    ]
    other_ratios = [
        (others['total_cds'] / others['total_genes']).mean(),
        (others['total_exons'] / others['total_genes']).mean(),
        (others['total_mrna'] / others['total_genes']).mean()
    ]
    ratio_names = ['CDS/Gene', 'Exons/Gene', 'Transcripts/Gene']
    
    x_pos2 = np.arange(len(ratios))
    bars3 = ax2.bar(x_pos2 - width/2, ratios, width,
                   label='Orthoptera', color='#E74C3C', alpha=0.8, edgecolor='black')
    bars4 = ax2.bar(x_pos2 + width/2, other_ratios, width,
                   label='Other Orders Average', color='#27AE60', alpha=0.8, edgecolor='black')
    
    ax2.set_title('B. Gene Structure Ratios Comparison', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Ratio Values', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(ratio_names, fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('gene_feature_orthoptera_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('gene_feature_orthoptera_analysis.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

def create_correlation_analysis_figure(df):
    """Create correlation analysis figure"""
    # Select features for correlation analysis
    features = ['avg_genes_per_genome', 'avg_gene_length', 'avg_genome_size', 
                'gene_density', 'gene_coverage']
    feature_names = ['Gene Number', 'Gene Length', 'Genome Size', 'Gene Density', 'Gene Coverage']
    
    # Calculate correlation matrix
    corr_matrix = df[features].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={'shrink': .8},
                annot_kws={'size': 12, 'weight': 'bold'})
    
    plt.title('Correlation Matrix of Genomic Features', fontsize=18, fontweight='bold', pad=20)
    plt.xticks(np.arange(len(feature_names)) + 0.5, feature_names, rotation=45, ha='right', fontsize=12)
    plt.yticks(np.arange(len(feature_names)) + 0.5, feature_names, rotation=0, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('gene_feature_correlations.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('gene_feature_correlations.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

def gene_feature_statistical_analysis(df):
    """Perform comprehensive statistical analysis of gene features"""
    print("GENE FEATURE STATISTICAL ANALYSIS")
    print("=" * 50)
    
    orthoptera = df.loc['Orthoptera']
    others = df[df.index != 'Orthoptera']
    
    print(f"\n1. GENE NUMBER ANALYSIS:")
    print("-" * 25)
    gene_ratio = orthoptera['avg_genes_per_genome'] / others['avg_genes_per_genome'].mean()
    print(f"Orthoptera gene number: {orthoptera['avg_genes_per_genome']:,.0f}")
    print(f"Other orders average: {others['avg_genes_per_genome'].mean():,.0f}")
    print(f"Expansion ratio: {gene_ratio:.2f}x")
    
    print(f"\n2. GENE LENGTH ANALYSIS:")
    print("-" * 25)
    length_ratio = orthoptera['avg_gene_length'] / others['avg_gene_length'].mean()
    print(f"Orthoptera gene length: {orthoptera['avg_gene_length']:,.0f} bp")
    print(f"Other orders average: {others['avg_gene_length'].mean():,.0f} bp")
    print(f"Expansion ratio: {length_ratio:.2f}x")
    
    print(f"\n3. GENE COMPLEXITY ANALYSIS:")
    print("-" * 30)
    orth_exon_ratio = orthoptera['total_exons'] / orthoptera['total_genes']
    other_exon_ratio = (others['total_exons'] / others['total_genes']).mean()
    print(f"Exons per gene:")
    print(f"  Orthoptera: {orth_exon_ratio:.2f}")
    print(f"  Other orders: {other_exon_ratio:.2f}")
    print(f"  Ratio: {orth_exon_ratio/other_exon_ratio:.2f}x")
    
    orth_transcript_ratio = orthoptera['total_mrna'] / orthoptera['total_genes']
    other_transcript_ratio = (others['total_mrna'] / others['total_genes']).mean()
    print(f"Transcripts per gene:")
    print(f"  Orthoptera: {orth_transcript_ratio:.2f}")
    print(f"  Other orders: {other_transcript_ratio:.2f}")
    print(f"  Ratio: {orth_transcript_ratio/other_transcript_ratio:.2f}x")

def main():
    """Main function for gene feature analysis"""
    print("Starting Gene Feature Analysis...")
    df = load_and_prepare_data()
    
    print("\nCreating visualizations...")
    create_gene_feature_main_figure(df)
    create_orthoptera_gene_analysis_figure(df)
    create_correlation_analysis_figure(df)
    
    print("\nPerforming statistical analysis...")
    gene_feature_statistical_analysis(df)
    
    print("\nAnalysis complete! Generated files:")
    print("  - gene_feature_analysis_main.png/pdf")
    print("  - gene_feature_orthoptera_analysis.png/pdf")
    print("  - gene_feature_correlations.png/pdf")

if __name__ == "__main__":
    main()
