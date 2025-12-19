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
    """Load and prepare genome size data"""
    df = pd.read_csv('insect_orders_genome_stats_with_size.csv', index_col=0)

    # Convert genome sizes to Mb for easier interpretation
    df['avg_genome_size_mb'] = df['avg_genome_size'] / 1e6
    df['min_genome_size_mb'] = df['min_genome_size'] / 1e6
    df['max_genome_size_mb'] = df['max_genome_size'] / 1e6

    return df

def create_genome_size_main_figure(df):
    """Create main genome size comparison figure"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # 1. Average Genome Size Comparison
    ax1 = axes[0, 0]
    df_sorted = df.sort_values('avg_genome_size_mb', ascending=False)
    colors = ['#E74C3C' if idx == 'Orthoptera' else '#3498DB' for idx in df_sorted.index]
    bars = ax1.bar(range(len(df_sorted)), df_sorted['avg_genome_size_mb'],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax1.set_title('A. Average Genome Size by Insect Order', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Genome Size (Mb)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Insect Orders', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(df_sorted)))
    ax1.set_xticklabels(df_sorted.index, rotation=45, ha='right', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                 f'{height:.1f} Mb', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. Genome Size Range
    ax2 = axes[0, 1]
    orders_sorted = df_sorted.index
    y_pos = np.arange(len(orders_sorted))

    for i, order in enumerate(orders_sorted):
        min_size = df.loc[order, 'min_genome_size_mb']
        max_size = df.loc[order, 'max_genome_size_mb']
        avg_size = df.loc[order, 'avg_genome_size_mb']

        color = '#E74C3C' if order == 'Orthoptera' else '#3498DB'
        ax2.plot([min_size, max_size], [i, i], color=color, linewidth=3, alpha=0.7)
        ax2.scatter(avg_size, i, color=color, s=80, zorder=5, edgecolor='black', linewidth=1)

    ax2.set_title('B. Genome Size Range and Distribution', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Genome Size (Mb)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Insect Orders', fontsize=14, fontweight='bold')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(orders_sorted, fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()

    # 3. Gene Density vs Genome Size
    ax3 = axes[1, 0]
    colors = ['#E74C3C' if idx == 'Orthoptera' else '#2ECC71' for idx in df.index]
    sizes = [120 if idx == 'Orthoptera' else 80 for idx in df.index]

    scatter = ax3.scatter(df['avg_genome_size_mb'], df['gene_density'],
                         s=sizes, c=colors, alpha=0.8, edgecolors='black', linewidth=0.5)

    # Add labels for all points
    for i, txt in enumerate(df.index):
        ax3.annotate(txt, (df['avg_genome_size_mb'].iloc[i], df['gene_density'].iloc[i]),
                    xytext=(8, 8), textcoords='offset points', fontsize=11,
                    fontweight='bold' if txt == 'Orthoptera' else 'normal',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

    ax3.set_title('C. Gene Density vs Genome Size', fontsize=16, fontweight='bold', pad=20)
    ax3.set_xlabel('Average Genome Size (Mb)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Gene Density (genes per Mb)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Genome Size vs Gene Coverage
    ax4 = axes[1, 1]
    scatter2 = ax4.scatter(df['avg_genome_size_mb'], df['gene_coverage'],
                          s=sizes, c=colors, alpha=0.8, edgecolors='black', linewidth=0.5)

    # Add labels for all points
    for i, txt in enumerate(df.index):
        ax4.annotate(txt, (df['avg_genome_size_mb'].iloc[i], df['gene_coverage'].iloc[i]),
                    xytext=(8, 8), textcoords='offset points', fontsize=11,
                    fontweight='bold' if txt == 'Orthoptera' else 'normal',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

    ax4.set_title('D. Gene Coverage vs Genome Size', fontsize=16, fontweight='bold', pad=20)
    ax4.set_xlabel('Average Genome Size (Mb)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Gene Coverage (% of genome)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('genome_size_analysis_main.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('genome_size_analysis_main.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

def create_cds_length_comparison_figure(df):
    """Create CDS average length comparison figure"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 1. CDS Average Length Comparison
    ax1 = axes[0]
    df_sorted_cds = df.sort_values('avg_cds_length', ascending=False)
    colors = ['#E74C3C' if idx == 'Orthoptera' else '#8E44AD' for idx in df_sorted_cds.index]
    
    # Convert to kb for better visualization
    cds_lengths_kb = df_sorted_cds['avg_cds_length'] / 1000
    
    bars = ax1.bar(range(len(df_sorted_cds)), cds_lengths_kb,
                   color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax1.set_title('A. Average CDS Length by Insect Order', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Average CDS Length (kb)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Insect Orders', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(df_sorted_cds)))
    ax1.set_xticklabels(df_sorted_cds.index, rotation=45, ha='right', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2f} kb', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. CDS Length vs Gene Length Comparison
    ax2 = axes[1]
    
    # Calculate CDS/Gene length ratio
    df['cds_gene_ratio'] = df['avg_cds_length'] / df['avg_gene_length']
    df_sorted_ratio = df.sort_values('cds_gene_ratio', ascending=False)
    
    colors2 = ['#E74C3C' if idx == 'Orthoptera' else '#27AE60' for idx in df_sorted_ratio.index]
    
    x_pos = np.arange(len(df_sorted_ratio))
    width = 0.35
    
    # Plot CDS length and gene length side by side
    cds_lengths = df_sorted_ratio['avg_cds_length'] / 1000
    gene_lengths = df_sorted_ratio['avg_gene_length'] / 1000
    
    bars2a = ax2.bar(x_pos - width/2, cds_lengths, width,
                     label='CDS Length', color='#8E44AD', alpha=0.8, edgecolor='black')
    bars2b = ax2.bar(x_pos + width/2, gene_lengths, width,
                     label='Gene Length', color='#3498DB', alpha=0.8, edgecolor='black')

    ax2.set_title('B. CDS Length vs Gene Length Comparison', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Length (kb)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Insect Orders', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(df_sorted_ratio.index, rotation=45, ha='right', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add ratio annotations
    for i, ratio in enumerate(df_sorted_ratio['cds_gene_ratio']):
        ax2.text(x_pos[i], max(cds_lengths.iloc[i], gene_lengths.iloc[i]) * 1.05,
                f'{ratio:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('cds_length_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('cds_length_comparison.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

def create_orthoptera_cds_analysis_figure(df):
    """Create specialized Orthoptera CDS analysis figure"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    orthoptera = df.loc['Orthoptera']
    others = df[df.index != 'Orthoptera']

    # 1. CDS Feature Comparison
    ax1 = axes[0]
    features = ['avg_cds_length', 'total_cds', 'total_cds_length']
    feature_names = ['CDS Length\n(bp)', 'Total CDS\nCount', 'Total CDS\nLength (Mb)']
    
    orthoptera_values = [orthoptera[feat] for feat in features]
    other_avg_values = [others[feat].mean() for feat in features]
    
    # Normalize for better visualization
    orthoptera_norm = [
        orthoptera_values[0] / 1000,  # CDS length in kb
        orthoptera_values[1] / 1000,  # CDS count in thousands
        orthoptera_values[2] / 1e6    # CDS length in Mb
    ]
    other_norm = [
        other_avg_values[0] / 1000,
        other_avg_values[1] / 1000,
        other_avg_values[2] / 1e6
    ]

    x_pos = np.arange(len(features))
    width = 0.35

    bars1 = ax1.bar(x_pos - width/2, orthoptera_norm, width,
                   label='Orthoptera', color='#E74C3C', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x_pos + width/2, other_norm, width,
                   label='Other Orders Average', color='#3498DB', alpha=0.8, edgecolor='black')

    ax1.set_title('A. Orthoptera CDS Features Comparison', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Values (bp, count, Mb)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(feature_names, fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add expansion ratios
    for i, (orth, other) in enumerate(zip(orthoptera_values, other_avg_values)):
        ratio = orth / other
        ax1.text(x_pos[i], max(orthoptera_norm[i], other_norm[i]) * 1.1,
                f'{ratio:.1f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 2. CDS Density and Coverage
    ax2 = axes[1]
    density_metrics = ['cds_density', 'avg_cds_length', 'total_cds_length']
    density_names = ['CDS Density\n(CDS/Mb)', 'Average CDS\nLength (bp)', 'Total CDS\nCoverage (Mb)']
    
    orthoptera_density = [orthoptera[metric] for metric in density_metrics]
    other_density = [others[metric].mean() for metric in density_metrics]
    
    # Normalize
    orthoptera_density_norm = [
        orthoptera_density[0],
        orthoptera_density[1] / 1000,
        orthoptera_density[2] / 1e6
    ]
    other_density_norm = [
        other_density[0],
        other_density[1] / 1000,
        other_density[2] / 1e6
    ]

    x_pos2 = np.arange(len(density_metrics))
    bars3 = ax2.bar(x_pos2 - width/2, orthoptera_density_norm, width,
                   label='Orthoptera', color='#E74C3C', alpha=0.8, edgecolor='black')
    bars4 = ax2.bar(x_pos2 + width/2, other_density_norm, width,
                   label='Other Orders Average', color='#27AE60', alpha=0.8, edgecolor='black')

    ax2.set_title('B. CDS Density and Coverage Comparison', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Values', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(density_names, fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('cds_orthoptera_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('cds_orthoptera_analysis.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

def genome_size_statistical_analysis(df):
    """Perform comprehensive statistical analysis"""
    print("GENOME SIZE STATISTICAL ANALYSIS")
    print("=" * 50)

    orthoptera = df.loc['Orthoptera']
    others = df[df.index != 'Orthoptera']

    # Basic statistics
    print(f"\n1. BASIC GENOME SIZE STATISTICS:")
    print("-" * 40)
    print(f"Orthoptera:")
    print(f"  Average genome size: {orthoptera['avg_genome_size_mb']:.2f} Mb")
    print(f"  Range: {orthoptera['min_genome_size_mb']:.2f} - {orthoptera['max_genome_size_mb']:.2f} Mb")
    print(f"  Gene density: {orthoptera['gene_density']:.1f} genes/Mb")
    print(f"  Gene coverage: {orthoptera['gene_coverage']:.2f}%")

    print(f"\nAll Orders Summary:")
    print(f"  Overall average: {df['avg_genome_size_mb'].mean():.2f} ± {df['avg_genome_size_mb'].std():.2f} Mb")
    print(f"  Range: {df['avg_genome_size_mb'].min():.2f} - {df['avg_genome_size_mb'].max():.2f} Mb")

    # Comparative analysis
    print(f"\n2. COMPARATIVE ANALYSIS:")
    print("-" * 30)
    size_ratio = orthoptera['avg_genome_size_mb'] / others['avg_genome_size_mb'].mean()
    print(f"Genome size ratio (Orthoptera/Other): {size_ratio:.2f}x")

    # Ranking
    size_ranking = df['avg_genome_size_mb'].sort_values(ascending=False)
    orth_rank = list(size_ranking.index).index('Orthoptera') + 1
    print(f"Genome size ranking: {orth_rank}/{len(df)}")

    # Statistical test
    print(f"\n3. STATISTICAL SIGNIFICANCE:")
    print("-" * 30)
    t_stat, p_value = stats.ttest_1samp(others['avg_genome_size_mb'], orthoptera['avg_genome_size_mb'])
    print(f"T-test p-value: {p_value:.4f}")
    print(f"Significant difference: {'YES' if p_value < 0.05 else 'NO'}")

def cds_length_statistical_analysis(df):
    """Perform comprehensive CDS length statistical analysis"""
    print("\n" + "=" * 50)
    print("CDS LENGTH STATISTICAL ANALYSIS")
    print("=" * 50)

    orthoptera = df.loc['Orthoptera']
    others = df[df.index != 'Orthoptera']

    print(f"\n1. CDS LENGTH STATISTICS:")
    print("-" * 25)
    print(f"Orthoptera CDS length: {orthoptera['avg_cds_length']:,.0f} bp")
    print(f"Other orders average: {others['avg_cds_length'].mean():,.0f} bp")
    
    cds_length_ratio = orthoptera['avg_cds_length'] / others['avg_cds_length'].mean()
    print(f"CDS length ratio (Orthoptera/Other): {cds_length_ratio:.2f}x")

    # Ranking
    cds_ranking = df['avg_cds_length'].sort_values(ascending=False)
    orth_cds_rank = list(cds_ranking.index).index('Orthoptera') + 1
    print(f"CDS length ranking: {orth_cds_rank}/{len(df)}")

    print(f"\n2. CDS TO GENE LENGTH RATIO:")
    print("-" * 30)
    orth_ratio = orthoptera['avg_cds_length'] / orthoptera['avg_gene_length']
    other_ratio = others['avg_cds_length'].mean() / others['avg_gene_length'].mean()
    print(f"Orthoptera CDS/Gene ratio: {orth_ratio:.2%}")
    print(f"Other orders average: {other_ratio:.2%}")
    print(f"Ratio difference: {orth_ratio/other_ratio:.2f}x")

    print(f"\n3. CDS DENSITY ANALYSIS:")
    print("-" * 25)
    print(f"Orthoptera CDS density: {orthoptera['cds_density']:.1f} CDS/Mb")
    print(f"Other orders average: {others['cds_density'].mean():.1f} CDS/Mb")
    print(f"Density ratio: {orthoptera['cds_density']/others['cds_density'].mean():.2f}x")

    print(f"\n4. STATISTICAL TESTS:")
    print("-" * 25)
    # T-test for CDS length
    t_stat_cds, p_value_cds = stats.ttest_1samp(others['avg_cds_length'], orthoptera['avg_cds_length'])
    print(f"CDS length T-test p-value: {p_value_cds:.4f}")
    print(f"Significant difference in CDS length: {'YES' if p_value_cds < 0.05 else 'NO'}")

    # T-test for CDS density
    t_stat_density, p_value_density = stats.ttest_1samp(others['cds_density'], orthoptera['cds_density'])
    print(f"CDS density T-test p-value: {p_value_density:.4f}")
    print(f"Significant difference in CDS density: {'YES' if p_value_density < 0.05 else 'NO'}")

    # Calculate correlation between CDS length and other metrics
    print(f"\n5. CORRELATIONS:")
    print("-" * 25)
    correlations = df[['avg_cds_length', 'avg_gene_length', 'avg_genome_size_mb', 
                      'cds_density', 'gene_density']].corr()
    
    print("Correlation of CDS length with:")
    print(f"  Gene length: {correlations.loc['avg_cds_length', 'avg_gene_length']:.3f}")
    print(f"  Genome size: {correlations.loc['avg_cds_length', 'avg_genome_size_mb']:.3f}")
    print(f"  CDS density: {correlations.loc['avg_cds_length', 'cds_density']:.3f}")

def main():
    """Main function for genome size analysis"""
    print("Starting Genome Size Analysis...")
    df = load_and_prepare_data()

    print("\nCreating visualizations...")
    create_genome_size_main_figure(df)
    create_cds_length_comparison_figure(df)
    create_orthoptera_cds_analysis_figure(df)

    print("\nPerforming statistical analysis...")
    genome_size_statistical_analysis(df)
    cds_length_statistical_analysis(df)

    print("\nAnalysis complete! Generated files:")
    print("  - genome_size_analysis_main.png/pdf")
    print("  - cds_length_comparison.png/pdf")
    print("  - cds_orthoptera_analysis.png/pdf")

if __name__ == "__main__":
    main()
