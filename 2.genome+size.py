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

def create_orthoptera_comparison_figure(df):
    """Create specialized Orthoptera comparison figure"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    orthoptera = df.loc['Orthoptera']
    others = df[df.index != 'Orthoptera']
    
    # 1. Genome Expansion Metrics
    ax1 = axes[0]
    metrics = ['avg_genome_size_mb', 'avg_genes_per_genome', 'avg_gene_length']
    metric_names = ['Genome Size\n(Mb)', 'Gene Number\nper Genome', 'Gene Length\n(bp)']
    
    orthoptera_values = [
        orthoptera['avg_genome_size_mb'],
        orthoptera['avg_genes_per_genome'],
        orthoptera['avg_gene_length']
    ]
    
    other_avg_values = [
        others['avg_genome_size_mb'].mean(),
        others['avg_genes_per_genome'].mean(),
        others['avg_gene_length'].mean()
    ]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, orthoptera_values, width, 
                   label='Orthoptera', color='#E74C3C', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x_pos + width/2, other_avg_values, width, 
                   label='Other Orders Average', color='#3498DB', alpha=0.8, edgecolor='black')
    
    ax1.set_title('A. Orthoptera Genome Expansion Patterns', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Values', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(metric_names, fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add ratio labels
    for i, (orth, other) in enumerate(zip(orthoptera_values, other_avg_values)):
        ratio = orth / other
        ax1.text(x_pos[i], max(orth, other) + max(orth, other)*0.1, 
                f'{ratio:.1f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 2. Density Metrics Comparison
    ax2 = axes[1]
    density_metrics = ['gene_density', 'cds_density', 'gene_coverage']
    density_names = ['Gene Density\n(genes/Mb)', 'CDS Density\n(CDS/Mb)', 'Gene Coverage\n(%)']
    
    orthoptera_density = [orthoptera[metric] for metric in density_metrics]
    other_density = [others[metric].mean() for metric in density_metrics]
    
    x_pos2 = np.arange(len(density_metrics))
    bars3 = ax2.bar(x_pos2 - width/2, orthoptera_density, width, 
                   label='Orthoptera', color='#E74C3C', alpha=0.8, edgecolor='black')
    bars4 = ax2.bar(x_pos2 + width/2, other_density, width, 
                   label='Other Orders Average', color='#27AE60', alpha=0.8, edgecolor='black')
    
    ax2.set_title('B. Genomic Density and Coverage Comparison', fontsize=16, fontweight='bold', pad=20)
    ax2.set_ylabel('Values', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(density_names, fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('genome_size_orthoptera_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('genome_size_orthoptera_comparison.pdf', bbox_inches='tight', facecolor='white')
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

def main():
    """Main function for genome size analysis"""
    print("Starting Genome Size Analysis...")
    df = load_and_prepare_data()
    
    print("\nCreating visualizations...")
    create_genome_size_main_figure(df)
    create_orthoptera_comparison_figure(df)
    
    print("\nPerforming statistical analysis...")
    genome_size_statistical_analysis(df)
    
    print("\nAnalysis complete! Generated files:")
    print("  - genome_size_analysis_main.png/pdf")
    print("  - genome_size_orthoptera_comparison.png/pdf")

if __name__ == "__main__":
    main()

