import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict

# Define insect orders list
orders = [
    "Blattodea", "Coleoptera", "Collembola", "Diptera", 
    "Ephemeroptera", "Hemiptera", "Hymenoptera", "Lepidoptera",
    "Neuroptera", "Odonata", "Orthoptera", "Phasmatodea", 
    "Psocodea", "Thysanoptera"
]

def calculate_genome_size_from_gff(gff_file):
    """Calculate genome size from GFF file by scanning all sequences"""
    scaffold_lengths = {}
    current_scaffold = None
    max_position = 0
    
    try:
        with open(gff_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                
                scaffold = parts[0]
                start = int(parts[3])
                end = int(parts[4])
                
                # Update maximum position for each scaffold
                if scaffold not in scaffold_lengths:
                    scaffold_lengths[scaffold] = end
                else:
                    if end > scaffold_lengths[scaffold]:
                        scaffold_lengths[scaffold] = end
    
    except Exception as e:
        print(f"Error calculating genome size for {gff_file}: {e}")
        return 0
    
    # Total genome size is sum of all scaffold lengths
    total_size = sum(scaffold_lengths.values())
    return total_size

def parse_gff_stats(gff_file):
    """Parse GFF files and extract genome features including genome size"""
    stats = {
        'genes': 0,
        'cds': 0,
        'exons': 0,
        'mrna': 0,
        'total_gene_length': 0,
        'total_cds_length': 0,
        'gene_lengths': [],
        'cds_lengths': [],
        'genome_size': 0,
        'scaffold_count': 0
    }
    
    scaffold_lengths = {}
    
    try:
        with open(gff_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                
                scaffold = parts[0]
                start = int(parts[3])
                end = int(parts[4])
                feature_type = parts[2]
                
                # Track scaffold lengths for genome size calculation
                if scaffold not in scaffold_lengths:
                    scaffold_lengths[scaffold] = end
                else:
                    if end > scaffold_lengths[scaffold]:
                        scaffold_lengths[scaffold] = end
                
                # Count genes
                if feature_type == 'gene':
                    stats['genes'] += 1
                    length = end - start + 1
                    stats['total_gene_length'] += length
                    stats['gene_lengths'].append(length)
                
                # Count CDS
                elif feature_type == 'CDS':
                    stats['cds'] += 1
                    length = end - start + 1
                    stats['total_cds_length'] += length
                    stats['cds_lengths'].append(length)
                
                # Count exons
                elif feature_type == 'exon':
                    stats['exons'] += 1
                
                # Count mRNA
                elif feature_type == 'mRNA':
                    stats['mrna'] += 1
        
        # Calculate genome size and scaffold count
        stats['genome_size'] = sum(scaffold_lengths.values())
        stats['scaffold_count'] = len(scaffold_lengths)
        stats['avg_scaffold_length'] = stats['genome_size'] / len(scaffold_lengths) if scaffold_lengths else 0
                    
    except Exception as e:
        print(f"Error parsing {gff_file}: {e}")
    
    return stats

def analyze_order(order_dir):
    """Analyze all GFF files in one order directory"""
    gff_files = glob.glob(os.path.join(order_dir, "*.gff"))
    gff_files.extend(glob.glob(os.path.join(order_dir, "*.gff3")))
    
    if not gff_files:
        print(f"No GFF files found in {order_dir}")
        return None
    
    order_stats = {
        'genome_count': 0,
        'total_genes': 0,
        'total_cds': 0,
        'total_exons': 0,
        'total_mrna': 0,
        'avg_genes_per_genome': 0,
        'avg_gene_length': 0,
        'avg_cds_length': 0,
        'total_genome_size': 0,
        'avg_genome_size': 0,
        'max_genome_size': 0,
        'min_genome_size': float('inf'),
        'avg_scaffold_count': 0,
        'avg_scaffold_length': 0,
        'gene_density': 0,  # genes per Mb
        'cds_density': 0,   # CDS per Mb
        'gene_coverage': 0, # percentage of genome covered by genes
        'gene_length_distribution': [],
        'cds_length_distribution': [],
        'genome_sizes': []
    }
    
    all_gene_lengths = []
    all_cds_lengths = []
    genome_sizes = []
    scaffold_counts = []
    scaffold_lengths = []
    
    for gff_file in gff_files:
        stats = parse_gff_stats(gff_file)
        if stats['genes'] > 0:  # Only count files with genes
            order_stats['genome_count'] += 1
            order_stats['total_genes'] += stats['genes']
            order_stats['total_cds'] += stats['cds']
            order_stats['total_exons'] += stats['exons']
            order_stats['total_mrna'] += stats['mrna']
            order_stats['total_genome_size'] += stats['genome_size']
            
            all_gene_lengths.extend(stats['gene_lengths'])
            all_cds_lengths.extend(stats['cds_lengths'])
            genome_sizes.append(stats['genome_size'])
            scaffold_counts.append(stats['scaffold_count'])
            scaffold_lengths.append(stats['avg_scaffold_length'])
            
            # Track min/max genome sizes
            if stats['genome_size'] > order_stats['max_genome_size']:
                order_stats['max_genome_size'] = stats['genome_size']
            if stats['genome_size'] < order_stats['min_genome_size']:
                order_stats['min_genome_size'] = stats['genome_size']
    
    if order_stats['genome_count'] > 0:
        # Basic statistics
        order_stats['avg_genes_per_genome'] = order_stats['total_genes'] / order_stats['genome_count']
        order_stats['avg_gene_length'] = sum(all_gene_lengths) / len(all_gene_lengths) if all_gene_lengths else 0
        order_stats['avg_cds_length'] = sum(all_cds_lengths) / len(all_cds_lengths) if all_cds_lengths else 0
        order_stats['avg_genome_size'] = order_stats['total_genome_size'] / order_stats['genome_count']
        order_stats['avg_scaffold_count'] = sum(scaffold_counts) / len(scaffold_counts) if scaffold_counts else 0
        order_stats['avg_scaffold_length'] = sum(scaffold_lengths) / len(scaffold_lengths) if scaffold_lengths else 0
        
        # Density calculations
        order_stats['gene_density'] = (order_stats['total_genes'] / order_stats['total_genome_size']) * 1e6  # genes per Mb
        order_stats['cds_density'] = (order_stats['total_cds'] / order_stats['total_genome_size']) * 1e6    # CDS per Mb
        
        # Gene coverage (total gene length as percentage of total genome size)
        total_gene_length = sum(all_gene_lengths)
        order_stats['gene_coverage'] = (total_gene_length / order_stats['total_genome_size']) * 100 if order_stats['total_genome_size'] > 0 else 0
        
        order_stats['gene_length_distribution'] = all_gene_lengths
        order_stats['cds_length_distribution'] = all_cds_lengths
        order_stats['genome_sizes'] = genome_sizes
    
    return order_stats

# Main analysis pipeline
all_results = {}

for order in orders:
    print(f"Analyzing {order}...")
    if os.path.exists(order):
        stats = analyze_order(order)
        if stats:
            all_results[order] = stats
            print(f"  Found {stats['genome_count']} genomes, {stats['total_genes']} genes, Avg genome size: {stats['avg_genome_size']/1e6:.2f} Mb")
    else:
        print(f"  Directory {order} not found")

# Save results to CSV
results_df = pd.DataFrame.from_dict(all_results, orient='index')
results_df.to_csv('insect_orders_genome_stats_with_size.csv')
print("Analysis complete! Results saved to insect_orders_genome_stats_with_size.csv")

