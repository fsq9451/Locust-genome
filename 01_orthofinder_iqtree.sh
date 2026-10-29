#!/bin/bash
# =============================================================
# 01_orthofinder_iqtree.sh — 18 物种建树（2026 拓扑约束）
# 输入: proteins18/  (每物种一个蛋白 .fa，文件名 = 物种名)
#       constraint18.nwk  (先填好直翅目内部拓扑!)
# 输出: tree18.nwk
# =============================================================
set -euo pipefail

THREADS=40
PROT_DIR="proteins18"

[ -d "$PROT_DIR" ] || { echo "错误: 找不到 $PROT_DIR 目录"; exit 1; }
[ -f constraint18.nwk ] || { echo "错误: 缺 constraint18.nwk"; exit 1; }

echo ">>> [1/4] OrthoFinder"
orthofinder -f "$PROT_DIR" -t "$THREADS" -a 8
OF_DIR=$(ls -d "$PROT_DIR"/OrthoFinder/Results_*/ | tail -1)


echo ">>> [2/4] 提取单拷贝 OG"
awk -F'\t' '{ok=1; for(i=2;i<=NF;i++){ if($i==""){ok=0;break}; c=split($i,a,", "); if(c!=1){ok=0;break} } if(ok) print $1}' \
  "$OF_DIR/Orthogroups/Orthogroups.tsv" > single_copy_OGs.txt
wc -l single_copy_OGs.txt
[ -s single_copy_OGs.txt ] || { echo "错误: 没有单拷贝 OG"; exit 1; }

echo ">>> [3/4] MAFFT + trimAl + 串联"
mkdir -p aln_trim
N=0
while read og; do
  N=$((N+1))
  mafft --auto --thread 4 "$OF_DIR/Orthogroup_Sequences/${og}.fa" 2>/dev/null \
    | trimal -in /dev/stdin -out "aln_trim/${og}.fa" -automated1
done < single_copy_OGs.txt
echo ">>> 完成比对 ${N} 个 OG"

ls "$PROT_DIR"/*.fa | sed 's/.*\///; s/\.fa$//' > species_order.txt

python3 - "$PWD/aln_trim" "$PWD/species_order.txt" "$PWD/CONCAT.aln" <<'PYEOF'
import sys, os
from collections import OrderedDict

aln_dir, sp_file, out_file = sys.argv[1], sys.argv[2], sys.argv[3]
species = [l.strip() for l in open(sp_file) if l.strip()]
seqs = OrderedDict((s, '') for s in species)

def parse_fasta(f):
    d = OrderedDict()
    name = None
    for line in open(f):
        line = line.strip()
        if not line: continue
        if line.startswith('>'):
            name = line[1:].split()[0]
            d[name] = ''
        elif name is not None:
            d[name] += line.upper()
    return d

files = [f for f in sorted(os.listdir(aln_dir)) if f.endswith('.fa')]
kept = 0
for f in files:
    d = parse_fasta(os.path.join(aln_dir, f))
    L = {len(v) for v in d.values()}
    if len(L) != 1: continue
    L = L.pop()
    if L == 0: continue
    for sp in species:
        seqs[sp] += d.get(sp, '-' * L)
    kept += 1

with open(out_file, 'w') as out:
    for sp in species:
        out.write('>%s\n%s\n' % (sp, seqs[sp]))
print("串联完成: %d 个 OG, 总长 %d aa" % (kept, len(next(iter(seqs.values())))))
PYEOF

echo ">>> [4/4] IQ-TREE 约束树"
iqtree2 -s CONCAT.aln -g constraint18.nwk -m MFP -bb 1000 -alrt 1000 \
        -T AUTO --prefix iqtree18 -redo
cp iqtree18.treefile tree18.nwk
echo ">>> 完成! tree18.nwk 已生成"
