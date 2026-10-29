#!/bin/bash
# =============================================================
# 04_repeatmasker.sh v4 — 合并库并行 RepeatMasker
# 输入: fa_files/*.fa        (24 个基因组, 4字母文件名)
#       RM/*-families.*      (RepeatModeler de novo 库, 18 个)
#       Dfam-1.embl          (绝对路径)
# 策略: 所有物种 de novo consensus 合并 -> cd-hit 去冗余(可选)
#        -> + Dfam = 统一库 all_insecta.lib
#        -> 24 个基因组共用同一库并行跑 RepeatMasker
# =============================================================
set -euo pipefail

THREADS=15           # 每个 RepeatMasker 任务核心数
PARALLEL_JOBS=12      # 同时运行的物种数
FA_DIR="fa_files"
RM_LIB_DIR="RM"
DFAM="/home/user001/data/Dfam/Dfam-1.embl"
RM_BIN="/pub/software/RepeatMasker-4.1.5"
PARSER_DIR="Parsing-RepeatMasker-Outputs"   # 可选; 存在才跑 parseRM
[ -d "$FA_DIR" ] || { echo "错误: 缺 $FA_DIR 目录"; exit 1; }
[ -d "$RM_LIB_DIR" ] || { echo "错误: 缺 $RM_LIB_DIR 目录"; exit 1; }
mkdir -p libs

# ---------- 1) 物种清单 ----------
ls "$FA_DIR"/*.fa | sed 's/.*\///; s/\.fa$//' | sort -u > species_list.txt
echo ">>> 共 $(wc -l < species_list.txt) 个物种"

# ---------- 2) 合并所有 de novo 库 ----------
: > libs/denovo_all.fa
: > no_denovo_species.txt
for sp in $(cat species_list.txt); do
  DN_LIB=""
  if [ -f "$RM_LIB_DIR/${sp}-families.fa" ]; then
    DN_LIB="$RM_LIB_DIR/${sp}-families.fa"
  elif [ -f "$RM_LIB_DIR/${sp}-consensi.fa" ]; then
    DN_LIB="$RM_LIB_DIR/${sp}-consensi.fa"
  elif [ -f "$RM_LIB_DIR/${sp}-families.stk" ]; then
    python3 - "$RM_LIB_DIR/${sp}-families.stk" "libs/${sp}.dn.fa" <<'PYEOF'
import sys
stk, out = sys.argv[1], sys.argv[2]
seqs = {}
order = []
for line in open(stk):
    line = line.rstrip()
    if not line or line.startswith('#'): continue
    parts = line.split()
    if len(parts) < 2: continue
    name, seq = parts[0], parts[1]
    if name not in seqs:
        seqs[name] = ''; order.append(name)
    seqs[name] += seq
with open(out, 'w') as fh:
    for n in order:
        fh.write('>%s\n%s\n' % (n, seqs[n]))
print("stk -> fa: %d 条" % len(order))
PYEOF
    DN_LIB="libs/${sp}.dn.fa"
  fi

  if [ -n "$DN_LIB" ] && [ -f "$DN_LIB" ]; then
    cat "$DN_LIB" >> libs/denovo_all.fa
    echo ">>> 已并入 ${sp} de novo 库"
  else
    echo "$sp" >> no_denovo_species.txt
    echo ">>> ${sp}: 无 de novo 库(仅靠 Dfam 覆盖)"
  fi
done
echo ">>> de novo 合并前条目数: $(grep -c '>' libs/denovo_all.fa)"
echo ">>> 无 de novo 库物种: $(cat no_denovo_species.txt | tr '\n' ' ')"

# ---------- 3) cd-hit 去冗余（可选, 装了才跑） ----------
if command -v cd-hit-est >/dev/null 2>&1; then
  echo ">>> cd-hit-est 去冗余 (80% 相似度)..."
  cd-hit-est -i libs/denovo_all.fa -o libs/denovo_all_clu.fa \
             -c 0.8 -n 5 -M 0 -T 60
  DENOVO=libs/denovo_all_clu.fa
  echo ">>> 去冗余后条目数: $(grep -c '>' $DENOVO)"
else
  DENOVO=libs/denovo_all.fa
  echo ">>> 未安装 cd-hit, 直接使用合并库(未去冗余)"
fi

# ---------- 4) 统一库 = de novo + Dfam ----------
cat "$DENOVO" "$DFAM" > libs/all_insecta.lib
echo ">>> 统一库 libs/all_insecta.lib 构建完成: $(grep -c '>' libs/all_insecta.lib) 条"

# ---------- 5) 并行 RepeatMasker（全部物种同一库） ----------
RM_HOME=$(dirname "$(command -v $RM_BIN)")

run_sp() {
  sp=$1
  [ -f "$FA_DIR/${sp}.fa" ] || { echo "警告: 缺 $FA_DIR/${sp}.fa, 跳过"; return 0; }
  echo ">>> [${sp}] RepeatMasker 开始 $(date)"
  mkdir -p "${sp}_out"
  $RM_BIN -pa "$THREADS" -e rmblast -lib libs/all_insecta.lib \
          -a -gff -xsmall -dir "${sp}_out" "$FA_DIR/${sp}.fa"

  echo ">>> [${sp}] calcDivergenceFromAlign"
  "$RM_HOME/util/calcDivergenceFromAlign.pl" -div "${sp}_out/${sp}.div" "${sp}_out/${sp}.fa.align"

  if [ -f "$PARSER_DIR/parseRM.pl" ]; then
    perl "$PARSER_DIR/parseRM.pl" -l 50 -g 50000 -div 1 "${sp}_out/${sp}.tbl" > "${sp}_out/${sp}_TE_table.txt"
  fi
  echo ">>> [${sp}] 完成 $(date)"
}
export -f run_sp
export FA_DIR RM_BIN THREADS RM_HOME PARSER_DIR

echo ">>> 开始并行处理: ${PARALLEL_JOBS} 个物种同时跑, 每任务 ${THREADS} 核"
cat species_list.txt | xargs -P "$PARALLEL_JOBS" -I {} bash -c 'run_sp "$1" > "${1}_run.log" 2>&1' _ {}

echo "========== 全部完成 =========="
echo "每物种日志: ${sp}_run.log ; 产物: ${sp}_out/${sp}.tbl / ${sp}.out / ${sp}.div"


