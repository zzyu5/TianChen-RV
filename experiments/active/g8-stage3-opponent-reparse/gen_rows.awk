# Emit 36-col T3 opponent-reparse rows. Spec FS = TAB. Fields:
# 1 board  2 rowkey(op|fmt|path|arity|shape)  3 vlen  4 ledger  5 opp_grade  6 opp_symbol  7 objdump_metric  8 hardgate  9 compiler_axis  10 note
BEGIN{ FS="\t"; OFS="," }
/^#/ || NF<9 { next }
{
  board=$1; rowkey=$2; vlen=$3; ledger=$4; grade=$5; sym=$6; metric=$7; hg=$8; cax=$9; note=(NF>=10?$10:"");
  oc="factory";
  if (grade ~ /scalar/) oc="scalar"; else if (grade ~ /generic|light/) oc="naive";
  cf=(board=="rvv")?"g8-stage3-opponent-reparse/rvv/objdump_metrics_rvv.txt":"g8-stage3-opponent-reparse/k1/objdump_metrics_k1.txt";
  bfp=(board=="rvv")?"rvv-openEuler-VLEN128-clang18.1.8-tcrv-canonical-fnoias-richmarch-core8_15":"k1-SpacemiTX60-VLEN256-clang18.1.8-Bianbu-core0_3";
  snap=(board=="rvv")?"ggml=e36a602(vericurve-2026-06-15)":"ggml=e36a602(vericurve)+stock-1x16.so";
  metric2=metric (note!=""?(";"note):"");
  print rowkey,"g8s3-opp-reparse","mechanism","opp-reparse-symbol-level","g8-stage3-opponent-reparse", \
    "n_a(opp-reparse-no-AB)","vlen=" vlen, metric2, \
    "pending-§6","pending-§6","pending-§6","n_a", \
    cf, oc, "pending-§6","pending-§6","pending-§6","n_a", \
    "n_a","n_a","n_a","n_a","n_a", \
    "opp-reparse-clang18-sym", bfp, "2026-07-14-g8-stage3-opp-reparse", cf, snap, \
    ledger, "pending-§6", grade, sym, cf, cax, "pending-§6", hg;
}
