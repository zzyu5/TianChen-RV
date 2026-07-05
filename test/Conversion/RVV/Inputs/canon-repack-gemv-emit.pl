#!/usr/bin/env perl
# Shared canonicalizer for the q4_0 16x1-REPACKED GEVM region-vs-monolith
# EMPIRICAL byte-diff lits (rvv-to-emitc-typed-repack-gemv-loop-body-*-full-body).
#
# The typed tcrv_rvv.typed_repack_gemv_loop_body region emitter and the monolithic
# emitRepackGemvQ4_0Q8_0 drive the SAME shared integer-core / dual-fp16 scale-fold
# leaves, so the emitted KERNEL BODY (the loop nest + every vle8 / nibble decode /
# vwmacc / vwadd / vle16 / vfwmul / vfcvt / vfmacc / vse32 intrinsic call, in
# order, with identical operand wiring) is byte-identical BY CONSTRUCTION on every
# resource arm. Three residues are NOT part of the kernel body and are normalized
# away here so an empirical `diff` proves the bodies equal:
#
#   1. the emitc.func NAME (the loop-body kernel vs the monolith kernel differ);
#   2. the source-op PROVENANCE strings the VerbatimOp comments carry
#      (route_source_op= / source_op=tcrv_rvv.<op>), which name the lowering's
#      source op and so differ by construction;
#   3. the monolith's trailing UNUSED-RESULT token (`vint32m1_t = vmv_v_x_i32m1(0,
#      1)` seeded so the GEMV op's result has a well-formed valueMap entry) -- the
#      loop-body op has NO result, so it emits no such token. This token lives in
#      the func entry block, so it also shifts every later SSA number by +3.
#
# After (3) is deleted and the SSA value names are canonicalized by first
# appearance, the two emissions are byte-for-byte identical (an empty `diff`).
# Canonicalizing (rather than blanking) the SSA names preserves def-use wiring, so
# the diff still catches a mis-wired operand, a dropped/added op, or a reordered
# node -- this is a structural byte-diff, not a loose token-set match.

use strict;
use warnings;

local $/;                      # slurp the whole module
my $txt = <STDIN>;

# (3) Delete the monolith's trailing unused-result token: the vmv_v_x_i32m1 call
# line plus its two immediately-preceding literal-operand lines (the 0 seed and
# the AVL 1). No-op on the region emit, which never has the token.
$txt =~ s/^[^\n]*\n[^\n]*\n[^\n]*__riscv_vmv_v_x_i32m1[^\n]*\n//mg;

# Canonicalize every %N / %argN SSA name by order of first appearance.
my %seen;
my $next = 0;
$txt =~ s{(\%arg\d+|\%\d+)}{ $seen{$1} //= "\%C" . $next++ }ge;

# (1) Normalize the emitc.func name.
$txt =~ s/\@tcrv_emitc_\w+\(/\@KERNEL(/g;

# (2) Normalize the source-op provenance the VerbatimOp comments carry.
$txt =~ s/(route_source_op|source_op)=tcrv_rvv\.\w+ /$1=OP /g;

print $txt;
