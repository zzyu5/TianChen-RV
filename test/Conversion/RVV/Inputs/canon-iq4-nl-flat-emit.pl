#!/usr/bin/env perl
# Shared canonicalizer for the iq4_nl (ggml IQ4_NL x Q8_0) CODEBOOK flat block-dot
# region-vs-monolith EMPIRICAL byte-diff lit
# (rvv-to-emitc-iq4-nl-q8-0-typed-flat-block-dot-loop-body).
#
# The typed tcrv_rvv.typed_flat_block_dot_loop_body codebook branch and the
# monolithic emitFlatBlockDot (via tcrv_rvv.iq4_nl_q8_0_block_dot) drive the SAME
# shared emitFlatBlockCore (-> emitFlatIntegerCore CodebookGatherNibble) +
# emitFlatFold (SumiTimesScales) leaves, so the emitted KERNEL BODY (the block loop,
# the codebook decl + broadcast, every vle8 / vand / vsrl / vrgather / vwmul /
# vwmacc / vwredsum / fp16 read / fold intrinsic call, in order, with identical
# operand wiring) is byte-identical BY CONSTRUCTION. Two residues are NOT part of the
# kernel body and are normalized away here so an empty `diff` proves the bodies
# equal:
#
#   1. the emitc.func NAME (the loop-body kernel vs the monolith kernel MAY differ);
#   2. the source-op PROVENANCE strings the VerbatimOp comments carry
#      (route_source_op= / source_op=tcrv_rvv.<op>), which name the lowering's
#      source op (typed_flat_block_dot_loop_body vs iq4_nl_q8_0_block_dot) and so
#      differ by construction.
#
# There is NO trailing unused-result token to strip (unlike the q4_0 repack GEVM):
# the iq4_nl block-dot op's i32-vector result maps to the already-emitted scalar
# store value, seeding no extra node, so the two emissions have identical SSA
# numbering natively. Canonicalizing (rather than blanking) the SSA names still
# preserves def-use wiring, so the diff catches a mis-wired operand, a dropped/added
# op, or a reordered node -- this is a structural byte-diff, not a loose token match.

use strict;
use warnings;

local $/;                      # slurp the whole module
my $txt = <STDIN>;

# Canonicalize every %N / %argN SSA name by order of first appearance.
my %seen;
my $next = 0;
$txt =~ s{(\%arg\d+|\%\d+)}{ $seen{$1} //= "\%C" . $next++ }ge;

# (1) Normalize the emitc.func name.
$txt =~ s/\@tcrv_emitc_\w+\(/\@KERNEL(/g;

# (2) Normalize the source-op provenance the VerbatimOp comments carry.
$txt =~ s/(route_source_op|source_op)=tcrv_rvv\.\w+ /$1=OP /g;

print $txt;
