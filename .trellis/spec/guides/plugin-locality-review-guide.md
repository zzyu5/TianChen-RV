# Plugin Locality Review Guide

Use before adding or reviewing an extension plugin.

## Checklist

- [ ] What capability does this plugin provide?
- [ ] What dialect, ops, types, and attrs does it register?
- [ ] What high-level op classes can it propose variants for?
- [ ] What legality rules are plugin-owned?
- [ ] What tuning parameters and cost inputs are plugin-owned?
- [ ] What emission/runtime/toolchain path is plugin-owned?
- [ ] What core APIs does it use?
- [ ] Did any core pass gain extension-specific branches?
- [ ] If core changed, is the change a generic interface extension or a concrete-extension workaround?

## Locality Evidence

When evaluating plugin integration, record:

- core pass modified LOC;
- plugin LOC;
- new capabilities;
- new ops/types;
- new variant generators;
- selection/dispatch reuse;
- verifier orchestration reuse;
- emission provider reuse.

## Red Flags

- `if hasRVV`, `if hasIME`, or `if hasSophgo` appears in core orchestration code.
- New extension lowering lives outside plugin adapter.
- Future extensions are forced into RVV or IME dialect for convenience.
- Pluginization is described as zero-work hardware support.
