# Notes

## Initial Context

Created from Hermes Direction Brief on 2026-05-13.

The key distinction from the previous completed task is composite dispatch
ownership:

```text
selected RVV callable component
  + selected scalar callable fallback
  -> RVV+scalar dispatch control
  -> common EmitC call route
  -> generated source/header/object bundle
  -> external caller on ssh rvv
```

Standalone direct RVV microkernel evidence is useful prior context, but it is
not sufficient evidence for this task.
