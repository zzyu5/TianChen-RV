#!/usr/bin/env python3
"""Export one registered benchmark kernel from the current Weft compiler.

This is runner orchestration, not a compiler implementation.  It deliberately
owns no formula, format semantics, selection policy, or C body.  Each recipe
starts at the production test/source entry for its family, invokes the current
family construction path, and asks the registered artifact lowering to emit C.

The official cell harnesses call this tool with ``--require-clean`` and write
the result only into an ephemeral ``mktemp`` directory.  The emitted seal makes
the measured symbol traceable to the current commit, compiler binaries, source
fixture, target capability tier, and generated artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILD_DIR = ROOT / "build" / "weft"
MLIR_TRANSLATE_CANDIDATES = (
    Path("/usr/bin/mlir-translate-20"),
    Path("/usr/lib/llvm-20/bin/mlir-translate"),
)


class ExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactRecipe:
    op: str
    fmt: str
    family: str
    fixture: str
    expected_symbol: str
    mode: str
    front_door: str = ""


def _grid_recipe(fmt: str) -> ArtifactRecipe:
    slug = fmt.replace("_", "-")
    return ArtifactRecipe(
        op="gemm_tile",
        fmt=fmt,
        family="rvv",
        fixture=(
            "test/Conversion/RVV/rvv-emit-quant-contraction-"
            f"{slug}-repack-gemm-prefill-vlen128.mlir"
        ),
        expected_symbol=(
            f"weft_emitc_ggml_repack_gemm_{fmt}_q8_K_kernel_"
            f"ggml_repack_gemm_{fmt}_q8_K"
        ),
        mode="rvv-quant-contraction",
    )


def _vec_dot_recipe(fmt: str) -> ArtifactRecipe:
    qslug = fmt.split("_", 1)[0].lower()
    front_door = f"weft-rvv-materialize-{qslug}-k-q8-k-block-dot-source-front-door"
    return ArtifactRecipe(
        op="vec_dot",
        fmt=fmt,
        family="rvv",
        fixture=(
            f"test/Target/RVV/{qslug}-k-q8-k-super-block-block-dot-"
            "full-pipeline-export-e2e.mlir"
        ),
        expected_symbol=(
            f"weft_emitc_ggml_vec_dot_{fmt}_q8_K_kernel_"
            f"rvv_{fmt}_q8_K_block_dot"
        ),
        mode="rvv-source-front-door",
        front_door=front_door,
    )


RECIPES = {
    (recipe.op, recipe.fmt): recipe
    for recipe in (
        *(_grid_recipe(fmt) for fmt in (
            "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s",
            "iq3_xxs", "iq3_s",
        )),
        *(_vec_dot_recipe(fmt) for fmt in (
            "q2_K", "q3_K", "q4_K", "q5_K", "q6_K",
        )),
        ArtifactRecipe(
            op="product_reduce",
            fmt="q4_0_nibble",
            family="rvv",
            fixture=(
                "test/Conversion/RVV/"
                "rvv-to-emitc-unsigned-nibble-x-i8-product-reduce.mlir"
            ),
            expected_symbol=(
                "weft_emitc_rvv_unsigned_nibble_q8_1_integer_core_kernel_"
                "rvv_unsigned_nibble_q8_1_integer_core"
            ),
            mode="rvv-exact-body",
        ),
        ArtifactRecipe(
            op="product_reduce",
            fmt="offset_binary_n3",
            family="rvv",
            fixture=(
                "test/Conversion/RVV/"
                "rvv-to-emitc-five-bit-offset-binary-x-i8-product-reduce.mlir"
            ),
            expected_symbol=(
                "weft_emitc_rvv_five_bit_q5_0_integer_core_kernel_"
                "rvv_five_bit_q5_0_integer_core"
            ),
            mode="rvv-exact-body",
        ),
        ArtifactRecipe(
            op="product_reduce",
            fmt="codebook_n3",
            family="rvv",
            fixture=(
                "test/Conversion/RVV/"
                "rvv-to-emitc-codebook-gather-x-i8-product-reduce.mlir"
            ),
            expected_symbol=(
                "weft_emitc_rvv_codebook_q8_0_integer_core_kernel_"
                "rvv_codebook_q8_0_integer_core"
            ),
            mode="rvv-exact-body",
        ),
        ArtifactRecipe(
            op="vec_dot",
            fmt="tq2_0",
            family="scalar",
            fixture="test/Target/Scalar/tq2-0-q8-k-ternary-vec-dot.mlir",
            expected_symbol="weft_emitc_tq2_0_kernel_scalar_fallback_first_slice",
            mode="scalar-source-front-door",
        ),
    )
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(args: Iterable[str], *, input_bytes: bytes | None = None) -> bytes:
    command = [str(arg) for arg in args]
    result = subprocess.run(
        command,
        cwd=ROOT,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise ExportError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{stderr}"
        )
    return result.stdout


def _git(*args: str) -> str:
    return _run(("git", *args)).decode("utf-8", errors="replace").strip()


def _resolve_mlir_translate() -> Path:
    override = os.environ.get("WEFT_MLIR_TRANSLATE")
    if override:
        path = Path(override).expanduser().resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
        raise ExportError(f"WEFT_MLIR_TRANSLATE is not executable: {path}")
    for path in MLIR_TRANSLATE_CANDIDATES:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    discovered = shutil.which("mlir-translate-20") or shutil.which("mlir-translate")
    if discovered:
        return Path(discovered).resolve()
    raise ExportError("mlir-translate-20 is required to render current EmitC")


def _march_for(board: str, family: str) -> str:
    if family == "scalar":
        if board != "scalar":
            raise ExportError("scalar recipe is valid only for board=scalar")
        return "rv64gc"
    if board == "rvv":
        return "rv64gcv"
    if board == "k1":
        return "rv64gcv_zvl256b"
    raise ExportError("RVV recipe is valid only for board=rvv|k1")


def _build_tools(build_dir: Path) -> None:
    if not (build_dir / "CMakeCache.txt").is_file():
        raise ExportError(f"configured Weft build directory is missing: {build_dir}")
    _run((
        "cmake", "--build", str(build_dir), "--target", "weft-opt",
        "weft-translate", "-j2",
    ))


def _check_clean(require_clean: bool) -> str:
    head = _git("rev-parse", "HEAD")
    if require_clean:
        # The official runner creates its immutable run directory before the
        # measure invocation.  That evidence is an output of this transaction,
        # not compiler input; every other tracked or untracked repository path
        # remains cleanliness-sensitive.
        dirty = _git(
            "status", "--porcelain", "--untracked-files=all", "--", ".",
            ":(exclude)experiments/runs/**",
        )
        if dirty:
            raise ExportError(
                "official current-artifact export requires a clean worktree; "
                "commit the compiler/task state before measurement"
            )
    return head


def _export_rvv(
    recipe: ArtifactRecipe,
    *,
    board: str,
    build_dir: Path,
    mlir_translate: Path,
) -> tuple[bytes, list[str]]:
    march = _march_for(board, recipe.family)
    weft_opt = build_dir / "bin" / "weft-opt"
    args = [str(weft_opt), str(ROOT / recipe.fixture)]
    if recipe.mode == "rvv-quant-contraction":
        args.append(f"--weft-rvv-lower-quant-contraction=march={march}")
    elif recipe.mode == "rvv-source-front-door":
        args.append(f"--{recipe.front_door}=march={march}")
        args.append("--weft-execution-planning-pipeline")
    elif recipe.mode != "rvv-exact-body":
        raise ExportError(f"unsupported RVV export mode: {recipe.mode}")
    args.append("--weft-rvv-lower-to-emitc")
    emitc_module = _run(args)
    source = _run((str(mlir_translate), "--mlir-to-cpp"), input_bytes=emitc_module)
    return source, args + ["|", str(mlir_translate), "--mlir-to-cpp"]


def _export_scalar(
    recipe: ArtifactRecipe, *, board: str, build_dir: Path
) -> tuple[bytes, list[str]]:
    _march_for(board, recipe.family)
    weft_translate = build_dir / "bin" / "weft-translate"
    args = [
        str(weft_translate),
        "--weft-scalar-emitc-to-cpp",
        str(ROOT / recipe.fixture),
    ]
    return _run(args), args


def export_current_artifact(
    recipe: ArtifactRecipe,
    *,
    board: str,
    output: Path,
    build_dir: Path,
    require_clean: bool,
    skip_build: bool,
) -> None:
    fixture = ROOT / recipe.fixture
    if not fixture.is_file():
        raise ExportError(f"registered current-artifact fixture is missing: {fixture}")
    if output.exists():
        raise ExportError(f"refusing to overwrite current-artifact output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    head = _check_clean(require_clean)
    if not skip_build:
        _build_tools(build_dir)

    weft_opt = build_dir / "bin" / "weft-opt"
    weft_translate = build_dir / "bin" / "weft-translate"
    for tool in (weft_opt, weft_translate):
        if not tool.is_file() or not os.access(tool, os.X_OK):
            raise ExportError(f"current compiler tool is missing: {tool}")

    march = _march_for(board, recipe.family)
    if recipe.family == "rvv":
        render_tool = _resolve_mlir_translate()
        source, command = _export_rvv(
            recipe,
            board=board,
            build_dir=build_dir,
            mlir_translate=render_tool,
        )
    else:
        render_tool = weft_translate
        source, command = _export_scalar(
            recipe, board=board, build_dir=build_dir
        )

    if not source.strip():
        raise ExportError("current artifact export produced empty source")
    if recipe.expected_symbol.encode() not in source:
        raise ExportError(
            "current artifact lost its declared deployed symbol: "
            f"{recipe.expected_symbol}"
        )
    # Provenance comments intentionally retain typed source-op names such as
    # `weft_rvv.*`; reject textual MLIR structure, not those honest comments.
    stripped = source.lstrip()
    if stripped.startswith(b"module {") or b'"emitc.func"()' in source:
        raise ExportError("current artifact export left compiler IR in generated C/C++")

    output.write_bytes(source)
    print(
        "# CURRENT_ARTIFACT"
        f" op={recipe.op} format={recipe.fmt} board={board} family={recipe.family}"
        f" march={march} head={head}"
        f" fixture_sha256={_sha256_file(fixture)}"
        f" weft_opt_sha256={_sha256_file(weft_opt)}"
        f" weft_translate_sha256={_sha256_file(weft_translate)}"
        f" render_tool={render_tool}"
        f" render_tool_sha256={_sha256_file(render_tool)}"
        f" artifact_sha256={_sha256_bytes(source)}"
        f" symbol={recipe.expected_symbol}"
    )
    print("# CURRENT_ARTIFACT_COMMAND " + " ".join(command))


def _load_bench_routes():
    bench_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(bench_dir))
    loader = importlib.machinery.SourceFileLoader(
        "weft_bench_contract", str(bench_dir / "bench")
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise ExportError("cannot load canonical bench route registry")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module.CELL_ROUTES


def self_test() -> None:
    if len(RECIPES) != len(set(RECIPES)):
        raise ExportError("duplicate current-artifact recipe")
    route_keys = set()
    for route in _load_bench_routes():
        for fmt in route["formats"]:
            route_keys.add((route["op"], fmt))
            if (route["op"], fmt) not in RECIPES:
                raise ExportError(
                    "canonical bench route lacks current-artifact recipe: "
                    f"{route['op']}/{fmt}"
                )
    extra = set(RECIPES) - route_keys
    if extra:
        raise ExportError(f"unregistered current-artifact recipes: {sorted(extra)}")
    for recipe in RECIPES.values():
        if not (ROOT / recipe.fixture).is_file():
            raise ExportError(f"missing fixture: {recipe.fixture}")
        if not recipe.expected_symbol or not recipe.mode:
            raise ExportError(f"incomplete recipe: {recipe.op}/{recipe.fmt}")
    print(
        "current-artifact exporter self-test: canonical runner routes and "
        "production export recipes are bijective"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="export a canonical benchmark kernel from the current compiler"
    )
    parser.add_argument("op", nargs="?")
    parser.add_argument("format", nargs="?")
    parser.add_argument("--board")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(os.environ.get("WEFT_BUILD_DIR", DEFAULT_BUILD_DIR)),
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="fail unless HEAD is committed and the worktree is clean",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="use already-built tools (tests/development only)",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
        return 0
    if not args.op or not args.format or not args.board or args.output is None:
        parser.error("op, format, --board and --output are required")
    recipe = RECIPES.get((args.op, args.format))
    if recipe is None:
        parser.error(
            f"no current-artifact recipe for ({args.op}, {args.format})"
        )
    export_current_artifact(
        recipe,
        board=args.board,
        output=args.output.resolve(),
        build_dir=args.build_dir.resolve(),
        require_clean=args.require_clean,
        skip_build=args.skip_build,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExportError as exc:
        print(f"CURRENT_ARTIFACT_EXPORT_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(2)
