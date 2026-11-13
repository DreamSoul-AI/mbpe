"""Configuration preview and the synthetic-grid run command."""

import argparse
import json

from ..structure.config import DEFAULT_EXPERIMENT_SEEDS, TokenizerConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description="MBPE research setup")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="print tokenizer configuration without running it")
    plan.add_argument("--patch-size", nargs="+", type=int, default=[1, 2, 2])
    plan.add_argument("--batch-size", type=int, default=4)
    run = commands.add_parser("run", help="train and encode the synthetic manifest")
    run.add_argument("--patch-size", nargs="+", type=int, default=[1, 2, 2])
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_EXPERIMENT_SEEDS))
    run.add_argument("--max-symbols", type=int, default=None)
    run.add_argument("--manifest-version", type=int, default=1)
    run.add_argument("--grid-name", default=None)
    run.add_argument("--output", default=None)
    controls = commands.add_parser("controls", help="rerun the three tile-grid comparisons")
    controls.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_EXPERIMENT_SEEDS))
    controls.add_argument("--max-symbols", type=int, default=None)
    controls.add_argument("--manifest-version", type=int, default=2)
    controls.add_argument("--grid-name", default="tile")
    controls.add_argument("--output", required=True)
    controls.add_argument("--diagnose", action="store_true",
                          help="record the first count-versus-spatial fork at symbol cap 280")
    args = parser.parse_args(argv)
    if args.command == "plan":
        return _preview(parser, args)
    if args.command == "controls":
        return _controls(parser, args)
    return _run(parser, args)


def _preview(parser, args):
    try:
        config = TokenizerConfig(max_codeword_size=tuple(args.patch_size), batch_size=args.batch_size)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"action": "preview", "config": config.to_dict(),
                      "tokenizer_kwargs": config.tokenizer_kwargs()}, indent=2))
    return 0


def _run(parser, args):
    from .run import default_output_directory, execute

    output = args.output if args.output is not None else default_output_directory()
    try:
        return execute(output, seeds=tuple(args.seeds), manifest_version=args.manifest_version,
                       max_codeword_size=tuple(args.patch_size), batch_size=args.batch_size,
                       max_symbols=args.max_symbols, grid_name=args.grid_name)
    except ValueError as error:
        parser.error(str(error))


def _controls(parser, args):
    from .tile_controls import execute_controls, execute_diagnose

    try:
        if args.diagnose:
            return execute_diagnose(args.output, seeds=tuple(args.seeds), manifest_version=args.manifest_version,
                                    grid_name=args.grid_name)
        return execute_controls(args.output, seeds=tuple(args.seeds), manifest_version=args.manifest_version,
                                grid_name=args.grid_name, max_symbols=args.max_symbols)
    except ValueError as error:
        parser.error(str(error))
