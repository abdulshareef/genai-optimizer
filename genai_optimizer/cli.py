"""Command line interface.

    genai-optimize prompt "your long prompt here"
    genai-optimize prompt --file brief.txt --level aggressive --json
    genai-optimize output --file answer.txt
    genai-optimize models --provider anthropic
    genai-optimize serve --port 8088
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .engine import OptimizationEngine
from .pricing import PRICES_UPDATED, format_usd, list_models


def _read(args) -> str:
    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            return fh.read()
    if args.text:
        return " ".join(args.text)
    return sys.stdin.read()


def cmd_prompt(args) -> int:
    engine = OptimizationEngine(level=args.level, security_mode=args.security,
                                priority=args.priority,
                                select_model=not args.no_select,
                                allowed_providers=args.providers)
    result = engine.optimize_prompt(_read(args))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(result.text)
        print("\n--- report ---")
        print(result.summary())
        for finding in result.security.findings:
            print("  ! %-9s %-16s %s" % (finding.category, finding.rule, finding.excerpt))
    return 0


def cmd_output(args) -> int:
    engine = OptimizationEngine(output_level=args.level)
    result = engine.optimize_output(_read(args))
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(result.text)
        print("\n--- report ---")
        print(result.summary())
    return 0


def cmd_models(args) -> int:
    rows = list_models(args.provider)
    print("Prices last checked on %s. Always verify with the vendor.\n" % PRICES_UPDATED)
    print("%-30s %-10s %10s %10s %9s %8s" % ("MODEL", "PROVIDER", "IN /1M",
                                             "OUT /1M", "CONTEXT", "QUALITY"))
    for m in rows:
        print("%-30s %-10s %10s %10s %9s %8.1f" % (
            m.id, m.provider, format_usd(m.input_price), format_usd(m.output_price),
            "%dk" % (m.context // 1000), m.quality))
    return 0


def cmd_serve(args) -> int:
    from .server import serve
    serve(host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="genai-optimize",
                                description="Optimise prompts and answers for any Gen AI model")
    p.add_argument("--version", action="version", version="genai-optimizer " + __version__)
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("prompt", help="optimise a prompt")
    sp.add_argument("text", nargs="*")
    sp.add_argument("-f", "--file")
    sp.add_argument("-l", "--level", default="balanced",
                    choices=["off", "safe", "balanced", "aggressive"])
    sp.add_argument("-s", "--security", default="warn",
                    choices=["warn", "redact", "block"])
    sp.add_argument("-p", "--priority", default="balanced",
                    choices=["cost", "balanced", "quality", "speed"])
    sp.add_argument("--providers", nargs="*", default=None)
    sp.add_argument("--no-select", action="store_true", help="skip model selection")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_prompt)

    so = sub.add_parser("output", help="optimise a model answer")
    so.add_argument("text", nargs="*")
    so.add_argument("-f", "--file")
    so.add_argument("-l", "--level", default="balanced",
                    choices=["off", "safe", "balanced", "aggressive"])
    so.add_argument("--json", action="store_true")
    so.set_defaults(func=cmd_output)

    sm = sub.add_parser("models", help="show the model catalog")
    sm.add_argument("--provider")
    sm.set_defaults(func=cmd_models)

    ss = sub.add_parser("serve", help="run the HTTP microservice")
    ss.add_argument("--host", default="127.0.0.1")
    ss.add_argument("--port", type=int, default=8088)
    ss.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
