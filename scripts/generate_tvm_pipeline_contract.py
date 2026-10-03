"""Extract the default TIR contract from the configured TVM source."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def generate(source):
    pipeline_path = source / "python/tvm/tir/pipeline.py"
    transform_path = source / "python/tvm/tir/transform/transform.py"
    pipeline = ast.parse(pipeline_path.read_text())
    transforms = {n.name: n for n in ast.parse(transform_path.read_text()).body if isinstance(n, ast.FunctionDef)}
    defaults = {}
    passes = []

    def value(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not value(node.operand)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "bool":
            return bool(value(node.args[0]))
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "config.get":
            if len(node.args) != 2 or node.keywords:
                raise ValueError("unsupported config access")
            key, default = map(value, node.args)
            if key in defaults and defaults[key] != default:
                raise ValueError("inconsistent default")
            defaults[key] = default
            return default
        raise ValueError(f"unsupported expression: {ast.unparse(node)}")

    def add(node, enabled):
        if not isinstance(node, ast.Call) or not ast.unparse(node.func).startswith("tir.transform."):
            raise ValueError("unsupported pass expression")
        name = node.func.attr
        function = transforms[name]
        parameters = function.args.args
        args = [value(n) for n in node.args]
        keywords = {n.arg: value(n.value) for n in node.keywords}
        first_default = len(parameters) - len(function.args.defaults)
        for i in range(len(args), len(parameters)):
            parameter = parameters[i].arg
            if parameter in keywords:
                args.append(keywords.pop(parameter))
            elif i >= first_default:
                args.append(value(function.args.defaults[i - first_default]))
            else:
                raise ValueError(f"missing argument: {name}.{parameter}")
        if keywords or function.args.vararg or function.args.kwarg or function.args.kwonlyargs or len(node.args) > len(parameters):
            raise ValueError("unsupported pass signature")
        if enabled:
            passes.append({"name": f"tir.transform.{name}", "args": [json.dumps(a) if not isinstance(a, str) else a for a in args]})

    def walk(statements, enabled=True):
        for node in statements:
            if isinstance(node, ast.If):
                selected = value(node.test)
                walk(node.body, enabled and selected)
                walk(node.orelse, enabled and not selected)
            elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "passes" for t in node.targets):
                for item in node.value.elts:
                    add(item, enabled)
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and ast.unparse(node.value.func.value) == "passes":
                call = node.value
                if call.func.attr == "append":
                    add(call.args[0], enabled)
                elif call.func.attr == "extend":
                    for item in call.args[0].elts:
                        add(item, enabled)
                else:
                    raise ValueError("unsupported pass mutation")
            elif isinstance(node, ast.Assign) and ast.unparse(node) in {
                "pass_ctx = tvm.transform.PassContext.current()",
                "config = pass_ctx.config",
                "mod = tvm.ir.transform.Sequential(passes)(mod)",
            }:
                continue
            elif isinstance(node, ast.Return) and ast.unparse(node.value) == "mod":
                continue
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            else:
                raise ValueError(f"unsupported pipeline statement: {ast.unparse(node)}")

    outer = next(n for n in pipeline.body if isinstance(n, ast.FunctionDef) and n.name == "default_tir_pipeline")
    inner = next(n for n in outer.body if isinstance(n, ast.FunctionDef))
    walk(inner.body)
    digest = hashlib.sha256(pipeline_path.read_bytes() + transform_path.read_bytes()).hexdigest()
    return {"source_digest": digest, "defaults": defaults, "passes": passes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    options = parser.parse_args()
    options.output.write_text(json.dumps(generate(options.source), sort_keys=True) + "\n")
