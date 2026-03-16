"""计算器工具模块。
负责提供受限表达式计算能力，避免直接执行不安全代码。
当前阶段仅支持基础四则运算和括号，不负责高级数学函数与变量上下文。
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Mapping
from typing import Final

from langchain_core.tools import tool

ALLOWED_BINARY_OPERATORS: Final[Mapping[type[ast.operator], object]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
ALLOWED_UNARY_OPERATORS: Final[Mapping[type[ast.unaryop], object]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate_expression(node: ast.AST) -> float:
    """递归计算受限表达式语法树。"""

    if isinstance(node, ast.Expression):
        return _evaluate_expression(node.body)

    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)

    if isinstance(node, ast.BinOp):
        binary_operator = ALLOWED_BINARY_OPERATORS.get(type(node.op))
        if binary_operator is None:
            raise ValueError("当前表达式包含不支持的运算符。")
        left_value = _evaluate_expression(node.left)
        right_value = _evaluate_expression(node.right)
        return float(binary_operator(left_value, right_value))

    if isinstance(node, ast.UnaryOp):
        unary_operator = ALLOWED_UNARY_OPERATORS.get(type(node.op))
        if unary_operator is None:
            raise ValueError("当前表达式包含不支持的一元运算。")
        operand_value = _evaluate_expression(node.operand)
        return float(unary_operator(operand_value))

    raise ValueError("当前表达式包含不支持的语法。")


@tool("calculator")
def calculator_tool(expression: str) -> str:
    """计算基础数学表达式并返回字符串结果。"""

    normalized_expression = expression.strip()
    if not normalized_expression:
        raise ValueError("表达式不能为空。")

    parsed_expression = ast.parse(normalized_expression, mode="eval")
    result = _evaluate_expression(parsed_expression)

    if result.is_integer():
        return str(int(result))
    return str(result)
