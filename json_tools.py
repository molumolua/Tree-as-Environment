# JSON 合法转义：\" \\ \/ \b \f \n \r \t \uXXXX；LaTeX 常用 \vec \langle \rangle 等是非法转义，需在解析前修复
from typing import List

_VALID_JSON_ESCAPE_NEXT = frozenset('"\\/bfnrtu')
def _fix_invalid_json_escapes(text: str) -> str:
    """
    在 JSON 字符串值内部，将非法反斜杠转义（如 \\vec 中的 \\v）多插一个 \\，
    使 \\v 变为 \\\\v，标准 JSON 解析器即可通过。只处理双引号字符串内的内容。
    """
    result: List[str] = []
    i = 0
    in_string = False
    escape_next = False
    while i < len(text):
        c = text[i]
        if escape_next:
            escape_next = False
            if in_string and c not in _VALID_JSON_ESCAPE_NEXT:
                result.append("\\")  # 非法转义：多插一个 \，使 \v 变为 \\v，解析后得到字面 \v
                result.append(c)
            elif in_string and c == "u":
                result.append(c)
                # 吞掉 \uXXXX 的 4 个十六进制字符
                for _ in range(4):
                    i += 1
                    if i < len(text) and text[i] in "0123456789aAbBcCdDeEfF":
                        result.append(text[i])
                i += 1
                continue
            else:
                result.append(c)
            i += 1
            continue
        if c == "\\" and in_string:
            escape_next = True
            result.append(c)
            i += 1
            continue
        if c == '"' and (i == 0 or text[i - 1] != "\\" or (i >= 2 and text[i - 2] == "\\")):
            # 简单判断：非转义的双引号切换 in_string（注意连续 \\ 结尾时 " 可能不转义）
            in_string = not in_string
            result.append(c)
            i += 1
            continue
        result.append(c)
        i += 1
    return "".join(result)