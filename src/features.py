import json
import re

import pandas as pd
from tqdm import tqdm

from .data_loading import extract_repo_id


def normalize_command_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_test_command(command: str) -> bool:
    cmd = normalize_command_text(command)

    test_patterns = [r"\bpytest\b", r"\bpy\.test\b", r"\bpython\s+-m\s+pytest\b", r"\bpython\s+-m\s+unittest\b",
                     r"\bunittest\b", r"\bnosetests?\b", r"\btox\b", r"\bnox\b", r"\bmake\s+test\b",
                     r"\bmake\s+check\b", r"\bctest\b", r"\bnpm\s+test\b", r"\byarn\s+test\b", r"\bpnpm\s+test\b",
                     r"\bjest\b", r"\bvitest\b", r"\bgo\s+test\b", r"\bcargo\s+test\b",
                     ]
    return any(re.search(pat, cmd) for pat in test_patterns)


def extract_edit_file_path(tool_call: dict) -> str:
    """
    Try to recover edited file path from edit-related tool call arguments.
    Returns normalized lowercase path or empty string if unavailable.
    """
    if not isinstance(tool_call, dict):
        return ""

    fn = tool_call.get("function", {})
    args = fn.get("arguments", "")

    if not isinstance(args, str) or not args.strip():
        return ""

    patterns = [r'"path"\s*:\s*"([^"]+)"', r'"file_path"\s*:\s*"([^"]+)"', r'"filename"\s*:\s*"([^"]+)"',
                r'"target_file"\s*:\s*"([^"]+)"',
                ]

    for pat in patterns:
        m = re.search(pat, args)
        if m:
            return m.group(1).strip().lower()
    return ""


def build_feature_df(dataset_split):
    rows = []
    for i in tqdm(range(len(dataset_split))):
        rows.append(extract_features(dataset_split[i]))
    return pd.DataFrame(rows)


def extract_features(row):
    patch = row["patch"] if row["patch"] is not None else ""
    patch_stripped = patch.strip()

    try:
        msgs = json.loads(row["messages"])
    except Exception:
        msgs = []

    # -----------------------------
    # Patch-level features
    # -----------------------------
    patch_is_empty = int(patch_stripped == "")
    patch_char_len = len(patch)

    num_lines_added = 0
    num_lines_removed = 0
    for line in patch.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            continue
        if line.startswith("+"):
            num_lines_added += 1
        elif line.startswith("-"):
            num_lines_removed += 1

    num_files_touched = patch.count("diff --git")
    touched_files = re.findall(r"diff --git a/(.*?) b/(.*?)\n", patch)

    test_file_modified = 0
    path_tokens = []
    top_level_dirs = set()

    num_test_files_touched = 0
    num_python_files_touched = 0
    touch_requirements_file = 0
    touch_config_file = 0
    touch_docs_file = 0
    touch_init_file = 0

    config_keywords = {"config", "setup.py", "pyproject.toml", "tox.ini", "pytest.ini", "mypy.ini", ".flake8",
                       "dockerfile", "makefile", "cmakelists.txt", "package.json",
                       }
    docs_keywords = {"readme", "docs/", ".md", ".rst"}

    for _, b_path in touched_files:
        path = b_path.lower()

        if (
            "test" in path
            or "tests/" in path
            or path.endswith("_test.py")
            or "/test_" in path
        ):
            test_file_modified = 1
            num_test_files_touched += 1

        if path.endswith(".py"):
            num_python_files_touched += 1

        if "requirements" in path:
            touch_requirements_file = 1

        if any(k in path for k in config_keywords):
            touch_config_file = 1

        if any(k in path for k in docs_keywords):
            touch_docs_file = 1

        if path.endswith("__init__.py"):
            touch_init_file = 1

        if "/" in path:
            top_level_dirs.add(path.split("/")[0])
        else:
            top_level_dirs.add(path)

        parts = re.split(r"[\\/._-]+", path)
        parts = [p for p in parts if p]
        path_tokens.extend(parts)

    touch_multiple_dirs = int(len(top_level_dirs) >= 2)
    file_path_text_clean = " ".join(path_tokens)

    code_lines = []
    for line in patch.splitlines():
        if line.startswith("+++ ") or line.startswith("--- ") or line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            content = line[1:].strip()
            if content:
                code_lines.append(content)

    contain_repeated_blocks = 0
    if code_lines:
        line_freq = {}
        for x in code_lines:
            line_freq[x] = line_freq.get(x, 0) + 1
        if max(line_freq.values()) >= 3:
            contain_repeated_blocks = 1

    patch_tokens = []
    for line in code_lines:
        toks = re.split(r"[^a-zA-Z0-9_]+", line.lower())
        toks = [t for t in toks if t]
        patch_tokens.extend(toks)
    patch_text_clean = " ".join(patch_tokens)

    patch_text_lower = "\n".join(code_lines).lower()
    contains_assert_change = int("assert" in patch_text_lower)
    contains_import_change = int(
        any(
            line.lstrip().startswith("import ") or line.lstrip().startswith("from ")
            for line in code_lines
        )
    )
    exception_keywords = ["try:", "except", "raise ", "finally:", "exception"]
    contains_exception_handling_change = int(
        any(kw in patch_text_lower for kw in exception_keywords)
    )

    # -----------------------------
    # Message-level features
    # -----------------------------
    num_messages = len(msgs)
    messages_char_len = len(row["messages"])

    num_action_messages = 0
    num_observation_messages = 0

    num_submit_calls = 0
    num_bash_calls = 0
    num_edit_calls = 0

    message_text_parts = []
    user_text_parts = []
    assistant_text_parts = []
    tool_text_parts = []
    tool_observation_text_parts = []

    edit_tool_names = {
        "str_replace_editor",
        "edit_file",
        "create_file",
        "insert",
        "replace",
        "write_file",
    }

    for msg_idx, msg in enumerate(msgs):
        role = msg.get("role", "")
        message_type = msg.get("message_type", "")

        if message_type == "action":
            num_action_messages += 1
        elif message_type == "observation":
            num_observation_messages += 1

        content = msg.get("content", "")

        tool_calls = msg.get("tool_calls", [])
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                fn_name = tc.get("function", {}).get("name", "") if isinstance(tc, dict) else ""
                fn_name = str(fn_name).lower()

                if fn_name == "submit":
                    num_submit_calls += 1

                if fn_name == "bash":
                    num_bash_calls += 1

                if fn_name in edit_tool_names:
                    num_edit_calls += 1

        def add_text_by_role(text_value, role_value, message_type_value):
            text_value = text_value.lower()
            message_text_parts.append(text_value)

            if role_value == "user":
                user_text_parts.append(text_value)
            elif role_value == "assistant":
                assistant_text_parts.append(text_value)
            elif role_value == "tool":
                tool_text_parts.append(text_value)
                if message_type_value == "observation":
                    tool_observation_text_parts.append(text_value)

        if isinstance(content, str):
            add_text_by_role(content, role, message_type)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    t = item.get("text", "")
                    if isinstance(t, str):
                        add_text_by_role(t, role, message_type)

    message_text_clean = " ".join(message_text_parts)
    user_text_clean = " ".join(user_text_parts)
    assistant_text_clean = " ".join(assistant_text_parts)
    tool_text_clean = " ".join(tool_text_parts)
    tool_observation_text_clean = " ".join(tool_observation_text_parts)

    return {
        "instance_id": row["instance_id"],
        "traj_id": row["traj_id"],
        "repo_id": extract_repo_id(row["instance_id"]),
        "model": row["model"],
        "resolved": int(bool(row["resolved"])),

        "patch_is_empty": patch_is_empty,
        "patch_char_len": patch_char_len,
        "num_lines_added": num_lines_added,
        "num_lines_removed": num_lines_removed,
        "num_files_touched": num_files_touched,
        "test_file_modified": test_file_modified,
        "contain_repeated_blocks": contain_repeated_blocks,
        "num_test_files_touched": num_test_files_touched,
        "num_python_files_touched": num_python_files_touched,
        "touch_requirements_file": touch_requirements_file,
        "touch_config_file": touch_config_file,
        "touch_docs_file": touch_docs_file,
        "touch_init_file": touch_init_file,
        "touch_multiple_dirs": touch_multiple_dirs,

        "num_messages": num_messages,
        "messages_char_len": messages_char_len,
        "num_action_messages": num_action_messages,
        "num_observation_messages": num_observation_messages,
        "num_submit_calls": num_submit_calls,
        "num_bash_calls": num_bash_calls,
        "num_edit_calls": num_edit_calls,

        "message_text_clean": message_text_clean,
        "user_text_clean": user_text_clean,
        "assistant_text_clean": assistant_text_clean,
        "tool_text_clean": tool_text_clean,
        "tool_observation_text_clean": tool_observation_text_clean,
        "file_path_text_clean": file_path_text_clean,
        "patch_text_clean": patch_text_clean,

        "contains_assert_change": contains_assert_change,
        "contains_import_change": contains_import_change,
        "contains_exception_handling_change": contains_exception_handling_change,
    }
