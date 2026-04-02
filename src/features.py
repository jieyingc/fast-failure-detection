import json
import re
import pandas as pd
from tqdm import tqdm
from .data_loading import extract_repo_id


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

    # Patch-level features
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
    file_paths = []
    path_tokens = []
    top_level_dirs = set()

    num_test_files_touched = 0
    num_python_files_touched = 0
    touch_requirements_file = 0
    touch_config_file = 0
    touch_docs_file = 0
    touch_init_file = 0

    config_keywords = {"config","setup.py","pyproject.toml","tox.ini","pytest.ini","mypy.ini",".flake8","dockerfile","makefile","cmakelists.txt","package.json",}

    docs_keywords = {"readme","docs/",".md",".rst",}

    for _, b_path in touched_files:
        path = b_path.lower()
        file_paths.append(path)

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

            # track top-level dirs
        if "/" in path:
            top_level_dirs.add(path.split("/")[0])
        else:
            top_level_dirs.add(path)

        # tokenize path for TF-IDF text
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

    # patch text for later TF-IDF
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

    function_names = []
    class_names = []
    for line in code_lines:
        m_func = re.search(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
        m_class = re.search(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if m_func:
            function_names.append(m_func.group(1).lower())
        if m_class:
            class_names.append(m_class.group(1).lower())

    function_name_text_clean = " ".join(function_names)
    class_name_text_clean = " ".join(class_names)

    import_names = []
    for line in code_lines:
        stripped = line.strip()

        # import x, y as z
        m_import = re.match(r"^import\s+(.+)$", stripped)
        if m_import:
            rhs = m_import.group(1)
            parts = [p.strip() for p in rhs.split(",")]

            for part in parts:
                # remove alias: "numpy as np" -> "numpy"
                part = re.sub(r"\s+as\s+\w+$", "", part).strip()
                if part:
                    import_names.append(part.lower())

                    # also split dotted module path: "a.b.c" -> a b c
                    dotted_parts = [x for x in part.split(".") if x]
                    import_names.extend([x.lower() for x in dotted_parts])

        # from x.y import a, b as c
        m_from = re.match(r"^from\s+([A-Za-z0-9_\.]+)\s+import\s+(.+)$", stripped)
        if m_from:
            module_name = m_from.group(1).strip()
            imported_rhs = m_from.group(2).strip()

            if module_name:
                import_names.append(module_name.lower())
                dotted_parts = [x for x in module_name.split(".") if x]
                import_names.extend([x.lower() for x in dotted_parts])

            imported_parts = [p.strip() for p in imported_rhs.split(",")]
            for part in imported_parts:
                part = re.sub(r"\s+as\s+\w+$", "", part).strip()
                if part and part != "*":
                    import_names.append(part.lower())

    import_name_text_clean = " ".join(import_names)

    # Message-level features
    num_messages = len(msgs)
    messages_char_len = len(row["messages"])

    num_assistant_messages = 0
    num_user_messages = 0
    num_tool_messages = 0
    num_system_messages = 0

    num_action_messages = 0
    num_observation_messages = 0

    has_submit = 0
    num_submit_calls = 0

    has_bash = 0
    num_bash_calls = 0

    has_str_replace_editor = 0
    num_edit_calls = 0
    num_error_messages = 0

    message_text_parts = []
    user_text_parts = []
    assistant_text_parts = []
    tool_text_parts = []
    tool_observation_text_parts = []

    edit_tool_names = {"str_replace_editor", "edit_file", "create_file", "insert", "replace", "write_file",}

    error_keywords = ["error", "exception", "traceback", "failed", "failure", "syntaxerror", "typeerror", "valueerror", "assertionerror",]

    for msg in msgs:
        role = msg.get("role", "")
        if role == "assistant":
            num_assistant_messages += 1
        elif role == "user":
            num_user_messages += 1
        elif role == "tool":
            num_tool_messages += 1
        elif role == "system":
            num_system_messages += 1

        message_type = msg.get("message_type", "")
        if message_type == "action":
            num_action_messages += 1
        elif message_type == "observation":
            num_observation_messages += 1

        action_text = str(msg.get("action", "")).lower()

        if "submit" in action_text:
            has_submit = 1
        if "bash" in action_text:
            has_bash = 1
        if "str_replace_editor" in action_text:
            has_str_replace_editor = 1

        tool_calls = msg.get("tool_calls", [])
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                fn_name = tc.get("function", {}).get("name", "") if isinstance(tc, dict) else ""
                fn_name = str(fn_name).lower()

                if fn_name == "submit":
                    has_submit = 1
                    num_submit_calls += 1
                if fn_name == "bash":
                    has_bash = 1
                    num_bash_calls += 1
                if fn_name == "str_replace_editor":
                    has_str_replace_editor = 1
                if fn_name in edit_tool_names:
                    num_edit_calls += 1

        text_blobs = []

        content = msg.get("content", "")

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
            text_blobs.append(content.lower())

        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    t = item.get("text", "")
                    if isinstance(t, str):
                        add_text_by_role(t, role, message_type)
                        text_blobs.append(t.lower())

        # thought is currently identical to content, so we do not add it to message_text_clean to avoid duplicated text in TF-IDF.
        # thought = msg.get("thought", "")
        #  if isinstance(thought, str):
        #     thought_lower = thought.lower()
        #     text_blobs.append(thought_lower)
        #     message_text_parts.append(thought_lower)

        text_blobs.append(action_text)
        joined_text = " ".join(text_blobs)

        for kw in error_keywords:
            num_error_messages += joined_text.count(kw)

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
        "num_assistant_messages": num_assistant_messages,
        "num_user_messages": num_user_messages,
        "num_tool_messages": num_tool_messages,
        "num_system_messages": num_system_messages,
        "num_action_messages": num_action_messages,
        "num_observation_messages": num_observation_messages,
        "has_submit": has_submit,
        "num_submit_calls": num_submit_calls,
        "has_bash": has_bash,
        "num_bash_calls": num_bash_calls,
        "has_str_replace_editor": has_str_replace_editor,
        "num_edit_calls": num_edit_calls,
        "num_error_messages": num_error_messages,

        "message_text_clean": message_text_clean,
        "user_text_clean": user_text_clean,
        "assistant_text_clean": assistant_text_clean,
        "tool_text_clean": tool_text_clean,
        "file_path_text_clean": file_path_text_clean,
        "patch_text_clean": patch_text_clean,
        "tool_observation_text_clean": tool_observation_text_clean,

        "contains_assert_change": contains_assert_change,
        "contains_import_change": contains_import_change,
        "contains_exception_handling_change": contains_exception_handling_change,

        "function_name_text_clean": function_name_text_clean,
        "class_name_text_clean": class_name_text_clean,
        "import_name_text_clean": import_name_text_clean,
    }
