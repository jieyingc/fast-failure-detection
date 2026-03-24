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
    for _, b_path in touched_files:
        path = b_path.lower()
        if (
            "test" in path
            or "tests/" in path
            or path.endswith("_test.py")
            or "/test_" in path
        ):
            test_file_modified = 1
            break

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
        if isinstance(content, str):
            content_lower = content.lower()
            text_blobs.append(content_lower)
            message_text_parts.append(content_lower)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    t = item.get("text", "")
                    if isinstance(t, str):
                        t_lower = t.lower()
                        text_blobs.append(t_lower)
                        message_text_parts.append(t_lower)

        thought = msg.get("thought", "")
        if isinstance(thought, str):
            thought_lower = thought.lower()
            text_blobs.append(thought_lower)
            message_text_parts.append(thought_lower)

        text_blobs.append(action_text)
        joined_text = " ".join(text_blobs)

        for kw in error_keywords:
            num_error_messages += joined_text.count(kw)

    message_text_clean = " ".join(message_text_parts)

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
    }
