

from aider.dump import dump

from rope.refactor.extract import ExtractVariable, ExtractMethod
from rope.base.change import Change, ChangeSet, ChangeContents

from extract import code_info_init, get_code_info_project, get_code_info_resource, get_code_info_changes_list
from extract import line_range_to_rope_offset
from extract import merge_change_list


repo_dir = '.'
fname = 'statement.py'


def extract_method(file_path, begin_line, end_line, new_function_name):
    begin_offset, end_offset = line_range_to_rope_offset(begin_line, end_line)
    extractor = ExtractMethod(get_code_info_project(), get_code_info_resource(), begin_offset, end_offset)
    changes = extractor.get_changes(new_function_name, similar=True, global_=False)
    dump(changes.get_description())
    return changes


def test_merge_two_changes_fail():
    code_info_init(root=repo_dir, fname=fname)
    code_info_changes_list: list[Change] = []

    # extract_method with args: {"file_path": "statement.py", "begin_line": 20, "end_line": 30, "new_function_name": "calculate_amount"}
    code_info_changes_list.append(extract_method("statement.py", 20, 30, "calculate_amount"))

    # extract_method with args: {"file_path": "statement.py", "begin_line": 32, "end_line": 35, "new_function_name": "calculate_volume_credits"}
    code_info_changes_list.append(extract_method("statement.py", 32, 35, "calculate_volume_credits"))

    dump(code_info_changes_list)

    the_change = merge_change_list(get_code_info_resource(), code_info_changes_list)

    dump(the_change.get_description())


def test_merge_two_changes_scope_mismatch():
    code_info_init(root=repo_dir, fname=fname)
    code_info_changes_list: list[Change] = []

    # Calling function: extract_method with args: {"file_path": "statement.py", "begin_line": 16, "end_line": 30, "new_function_name": "calculate_amount_for_performance"}
    code_info_changes_list.append(extract_method("statement.py", 16, 30, "calculate_amount_for_performance"))

    # Calling function: extract_method with args: {"file_path": "statement.py", "begin_line": 32, "end_line": 35, "new_function_name": "calculate_volume_credits"}
    code_info_changes_list.append(extract_method("statement.py", 32, 35, "calculate_volume_credits"))

    dump(code_info_changes_list)

    the_change = merge_change_list(get_code_info_resource(), code_info_changes_list)

    dump(the_change.get_description())


if __name__ == '__main__':
    test_merge_two_changes_fail()
    test_merge_two_changes_scope_mismatch()
