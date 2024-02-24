#!python
# %pip install -qU aider-chat intervaltree rope llama-index diff-match-patch httpx httpcore

import argparse
import os
import sys
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap

import intervaltree as it
import tree_sitter as ts
from aider import models
from aider.dump import dump
from aider.io import InputOutput
from aider.repomap import RepoMap
from diff_match_patch import diff_match_patch
from grep_ast import filename_to_lang
from llama_index.agent import OpenAIAgent
from llama_index.bridge.pydantic import Field
from llama_index.llms import OpenAI
from llama_index.tools import FunctionTool
from llama_index.tools.utils import create_schema_from_function
from rope.base.change import Change, ChangeContents, ChangeSet
from rope.base.codeanalyze import SourceLinesAdapter
from rope.base.exceptions import RopeError
from rope.base.project import Project
from rope.refactor.extract import ExtractMethod
from tree_sitter_languages import get_language, get_parser


def add_function_bodies(node: ts.Node, body_lines: it.IntervalTree):
    if node is None:
        return
    if "function_definition" in node.type:
        for body in node.children_by_field_name("body"):
            start_line = body.start_point[0]
            limit_line = body.end_point[0] + 1
            body_lines.addi(start_line, limit_line)
    for child in node.children:
        add_function_bodies(child, body_lines)
    return


def add_function_definitions(node: ts.Node, definition_lines: it.IntervalTree):
    if node is None:
        return
    if "function_definition" in node.type:
        for child in node.children:
            if child.type == "body":
                break
            start_line = child.start_point[0]
            limit_line = child.end_point[0] + 1
            definition_lines.addi(start_line, limit_line)
            if child.type == ":":
                break
    for child in node.children:
        add_function_definitions(child, definition_lines)
    return


def enumerate_extract_begin_lines(node: ts.Node) -> it.IntervalTree:
    extract_begin_lines = it.IntervalTree()
    add_function_bodies(node, extract_begin_lines)
    extract_begin_lines.merge_overlaps()
    definition_lines = it.IntervalTree()
    add_function_definitions(node, definition_lines)
    extract_begin_lines.difference_update(definition_lines)
    return extract_begin_lines


Ref = namedtuple("Ref", ["tag", "name", "line", "node"])


@dataclass()
class CodeIntervalInfo:
    node: ts.Node
    token_count: int


@dataclass()
class CodeTreeInfo:
    rm: RepoMap
    lang: str
    fname: str
    lines: list[str]
    token_counts: list[int]
    tree: ts.Tree
    interval_tree_map: it.IntervalTree
    file_mtime: float | None = None
    valid_begin_lines: it.IntervalTree = None
    encoding: str = "utf-8"
    number_separator: str = "|"

    def __init__(self, rm: RepoMap, fname: str, lazy: bool = False):
        self.rm = rm
        self.lang = filename_to_lang(fname)
        self.fname = fname
        if not lazy:
            self.load(force=True)

    def load(self, force=False):
        global code_info_project, code_info_resource, rope_lines
        current_mtime = self.rm.get_mtime(self.fname)
        if current_mtime is None:
            raise ValueError(f"File {self.fname} not found")
        if not force and self.file_mtime and (current_mtime < self.file_mtime):
            if self.rm.verbose:
                self.rm.io.tool_output(f"File {self.fname} not modified")
            return
        code = self.rm.io.read_text(self.fname)
        if not code:
            raise ValueError(f"File {self.fname} is empty")
        self.lines = code.splitlines()
        self.token_counts = [self.rm.token_count(line) for line in self.lines]
        self.interval_tree_map = it.IntervalTree()
        parser = get_parser(self.lang)
        self.tree = parser.parse(bytes(code, "utf-8"))
        self.add_definitions_to_map(self.tree.root_node)
        self.file_mtime = current_mtime
        code_info_project = Project(self.rm.root)
        code_info_resource = code_info_project.get_resource(self.fname)
        rope_lines = SourceLinesAdapter(code_info_resource.read())

        self.valid_begin_lines = enumerate_extract_begin_lines(self.tree.root_node)

    def add_definitions_to_map(self, node: ts.Node):
        if node is None:
            return
        if "definition" in node.type:
            start_line = node.start_point[0]
            limit_line = node.end_point[0] + 1
            info = CodeIntervalInfo(node, sum(self.token_counts[start_line:limit_line]))
            self.interval_tree_map.add(it.Interval(start_line, limit_line, info))
        for child in node.children:
            self.add_definitions_to_map(child)

    def list_references(self):
        # Load the tags queries
        scm_fname = os.path.join("aider/queries", f"tree-sitter-{self.lang}-tags.scm")
        query_scm = Path(scm_fname)
        if not query_scm.exists():
            if not self.rm.verbose:
                self.rm.io.tool_output(f"No tags query file found for {self.lang}: {scm_fname}")
            return
        query_scm = query_scm.read_text()

        # Run the tags queries
        language = get_language(self.lang)
        query = language.query(query_scm)
        captures = query.captures(self.tree.root_node)

        captures = list(captures)

        for node, tag in captures:
            if "name.reference" not in tag:
                continue

            yield Ref(tag=tag, name=node.text.decode("utf-8"), line=node.start_point[0], node=node)

    def numbered_lines(
        self, included_lines: it.IntervalTree = None, start_line: int = 0, limit_line: int = None
    ) -> str:
        if limit_line is None:
            limit_line = len(self.lines)
        if included_lines is None:
            included_lines = it.IntervalTree()
            included_lines.addi(start_line, limit_line)
        return "\n".join([
            (f"{i:04d}:{line}" if included_lines.overlaps(i) else f"----:{line}")
            for i, line in enumerate(self.lines[start_line:limit_line], start_line)
        ])


def code_info_init(root: str, fname: str):
    global rm, code_info, code_info_changes_list
    rm = RepoMap(root=root, io=InputOutput(), main_model=models.Model.create("gpt-4-1106-preview"))
    code_info = CodeTreeInfo(rm=rm, fname=fname)
    code_info_changes_list = []
    return code_info


def get_code_info_project() -> Project:
    global code_info_project
    return code_info_project


def get_code_info_resource():
    global code_info_resource
    return code_info_resource


def get_code_info_changes_list() -> list[Change]:
    global code_info_changes_list
    return code_info_changes_list


def print_identifiers(node):
    if node is None:
        return
    if "identifier" in node.type:
        print(f"identifier: {node.text.decode(code_info.encoding)}")
    for child in node.children:
        print_identifiers(child)


def wraplines(lines: str, width: int = 80):
    return "\n".join(["\n".join(wrap(line, width)) for line in lines.splitlines()])


def print_node(node: ts.Node, indent: int = 0):
    print(f"{' '*indent}{node.type} {node.start_point}..{node.end_point}")
    for child in node.children:
        print_node(child, indent=indent + 2)


def extract_method_problems(code_info: CodeTreeInfo, fname: str, begin_line: int, end_line: int):
    limit_line = end_line + 1
    if fname not in code_info.fname:
        code_info.rm.io.tool_output(f"{fname} not in {code_info.fname}")
    for node in code_info.interval_tree_map.overlap(begin_line, limit_line):
        if node.data.node.type != "function_definition":
            continue
        if not code_info.valid_begin_lines.overlaps(begin_line):
            return "Extraction must begin at a valid (i.e. numbered) line."
        if not code_info.valid_begin_lines.overlaps(end_line):
            return "Extraction must end at a valid (i.e. numbered) line."
        if node.data.node.start_point[0] > end_line:
            continue
        if node.data.node.end_point[0] < begin_line:
            continue
        if node.data.node.start_point[0] < begin_line:
            if node.data.node.end_point[0] < end_line:
                return f"Extraction range invalid.  Begins ({begin_line}) inside body but end ({end_line}) is beyond the function's end ({node.data.node.start_point[0]}..{node.data.node.end_point[0]})."
        extraction_line_count = end_line - begin_line + 1
        body_line_count = node.data.node.end_point[0] - node.data.node.start_point[0] + 1
        if extraction_line_count == body_line_count:
            return "Extracting whole body."
        if extraction_line_count / body_line_count > 0.75:
            return "Extracting more than 75% of function body."
        if extraction_line_count > body_line_count - 2:
            return "Extracting too much of function body."
        return None
    return f"Extraction range ({begin_line}..{end_line}) is not within any function definition."


def line_range_to_rope_offset(start, end):
    global rope_lines
    return rope_lines.get_line_start(start + 1), rope_lines.get_line_end(end + 1)


# @trace_method
def extract_method_fn(
    file_path: str = Field(description="Path to the file to extract the method from"),
    begin_line: int = Field(description="Number of first line to extract"),
    end_line: int = Field(description="Number of last line to extract"),
    new_function_name: str = Field(description="Name for the extracted method"),
    replace_similar: bool = Field(
        default=True, description="Replace similar code with a call to the extracted method"
    ),
    global_def: bool = Field(default=False, description="Extract as a global function"),
):
    print(f"Extract method {new_function_name} from {file_path} lines {begin_line}..{end_line}")
    global code_info, code_info_project, code_info_resource, code_info_changes_list
    code_info.rm.io.tool_output(
        f"Extract method {new_function_name} from {file_path} lines {begin_line}..{end_line}"
    )
    problems = extract_method_problems(code_info, file_path, begin_line, end_line)
    if problems:
        code_info.rm.io.tool_output(f"Error: {problems}")
        return f"ERROR: {problems}.  Your extraction selection should be a portion of the function body with a narrower purpose than the existing function as a whole."
    try:
        begin_offset, end_offset = line_range_to_rope_offset(begin_line, end_line)
        extractor = ExtractMethod(code_info_project, code_info_resource, begin_offset, end_offset)
        changes = extractor.get_changes(
            new_function_name, similar=replace_similar, global_=global_def
        )
        dump(changes.get_description())
        code_info_changes_list.append(changes)
        print(f"Extracted method {new_function_name}.")
        return f"Extracted method {new_function_name}."
    except RopeError as e:
        code_info.rm.io.tool_output(f"Error: {e}")
        return f"ERROR: {e}. Note that a valid extraction must begin within a method body and end on the same syntactic level (i.e. same indentation level for Python) as the beginning."
    raise Exception("Shouldn't get here")

# extract_tool_description = """Use this function to refactor by extracting a new method from a range of lines of code that are part of an existing method.
# The lines of code to be extracted must be a syntactically contiguous block.
# IOW the lines must be a single statement or a single block of statements (IOW cannot be at different indentation levels).
# DO NOT simply extract the whole body of a function.  That would be a rename and is not supported by this tool which is for extracting a portion of a method body.
# Note that an extraction can never begin at the line that defines the existing function's name.
# The size of the extraction is also limited to less than 75% of the existing function's body."""


extract_tool_description = """Use this function to refactor the code by extracting a new method from a range of lines of code."""

extract_method_tool = FunctionTool.from_defaults(
    fn=extract_method_fn,
    name="extract_method",
    description=extract_tool_description,
    fn_schema=create_schema_from_function("extract_method_schema", extract_method_fn),
)


def insert_comment_fn(
    file_path: str = Field(description="Path to the file to extract the method from"),
    line: int = Field(description="Line number to insert the comment at"),
    comment: str = Field(description="The comment to insert"),
):
    global code_info
    code_info.rm.io.tool_output(f"Comment at {file_path} line {line}: {comment}")
    return f"Inserted comment at {line}."


INSERT_COMMENT_TOOL = FunctionTool.from_defaults(
    fn=insert_comment_fn,
    name="insert_comment",
    description="""Use this function to insert explanatory comments into the code.""",
    fn_schema=create_schema_from_function("insert_comment_schema", insert_comment_fn),
)

# swe_instructions = """You are an expert software engineer.
# Examine the code and refactor when you find worthwhile opportunities.
# The code lines are prefixed by their lines numbers.  Refactoring tool line number parameters can only have values for numbered lines.
# A refactoring should serve a useful purpose in making the code more readable.
# Be careful to consider the syntactic boundary conditions when refactoring.
# A method extraction can never begin at the first line of the function's definition itself (IOW the 'def foo(...) itself').
# Don't simply extract the whole body of a method into a new method.
# It can help to think in terms of "code paragraphs" when deciding what to extract.
# Other concepts to consider in deciding what code to keep or remove in a particular function is
# conceptual level of abstraction and the Single Responsibility Principle.
# """

# agent = OpenAIAssistantAgent.from_new(
#     name="Software Engineering Assistant",
#     instructions=swe_instructions,
#     tools=[extract_method_tool],
#     verbose=True,
#     run_retrieve_sleep_time=1.0,
# )

# swe_instructions = """You are an expert software engineer starting work on a legacy codebase.
# Examine this code which has line number prefixes for easy reference and insert comments where you think they are needed to improve clarity and understanding.
# A helpful concept to apply is the "code paragraph" which is a few statements that combine to serve a single purpose.
# """

# agent = OpenAIAssistantAgent.from_new(
#     name="Software Engineering Assistant",
#     instructions=swe_instructions,
#     tools=[insert_comment_tool],
#     verbose=True,
#     run_retrieve_sleep_time=1.0,
# )

swe_instructions = """
You are a refactoring expert, specializing in extracting code paragraphs from long methods.

In the provided Python code, extract method opportunities using the refactoring tool.

# Criteria

Look at the biggest methods first and find a reasonably sized chunk to pull out.

DO NOT extract an entire method body.

If you don't see any actions that improve the code, you can also respond `nothing important to do here`

# Process
Please do at most 5."""


def run_agent(swe_instructions, streaming: bool = True):
    global extract_method_tool
    llm = OpenAI(model="gpt-4-1106-preview")
    # Default max_function_calls is 5.
    agent = OpenAIAgent.from_tools(llm=llm, tools=[extract_method_tool], verbose=True)

    prompt = (
        swe_instructions
        + "\n\n"
        + code_info.fname
        + "\n"
        + code_info.numbered_lines(code_info.valid_begin_lines)
    )
    print(prompt)

    code_info_changes_list = []

    if streaming:
        response = agent.stream_chat(prompt)
        response_gen = response.response_gen

        for token in response_gen:
            print(token, end="")
    else:
        response = agent.chat(prompt)

    print(wraplines(str(response), width=120))
    return code_info_changes_list


def merge(v0: str, a: str, b: str) -> str:
    # BillCompton
    # Mar 29, 2011, 2:26:47 AM
    # to Diff Match Patch
    # I need a three-way merge. That is, given three versions of a doc, V0,
    # V1, and V2, where V1 and V2 are changes from V0, merge V1 and V2 into
    # V3, including detecting and marking "conflicts" (sections changed by
    # both V1 and V2).
    #
    # Does Google Diff Match Patch already provide this? If not, is Google
    # Diff Match Patch a good starting-point for implementing this?
    #
    # Thanks in advance!
    # Neil Fraser
    # Mar 29, 2011, 11:14:09 AM
    # to Diff Match Patch
    # Yes, this is what DMP was originally built for. Here's the
    # pseudocode:
    # The result list is an array of true/false values. If it's all true,
    # then the merge worked great. If there's a false, then one of the
    # patches could not be applied. In that case it might be worth swapping
    # V1 and V2, trying again and seeing if the results are better.

    dmp = diff_match_patch()
    patches = dmp.patch_make(v0, a)
    (v3, result) = dmp.patch_apply(patches, b)
    if not all(result):
        print("First try at merge failed.  Swapping V1 and V2 and trying again.")
        patches = dmp.patch_make(v0, b)
        (v3, result) = dmp.patch_apply(patches, a)
        if not all(result):
            print("*** MERGE FAILED!!! ***")
    return v3


def merge_change_list(resource, change_list: list[Change]) -> Change:
    source = resource.read()
    base = source
    for change in change_list:
        if isinstance(change, ChangeSet):
            source = merge_changes(resource, base, source, change.changes)
        elif isinstance(change, ChangeContents):
            source = merge(base, source, change.new_contents)
    return ChangeContents(resource, source)


def merge_changes(resource, base: str, source: str, change_list: list[Change]) -> str:
    for change in change_list:
        if isinstance(change, ChangeSet):
            source = merge_changes(resource, base, source, change.changes)
        elif isinstance(change, ChangeContents):
            source = merge(base, source, change.new_contents)
    return source


def main(cli_args):
    global code_info, code_info_project, code_info_resource, code_info_changes_list, rope_lines

    if "OPENAI_API_KEY" not in os.environ:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)

    # Create the parser
    parser = argparse.ArgumentParser(description="Process some files.")

    # Add the arguments
    parser.add_argument("file", type=str, help="The file to refactor")
    parser.add_argument(
        "--repo_dir",
        type=str,
        default=os.getcwd(),
        help="The directory to process (defaults to current directory)",
    )
    parser.add_argument("--no_streaming", action="store_true", help="Disable streaming")
    parser.add_argument(
        "--no_commit", action="store_true", help="Disable commit (i.e. update file)"
    )
    parser.add_argument("--no_undo", action="store_true", help="Disable undo commit")

    args = parser.parse_args(args=cli_args)

    fname = args.file
    repo_dir = args.repo_dir

    # Check if the file exists
    if not os.path.isfile(fname):
        print(f"The file {fname} does not exist.")
        exit(1)

    # Check if the directory exists
    if not os.path.isdir(repo_dir):
        print(f"The directory {repo_dir} does not exist.")
        exit(1)

    code_info_init(root=repo_dir, fname=fname)

    run_agent(swe_instructions, streaming=not args.no_streaming)

    if code_info_changes_list:
        if len(code_info_changes_list) == 1:
            the_change = code_info_changes_list[0]
        else:
            the_change = merge_change_list(code_info_resource, code_info_changes_list)

        print(the_change.get_description())

        if not args.no_commit:
            print("File updated.")
            code_info_project.do(the_change)
            if not args.no_undo:
                input("Press Enter to undo commit.")
                code_info_project.history.undo()
    else:
        print("No changes to commit")


if __name__ == "__main__":
    main(sys.argv[1:])
