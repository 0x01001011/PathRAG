import argparse
import os

from . import PathRAG, QueryParam
from .llm import gpt_4o_mini_complete


def build_parser() -> argparse.ArgumentParser:
    """Create and return the CLI argument parser."""

    parser = argparse.ArgumentParser(description="PathRAG command line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    insert_p = subparsers.add_parser("insert", help="Insert documents")
    insert_group = insert_p.add_mutually_exclusive_group(required=True)
    insert_group.add_argument("--file", type=str, help="Path to a text file to insert")
    insert_group.add_argument("--directory", type=str, help="Directory containing text files")
    insert_p.add_argument("--working-dir", required=True, help="PathRAG working directory")

    query_p = subparsers.add_parser("query", help="Query the knowledge base")
    query_p.add_argument("--question", required=True, help="Question to ask")
    query_p.add_argument("--working-dir", required=True)
    query_p.add_argument("--mode", default="hybrid", help="Query mode")

    delete_p = subparsers.add_parser("delete", help="Delete an entity")
    delete_p.add_argument("--entity", required=True, help="Entity name")
    delete_p.add_argument("--working-dir", required=True)

    return parser


def load_files_from_dir(directory: str) -> list[str]:
    """Load all ``.txt`` files from ``directory`` and return their contents."""

    files = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and path.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                files.append(f.read())
    return files


def main() -> None:
    """Entry point for the PathRAG CLI."""

    parser = build_parser()
    args = parser.parse_args()

    rag = PathRAG(working_dir=args.working_dir, llm_model_func=gpt_4o_mini_complete)

    if args.command == "insert":
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                rag.insert(f.read())
        else:
            texts = load_files_from_dir(args.directory)
            for text in texts:
                rag.insert(text)
    elif args.command == "query":
        param = QueryParam(mode=args.mode)
        print(rag.query(args.question, param=param))
    elif args.command == "delete":
        rag.delete_by_entity(args.entity)


if __name__ == "__main__":
    main()
