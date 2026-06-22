import argparse
import sys
from importlib.metadata import entry_points

from worldscore.benchmark.utils.utils import type2model
from worldscore.common.utils import print_banner


def show_model_list():
    """Display available models in type2model"""
    print("\nAvailable models:")
    print(
        "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
    )
    for model_type, model_names in type2model.items():
        print(f"Model Type: {model_type}")
        print(f"  Model Name: {model_names}")
        print(
            "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
        )


def list_commands():
    """List all available worldscore commands"""
    print("Available WorldScore commands:")
    print("\nUsage: <command> [options]")
    print("\nCommands:")

    eps = entry_points(group="console_scripts")
    for ep in eps:
        if ep.name.startswith("worldscore-"):
            command = ep.name
            print(f"  {command:<15} - WorldScore {command} command")


def main():
    print_banner("CLI")

    if len(sys.argv) >= 2 and (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
        list_commands()
        show_model_list()
        return

    print(
        "No command provided. Use '--help' or '-h' to see available commands and models."
    )
    return


if __name__ == "__main__":
    main()
