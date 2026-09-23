"""Manage the bnlearn Agent Skill."""

from pathlib import Path
import argparse
import shutil


KNOWN_HARNESSES = {
    "claude": ".claude",
    "opencode": ".opencode",
    "agents": ".agents",
}


def skill_path():
    """Return the path to the bundled bnlearn Agent Skill."""
    return Path(__file__).resolve().parent / "skills" / "bnlearn"


def install_skill(harness="claude"):
    """Install the bnlearn Agent Skill into the current project.

    Parameters
    ----------
    harness : str, default='claude'
        Name of the AI coding harness. The skill is installed to:

            ./.<harness>/skills/bnlearn/

        Any harness name is accepted.
    """
    source = skill_path()

    # Keep the harness generic. A leading dot is added automatically.
    harness = harness.lstrip(".")
    destination = Path.cwd() / f".{harness}" / "skills" / "bnlearn"

    if harness not in KNOWN_HARNESSES:
        print(
            f"Warning: '{harness}' is not a known AI harness. "
            f"Installing anyway to:\n{destination}"
        )

    if not source.exists():
        raise FileNotFoundError(
            f"Bundled bnlearn skill not found: {source}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        shutil.rmtree(destination)

    shutil.copytree(source, destination)

    print(f"bnlearn skill installed to:\n{destination}")


def main():
    """Command-line interface for bnlearn."""
    parser = argparse.ArgumentParser(
        prog="bnlearn",
        description="bnlearn command-line interface.",
    )

    subparsers = parser.add_subparsers(dest="command")

    # ------------------------------------------------------------------
    # bnlearn install ...
    # ------------------------------------------------------------------
    install_parser = subparsers.add_parser(
        "install",
        help="Install bnlearn components.",
    )

    install_subparsers = install_parser.add_subparsers(
        dest="install_command",
    )

    skill_install_parser = install_subparsers.add_parser(
        "skill",
        help="Install the bnlearn Agent Skill.",
    )

    skill_install_parser.add_argument(
        "--harness",
        default="claude",
        help="AI coding harness name (default: claude).",
    )

    # ------------------------------------------------------------------
    # bnlearn skill ...
    # ------------------------------------------------------------------
    skill_parser = subparsers.add_parser(
        "skill",
        help="Manage the bnlearn Agent Skill.",
    )

    skill_subparsers = skill_parser.add_subparsers(
        dest="skill_command",
    )

    skill_subparsers.add_parser(
        "path",
        help="Show the path to the bundled bnlearn Agent Skill.",
    )

    args = parser.parse_args()

    # bnlearn install skill
    if args.command == "install":
        if args.install_command == "skill":
            install_skill(harness=args.harness)
        else:
            install_parser.print_help()

    # bnlearn skill path
    elif args.command == "skill":
        if args.skill_command == "path":
            print(skill_path())
        else:
            skill_parser.print_help()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()