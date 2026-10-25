"""Portable run metadata and explicit checkpoint/output-directory handling."""

from pathlib import Path
import os
import shutil


def metadata_for_save(value, directory):
    """Store paths relative to the JSON file's directory; leave other values alone."""
    if isinstance(value, dict):
        return {key: metadata_for_save(item, directory) for key, item in value.items()
                if key != "_overwrite"}
    if isinstance(value, (list, tuple)):
        return [metadata_for_save(item, directory) for item in value]
    if isinstance(value, (str, os.PathLike)) and os.path.isabs(value):
        return Path(os.path.relpath(value, directory)).as_posix()
    return value


def metadata_from_disk(value, directory, key=""):
    """Resolve path fields against the containing JSON directory, never the CWD."""
    if isinstance(value, dict):
        return {name: metadata_from_disk(item, directory, name) for name, item in value.items()}
    if isinstance(value, list):
        return [metadata_from_disk(item, directory, key) for item in value]
    is_path = key in {"actor", "critic"} or key.endswith(("_path", "_paths", "_dir"))
    if is_path and isinstance(value, str) and value:
        path = Path(value)
        if value.startswith("output/"):
            # Older records used repository-relative paths, before JSON-relative paths.
            local = Path(directory) / path.name
            if key in {"actor", "critic", "actor_path", "critic_path"} and local.is_file():
                path = local
            else:
                path = Path(__file__).resolve().parents[1] / path
        elif path.is_absolute():
            # Compatibility with old metadata after moving a run directory.
            local = Path(directory) / path.name
            if key in {"actor", "critic", "actor_path", "critic_path"} and local.is_file():
                path = local
        else:
            path = Path(directory) / path
        return str(path.resolve())
    return value


def check_checkpoint_paths(actor, critic, *, multiplayer=False):
    """Validate checkpoint paths for both Python callers and command-line entrypoints."""
    if bool(actor) != bool(critic):
        raise ValueError("provide both --actor_path and --critic_path")
    if not actor:
        return
    directories = []
    for path in (actor, critic):
        path = Path(path)
        if not (path.is_file() or (multiplayer and path.is_dir())):
            raise FileNotFoundError(f"checkpoint not found: {path}")
        directories.append(path.resolve() if path.is_dir() else path.resolve().parent)
    if directories[0] != directories[1]:
        raise ValueError("actor and critic checkpoints must belong to the same run directory")


def validate_checkpoint_args(parser, args, *, multiplayer=False):
    """Reject missing pairs and invalid paths before model construction or writes."""
    if getattr(args, "overwrite", False) and getattr(args, "dry_run", False):
        parser.error("--overwrite cannot be combined with --dry-run")
    try:
        check_checkpoint_paths(args.actor_path, args.critic_path, multiplayer=multiplayer)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.resume and not args.actor_path:
        parser.error("--resume requires actor/critic checkpoints")
    if args.actor_path and getattr(args, "overwrite", False):
        parser.error("--overwrite is only for fresh training; omit checkpoint paths")


def prepare_run_directory(directory, *, overwrite=False, resume=False):
    """Require an explicit choice before fresh training reuses a nonempty directory."""
    directory = Path(directory).resolve()
    if directory.exists() and not directory.is_dir():
        raise NotADirectoryError(directory)
    if directory.is_dir() and any(directory.iterdir()) and not resume:
        if not overwrite:
            raise FileExistsError(
                f"Run directory already contains results: {directory}. "
                "Choose a new --output-dir, use --resume with checkpoints, "
                "or explicitly use --overwrite to replace this run."
            )
        repository = Path(__file__).resolve().parents[1]
        if directory == Path.home() or directory == Path(directory.anchor) or directory in repository.parents or directory == repository or (directory / ".git").exists():
            raise ValueError(f"Refusing to overwrite a home, repository or ancestor directory: {directory}")
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
