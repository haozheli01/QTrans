"""QTrans io.filesystem helpers."""

from pathlib import Path
from typing import Optional
from typing import Tuple
from typing import Union
import os


def check_path_exist(
    path: str,
    check_type: str = "file",
    create_if_missing: bool = False,
    check_access: Optional[str] = None,
) -> Union[bool, Tuple[bool, bool]]:
    """Check whether a given path exists, with an optional access check.

    Args:
        path:              The path to check. Can be a file or a directory.
        check_type:        Type of path to check — "file" or "dir".
                           Defaults to "file".
        create_if_missing: If True and check_type is "dir", create the
                           directory (including parents) when it does not
                           exist.  Defaults to False.
        check_access:      Optional combination of "r", "w", "x" to verify.
                           e.g. "r", "rw", "rx", "rwx", "w", "x", "wx".
                           Defaults to None (no access check).

    Returns:
        bool:              Path exists or not, when check_access is None.
        Tuple[bool, bool]: (exists, access_ok) when check_access is set.
                           access_ok is always False if the path doesn't exist.

    Raises:
        ValueError: If check_type or check_access is invalid.

    Examples:
        >>> check_path_exist("/tmp", check_type="dir")
        True

        >>> check_path_exist("run.sh", check_access="rx")
        (True, True)
    """
    if check_type not in ("file", "dir"):
        raise ValueError(f"check_type must be 'file' or 'dir', got '{check_type}'")

    _flag_map = {"r": os.R_OK, "w": os.W_OK, "x": os.X_OK}
    if check_access is not None:
        if not check_access or not set(check_access).issubset(_flag_map):
            raise ValueError(
                f"check_access must be a combination of 'r', 'w', 'x', got '{check_access}'"
            )

    path_obj = Path(path)
    is_exist = False

    if check_type == "dir":
        if path_obj.is_dir():
            is_exist = True
        elif create_if_missing:
            path_obj.mkdir(parents=True, exist_ok=True)
            print(f"Directory did not exist — created: {path}")
            is_exist = True
    else:
        if path_obj.is_file():
            is_exist = True

    if check_access is None:
        return is_exist

    access_ok = is_exist and all(os.access(path, _flag_map[c]) for c in set(check_access))
    return is_exist, access_ok


def make_directory(path_str:str, parents:bool=False, exist_ok:bool=True):
    folder_path = Path(path_str)
    try:
        folder_path.mkdir(parents=parents, exist_ok=exist_ok)
    except Exception as e:
        raise RuntimeError(f"Fail to make directory: {path_str}")
