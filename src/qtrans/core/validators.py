"""QTrans input validators."""

from qtrans.io.filesystem import check_path_exist
import numpy as np
import os


def _check_type(value, expected, name):
    if not isinstance(value, expected) or isinstance(value, bool):
        raise TypeError(
            f"{name} must be {expected}, but got {type(value).__name__}: {value!r}"
        )


def validate_int(value, name):
    _check_type(value, int, name)
    return value


def validate_positive_int(value, name):
    _check_type(value, int, name)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, but got {value!r}")
    return value


def validate_num(value, name):
    _check_type(value, (int, float), name)
    return value


def validate_positive_num(value, name):
    _check_type(value, (int, float), name)
    if value <= 0:
        raise ValueError(f"{name} must be a positive number, but got {value!r}")
    return value


def validate_non_negative_num(value, name):
    _check_type(value, (int, float), name)
    if value < 0:
        raise ValueError(f"{name} must be a non-negative number, but got {value!r}")
    return value


def validate_bool(value, name):
    if not isinstance(value, bool):
        raise TypeError(
            f"{name} must be a boolean, but got {type(value).__name__}: {value!r}"
        )
    return value


def _validate_typed_list(value, length, name, elem_type, type_desc):
    if (
        not isinstance(value, list)
        or len(value) != length
        or not all(isinstance(v, elem_type) and not isinstance(v, bool) for v in value)
    ):
        raise ValueError(
            f"{name} must be a list of {length} {type_desc}, but got {value!r}"
        )
    return value


def validate_num_list(value, name, length):
    return _validate_typed_list(value, length, name, (int, float), "numbers")

def validate_int_list(value, name, length):
    return _validate_typed_list(value, length, name, int, "integers")

def validate_positive_int_list(value, name, length):
    _validate_typed_list(value, length, name, int, "positive integers")
    if not all(v > 0 for v in value):
        raise ValueError(
            f"{name} must be a list of {length} positive integers, but got {value!r}"
        )
    return value

def validate_str_list(value, name, length):
    return _validate_typed_list(value, length, name, str, "string")


def validate_numpy_array(value, name, shape=None, dtype=float):
    try:
        value = np.asarray(value, dtype=dtype)
    except Exception:
        raise ValueError(
            f"{name} must be a {shape} array with {dtype} elements, but got {value!r}"
        )
    if shape is not None and value.shape != shape:
        raise ValueError(
            f"{name} must be a {shape} array with {dtype} elements, but got shape {value.shape}"
        )
    return value


def validate_readable_path(value, name):
    value = str(value)
    value = os.path.abspath(value) 
    is_exist, is_readable = check_path_exist(value, check_access='r')
    if not (is_exist and is_readable):
        raise ValueError(
            f"The {name} path '{value}' either does not exist or is not readable."
        )
    return value

def validate_existed_dir(value, name):
    value = str(value)
    value = os.path.abspath(value) 
    is_exist = check_path_exist(value, check_type='dir')
    if not is_exist:
        raise ValueError(
            f"The {name} folder '{value}' does not exist."
        )
    return value
