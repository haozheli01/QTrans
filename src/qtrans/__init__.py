"""Public package interface for QTrans."""

from ._version import __version__

__all__ = ["QTransCalculator", "QTransPreparer", "__version__"]


def __getattr__(name):
    if name == "QTransCalculator":
        from .calculator import QTransCalculator

        return QTransCalculator
    if name == "QTransPreparer":
        from .preparer import QTransPreparer

        return QTransPreparer
    raise AttributeError("module 'qtrans' has no attribute {!r}".format(name))

