"""A workbench plugin whose entry point raises: the workbench must report it and carry on."""
from __future__ import annotations


def register():
    raise RuntimeError("this plugin is broken on purpose")
