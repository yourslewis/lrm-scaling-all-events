import pytest


def pytest_addoption(parser):
    parser.addoption("--data_dir", action="store", default="/tmp/hstu_local/data")
    parser.addoption("--embd_dir", action="store", default="/tmp/hstu_local/embds")
