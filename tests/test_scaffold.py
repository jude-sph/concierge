import sys

def test_python_version_is_310():
    assert sys.version_info[:2] == (3, 10), (
        "SoulX-Duplug pins Python 3.10; newer versions lack wheels for its deps"
    )

def test_package_imports():
    import rtvoice  # noqa: F401
