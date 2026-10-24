import os
import sys

# lets the tests run from a source checkout without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
