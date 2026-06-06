import sys
from pathlib import Path

# Add the SNPbag package directory to sys.path so tests can import from it
sys.path.insert(0, str(Path(__file__).parent.parent))
