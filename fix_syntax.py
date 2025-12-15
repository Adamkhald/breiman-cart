# fix_syntax.py
import re

# Read the file
with open('breiman_cart/breiman_cart.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and fix the malformed docstring around line 1371
# Look for the pattern: """(self, filepath: str) -> None:
content = re.sub(
    r'"""(\(self, filepath: str\) -> None:)',
    r'"""\n        Save the model to a file.\n        \n        Parameters\n        ----------\n        filepath : str\n            Path where the model should be saved\n        \n        Examples\n        --------\n        >>> inference.save_model("my_model_v2.pkl")\n        """\n        self.model.save(filepath)\n    \n    def export_to_json(self, filepath: str) -> None:\n        """',
    content
)

# Write back
with open('breiman_cart/breiman_cart.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed syntax error!")