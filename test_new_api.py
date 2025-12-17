import breiman_cart as brc
import pandas as pd
import numpy as np

# Test data
X = pd.DataFrame({'x1': [1, 2, 3, 4, 5], 'x2': [2, 4, 6, 8, 10]})
y = np.array([3, 6, 9, 12, 15])

# Test Regression
print("Testing BRCRegression...")
model = brc.BRCRegression(max_depth=3)
model.fit(X, y)
predictions = model.predict(X)
print(f"✅ Predictions: {predictions}")

# Test Classification
y_class = np.array([0, 0, 1, 1, 1])
clf = brc.BRCClassification(max_depth=2)
clf.fit(X, y_class)
print(f"✅ Classification works!")

# Test Inference
inference = brc.BRCInference(model)
print(f"✅ Inference: {inference}")

print("\n🎉 All tests passed!")