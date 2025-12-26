# test_classifier.py
from PIL import Image
import numpy as np
from model import PathologyClassifier
import os

# Test classifier
classifier = PathologyClassifier()

# Check what files actually exist
print("Files in examples/:")
if os.path.exists("examples"):
    files = os.listdir("examples")
    for f in files:
        print(f"  - {f}")
else:
    print("❌ examples/ folder doesn't exist!")
    exit()

# Test with actual filenames
test_images = [
    "examples/malignant.png",
    "examples/benign.png",
    "examples/suspicious.jpg"
]

print(f"\nTesting images...\n")

for img_path in test_images:
    if not os.path.exists(img_path):
        print(f"⚠️  Skipping {img_path} - file not found")
        continue
        
    try:
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Convert RGBA to RGB if needed (PNG can have alpha channel)
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
        
        result = classifier.classify(img_array)
        
        print(f"{'='*60}")
        print(f"📁 Image: {img_path}")
        print(f"🔬 Classification: {result['classification']}")
        print(f"📊 Confidence: {result['confidence']:.1%}")
        print(f"⚠️  Severity: {result['severity']}")
        print(f"🧬 Tumor Type: {result['tumor_type']}")
        print('='*60)
        print()
    except Exception as e:
        print(f"❌ Error with {img_path}: {e}")
        import traceback
        traceback.print_exc()