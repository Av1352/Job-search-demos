from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_ui_mockup(filename, shift=False):
    # Create 800x600 image
    img = Image.new('RGB', (800, 600), color=(30, 41, 59))  # Dark background
    draw = ImageDraw.Draw(img)
    
    # Header bar
    header_y = 20 if not shift else 32  # Shift if current version
    draw.rectangle([20, header_y, 780, header_y + 60], fill=(16, 185, 129))
    
    # Buttons
    button_x = 50 if not shift else 62  # Shift button
    draw.rectangle([button_x, 120, button_x + 120, 170], fill=(59, 130, 246))
    draw.rectangle([220, 120, 340, 170], fill=(59, 130, 246))
    
    # Content area
    draw.rectangle([20, 200, 780, 550], fill=(51, 65, 85))
    
    # Add text
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    draw.text((60, header_y + 20), "VisionTest UI", fill='white', font=font)
    
    # Missing element in current version
    if not shift:
        draw.ellipse([700, 520, 730, 550], fill=(239, 68, 68))  # Search icon
    
    img.save(f'examples/{filename}')
    print(f"✅ Created {filename}")

# Create examples folder
import os
os.makedirs('examples', exist_ok=True)

# Generate test images
create_ui_mockup('baseline.png', shift=False)
create_ui_mockup('current.png', shift=True)

print("\n✅ Test images created!")
print("Baseline: No shifts, all elements present")
print("Current: Header shifted, button moved, search icon missing") 