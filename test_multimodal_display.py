#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script to verify multimodal image display is working correctly."""

def test_image_display():
    """Test that base64 images are properly decoded for display."""
    print("Testing multimodal image display...\n")
    
    # Create a simple test image
    from PIL import Image
    import io
    import base64
    
    # Create a small red square
    img = Image.new('RGB', (100, 100), color='red')
    
    # Convert to base64 (like our multimodal module does)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    print("1. Testing IPython Image display with base64 string (should fail):")
    try:
        from IPython.display import Image as IPImage
        # This should fail with the old code
        display(IPImage(data=img_data, format='png'))
        print("   [FAIL] Unexpectedly succeeded - this should have failed!")
    except Exception as e:
        print(f"   [OK] Expected failure: {type(e).__name__}")
    
    print("\n2. Testing IPython Image display with decoded bytes (should work):")
    try:
        # This is the fixed approach
        img_bytes = base64.b64decode(img_data)
        display(IPImage(data=img_bytes, format='png'))
        print("   [OK] Success! Image should be displayed above.")
    except Exception as e:
        print(f"   [FAIL] Failed: {e}")
    
    print("\n3. Testing multimodal content structure:")
    # Simulate what multimodal.py creates
    img_content = {
        'type': 'image',
        'data': img_data,  # This is base64 string
        'size': (100, 100),
        'source': 'test'
    }
    
    print(f"   Content type: {img_content['type']}")
    print(f"   Data type: {type(img_content['data'])}")
    print(f"   Data preview: {img_content['data'][:50]}...")
    print(f"   Size: {img_content['size']}")
    
    return img_content

if __name__ == "__main__":
    print("Run this in a Jupyter notebook cell:")
    print("exec(open('test_multimodal_display.py').read())")
    print("img_content = test_image_display()")
    print("\nThen test the magic command:")
    print("%llm_paste --show")