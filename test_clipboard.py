"""Test clipboard functionality directly."""

def test_clipboard():
    print("Testing clipboard image handling...\n")
    
    try:
        from llm_kernel.multimodal import MultimodalContent
        
        # Create a mock kernel object
        class MockKernel:
            def __init__(self):
                self.log = None
        
        kernel = MockKernel()
        mm = MultimodalContent(kernel)
        
        print("1. Checking clipboard for image...")
        img_content = mm.get_clipboard_image()
        
        if img_content:
            print(f"   Found image: {img_content['size']}")
            print(f"   Data type: {type(img_content['data'])}")
            print(f"   Data preview: {img_content['data'][:50]}...")
            
            # Test display
            print("\n2. Testing display with IPython Image...")
            from IPython.display import Image as IPImage, display
            import base64
            
            # This should work
            img_bytes = base64.b64decode(img_content['data'])
            display(IPImage(data=img_bytes, format='png', width=200))
            print("   [OK] Image displayed successfully!")
            
        else:
            print("   No image found in clipboard")
            print("   Try copying an image to clipboard first")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Run in Jupyter cell: exec(open('test_clipboard.py').read())")
    print("Then: test_clipboard()")