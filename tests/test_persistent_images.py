"""Test persistent image context in LLM Kernel."""

print("""
Test Instructions:
==================

1. First, copy an image to your clipboard

2. Run in a Jupyter cell:
   %llm_paste
   
3. In a NEW cell, ask about the image:
   What's in the image I just pasted?
   
4. The LLM should be able to see the image even though it's in a different cell!

5. You can also check the context to verify the image is there:
   %llm_context
   
The image should now persist in the conversation context across cells,
just like in regular LLM chat interfaces!
""")