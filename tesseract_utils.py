import pytesseract
from PIL import Image

from .utils import upsample_image

def extract_text_tesseract(
    image_path: str,
    config: str = '--psm 8',
    target_width: int = None,
    target_height: int = None,
    upsample: bool = False
) -> str:
    """
    Extract text from image using Tesseract OCR.
    
    Args:
        image_path: Path to the image file
        config: Tesseract configuration string
               --psm 7: Treat the image as a single text line
        target_width: Target width for upsampling (maintains aspect ratio)
        target_height: Target height for upsampling (maintains aspect ratio)
        upsample: Whether to upsample the image before OCR
    
    Returns:
        Extracted text string
    """
    try:
        img = Image.open(image_path)
        
        # Upsample if requested
        if upsample:
            if target_width is None and target_height is None:
                raise ValueError("target_width or target_height must be provided when upsample=True")
            img = upsample_image(img, target_width=target_width, target_height=target_height)
        
        text = pytesseract.image_to_string(img, config=config)
        # Strip whitespace and newlines
        return text.strip()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return ""
