import pydicom
import numpy as np
from PIL import Image
import io

def dicom_to_preview(file):

    file.seek(0)

    dicom = pydicom.dcmread(file, force=True)

    image = dicom.pixel_array.astype(np.float32)

    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    image = (image * 255).astype(np.uint8)

    if hasattr(dicom, "PhotometricInterpretation"):
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            image = np.max(image) - image

    pil_img = Image.fromarray(image).convert("RGB")

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG")
    buffer.seek(0)

    return buffer