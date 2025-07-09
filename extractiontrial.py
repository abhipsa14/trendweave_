import os
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from tqdm import tqdm

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pdf_folder = r"C:\Users\yasha\Downloads\wgsn_pdfs\wgsn_dataset"
output_folder = "extracted_data"
os.makedirs(output_folder, exist_ok=True)

pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

for pdf_file in tqdm(pdf_files, desc="PDFs processed"):
    try:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        pdf_name = os.path.splitext(pdf_file)[0]
        pdf_output_dir = os.path.join(output_folder, pdf_name)
        os.makedirs(pdf_output_dir, exist_ok=True)

        # 1. Extract images using PyMuPDF
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = os.path.join(
                    pdf_output_dir, f"page_{page_num+1}_img_{img_index+1}.{image_ext}"
                )
                with open(image_filename, "wb") as img_file:
                    img_file.write(image_bytes)
        doc.close()

        # 2. Convert PDF pages to images for OCR
        page_images = convert_from_path(pdf_path)
        for page_num, image in enumerate(tqdm(page_images, desc=f"Pages in {pdf_file}", leave=False)):
            page_img_path = os.path.join(pdf_output_dir, f"page_{page_num+1}_ocr.jpg")
            image.save(page_img_path, "JPEG")

            # 3. OCR: Extract text from image
            text = pytesseract.image_to_string(Image.open(page_img_path))
            text_file = os.path.join(pdf_output_dir, f"page_{page_num+1}_ocr.txt")
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(text)
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")

print("OCR extraction complete.")
