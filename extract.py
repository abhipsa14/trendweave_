import os
import fitz  # PyMuPDF
import json

PDF_DIR = "pdfs"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_from_pdf(file_path, pdf_name):
    doc = fitz.open(file_path)
    total_images = 0
    page_metadata = {}

    pdf_output_dir = os.path.join(OUTPUT_DIR, pdf_name)
    os.makedirs(pdf_output_dir, exist_ok=True)

    for page_num, page in enumerate(doc):
        page_id = f"page{page_num + 1}"
        page_dir = os.path.join(pdf_output_dir, page_id)
        os.makedirs(page_dir, exist_ok=True)

        # Save text
        text = page.get_text()
        with open(os.path.join(page_dir, "text.txt"), "w", encoding="utf-8") as f:
            f.write(text)

        # Save images
        image_count = 0
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            img_filename = f"img{img_index + 1}.{image_ext}"
            with open(os.path.join(page_dir, img_filename), "wb") as img_file:
                img_file.write(image_bytes)
            image_count += 1
            total_images += 1

        # Store metadata for page
        page_metadata[page_id] = {
            "word_count": len(text.split()),
            "image_count": image_count
        }

    # Save metadata for the entire PDF
    metadata = {
        "pdf_name": pdf_name,
        "total_pages": len(doc),
        "total_images": total_images,
        "page_summary": page_metadata
    }

    with open(os.path.join(pdf_output_dir, "metadata.json"), "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=4)

    doc.close()

# Run on all PDFs in the folder
for filename in os.listdir(PDF_DIR):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(PDF_DIR, filename)
        pdf_name = os.path.splitext(filename)[0]
        print(f"📄 Processing: {filename}")
        extract_from_pdf(pdf_path, pdf_name)

print("\n✅ Extraction complete. Check the 'output' folder.")
