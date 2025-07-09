import os
import glob
import  fitz
from datetime import datetime
import json

class PDFExtractor:
    def __init__(self, output_base_dir="extracted_content"):
        self.output_base_dir = output_base_dir
        self.create_output_directory()
    
    def create_output_directory(self):
        """Create the main output directory if it doesn't exist"""
        if not os.path.exists(self.output_base_dir):
            os.makedirs(self.output_base_dir)
            print(f"Created output directory: {self.output_base_dir}")
    
    def get_pdf_list(self, source_path):
        """Get list of PDF files from various sources"""
        pdf_files = []
        
        if os.path.isfile(source_path):
            # Single file
            if source_path.lower().endswith('.pdf'):
                pdf_files.append(source_path)
            else:
                print(f"Error: {source_path} is not a PDF file")
        elif os.path.isdir(source_path):
            # Directory - find all PDFs
            pdf_files = glob.glob(os.path.join(source_path, "*.pdf"))
            # Also check subdirectories
            for root, dirs, files in os.walk(source_path):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(root, file))
        else:
            print(f"Error: {source_path} is not a valid file or directory")
        
        return list(set(pdf_files))  # Remove duplicates
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file"""
        try:
            doc = fitz.open(pdf_path)
            all_text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                
                if page_text.strip():  # Only add if there's actual text
                    all_text += f"\n{'='*50}\n"
                    all_text += f"PAGE {page_num + 1}\n"
                    all_text += f"{'='*50}\n"
                    all_text += page_text
            
            doc.close()
            return all_text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return None
    
    def extract_images_from_pdf(self, pdf_path, output_dir):
        """Extract images from a PDF file"""
        try:
            doc = fitz.open(pdf_path)
            image_info = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Create filename
                        image_filename = f"page{page_num + 1}_img{img_index + 1}.{image_ext}"
                        image_path = os.path.join(output_dir, image_filename)
                        
                        # Save image
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        # Store image metadata
                        image_info.append({
                            "filename": image_filename,
                            "page": page_num + 1,
                            "index": img_index + 1,
                            "width": base_image["width"],
                            "height": base_image["height"],
                            "size_bytes": len(image_bytes),
                            "format": image_ext
                        })
                        
                    except Exception as e:
                        print(f"Error extracting image {img_index + 1} from page {page_num + 1}: {str(e)}")
                        continue
            
            doc.close()
            return image_info
        except Exception as e:
            print(f"Error extracting images from {pdf_path}: {str(e)}")
            return []
    
    def process_single_pdf(self, pdf_path):
        """Process a single PDF file"""
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_output_dir = os.path.join(self.output_base_dir, pdf_name)
        
        # Create directory for this PDF
        if not os.path.exists(pdf_output_dir):
            os.makedirs(pdf_output_dir)
        
        print(f"\nProcessing: {pdf_name}")
        print("-" * 50)
        
        # Extract text
        print("1. Extracting text...")
        text_content = self.extract_text_from_pdf(pdf_path)
        
        if text_content:
            text_file_path = os.path.join(pdf_output_dir, f"{pdf_name}_text.txt")
            with open(text_file_path, "w", encoding="utf-8") as text_file:
                text_file.write(text_content)
            print(f"   ✓ Text saved to: {text_file_path}")
        else:
            text_file_path = None
            print("   ✗ No text extracted or error occurred")
        
        # Extract images
        print("2. Extracting images...")
        image_info = self.extract_images_from_pdf(pdf_path, pdf_output_dir)
        
        if image_info:
            print(f"   ✓ Extracted {len(image_info)} images")
            
            # Save image metadata
            metadata_path = os.path.join(pdf_output_dir, f"{pdf_name}_image_metadata.json")
            with open(metadata_path, "w") as meta_file:
                json.dump(image_info, meta_file, indent=2)
            print(f"   ✓ Image metadata saved to: {metadata_path}")
        else:
            print("   ✗ No images found or error occurred")
        
        return {
            "pdf_name": pdf_name,
            "output_directory": pdf_output_dir,
            "text_file": text_file_path,
            "images_extracted": len(image_info),
            "image_metadata": image_info
        }
    
    def process_multiple_pdfs(self, source_path):
        """Process multiple PDF files"""
        pdf_files = self.get_pdf_list(source_path)
        
        if not pdf_files:
            print("No PDF files found!")
            return {}
        
        print(f"Found {len(pdf_files)} PDF file(s)")
        print("=" * 60)
        
        results = {}
        
        for pdf_path in pdf_files:
            try:
                result = self.process_single_pdf(pdf_path)
                results[result["pdf_name"]] = result
            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")
                continue
        
        # Create summary report
        self.create_summary_report(results)
        
        return results
    
    def create_summary_report(self, results):
        """Create a summary report of the extraction process"""
        summary_path = os.path.join(self.output_base_dir, "extraction_summary.txt")
        
        with open(summary_path, "w") as summary_file:
            summary_file.write("PDF EXTRACTION SUMMARY REPORT\n")
            summary_file.write("=" * 50 + "\n")
            summary_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            total_pdfs = len(results)
            total_images = sum(result["images_extracted"] for result in results.values())
            
            summary_file.write(f"Total PDFs processed: {total_pdfs}\n")
            summary_file.write(f"Total images extracted: {total_images}\n\n")
            
            for pdf_name, result in results.items():
                summary_file.write(f"PDF: {pdf_name}\n")
                summary_file.write(f"  - Images extracted: {result['images_extracted']}\n")
                summary_file.write(f"  - Text file: {'Yes' if result['text_file'] else 'No'}\n")
                summary_file.write(f"  - Output directory: {result['output_directory']}\n\n")
        
        print(f"\nSummary report saved to: {summary_path}")
        ##text file generated to get the extracted info


def main():
    """Main function to demonstrate usage"""
    extractor = PDFExtractor("extracted_content")
    print("PDF Text and Image Extractor")
    print("=" * 50)
    
    pdf_directory="dataset" ## directory given to process multiple files
    pdf_files=glob.glob(os.path.join(pdf_directory,"*.pdf"))
    
    results = {}
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            result = extractor.process_single_pdf(pdf_file)
            results[result["pdf_name"]] = result
    
    if results:
        extractor.create_summary_report(results)
    
    print("\n✅ Extraction complete!")


if __name__ == "__main__":
    main()