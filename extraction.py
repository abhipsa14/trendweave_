import os
import glob
import fitz  # PyMuPDF
from datetime import datetime
import json

from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import exifread

import cv2
import ffmpeg
import whisper
from tqdm import tqdm

class PDFExtractor:
    def __init__(self, output_base_dir="extracted_content"):
        self.output_base_dir = output_base_dir
        self.create_output_directory()
        pytesseract.pytesseract.tesseract_cmd = r"Tesseract-OCR/tesseract.exe"

    def create_output_directory(self):
        if not os.path.exists(self.output_base_dir):
            os.makedirs(self.output_base_dir)
            print(f"Created output directory: {self.output_base_dir}")

    def get_pdf_list(self, source_path):
        pdf_files = []
        if os.path.isfile(source_path):
            if source_path.lower().endswith('.pdf'):
                pdf_files.append(source_path)
        elif os.path.isdir(source_path):
            pdf_files = glob.glob(os.path.join(source_path, "*.pdf"))
            for root, dirs, files in os.walk(source_path):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(root, file))
        return list(set(pdf_files))

    def extract_text_with_ocr(self, pdf_path, pdf_output_dir):
        text_file = os.path.join(pdf_output_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_ocr.txt")
        if os.path.exists(text_file):
            print("   ✓ OCR text already exists, skipping.")
            return text_file
        try:
            page_images = convert_from_path(pdf_path)
            all_text = ""
            for page_num, image in enumerate(page_images):
                page_folder = os.path.join(pdf_output_dir, f"page_{page_num+1}")
                os.makedirs(page_folder, exist_ok=True)
                page_img_path = os.path.join(page_folder, "ocr.jpg")
                if not os.path.exists(page_img_path):
                    image.save(page_img_path, "JPEG")
                text = pytesseract.image_to_string(Image.open(page_img_path))
                text = text.replace("-\n", "")
                all_text += f"\n\n--- Page {page_num+1} ---\n\n{text}"
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(all_text)
            print(f"   ✓ OCR text extracted and saved to: {text_file}")
            return text_file
        except Exception as e:
            print(f"Error during OCR extraction: {e}")
            return None

    def extract_images_from_pdf(self, pdf_path, output_dir):
        image_metadata_list = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(doc.page_count):
                page_folder = os.path.join(output_dir, f"page_{page_num+1}")
                os.makedirs(page_folder, exist_ok=True)
                images = doc.load_page(page_num).get_images(full=True)
                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_filename = os.path.join(page_folder, f"img_{img_index+1}.{image_ext}")
                    if not os.path.exists(image_filename):
                        with open(image_filename, "wb") as img_file:
                            img_file.write(image_bytes)
                    tags_dict = {}
                    try:
                        with open(image_filename, 'rb') as imgf:
                            tags = exifread.process_file(imgf)
                            tags_dict = {str(tag): str(tags[tag]) for tag in tags.keys()}
                    except Exception:
                        pass
                    image_metadata_list.append({
                        "page": page_num + 1,
                        "image_index": img_index + 1,
                        "image_filename": os.path.relpath(image_filename, output_dir),
                        "exif_metadata": tags_dict
                    })
            doc.close()
            all_image_meta_path = os.path.join(output_dir, "all_images_metadata.json")
            if not os.path.exists(all_image_meta_path):
                with open(all_image_meta_path, "w", encoding="utf-8") as meta_file:
                    json.dump(image_metadata_list, meta_file, indent=2)
            return image_metadata_list
        except Exception as e:
            print(f"Error extracting images from {pdf_path}: {str(e)}")
            return []

    def extract_text_from_pdf(self, pdf_path):
        try:
            doc = fitz.open(pdf_path)
            all_text = ""
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    all_text += f"\n{'='*50}\n"
                    all_text += f"PAGE {page_num + 1}\n"
                    all_text += f"{'='*50}\n"
                    all_text += page_text
            doc.close()
            return all_text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return None

    def process_single_pdf(self, pdf_path):
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        pdf_output_dir = os.path.join(self.output_base_dir, pdf_name)
        if not os.path.exists(pdf_output_dir):
            os.makedirs(pdf_output_dir)
        print(f"\nProcessing: {pdf_name}")
        print("-" * 50)
        print("1. Extracting selectable text...")
        text_content = self.extract_text_from_pdf(pdf_path)
        text_file_path = None
        if text_content and text_content.strip():
            text_file_path = os.path.join(pdf_output_dir, f"{pdf_name}_text.txt")
            with open(text_file_path, "w", encoding="utf-8") as text_file:
                text_file.write(text_content)
            print(f"   ✓ Text saved to: {text_file_path}")
        else:
            print("   ✗ No selectable text found, using OCR...")
            text_file_path = self.extract_text_with_ocr(pdf_path, pdf_output_dir)
        print("2. Extracting images and metadata...")
        image_info = self.extract_images_from_pdf(pdf_path, pdf_output_dir)
        if image_info:
            print(f"   ✓ Extracted {len(image_info)} images")
            metadata_path = os.path.join(pdf_output_dir, f"{pdf_name}_image_metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as meta_file:
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
        self.create_summary_report(results)
        return results

    def create_summary_report(self, results):
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

class VideoExtractor:
    def __init__(self, output_base_dir="extracted_content"):
        self.output_base_dir = output_base_dir
        self.whisper_model = whisper.load_model("base")

    def extract_frames(self, video_path, frame_interval=30):
        """
        Extract frames from video at given interval.
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(self.output_base_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        extracted = 0
        frame_paths = []
        print(f"   Extracting frames from {video_name} every {frame_interval} frames...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_filename = os.path.join(video_output_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)
                frame_paths.append(frame_filename)
                extracted += 1
            frame_count += 1
        cap.release()
        print(f"   ✓ Extracted {extracted} frames")
        return frame_paths

    def extract_audio(self, video_path):
        """
        Extract audio as .wav file using ffmpeg.
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(self.output_base_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        audio_path = os.path.join(video_output_dir, f"{video_name}.wav")
        if not os.path.exists(audio_path):
            try:
                (
                    ffmpeg
                    .input(video_path)
                    .output(audio_path, acodec='pcm_s16le', ac=1, ar='16000')
                    .run(overwrite_output=True, quiet=True)
                )
                print(f"   ✓ Audio extracted to {audio_path}")
            except Exception as e:
                print(f"   ✗ Audio extraction failed: {e}")
                return None
        else:
            print(f"   ✓ Audio already exists: {audio_path}")
        return audio_path

    def transcribe_audio(self, audio_path):
        """
        Transcribe audio to text using Whisper.
        """
        transcript_path = audio_path.replace(".wav", "_transcript.txt")
        segments_path = audio_path.replace(".wav", "_segments.json")
        if os.path.exists(transcript_path) and os.path.exists(segments_path):
            print(f"   ✓ Transcript already exists: {transcript_path}")
            return transcript_path, segments_path
        print("   Transcribing audio with Whisper...")
        try:
            result = self.whisper_model.transcribe(audio_path)
            transcript_text = result["text"]
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            with open(segments_path, "w", encoding="utf-8") as f:
                json.dump(result["segments"], f, indent=2)
            print(f"   ✓ Transcript saved: {transcript_path}")
            return transcript_path, segments_path
        except Exception as e:
            print(f"   ✗ Transcription failed: {e}")
            return None, None

    def extract_video_metadata(self, video_path):
        """
        Extract video metadata using ffmpeg.probe.
        """
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(self.output_base_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        meta_path = os.path.join(video_output_dir, f"{video_name}_metadata.json")
        if not os.path.exists(meta_path):
            try:
                probe = ffmpeg.probe(video_path)
                with open(meta_path, "w", encoding="utf-8") as meta_file:
                    json.dump(probe, meta_file, indent=2)
                print(f"   ✓ Metadata saved: {meta_path}")
            except Exception as e:
                print(f"   ✗ Metadata extraction failed: {e}")
                return None
        else:
            print(f"   ✓ Metadata already exists: {meta_path}")
        return meta_path

    def process_single_video(self, video_path, frame_interval=30):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\nProcessing video: {video_name}")
        print("-" * 50)
        frame_paths = self.extract_frames(video_path, frame_interval)
        audio_path = self.extract_audio(video_path)
        transcript_path, segments_path = (None, None)
        if audio_path:
            transcript_path, segments_path = self.transcribe_audio(audio_path)
        meta_path = self.extract_video_metadata(video_path)
        return {
            "video_name": video_name,
            "frame_paths": frame_paths,
            "audio_path": audio_path,
            "transcript_path": transcript_path,
            "segments_path": segments_path,
            "metadata_path": meta_path
        }

    def process_multiple_videos(self, source_path, frame_interval=30):
        video_files = []
        if os.path.isdir(source_path):
            for ext in ('.mp4', '.avi', '.mov', '.mkv'):
                video_files.extend(glob.glob(os.path.join(source_path, f"*{ext}")))
        elif os.path.isfile(source_path):
            video_files = [source_path]
        else:
            print("No video folder found or provided, skipping video processing.")
            return {}
        print(f"Found {len(video_files)} video file(s)")
        results = {}
        for video_path in tqdm(video_files, desc="Videos processed"):
            try:
                result = self.process_single_video(video_path, frame_interval)
                results[result["video_name"]] = result
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
        return results

def main():
    # PDF Extraction
    extractor = PDFExtractor("extracted_content")
    print("PDF Text, Image, and OCR Extractor")
    print("=" * 50)
    pdf_directory = "dataset"  # Directory to process multiple files
    if os.path.exists(pdf_directory):
        extractor.process_multiple_pdfs(pdf_directory)
    else:
        print("PDF directory not found.")

    # Video Extraction
    video_directory = "videos"  # Change to your video folder
    if os.path.exists(video_directory):
        vextractor = VideoExtractor("extracted_content")
        vextractor.process_multiple_videos(video_directory, frame_interval=30)
    else:
        print("Video directory not found.")

    print("\n✅ Extraction complete!")

if __name__ == "__main__":
    main()