import requests
import os
import PyPDF2

def check_pdf_integrity(pdf_path):
    """Check the integrity of a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            if reader.pages:
                return True
    except (PyPDF2.errors.PdfReadError, OSError):
        return False
    return True

def find_corrupt_files(directory):
    """Find and list corrupt PDF files in a directory."""
    corrupt_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            if not check_pdf_integrity(pdf_path):
                corrupt_files.append(pdf_path)
    return corrupt_files


corrupt_pdfs = find_corrupt_files('/lstr/sahara/datalab-ml/z1974769/cs_pap1_2015')
# print("Corrupt PDF files found:", corrupt_pdfs)


def delete_files(file_list):
    """Delete files from the filesystem."""
    for file_path in file_list:
        os.remove(file_path)
        print(f"Deleted {file_path}")

# Optionally delete corrupt files
delete_files(corrupt_pdfs)

download_map = dict()

for file_path in corrupt_pdfs:
    file_name = file_path.split("/")[-1]
    download_map[file_name] = f"https://arxiv.org/pdf/{file_name}"


print(list(download_map.items())[0:10])
# def redownload_files(file_list, download_map):
#     """Redownload files based on a mapping from filenames to URLs."""
#     for file_path in file_list:
#         file_name = os.path.basename(file_path)
#         url = download_map.get(file_name)
#         if url:
#             response = requests.get(url)
#             if response.status_code == 200:
#                 with open(file_path, 'wb') as f:
#                     f.write(response.content)
#                 print(f"Redownloaded {file_name} from {url}")

# # download_map = {'example_corrupt_file.pdf': 'http://example.com/path_to_file.pdf'}
# redownload_files(corrupt_pdfs, download_map)
