from datasets import load_dataset
from collections import defaultdict
from datasets import Dataset
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fitz
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import random



arxive_data = load_dataset("json", data_files="/lstr/sahara/datalab-ml/z1974769/arxiv-metadata-oai-snapshot.json")
# cs_data_after_2015x = archive_data.filter(lambda data_point: "cs." in data_point["categories"] and int(data_point["id"][0:2]) >= 15)
# unique_ids = list()
def is_cs_and_year_2015_or_later(data_point):
    if "cs." in data_point["categories"]:
        try:
            year_prefix = int(data_point["id"].split(".")[0][:2])
            # unique_ids.append(data_point["id"])
            return year_prefix >= 15
        except ValueError:
            return False
    else:
        return False

cs_data_after_2015 = arxive_data.filter(is_cs_and_year_2015_or_later)

print(cs_data_after_2015)


unique_ids = dict()

for paper in cs_data_after_2015["train"]:
    if unique_ids.get(int(paper["id"].split(".")[0][:2])) is not None:
        unique_ids[int(paper["id"].split(".")[0][:2])].append(paper["id"])
    else :
        unique_ids[int(paper["id"].split(".")[0][:2])] = [paper["id"]]

print(len(unique_ids), "unique ids")

unique_ids2 = []

for year in list(unique_ids.keys()):
    if year in unique_ids.keys():
        ids_for_year = unique_ids[year]
        num_ids_to_select = min(100, len(ids_for_year))
        selected_ids = random.sample(ids_for_year, num_ids_to_select)
        unique_ids2.extend(selected_ids)


def download_paper(arxiv_id, save_dir):
    try:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        response = requests.get(pdf_url, timeout=30, stream=True)
        if response.status_code == 200:
            file_path = Path(save_dir) / f"{arxiv_id}.pdf"
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {arxiv_id}")
        else:
            print(f"Failed to download {arxiv_id}")
    except requests.RequestException as e:
        print(f"Error downloading {arxiv_id}: {e}")


arxiv_ids = unique_ids2

save_directory = "/lstr/sahara/datalab-ml/z1974769/cs_pap1_2015"



with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(download_paper, arxiv_id, save_directory) for arxiv_id in arxiv_ids]
    for future in futures:
        future.result()

print("All downloads completed.")



# def download_paper(arxiv_id, save_dir):
#     pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
#     file_path = Path(save_dir) / f"{arxiv_id}.pdf"
    
#     session = requests.Session()
#     # Define a Retry object with backoff strategy
#     retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504], method_whitelist=["GET"])
#     session.mount("http://", HTTPAdapter(max_retries=retries))
#     session.mount("https://", HTTPAdapter(max_retries=retries))

#     try:
#         with session.get(pdf_url, timeout=30, stream=True) as response:
#             response.raise_for_status()
#             with open(file_path, "wb") as f:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     f.write(chunk)
#             print(f"Downloaded {arxiv_id}")
#     except requests.exceptions.HTTPError as errh:
#         print(f"Http Error for {arxiv_id}: {errh}")
#     except requests.exceptions.ConnectionError as errc:
#         print(f"Error Connecting for {arxiv_id}: {errc}")
#     except requests.exceptions.Timeout as errt:
#         print(f"Timeout Error for {arxiv_id}: {errt}")
#     except requests.exceptions.RequestException as err:
#         print(f"Error downloading {arxiv_id}: {err}")

# arxiv_ids = unique_ids[0:len(unique_ids)//2]
# save_directory = "/lstr/sahara/datalab-ml/z1974769/cs_pap1_2015"

# with ThreadPoolExecutor(max_workers=10) as executor:
#     futures = [executor.submit(download_paper, arxiv_id, save_directory) for arxiv_id in arxiv_ids]
#     for future in futures:
#         # To handle exceptions inside threads, consider future.result()
#         try:
#             future.result()
#         except Exception as e:
#             print(f"Exception in downloading: {e}")

# print("All downloads completed.")
