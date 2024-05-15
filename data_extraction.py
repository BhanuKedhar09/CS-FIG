from datasets import load_dataset
from collections import defaultdict
from datasets import Dataset
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

archive_data = load_dataset("json", data_files="/lstr/sahara/datalab-ml/z1974769/arxiv-metadata-oai-snapshot.json")
cs_data = archive_data.filter(lambda data_point: data_point["categories"].startswith("cs"))


selected_papers_by_category = defaultdict(list)

for paper in cs_data["train"]:
    categories_of_paper = paper['categories'].split()
    for category in categories_of_paper:
        if category.startswith('cs.') and len(selected_papers_by_category[category]) < 100:
            selected_papers_by_category[category].append(paper)
            
            
print("printting example categry papers on Artifical inteligence")
print(selected_papers_by_category["cs.AI"])


papers_list = []
for category, papers in selected_papers_by_category.items():
    for paper in papers:
        paper["unique_category"] = category
        papers_list.append(paper)

cs_dataset_per_category = Dataset.from_dict({"data" : papers_list})
cs_dataset_per_category.to_json("intermediate_data.json")


# for paper in cs_archive_intermediate_data["train"]:
#     id  = paper["data"]["id"]
#     url_template = f"https://arxiv.org/pdf/{id}.pdf"


unique_ids = set()
for paper in papers_list:
    unique_ids.add(paper["id"])
len(unique_ids)

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


arxiv_ids = unique_ids
save_directory = "./cs_intermediate_papers"


with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(download_paper, arxiv_id, save_directory) for arxiv_id in arxiv_ids]
    for future in futures:
        future.result()

print("All downloads completed.")
