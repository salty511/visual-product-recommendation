import os
import shutil
from tqdm.auto import tqdm
from PIL import Image


def load_dupe_groups(dupes_path):
	with open(dupes_path, "r") as f:
		lines = [line.strip() for line in f.readlines() if line.strip()]
	return [line.split() for line in lines]


def remove_dupes(
	images_dir="data/images",
	dupes_path="dupes.txt",
	clean_dir="data/clean",
):
	dupe_groups = load_dupe_groups(dupes_path)
	keep = {group[0] for group in dupe_groups if group}
	remove = {name for group in dupe_groups for name in group[1:]}

	os.makedirs(clean_dir, exist_ok=True)

	copied = []
	for name in tqdm(sorted(os.listdir(images_dir)), desc="Copying clean"):
		src_path = os.path.join(images_dir, name)
		im = Image.open(src_path)

		# Check for empty image
		if(im.size[0] <= 1 or im.size[1] <= 1):
			continue
		
		# Check if image is a dupe
		if name in remove or not os.path.isfile(src_path):
			continue

		# Copy to clean dir
		dst_path = os.path.join(clean_dir, name)
		if not os.path.exists(dst_path):
			if os.path.exists(src_path):
				shutil.copy2(src_path, dst_path)
				copied.append(name)

	return {
		"keep_count": len(keep),
		"remove_count": len(remove),
		"copied": copied,
		"clean_dir": clean_dir,
	}

def run_data_pipline():
	result = remove_dupes()
	print(
		f"Copied {len(result['copied'])} clean images to {result['clean_dir']}; "
	)

if __name__ == "__main__":
	