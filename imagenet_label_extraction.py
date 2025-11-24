import json
import csv
import scipy.io as sio

############################################
# 1. Load meta.mat (contains synset info)
############################################

meta = sio.loadmat("data/meta.mat")["synsets"]

wnids = []
words = []
class_ids = []

for i in range(len(meta)):
    class_ids.append(int(meta[i][0][0][0][0]))
    wnids.append(str(meta[i][0][1][0]))
    words.append(str(meta[i][0][2][0]))

# ClassID → Label mapping (canonical)
id_to_label = {cid: w for cid, w in zip(class_ids, words)}

############################################
# 2. Read validation ground truth labels
############################################

with open("data/ILSVRC2012_validation_ground_truth.txt") as f:
    val_gt = [int(x.strip()) for x in f]

############################################
# 3. Build mapping: val image → human label
############################################

val_mapping = {}

for i, cid in enumerate(val_gt, start=1):
    filename = f"ILSVRC2012_val_{i:08d}.JPEG"
    val_mapping[filename] = id_to_label[cid]

############################################
# 4. SAVE OUTPUT FILES
############################################

# --- A. Save ClassID → label mapping ---
with open("imagenet_classid_to_label.json", "w") as f:
    json.dump(id_to_label, f, indent=2)

with open("imagenet_classid_to_label.txt", "w") as f:
    for cid, lbl in id_to_label.items():
        f.write(f"{cid}\t{lbl}\n")

# --- B. Save val-image → label mapping ---
with open("imagenet_val_filename_to_label.json", "w") as f:
    json.dump(val_mapping, f, indent=2)

with open("imagenet_val_filename_to_label.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    for fn, lbl in val_mapping.items():
        writer.writerow([fn, lbl])

with open("imagenet_val_filename_to_label.txt", "w") as f:
    for fn, lbl in val_mapping.items():
        f.write(f"{fn}\t{lbl}\n")

print("Saved:")
print(" - imagenet_classid_to_label.json")
print(" - imagenet_classid_to_label.txt")
print(" - imagenet_val_filename_to_label.json")
print(" - imagenet_val_filename_to_label.csv")
print(" - imagenet_val_filename_to_label.txt")


