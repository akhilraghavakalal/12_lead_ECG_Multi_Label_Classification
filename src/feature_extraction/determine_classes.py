import os
import json
from collections import Counter
from tqdm import tqdm

def get_unique_classes(data_dir):
    print("Inside determine classes")
    all_classes = []
    total_files = sum(len([f for f in os.listdir(os.path.join(data_dir, subset)) if f.endswith('_header.json')]) for subset in ['train', 'validation'])
    
    with tqdm(total=total_files, desc="Processing header files") as pbar:
        for subset in ['train', 'validation']:
            subset_dir = os.path.join(data_dir, subset)
            for file in os.listdir(subset_dir):
                if file.endswith('_header.json'):
                    with open(os.path.join(subset_dir, file), 'r') as f:
                        header = json.load(f)
                        dx = header.get('Dx', [])
                        if isinstance(dx, str):
                            dx = dx.split(',')
                        all_classes.extend(dx)
                    pbar.update(1)
    
    class_counts = Counter(all_classes)
    print("Total unique classes:", len(class_counts))
    print("Top 27 classes by frequency:")
    for cls, count in class_counts.most_common(27):
        print(f"{cls}: {count}")

    return [cls for cls, _ in class_counts.most_common(27)]