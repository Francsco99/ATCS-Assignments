import os
import json
from collections import defaultdict, Counter

def analyze_dataset(dataset_path):
    sources = os.listdir(dataset_path)
    total_sources = len(sources)
    
    total_objects = 0
    objects_per_source = []
    attribute_counter = Counter()
    attribute_values = defaultdict(list)
    
    for source in sources:
        source_path = os.path.join(dataset_path, source)
        if os.path.isdir(source_path):
            files = os.listdir(source_path)
            num_files = len(files)
            objects_per_source.append(num_files)
            total_objects += num_files
            
            for file in files:
                file_path = os.path.join(source_path, file)
                with open(file_path, 'r') as f:
                    obj = json.load(f)
                    for attr, value in obj.items():
                        attribute_counter[attr] += 1
                        attribute_values[attr].append(value)
    
    avg_objects_per_source = total_objects / total_sources
    min_objects_per_source = min(objects_per_source)
    max_objects_per_source = max(objects_per_source)
    
    avg_attributes_per_object = sum(attribute_counter.values()) / total_objects
    min_attributes_per_object = min(attribute_counter.values())
    max_attributes_per_object = max(attribute_counter.values())
    
    most_common_attributes = attribute_counter.most_common(10)
    
    stats = {
        "total_sources": total_sources,
        "total_objects": total_objects,
        "avg_objects_per_source": avg_objects_per_source,
        "min_objects_per_source": min_objects_per_source,
        "max_objects_per_source": max_objects_per_source,
        "avg_attributes_per_object": avg_attributes_per_object,
        "min_attributes_per_object": min_attributes_per_object,
        "max_attributes_per_object": max_attributes_per_object,
        "most_common_attributes": most_common_attributes
        #"attribute_values": attribute_values
    }
    
    return stats

# Esempio di utilizzo
dataset_path = "monitor/monitor_specs"
stats = analyze_dataset(dataset_path)
for k,v in stats.items():
    print(k,v)
