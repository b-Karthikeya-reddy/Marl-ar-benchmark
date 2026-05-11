import os
import re
import csv

def parse_furniture_txt(txt_path, category, room):
    """Parse IKEA furniture dimensions from txt file"""
    furniture_items = []
    
    with open(txt_path, 'r') as f:
        content = f.read()
    
    # Split by dashes
    items = content.split('---')
    
    for item in items:
        if not item.strip():
            continue
            
        furniture = {}
        furniture['room'] = room
        furniture['category'] = category
        
        # Get dimensions
        length = re.search(r'Length:\s*([\d.]+)\s*cm', item)
        width = re.search(r'Width:\s*([\d.]+)\s*cm', item)
        height = re.search(r'Height:\s*([\d.]+)\s*cm', item)
        
        if length: furniture['length'] = float(length.group(1))
        if width: furniture['width'] = float(width.group(1))
        if height: furniture['height'] = float(height.group(1))
        
        if len(furniture) > 2:
            furniture_items.append(furniture)
    
    return furniture_items

def load_all_rooms(base_path):
    """Load all furniture data from all unzipped room folders"""
    all_furniture = []
    
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue
            
        # Go into the room subfolder
        for room_folder in os.listdir(folder_path):
            room_path = os.path.join(folder_path, room_folder)
            if not os.path.isdir(room_path):
                continue
                
            # Go into each category
            for category in os.listdir(room_path):
                category_path = os.path.join(room_path, category)
                if not os.path.isdir(category_path):
                    continue
                    
                # Parse txt files
                for f in os.listdir(category_path):
                    if f.endswith('.txt'):
                        txt_path = os.path.join(category_path, f)
                        items = parse_furniture_txt(txt_path, category, room_folder)
                        all_furniture.extend(items)
    
    return all_furniture

# Load all rooms
base_path = "datasets/IKEA-Dataset-master"
all_furniture = load_all_rooms(base_path)

# Save to CSV
csv_path = "datasets/ikea_furniture.csv"
if all_furniture:
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['room', 'category', 'length', 'width', 'height'])
        writer.writeheader()
        writer.writerows(all_furniture)

print(f"Total furniture items loaded: {len(all_furniture)}")
print(f"Saved to {csv_path}")
print("\nSample items:")
for item in all_furniture[:5]:
    print(item)