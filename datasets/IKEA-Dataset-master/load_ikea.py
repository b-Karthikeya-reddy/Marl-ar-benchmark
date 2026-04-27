import os
import re

def parse_furniture_txt(txt_path):
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
        
        # Get dimensions
        length = re.search(r'Length:\s*([\d.]+)\s*cm', item)
        width = re.search(r'Width:\s*([\d.]+)\s*cm', item)
        height = re.search(r'Height:\s*([\d.]+)\s*cm', item)
        
        if length: furniture['length'] = float(length.group(1))
        if width: furniture['width'] = float(width.group(1))
        if height: furniture['height'] = float(height.group(1))
        
        if furniture:
            furniture_items.append(furniture)
    
    return furniture_items

# Test it
txt_file = "datasets/IKEA-Dataset-master/Living_Room_1/Living Room 1/Coffee and Side Tables"
for f in os.listdir(txt_file):
    if f.endswith('.txt'):
        items = parse_furniture_txt(os.path.join(txt_file, f))
        print(f"Found {len(items)} furniture items")
        for item in items[:3]:
            print(item)