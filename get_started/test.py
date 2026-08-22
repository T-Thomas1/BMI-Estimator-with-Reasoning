from datasets import load_dataset

ds = load_dataset("Dongrae/celeb-fbi")

# Check what the image column contains
sample = ds['train'][0]
print("Type of 'image':", type(sample['image']))
if isinstance(sample['image'], dict):
    print("Keys in image dict:", sample['image'].keys())
    for key, value in sample['image'].items():
        print(f"  {key}: {type(value)}")
else:
    print("Image is:", sample['image'])