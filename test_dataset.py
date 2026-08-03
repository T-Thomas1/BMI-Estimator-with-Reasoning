# test_dataset.py
from datasets import load_dataset

print("Testing access to UniqueData/body-measurements-dataset...")

try:
    dataset = load_dataset("UniqueData/body-measurements-dataset", split="train", streaming=True)
    first_sample = next(iter(dataset))

    print("\n✅ SUCCESS! You have access to the dataset!")
    print(f"\nKeys: {first_sample.keys()}")

    # Check what's inside 'label'
    print(f"\n'label' contents:")
    label = first_sample['label']
    print(f"  Type: {type(label)}")

    if isinstance(label, dict):
        print(f"  Keys in label: {label.keys()}")
        print(f"\n  Sample values:")
        for key, value in label.items():
            if isinstance(value, (int, float, str)):
                print(f"    {key}: {value}")
            else:
                print(f"    {key}: {type(value)}")
    else:
        print(f"  Value: {label}")

    # Also check 'image'
    print(f"\n'image' contents:")
    image = first_sample['image']
    print(f"  Type: {type(image)}")
    if hasattr(image, 'size'):
        print(f"  Size: {image.size}")

except Exception as e:
    print(f"\n❌ Failed: {e}")