import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_bedlam_measurements(data_path="D:/BMI-Dataset/be_imagedata_download"):
    """Extract body measurements from BEDLAM 3D skeleton data"""
    print("\n📥 Loading BEDLAM 3D measurements...")

    data_path = Path(data_path)
    csv_files = list(data_path.glob("**/be_seq.csv"))

    if not csv_files:
        print("No BEDLAM CSV files found")
        return None

    all_data = []

    for csv_file in csv_files[:100]:  # Process first 100 files
        try:
            df = pd.read_csv(csv_file)

            if 'Body' not in df.columns:
                continue

            # Extract key body parts
            body_parts = {}
            for _, row in df.iterrows():
                body_part = row['Body']
                x, y, z = row['X'], row['Y'], row['Z']
                body_parts[body_part] = (x, y, z)

            # Find body landmarks
            left_shoulder = right_shoulder = left_hip = right_hip = head = ankle = None

            for part, coords in body_parts.items():
                part_lower = str(part).lower()
                if 'shoulder' in part_lower:
                    if 'left' in part_lower or 'l_' in part_lower:
                        left_shoulder = coords
                    elif 'right' in part_lower or 'r_' in part_lower:
                        right_shoulder = coords
                elif 'hip' in part_lower or 'pelvis' in part_lower:
                    if 'left' in part_lower or 'l_' in part_lower:
                        left_hip = coords
                    elif 'right' in part_lower or 'r_' in part_lower:
                        right_hip = coords
                elif 'head' in part_lower or 'nose' in part_lower:
                    head = coords
                elif 'ankle' in part_lower or 'foot' in part_lower:
                    if 'left' in part_lower or 'l_' in part_lower:
                        ankle = coords

            # Calculate measurements
            if left_shoulder and right_shoulder:
                shoulder_width = np.sqrt(
                    (right_shoulder[0] - left_shoulder[0])**2 +
                    (right_shoulder[1] - left_shoulder[1])**2 +
                    (right_shoulder[2] - left_shoulder[2])**2
                )
            else:
                continue

            if left_hip and right_hip:
                hip_width = np.sqrt(
                    (right_hip[0] - left_hip[0])**2 +
                    (right_hip[1] - left_hip[1])**2 +
                    (right_hip[2] - left_hip[2])**2
                )
            else:
                continue

            if head and ankle:
                height = np.sqrt(
                    (ankle[0] - head[0])**2 +
                    (ankle[1] - head[1])**2 +
                    (ankle[2] - head[2])**2
                )
            else:
                height = 170  # Default fallback

            # Calculate ratios
            swr = shoulder_width / hip_width if hip_width > 0 else 0
            whr = hip_width / height if height > 0 else 0

            # Estimate BMI from body proportions
            bmi = 35 * whr + 10 * (1.4 - swr)
            bmi = np.clip(bmi, 15, 45)

            all_data.append({
                'swr': swr,
                'whr': whr,
                'bmi': bmi,
                'source': 'BEDLAM'
            })

        except Exception as e:
            continue

    if all_data:
        df = pd.DataFrame(all_data)
        print(f"✓ Loaded {len(df)} samples from BEDLAM")
        print(f"  SWR range: {df['swr'].min():.2f} - {df['swr'].max():.2f}")
        print(f"  WHR range: {df['whr'].min():.2f} - {df['whr'].max():.2f}")
        print(f"  BMI range: {df['bmi'].min():.1f} - {df['bmi'].max():.1f}")
        return df

    return None

def load_unidata_dataset(data_path=None):
    """
    Load UniqueData dataset once purchased.
    This is a placeholder - update with actual file paths and column names.
    """
    if data_path is None:
        print("\n⚠️ UniqueData dataset not available yet")
        print("   Will be added after purchase")
        return None

    print("\n📥 Loading UniqueData dataset...")
    data_path = Path(data_path)

    # TODO: Update these paths and column names based on actual dataset structure
    # Example:
    # df = pd.read_csv(data_path / "measurements.csv")
    # df['swr'] = df['shoulder_width'] / df['hip_width']
    # df['whr'] = df['waist'] / df['height']

    return None

def combine_datasets():
    """Combine BEDLAM and UniqueData (when available)"""
    print("=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    all_data = []

    # Load BEDLAM data
    bedlam_df = load_bedlam_measurements()
    if bedlam_df is not None:
        all_data.append(bedlam_df)

    # Load UniqueData when available
    unidata_df = load_unidata_dataset()
    if unidata_df is not None:
        all_data.append(unidata_df)

    if not all_data:
        print("\n⚠️ No datasets loaded. Using synthetic data fallback.")
        return create_synthetic_fallback()

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ TOTAL DATA: {len(combined_df)} samples")
    print(f"   Sources: {combined_df['source'].value_counts().to_dict()}")

    # Save combined dataset
    Path("data").mkdir(exist_ok=True)
    combined_df.to_csv('data/combined_dataset.csv', index=False)
    print(f"   Saved to: data/combined_dataset.csv")

    return combined_df

def create_synthetic_fallback():
    """Fallback to synthetic data"""
    print("\n📊 Creating synthetic dataset...")
    np.random.seed(42)
    n_samples = 2000

    swr = np.random.uniform(1.2, 1.6, n_samples)
    whr = np.random.uniform(0.35, 0.65, n_samples)
    bmi = 35 * whr + 10 * (1.4 - swr) + np.random.normal(0, 1.5, n_samples)
    bmi = np.clip(bmi, 15, 45)

    df = pd.DataFrame({
        'swr': swr,
        'whr': whr,
        'bmi': bmi,
        'source': 'synthetic'
    })

    print(f"✓ Created {len(df)} synthetic samples")
    return df

def prepare_training_data(df):
    """Prepare data for model training"""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = df[['swr', 'whr']].values
    y = df['bmi'].values

    # Remove NaN values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n📊 Training data ready:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    df = combine_datasets()
    print(f"\n📊 Dataset summary:")
    print(df[['swr', 'whr', 'bmi']].describe())