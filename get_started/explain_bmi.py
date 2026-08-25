import argparse
import os
import numpy as np
import torch
from openai import OpenAI
from model import SEDensenet121, SEDensenet201, load_pretrained_densenet #, load_pretrained_densenet201
from dataset import BMIDataset, df_test, df_train
from predict_bmi import load_model, predict_bmi

def load_api_key():
    """OpenAI key: env var first, else a gitignored local file."""
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    if os.path.exists("openai_key.txt"):
        return open("openai_key.txt").read().strip()
    raise RuntimeError(
        "No OpenAI key found. Set OPENAI_API_KEY, or create openai_key.txt in the repo root."
    )

def explain_bmi(client, pred, lower, upper, age, gender):
    system = (
        "You are a clear, plain-English explainer for a BMI estimation feature. "
        "A person uploaded a photo and the system produced a BMI estimate: your job "
        "is to explain that estimate honestly and kindly.\n"
        "Rules: use 2-3 short plain sentences; no jargon. Be honest about uncertainty. "
        "Give no medical advice, diagnosis, or treatment suggestions. Be non-judgemental "
        "about weight or body size -- BMI is a rough population metric, not a personal "
        "health verdict. End with a caveat only when it is genuinely relevant."
    )
    user = (
        f"A vision model estimated a person's BMI from one photo.\n"
        f"- Predicted BMI: {pred:.1f}\n"
        f"- 90% confidence interval: [{lower:.1f}, {upper:.1f}] "
        f"(the true BMI is in this range about 90% of the time)\n"
        f"- Age: {age}, gender: {gender}\n\n"
        f"Context: adult BMI under 18.5 is underweight, 18.5-24.9 health, 25-29.9 "
        f"overweight, 30+ obese. This model systematically under-predicts very high or "
        f"very low BMIs, so edge-of-range estimates are less trustworthy. A wider interval "
        f"means lower certainty.\n\n"
        f"Explain in 2-3 plain sentences: what this estimate means, how much confidence "
        f"the person should place in it, and one relevant caveat if the interval is wide "
        f"or the value is extreme."
    )
    resp = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                 temperature=0.3,
    )
    return resp.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description='BMI Prediction using DenseNet')
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument('--model_path', type=str, default='weights/best_model.ckpt',
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='densenet121',
                        choices=['densenet121', 'densenet201'],
                        help='Type of model to use')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for computation')
    parser.add_argument('--q_hat_path', type=str, default='weights/q_hat.npy', 
                        help='Path to saved conformal q_hat (from calibrate_bmi')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else: 
        device = args.device

    model = load_model(args.model_path, args.model_type, device) #Load our model
    q_hat = float(np.load(args.q_hat_path)) #Load our q_hat

    ds = BMIDataset(df_test) #pull one sample and run inference on it 
    image, bmi, sample_id = ds[args.index]
    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(device, dtype=torch.float32)).cpu().numpy().flatten()[0]

    row = next(r for r in df_test if r["id"] == sample_id) #Map back to the raw dataset
    age, gender = row ["age"], row["gender"]
    actual = bmi

    lower, upper, = pred - q_hat, pred + q_hat #Conformal interval. Prediction plus or minus the calibration
    print(f"id {sample_id} | age {age} | gender {gender}")
    print(f"predicted BMI {pred:.1f} | 90% interval [{lower:.1f}, {upper:.1f}] | actual {actual:.1f}")

    client = OpenAI(api_key=load_api_key())
    print("\n--- explanation ---")
    print(explain_bmi(client, pred, lower, upper, age, gender))

if __name__ == "__main__":
    main()