import argparse
import os
import json
import time
import PIL
import torch
import models
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Demo Inference for deepfake_detection model')
    # Input and Output Paths
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input video/audio/image file or folder')
    parser.add_argument('--output_path', type=str, required=True, help='Directory where results will be saved. Inference results will be saved as JSON files.')
    parser.add_argument('--device', type=str, default='cuda', help='Set to "cuda" for GPU or "cpu"')
    parser.add_argument('--model_input_size', type=int, default=384, help='Input size of the model (224 or 384)', choices=[224, 384])
    parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from', default="pretrained_weights/model_v11_ViT_384_base_ckpt.pt")
    return parser.parse_args()

def load_model(ckpt_path='pretrained_weights/model_v11_ViT_384_base_ckpt.pt', input_size=384, device='cuda'):
    """
    Loads the model
    Args:
        ckpt_path: path to checkpoint
        input_size: input size of the model (224 or 384)
        device: device rank to load the model
    Returns:
        model: loaded model
    """
    device = torch.device(device)
    ckpt=torch.load(ckpt_path, map_location=device)

    args = argparse.Namespace()
    args.model_size='small'
    args.patch_size=16
    args.freeze_backbone=False
    args.input_size=input_size
    model = models.ViTClassifier(args, device=device)
    model.load_state_dict(ckpt['model'])

    return model.to(device)

def get_result_description(prob):
    if prob > 0.9:
        return "This sample is classified as generated with high confidence."
    elif prob > 0.7:
        return "This sample is classified as generated with moderate confidence."
    elif prob > 0.6:
        return "This sample is classified as generated with low confidence."
    elif prob > 0.5:
        return "This sample is classified as generated with very low confidence."
    elif prob > 0.4:
        return "This sample is classified as real with very low confidence."
    elif prob > 0.3:
        return "This sample is classified as real with low confidence."
    elif prob > 0.1:
        return "This sample is classified as real with moderate confidence."
    else:
        return "This sample is classified as real with high confidence."

def run_inference(args, input_path, output_path, model=None, device='cuda'):
    start_time = time.time()

    if model == None:
        model = load_model(device=device, ckpt_path=args.checkpoint_path, input_size=args.model_input_size)
        model.eval()

    input_image = PIL.Image.open(input_path).convert('RGB')
    fake_prob = model.forward(input_image)
    fake_prob = fake_prob.item() # Convert to float

    # Save result as JSON
    json_result = {
        "Input File": os.path.abspath(input_path), # Input file name
        "Date": str(datetime.now()), # Current date and time
        "Result": {"Fake Probability": fake_prob}, # Probabilities
        "Result Description": get_result_description(fake_prob), # description
        "Analysis Time in Second": round(time.time() - start_time, 2), # Time taken for analysis
        "Device": device # cuda or cpu
    }

    # Save the result in the output path
    result_path = os.path.join(output_path, f"result_{datetime.now().strftime('%m%d%Y_%H%M%S')}_{os.path.basename(input_path).replace('.','_')}.json")
    with open(result_path, 'w') as json_file:
        json.dump(json_result, json_file, indent=4)

def main():
    args = parse_args()
    if args.device.isdigit():
        args.device = int(args.device)
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # load model
    model = load_model(ckpt_path=args.checkpoint_path, input_size=args.model_input_size, device=args.device)
    model.eval()
    supported_extensions = ["jpeg","jpg","png","bmp","gif","tiff","webp","apng","mpo"]

    # check if input is a folder and handle it
    if os.path.isdir(args.input_path):
        # If input is a folder, iterate over all files
        input_files = os.listdir(args.input_path)
        input_files = [file for file in input_files if file.split('.')[-1] in supported_extensions] # filter out supported extensions
        for file in input_files:
            file_path = os.path.join(args.input_path, file)
            run_inference(args, file_path, args.output_path, model, args.device)
    else:
        # If input is a single file
        if os.path.basename(args.input_path).split('.')[-1] not in supported_extensions:
            print(f"WARNING: Unsupported file format `{os.path.basename(args.input_path).split('.')[-1]}`. Supported formats are: {supported_extensions}.")
        run_inference(args, args.input_path, args.output_path, model, device=args.device)

if __name__ == "__main__":
        main()