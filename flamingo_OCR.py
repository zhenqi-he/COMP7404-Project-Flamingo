from open_flamingo import create_model_and_transforms
from PIL import Image
import requests
import torch
import os
import glob
from huggingface_hub import hf_hub_download
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_paths',required=True, help="The path for images") 
    

    args = parser.parse_args()
    image_paths = args.image_paths
   

    # image_paths = "/Users/zhenqihe/Desktop/Master/COMP7404/COMP7404-Project/OCR"
    images_list = glob.glob(os.path.join(image_paths,'*.jpeg')) + glob.glob(os.path.join(image_paths,'*.png'))
    class_lists = [i.split('/')[-1].split('.')[0].replace('_',' ') for i in images_list]
    img_list = [ Image.open(i) for i in images_list]



    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
        tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
        cross_attn_every_n_layers=1,
        # cache_dir="PATH/TO/CACHE/DIR"  # Defaults to ~/.cache
    )
    checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)

    vision_x = [image_processor(i).unsqueeze(0) for i in img_list]
    vision_x = torch.cat(vision_x, dim=0)
    vision_x = vision_x.unsqueeze(1).unsqueeze(0)


    # vision_x_1 = torch.cat(vision_x_1, dim=0)
    # vision_x_1 = vision_x_1.unsqueeze(1).unsqueeze(0)

    tokenizer.padding_side = "left" # For generation padding tokens should be on the left
    tokens = ''
    for i in class_lists[:-1]:
        tokens += f"<image>This is the logo of {i}<|endofchunk|>]"
    tokens += '<image>This is the logo of'
    tokenizer_list = [tokens]
    print(tokenizer_list)
    lang_x = tokenizer(
        tokenizer_list,
        return_tensors="pt",
    )

    generated_text = model.generate(
        vision_x=vision_x,
        lang_x=lang_x["input_ids"],
        attention_mask=lang_x["attention_mask"],
        max_new_tokens=5,
        num_beams=3,
    )

    print("Generated text: ", tokenizer.decode(generated_text[0]))

if __name__=='__main__':
    main()