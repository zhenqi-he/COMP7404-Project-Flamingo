from open_flamingo import create_model_and_transforms
from PIL import Image
import requests
from huggingface_hub import hf_hub_download
import torch
import os
import glob
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_1_path',required=True, help="The path for images of the first category") 
    parser.add_argument('--class_2_path',required=True, help="The path for images of the second category")
    parser.add_argument('--test_cases_path',required=True, help="The path for images of testcases")

    args = parser.parse_args()
    class_1_path = args.class_1_path
    class_2_path = args.class_2_path
    test_cases_path = args.test_cases_path
   
    # class_1_path = "/Users/zhenqihe/Desktop/Master/COMP7404/COMP7404-Project/hanbao"
    class1_name = class_1_path.split('/')[-1]
    # class_2_path = "/Users/zhenqihe/Desktop/Master/COMP7404/COMP7404-Project/tuanzi"
    class2_name = class_2_path.split('/')[-1]
    # test_cases_path = "/Users/zhenqihe/Desktop/Master/COMP7404/COMP7404-Project/test_cases"


    img_1_list = glob.glob(os.path.join(class_1_path,'*.jpg'))
    img_1_list = [ Image.open(i) for i in img_1_list]

    img_2_list = glob.glob(os.path.join(class_2_path,'*.jpg'))
    img_2_list = [ Image.open(i) for i in img_2_list]



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



    for i in range(1,len(glob.glob(os.path.join(test_cases_path,'*.jpg')))):
    # if 1:
        # i = 2

        img_3_list = [ Image.open(f"/Users/zhenqihe/Desktop/Master/COMP7404/COMP7404-Project/test_cases/{i}.jpg")]


        vision_x_1 = [image_processor(i).unsqueeze(0) for i in img_1_list]
        vision_x_2 = [image_processor(i).unsqueeze(0) for i in img_2_list]
        vision_x_3 = [image_processor(i).unsqueeze(0) for i in img_3_list]
        vision_x = vision_x_1 + vision_x_2 + vision_x_3
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)


        # vision_x_1 = torch.cat(vision_x_1, dim=0)
        # vision_x_1 = vision_x_1.unsqueeze(1).unsqueeze(0)

        tokenizer.padding_side = "left" # For generation padding tokens should be on the left
        tokenizer_list = [f"<image>An image of a cat named {class1_name}.<|endofchunk|>]"*len(img_1_list) + f"<image>An image of a cat named {class2_name}.<|endofchunk|>]"*len(img_2_list) + "<image>An image of a cat named" ]
        print(tokenizer_list)
        lang_x = tokenizer(
            tokenizer_list,
            return_tensors="pt",
        )

        generated_text = model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            max_new_tokens=3,
            num_beams=5,
        )

        print("Generated text: ", tokenizer.decode(generated_text[0]))

if __name__=='__main__':
    main()