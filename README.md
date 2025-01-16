Community Forensics: \
Using Thousands of Generators to Train Fake Image Detectors
---

Repository for [Community Forensics: Using Thousands of Generators to Train Fake Image Detectors](https://arxiv.org/abs/2411.04125). \
([Project page](https://jespark.net/projects/2024/community_forensics/))

Currently, we release a simple evaluation pipeline that outputs a probability of an input image being generated. 

## Usage example

1. Download the checkpoints. [Link (Dropbox)](https://www.dropbox.com/scl/fi/e8titz35ci9a2ij1oq5mu/model_weights.tar?rlkey=tmyz3tjqf7b4dg071kypsgoal&st=09ud9hdj&dl=0)
2. Install the required libraries -- `torch`, `torchvision`, `pillow`, `timm`
    - `pip install -r requirements.txt`
3. Run the evaluation pipeline.
    - If evaluating a single file:
      - `python main.py --input_path="test_image.jpeg" --output_path="./results_dir" --device="cuda" --checkpoint_path="pretrained_weights/model_v11_ViT_384_base_ckpt.pt"`
    - If evaluating a folder:
      - `python main.py --input_path="./path_to_test_images" --output_path="./results_dir" --device="cuda" --checkpoint_path="pretrained_weights/model_v11_ViT_384_base_ckpt.pt"`
4. Check the `.json` files containing the results under the output path designated by `--output_path` argument.

## Citation

```
@misc{park2024communityforensics,
    title={Community Forensics: Using Thousands of Generators to Train Fake Image Detectors}, 
    author={Jeongsoo Park and Andrew Owens},
    year={2024},
    eprint={2411.04125},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2411.04125}, 
}
```
