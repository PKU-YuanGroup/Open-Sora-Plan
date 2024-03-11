import argparse
from caption_refiner import CaptionRefiner
from gpt_combinator import caption_summary, caption_qa

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--root_path", required=True, help="The path to repo.")
    parser.add_argument("--api_key", required=True, help="OpenAI API key.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    myrefiner = CaptionRefiner(
        sample_num=6, add_detect=True, add_pos=True, add_attr=True,
        openai_api_key = args.api_key,
        openai_api_base = "https://one-api.bltcy.top/v1",
    )

    results = myrefiner.caption_refine(
        video_path="./dataset/test_videos/video1.gif",
        org_caption="A red mustang parked in a showroom with american flags hanging from the ceiling.",
        model_path = args.root_path + "/ckpts/SPHINX-Tiny",
    )

    final_caption = myrefiner.gpt_summary(results)
    
    print(final_caption)
