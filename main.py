import torch

from PIL import Image

from transformers import LlavaForConditionalGeneration, GenerationConfig
from processor import MFCDProcessor


def main():
    model_name_or_path = "llava-hf/llava-1.5-7b-hf"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    processor = MFCDProcessor.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
        use_fast=True,
        high_pass_cutoff=0.1,
        low_pass_cutoff=0.9,
        device=device,
    )

    image = Image.open('./test.jpg')
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": "Please describe this image in detail."
                }
            ]
        }
    ]

    query = processor.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=False,
    ),

    inputs = processor.__call__(
        text=query,
        images=image,
        return_tensors="pt",
        padding=True,
        padding_side="left",
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path,
    ).to(device=device)

    inputs = inputs.to(device=model.device, dtype=model.dtype)

    generation_config = GenerationConfig(
        temperature=1.2,
        do_sample=True,
        use_cache=True,
        max_new_tokens=4096,
        mfc_low_alpha=1.0,
        mfc_high_alpha=1.0,
        mfc_beta=1.0,
        mfc_jsd=True,
        mfc_entropy=True,
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )

    responses = []
    responses.extend(
        (
            processor.decode(outputs[i][inputs["input_ids"][i].shape[0]:], skip_special_tokens=True)
            for i in range(outputs.size(0))
        )
    )

    print("\n\n".join(responses))


if __name__ == '__main__':
    main()
