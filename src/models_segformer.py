from transformers import SegformerForSemanticSegmentation

def load_segformer(num_classes: int):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        )
    return model
