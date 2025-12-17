from transformers import UperNetForSemanticSegmentation

def load_fair_cnn(num_classes: int):
    return UperNetForSemanticSegmentation.from_pretrained(
        "openmmlab/upernet-convnext-tiny",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

def load_fair_vit(num_classes: int):
    return UperNetForSemanticSegmentation.from_pretrained(
        "openmmlab/upernet-swin-tiny",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
