## Introduction

DSCENet: Dynamic Screening and Clinical-Enhanced Multimodal Fusion for MPNs Subtype Classification is accepted by MICCAI 2024, oral.
You can access the full article [here](https://arxiv.org/abs/2407.08167).

We propose a Dynamic Screening and Clinical-Enhanced Network (DSCENet) for the subtype classification of MPNs on the multimodal fusion of whole slide images (WSIs) and clinical information. 


## Usage
We have uploaded part of the code, in which the core model is located in Model.DSCE
```
        from Models.model_DSCE import DSCE
        model_dict = {
            "clinic_factor": args.clinic_factor,
            "n_classes": args.n_classes,
            "fusion": args.fusion,
        }
        model = DSCE(**model_dict)
```
Our code is currently undergoing generalizability testing. Stay tuned.


## Availability of dataset
We are trying to communicate and make the feature of the data open source.. Stay tuned.


## Citation

```
@InProceedings{Zha_DSCENet_MICCAI2024,
        author = { Zhang, Yuan and Qi, Yaolei and Qi, Xiaoming and Wei, Yongyue and Yang, Guanyu},
        title = { { DSCENet: Dynamic Screening and Clinical-Enhanced Multimodal Fusion for MPNs Subtype Classification } },
        booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2024},
        year = {2024},
        publisher = {Springer Nature Switzerland},
        volume = {LNCS 15004},
        month = {October},
        page = {pending}
}
```





