# StreamVC
An unofficial pytorch implementation of [STREAMVC: REAL-TIME LOW-LATENCY VOICE CONVERSION](https://arxiv.org/pdf/2401.03078.pdf).
It was created for learning purposes, it isn't feature complete, and it doesn't replicate the results from the paper.

The streaming infernce isn't fully implemented.

```mermaid
flowchart LR 
    TS[Training Sample] -.-> SP
    SP -.-> HB[["(8) HubBert based\npseudo labels"]]
    CE -.-> LN[[LN+Linear+Softmax]] -.->L1((cross\nentropy\nloss))
    HB -.->L1
    subgraph Online Inference
        SP[Source\nSpeech] --> CE[["(1) Content\nEncoder"]] -->|"// (grad-stop)"| CL[Content\nLatent] --> CAT(( )) -->D[["(3) Decoder"]]
        SP --> f0[["(4) f0 estimation"]] --> f0y[["(5) f0 whitening"]] --> CAT
        SP --> FE[["(6) Frame Energy\nEstimation"]] --> CAT
    end
    subgraph "Offline Inference (Target Speaker)"
        TP[Target\nSpeech] --> SE[["(2) Speech\nEncoder"]] --- LP[["(7) Learnable\nPooling"]]-->SL[Speaker\nLatent]
    end
    TS -.-> TP
    SL --> |Conditioning| D
    D -.-> Dis[["(9) Discriminator"]]
    TS2[Training Sample] -.->Dis
    Dis -.-> L2((adversarial\nloss))
    Dis -.-> L3((feature\nloss))
    D -.-> L4((reconstruction\nloss))
    TS2 -.-> L4 
    classDef train fill:#337,color:#ccc;
    class TS,TS2,HB,LN,Dis train;
    classDef off fill:#733,color:#ccc;
    class TP,SE,LP,SL off;
    classDef else fill:#373,color:#ccc;
    class SP,CE,CL,D,f0,f0y,FE,CAT else;
```

## Example Usage
### Training
#### Requirements
To install the requirements for training run:
```bash
pip install -r requirements-training.txt
```
#### Running the training script
`train.py` is the python script for training, it uses [🤗 Accelerate](https://huggingface.co/docs/accelerate).
To cofigure Accelerate to your enviroenment use [`accelerator config`](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-config).

To launch the script, run:
```bash
accelerate launch [ACCELERATE-OPTIONS] train.py [TRAINING-OPTIONS]
```
To see the available training options, run: 

```
python train.py --help
```
### Inference
#### Requirements
To install the requirements for inference run:
```bash
pip install -r requirements-inference.txt
```
#### Running the script
 `inference.py` is the python script for inference on a single source & target combo.


To launch the script, run:
```bash
python inference.py [INFERENCE-OPTIONS] -s <source_speech> -t <target_speech>
-o <output_file>
```
To see the available inference options, run: 

```
python inference.py --help
```

## Acknowledgements
This project was made possible by the following open source projects:

 - For the encoder-decoder architecture (based on SoundStream) we based our code on [AudioLM's official implementation](https://github.com/lucidrains/audiolm-pytorch).
 - For the multi-scale discriminator and the discriminator losses we based our code on [MelGan's official implementation](https://github.com/descriptinc/melgan-neurips).
 -  For the HuBert discrete units computation we used the HuBert + KMeans implementation from [SoftVC's official implementation](https://github.com/bshall/soft-vc).
 - For the Yin algorithm we based our implementation on the [torch-yin package](https://github.com/brentspell/torch-yin).