## remote inference

VCM (video coding for machines) type pipeline 
     ┌───────────┐       ┌───────────┐      ┌─────────────┐
     │           │       │           │      │             │
────►│  Encoder  ├──────►│  Decoder  ├─────►│   NN Task   ├────►
     │           │       │           │      │             │
     └───────────┘       └───────────┘      └─────────────┘
                      <-------------- Remote Server ----------->


### split inference
FCM (feature coding for machines) type pipeline 
     ┌─────────────────┐                                         ┌─────────────────┐
     │                 │     ┌───────────┐     ┌───────────┐     │                 │
     │     NN Task     │     │           │     │           │     │      NN Task    │
────►│                 ├────►│  Encoder  ├────►│  Decoder  ├────►│                 ├────►
     │      Part 1     │     │           │     │           │     │      Part 2     │
     │                 │     └───────────┘     └───────────┘     │                 │
     └─────────────────┘                                         └─────────────────┘

Evaluation using Detectron2 API _only_

### fo_vcm

Legacy VCM type pipeline using FyftyOne library for evaluation