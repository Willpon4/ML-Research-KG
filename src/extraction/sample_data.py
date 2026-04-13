"""
Generate sample paper data for testing the pipeline.
Creates realistic-looking paper/author data without needing API calls.
This allows testing the KG builder, SPARQL queries, and ML pipeline.
"""

import json
import random
import os
from pathlib import Path

random.seed(42)

# ============================================================
# Seed data: real influential ML papers and authors
# ============================================================

PAPERS = [
    # Transformers & Attention
    {"paperId": "p001", "title": "Attention Is All You Need", "year": 2017,
     "citationCount": 95000, "venue": "NeurIPS",
     "authors": [
         {"authorId": "a001", "name": "Ashish Vaswani"},
         {"authorId": "a002", "name": "Noam Shazeer"},
         {"authorId": "a003", "name": "Niki Parmar"},
         {"authorId": "a004", "name": "Jakob Uszkoreit"},
     ],
     "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
     "fieldsOfStudy": ["Computer Science"],
     "references": [], "citations": ["p002", "p003", "p004", "p005", "p006", "p007", "p009", "p010"]},

    {"paperId": "p002", "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "year": 2019,
     "citationCount": 72000, "venue": "NAACL",
     "authors": [
         {"authorId": "a005", "name": "Jacob Devlin"},
         {"authorId": "a006", "name": "Ming-Wei Chang"},
         {"authorId": "a007", "name": "Kenton Lee"},
     ],
     "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p001"], "citations": ["p004", "p007", "p010"]},

    {"paperId": "p003", "title": "Language Models are Few-Shot Learners", "year": 2020,
     "citationCount": 28000, "venue": "NeurIPS",
     "authors": [
         {"authorId": "a008", "name": "Tom Brown"},
         {"authorId": "a009", "name": "Benjamin Mann"},
         {"authorId": "a010", "name": "Nick Ryder"},
         {"authorId": "a002", "name": "Noam Shazeer"},
     ],
     "abstract": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. We show that scaling up language models greatly improves task-agnostic, few-shot performance.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p001", "p002"], "citations": ["p010", "p035"]},

    # GANs
    {"paperId": "p004", "title": "Generative Adversarial Nets", "year": 2014,
     "citationCount": 55000, "venue": "NeurIPS",
     "authors": [
         {"authorId": "a011", "name": "Ian Goodfellow"},
         {"authorId": "a012", "name": "Jean Pouget-Abadie"},
         {"authorId": "a013", "name": "Mehdi Mirza"},
         {"authorId": "a014", "name": "Yoshua Bengio"},
     ],
     "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability.",
     "fieldsOfStudy": ["Computer Science"],
     "references": [], "citations": ["p005", "p020"]},

    {"paperId": "p005", "title": "A Style-Based Generator Architecture for Generative Adversarial Networks", "year": 2019,
     "citationCount": 12000, "venue": "CVPR",
     "authors": [
         {"authorId": "a015", "name": "Tero Karras"},
         {"authorId": "a016", "name": "Samuli Laine"},
         {"authorId": "a017", "name": "Timo Aila"},
     ],
     "abstract": "We propose an alternative generator architecture for generative adversarial networks, borrowing from style transfer literature. The new architecture leads to an automatically learned, unsupervised separation of high-level attributes.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p001", "p004"], "citations": ["p020"]},

    # Graph Neural Networks
    {"paperId": "p006", "title": "Semi-Supervised Classification with Graph Convolutional Networks", "year": 2017,
     "citationCount": 25000, "venue": "ICLR",
     "authors": [
         {"authorId": "a018", "name": "Thomas Kipf"},
         {"authorId": "a019", "name": "Max Welling"},
     ],
     "abstract": "We present a scalable approach for semi-supervised learning on graph-structured data that is based on an efficient variant of convolutional neural networks which operate directly on graphs.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p001"], "citations": ["p007", "p008"]},

    {"paperId": "p007", "title": "Graph Attention Networks", "year": 2018,
     "citationCount": 18000, "venue": "ICLR",
     "authors": [
         {"authorId": "a020", "name": "Petar Velickovic"},
         {"authorId": "a021", "name": "Guillem Cucurull"},
         {"authorId": "a014", "name": "Yoshua Bengio"},
     ],
     "abstract": "We present graph attention networks (GATs), novel neural network architectures that operate on graph-structured data, leveraging masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p001", "p002", "p006"], "citations": ["p008"]},

    {"paperId": "p008", "title": "Inductive Representation Learning on Large Graphs", "year": 2017,
     "citationCount": 15000, "venue": "NeurIPS",
     "authors": [
         {"authorId": "a022", "name": "William Hamilton"},
         {"authorId": "a023", "name": "Zhitao Ying"},
         {"authorId": "a024", "name": "Jure Leskovec"},
     ],
     "abstract": "Low-dimensional embeddings of nodes in large graphs have proved extremely useful in a variety of prediction tasks. We propose GraphSAGE, a general inductive framework that leverages node feature information to efficiently generate node embeddings.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p006", "p007"], "citations": ["p027"]},

    # Computer Vision
    {"paperId": "p009", "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", "year": 2021,
     "citationCount": 30000, "venue": "ICLR",
     "authors": [
         {"authorId": "a025", "name": "Alexey Dosovitskiy"},
         {"authorId": "a004", "name": "Jakob Uszkoreit"},
     ],
     "abstract": "While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. We show that a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p001"], "citations": ["p011"]},

    {"paperId": "p010", "title": "Training language models to follow instructions with human feedback", "year": 2022,
     "citationCount": 8500, "venue": "NeurIPS",
     "authors": [
         {"authorId": "a026", "name": "Long Ouyang"},
         {"authorId": "a027", "name": "Jeff Wu"},
     ],
     "abstract": "Making language models bigger does not inherently make them better at following a user's intent. We show that fine-tuning with human feedback significantly improves language model behavior through reinforcement learning from human feedback (RLHF).",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p001", "p002", "p003"], "citations": ["p035"]},

    # Deep Learning Fundamentals
    {"paperId": "p011", "title": "Deep Residual Learning for Image Recognition", "year": 2016,
     "citationCount": 165000, "venue": "CVPR",
     "authors": [
         {"authorId": "a028", "name": "Kaiming He"},
         {"authorId": "a029", "name": "Xiangyu Zhang"},
         {"authorId": "a030", "name": "Shaoqing Ren"},
         {"authorId": "a031", "name": "Jian Sun"},
     ],
     "abstract": "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions.",
     "fieldsOfStudy": ["Computer Science"],
     "references": [], "citations": ["p005", "p009"]},

    {"paperId": "p012", "title": "ImageNet Large Scale Visual Recognition Challenge", "year": 2015,
     "citationCount": 42000, "venue": "IJCV",
     "authors": [
         {"authorId": "a032", "name": "Olga Russakovsky"},
         {"authorId": "a033", "name": "Jia Deng"},
         {"authorId": "a034", "name": "Hao Su"},
     ],
     "abstract": "The ImageNet Large Scale Visual Recognition Challenge is a benchmark in object category classification and detection on hundreds of object categories and millions of images.",
     "fieldsOfStudy": ["Computer Science"],
     "references": [], "citations": ["p011"]},

    # Diffusion Models
    {"paperId": "p013", "title": "Denoising Diffusion Probabilistic Models", "year": 2020,
     "citationCount": 12000, "venue": "NeurIPS",
     "authors": [
         {"authorId": "a035", "name": "Jonathan Ho"},
         {"authorId": "a036", "name": "Ajay Jain"},
         {"authorId": "a037", "name": "Pieter Abbeel"},
     ],
     "abstract": "We present high quality image synthesis results using diffusion probabilistic models, a class of latent variable models inspired by considerations from nonequilibrium thermodynamics.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p004"], "citations": ["p014", "p015"]},

    {"paperId": "p014", "title": "High-Resolution Image Synthesis with Latent Diffusion Models", "year": 2022,
     "citationCount": 9000, "venue": "CVPR",
     "authors": [
         {"authorId": "a038", "name": "Robin Rombach"},
         {"authorId": "a039", "name": "Andreas Blattmann"},
         {"authorId": "a040", "name": "Dominik Lorenz"},
     ],
     "abstract": "By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models achieve state-of-the-art synthesis results. We apply diffusion models in the latent space of powerful pretrained autoencoders.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p013", "p004"], "citations": ["p015"]},

    {"paperId": "p015", "title": "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding", "year": 2022,
     "citationCount": 5000, "venue": "NeurIPS",
     "authors": [
         {"authorId": "a041", "name": "Chitwan Saharia"},
         {"authorId": "a002", "name": "Noam Shazeer"},
     ],
     "abstract": "We present Imagen, a text-to-image diffusion model with an unprecedented degree of photorealism and a deep level of language understanding. Imagen builds on the power of large transformer language models.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p013", "p014", "p001", "p003"], "citations": []},

    # Knowledge Graphs
    {"paperId": "p016", "title": "Translating Embeddings for Modeling Multi-relational Data", "year": 2013,
     "citationCount": 8500, "venue": "NeurIPS",
     "authors": [
         {"authorId": "a042", "name": "Antoine Bordes"},
         {"authorId": "a043", "name": "Nicolas Usunier"},
         {"authorId": "a044", "name": "Jason Weston"},
     ],
     "abstract": "We consider the problem of embedding entities and relationships of multi-relational data in low-dimensional vector spaces. Our model, TransE, learns vector embeddings so that relationships correspond to translations in the embedding space.",
     "fieldsOfStudy": ["Computer Science"],
     "references": [], "citations": ["p017", "p018"]},

    {"paperId": "p017", "title": "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space", "year": 2019,
     "citationCount": 3200, "venue": "ICLR",
     "authors": [
         {"authorId": "a045", "name": "Zhiqing Sun"},
         {"authorId": "a046", "name": "Zhi-Hong Deng"},
         {"authorId": "a024", "name": "Jure Leskovec"},
     ],
     "abstract": "We study the problem of learning representations of entities and relations in knowledge graphs. We propose RotatE, which defines each relation as a rotation from the source entity to the target entity in the complex vector space.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p016"], "citations": ["p018"]},

    {"paperId": "p018", "title": "Complex Embeddings for Simple Link Prediction", "year": 2016,
     "citationCount": 3000, "venue": "ICML",
     "authors": [
         {"authorId": "a047", "name": "Theo Trouillon"},
         {"authorId": "a048", "name": "Johannes Welbl"},
     ],
     "abstract": "In statistical relational learning, knowledge graph completion deals with automatically understanding the structure of large knowledge graphs. We present ComplEx, an extension of DistMult that uses complex-valued embeddings for link prediction.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p016", "p017"], "citations": []},

    # Reinforcement Learning
    {"paperId": "p019", "title": "Playing Atari with Deep Reinforcement Learning", "year": 2013,
     "citationCount": 15000, "venue": "NIPS Workshop",
     "authors": [
         {"authorId": "a049", "name": "Volodymyr Mnih"},
         {"authorId": "a050", "name": "Koray Kavukcuoglu"},
         {"authorId": "a051", "name": "David Silver"},
     ],
     "abstract": "We present the first deep learning model to successfully learn control policies directly from high-dimensional sensory input using reinforcement learning. We use a deep Q-network (DQN) to learn to play Atari 2600 games.",
     "fieldsOfStudy": ["Computer Science"],
     "references": [], "citations": ["p021"]},

    {"paperId": "p020", "title": "Conditional Image Generation with PixelCNN Decoders", "year": 2016,
     "citationCount": 4500, "venue": "NeurIPS",
     "authors": [
         {"authorId": "a052", "name": "Aaron van den Oord"},
         {"authorId": "a053", "name": "Nal Kalchbrenner"},
     ],
     "abstract": "This work explores conditional image generation with a new image density model based on the PixelCNN architecture. We show that a single model can generate diverse, globally coherent images.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p004", "p005"], "citations": ["p013"]},

    {"paperId": "p021", "title": "Proximal Policy Optimization Algorithms", "year": 2017,
     "citationCount": 14000, "venue": "arXiv",
     "authors": [
         {"authorId": "a054", "name": "John Schulman"},
         {"authorId": "a055", "name": "Filip Wolski"},
         {"authorId": "a037", "name": "Pieter Abbeel"},
     ],
     "abstract": "We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a surrogate objective function using stochastic gradient ascent.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p019"], "citations": ["p010"]},

    # Self-Supervised Learning
    {"paperId": "p022", "title": "A Simple Framework for Contrastive Learning of Visual Representations", "year": 2020,
     "citationCount": 14000, "venue": "ICML",
     "authors": [
         {"authorId": "a056", "name": "Ting Chen"},
         {"authorId": "a057", "name": "Simon Kornblith"},
         {"authorId": "a058", "name": "Geoffrey Hinton"},
     ],
     "abstract": "This paper presents SimCLR: a simple framework for contrastive learning of visual representations. We simplify recently proposed contrastive self-supervised learning algorithms without requiring specialized architectures.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p011"], "citations": ["p023"]},

    {"paperId": "p023", "title": "Momentum Contrast for Unsupervised Visual Representation Learning", "year": 2020,
     "citationCount": 12000, "venue": "CVPR",
     "authors": [
         {"authorId": "a028", "name": "Kaiming He"},
         {"authorId": "a059", "name": "Haoqi Fan"},
     ],
     "abstract": "We present Momentum Contrast (MoCo) for unsupervised visual representation learning. We view contrastive learning as dictionary look-up, and build a dynamic dictionary with a queue and a moving-averaged encoder.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p011", "p022"], "citations": []},

    # Federated Learning
    {"paperId": "p024", "title": "Communication-Efficient Learning of Deep Networks from Decentralized Data", "year": 2017,
     "citationCount": 18000, "venue": "AISTATS",
     "authors": [
         {"authorId": "a060", "name": "H. Brendan McMahan"},
         {"authorId": "a061", "name": "Eider Moore"},
     ],
     "abstract": "Modern mobile devices have access to a wealth of data suitable for learning models, which in turn can greatly improve the user experience on the device. We propose federated learning, a method for training deep networks on decentralized data.",
     "fieldsOfStudy": ["Computer Science"],
     "references": [], "citations": ["p025"]},

    {"paperId": "p025", "title": "Advances and Open Problems in Federated Learning", "year": 2021,
     "citationCount": 6000, "venue": "Foundations and Trends in ML",
     "authors": [
         {"authorId": "a060", "name": "H. Brendan McMahan"},
         {"authorId": "a062", "name": "Peter Kairouz"},
     ],
     "abstract": "Federated learning is a machine learning setting where multiple entities collaborate in solving a machine learning problem under the coordination of a central server. This paper discusses recent advances and enumerates important open problems.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p024"], "citations": []},

    # NAS
    {"paperId": "p026", "title": "Neural Architecture Search with Reinforcement Learning", "year": 2017,
     "citationCount": 8000, "venue": "ICLR",
     "authors": [
         {"authorId": "a063", "name": "Barret Zoph"},
         {"authorId": "a064", "name": "Quoc Le"},
     ],
     "abstract": "Neural network design requires a lot of expert knowledge. We use a recurrent neural network to generate the model descriptions of neural networks and train this RNN with reinforcement learning to maximize the expected accuracy of generated architectures.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p019"], "citations": ["p027"]},

    {"paperId": "p027", "title": "DARTS: Differentiable Architecture Search", "year": 2019,
     "citationCount": 5500, "venue": "ICLR",
     "authors": [
         {"authorId": "a065", "name": "Hanxiao Liu"},
         {"authorId": "a066", "name": "Karen Simonyan"},
     ],
     "abstract": "We address the scalability challenge of neural architecture search by formulating the task in a differentiable manner. We propose DARTS, an efficient architecture search algorithm that uses continuous relaxation of the discrete search space.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p008", "p026"], "citations": []},

    # Object Detection
    {"paperId": "p028", "title": "You Only Look Once: Unified, Real-Time Object Detection", "year": 2016,
     "citationCount": 38000, "venue": "CVPR",
     "authors": [
         {"authorId": "a067", "name": "Joseph Redmon"},
         {"authorId": "a068", "name": "Santosh Divvala"},
         {"authorId": "a069", "name": "Ali Farhadi"},
     ],
     "abstract": "We present YOLO, a new approach to object detection. We frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p012"], "citations": ["p029"]},

    {"paperId": "p029", "title": "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", "year": 2015,
     "citationCount": 45000, "venue": "NeurIPS",
     "authors": [
         {"authorId": "a030", "name": "Shaoqing Ren"},
         {"authorId": "a028", "name": "Kaiming He"},
         {"authorId": "a070", "name": "Ross Girshick"},
         {"authorId": "a031", "name": "Jian Sun"},
     ],
     "abstract": "State-of-the-art object detection networks depend on region proposal algorithms. We introduce a Region Proposal Network that shares full-image convolutional features with the detection network, enabling nearly cost-free region proposals.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p012"], "citations": ["p028"]},

    # Transfer Learning
    {"paperId": "p030", "title": "How transferable are features in deep neural networks?", "year": 2014,
     "citationCount": 12000, "venue": "NeurIPS",
     "authors": [
         {"authorId": "a071", "name": "Jason Yosinski"},
         {"authorId": "a072", "name": "Jeff Clune"},
         {"authorId": "a014", "name": "Yoshua Bengio"},
     ],
     "abstract": "Many deep neural networks trained on natural images exhibit a curious phenomenon in common: on the first layer they learn features similar to Gabor filters and color blobs. We study the transferability of features learned at different layers.",
     "fieldsOfStudy": ["Computer Science"],
     "references": [], "citations": ["p002", "p022"]},

    # Additional recent papers
    {"paperId": "p031", "title": "Scaling Laws for Neural Language Models", "year": 2020,
     "citationCount": 4000, "venue": "arXiv",
     "authors": [
         {"authorId": "a073", "name": "Jared Kaplan"},
         {"authorId": "a074", "name": "Sam McCandlish"},
     ],
     "abstract": "We study empirical scaling laws for language model performance on the cross-entropy loss. The loss scales as a power-law with model size, dataset size, and the amount of compute used for training.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p001", "p003"], "citations": ["p035"]},

    {"paperId": "p032", "title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", "year": 2014,
     "citationCount": 38000, "venue": "JMLR",
     "authors": [
         {"authorId": "a075", "name": "Nitish Srivastava"},
         {"authorId": "a058", "name": "Geoffrey Hinton"},
     ],
     "abstract": "Deep neural nets with a large number of parameters are very powerful machine learning systems. However, overfitting is a serious problem. We propose dropout, a technique for addressing this problem by randomly dropping units during training.",
     "fieldsOfStudy": ["Computer Science"],
     "references": [], "citations": ["p011", "p006"]},

    {"paperId": "p033", "title": "Adam: A Method for Stochastic Optimization", "year": 2015,
     "citationCount": 130000, "venue": "ICLR",
     "authors": [
         {"authorId": "a076", "name": "Diederik Kingma"},
         {"authorId": "a077", "name": "Jimmy Ba"},
     ],
     "abstract": "We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments.",
     "fieldsOfStudy": ["Computer Science", "Mathematics"],
     "references": [], "citations": ["p001", "p006", "p011", "p013"]},

    {"paperId": "p034", "title": "Batch Normalization: Accelerating Deep Network Training", "year": 2015,
     "citationCount": 48000, "venue": "ICML",
     "authors": [
         {"authorId": "a078", "name": "Sergey Ioffe"},
         {"authorId": "a079", "name": "Christian Szegedy"},
     ],
     "abstract": "Training Deep Neural Networks is complicated by the fact that the distribution of each layer's inputs changes during training. We propose Batch Normalization, which allows us to use much higher learning rates and be less careful about initialization.",
     "fieldsOfStudy": ["Computer Science"],
     "references": [], "citations": ["p011"]},

    {"paperId": "p035", "title": "Constitutional AI: Harmlessness from AI Feedback", "year": 2022,
     "citationCount": 2000, "venue": "arXiv",
     "authors": [
         {"authorId": "a080", "name": "Yuntao Bai"},
         {"authorId": "a081", "name": "Saurav Kadavath"},
     ],
     "abstract": "We experiment with methods for training a harmless AI assistant through a process we call Constitutional AI. The approach involves both a supervised learning and a reinforcement learning phase using AI-generated feedback.",
     "fieldsOfStudy": ["Computer Science"],
     "references": ["p003", "p010", "p031"], "citations": []},
]


# Author details with affiliations
AUTHOR_DETAILS = {
    "a001": {"authorId": "a001", "name": "Ashish Vaswani", "affiliations": ["Google Brain"], "paperCount": 45, "citationCount": 110000, "hIndex": 28},
    "a002": {"authorId": "a002", "name": "Noam Shazeer", "affiliations": ["Google Brain"], "paperCount": 60, "citationCount": 150000, "hIndex": 35},
    "a003": {"authorId": "a003", "name": "Niki Parmar", "affiliations": ["Google Brain"], "paperCount": 25, "citationCount": 100000, "hIndex": 18},
    "a004": {"authorId": "a004", "name": "Jakob Uszkoreit", "affiliations": ["Google Brain"], "paperCount": 30, "citationCount": 120000, "hIndex": 22},
    "a005": {"authorId": "a005", "name": "Jacob Devlin", "affiliations": ["Google AI"], "paperCount": 35, "citationCount": 85000, "hIndex": 22},
    "a006": {"authorId": "a006", "name": "Ming-Wei Chang", "affiliations": ["Google AI"], "paperCount": 55, "citationCount": 90000, "hIndex": 30},
    "a007": {"authorId": "a007", "name": "Kenton Lee", "affiliations": ["Google AI"], "paperCount": 40, "citationCount": 80000, "hIndex": 25},
    "a008": {"authorId": "a008", "name": "Tom Brown", "affiliations": ["OpenAI"], "paperCount": 20, "citationCount": 35000, "hIndex": 15},
    "a009": {"authorId": "a009", "name": "Benjamin Mann", "affiliations": ["OpenAI"], "paperCount": 10, "citationCount": 30000, "hIndex": 8},
    "a010": {"authorId": "a010", "name": "Nick Ryder", "affiliations": ["OpenAI"], "paperCount": 8, "citationCount": 29000, "hIndex": 6},
    "a011": {"authorId": "a011", "name": "Ian Goodfellow", "affiliations": ["Google Brain", "Apple"], "paperCount": 80, "citationCount": 200000, "hIndex": 55},
    "a012": {"authorId": "a012", "name": "Jean Pouget-Abadie", "affiliations": ["Universite de Montreal"], "paperCount": 15, "citationCount": 60000, "hIndex": 10},
    "a013": {"authorId": "a013", "name": "Mehdi Mirza", "affiliations": ["Universite de Montreal"], "paperCount": 20, "citationCount": 62000, "hIndex": 12},
    "a014": {"authorId": "a014", "name": "Yoshua Bengio", "affiliations": ["Universite de Montreal", "Mila"], "paperCount": 500, "citationCount": 600000, "hIndex": 150},
    "a015": {"authorId": "a015", "name": "Tero Karras", "affiliations": ["NVIDIA"], "paperCount": 20, "citationCount": 30000, "hIndex": 14},
    "a016": {"authorId": "a016", "name": "Samuli Laine", "affiliations": ["NVIDIA"], "paperCount": 45, "citationCount": 25000, "hIndex": 20},
    "a017": {"authorId": "a017", "name": "Timo Aila", "affiliations": ["NVIDIA"], "paperCount": 40, "citationCount": 22000, "hIndex": 18},
    "a018": {"authorId": "a018", "name": "Thomas Kipf", "affiliations": ["University of Amsterdam", "Google Brain"], "paperCount": 25, "citationCount": 40000, "hIndex": 15},
    "a019": {"authorId": "a019", "name": "Max Welling", "affiliations": ["University of Amsterdam", "Microsoft Research"], "paperCount": 250, "citationCount": 80000, "hIndex": 70},
    "a020": {"authorId": "a020", "name": "Petar Velickovic", "affiliations": ["University of Cambridge", "DeepMind"], "paperCount": 50, "citationCount": 30000, "hIndex": 25},
    "a021": {"authorId": "a021", "name": "Guillem Cucurull", "affiliations": ["Facebook AI"], "paperCount": 15, "citationCount": 20000, "hIndex": 10},
    "a022": {"authorId": "a022", "name": "William Hamilton", "affiliations": ["McGill University"], "paperCount": 40, "citationCount": 25000, "hIndex": 20},
    "a023": {"authorId": "a023", "name": "Zhitao Ying", "affiliations": ["Stanford University"], "paperCount": 20, "citationCount": 18000, "hIndex": 12},
    "a024": {"authorId": "a024", "name": "Jure Leskovec", "affiliations": ["Stanford University"], "paperCount": 300, "citationCount": 150000, "hIndex": 90},
    "a025": {"authorId": "a025", "name": "Alexey Dosovitskiy", "affiliations": ["Google Brain"], "paperCount": 30, "citationCount": 50000, "hIndex": 22},
    "a026": {"authorId": "a026", "name": "Long Ouyang", "affiliations": ["OpenAI"], "paperCount": 8, "citationCount": 10000, "hIndex": 6},
    "a027": {"authorId": "a027", "name": "Jeff Wu", "affiliations": ["OpenAI"], "paperCount": 12, "citationCount": 12000, "hIndex": 8},
    "a028": {"authorId": "a028", "name": "Kaiming He", "affiliations": ["Meta AI", "MIT"], "paperCount": 80, "citationCount": 350000, "hIndex": 60},
    "a029": {"authorId": "a029", "name": "Xiangyu Zhang", "affiliations": ["MEGVII Technology"], "paperCount": 40, "citationCount": 200000, "hIndex": 25},
    "a030": {"authorId": "a030", "name": "Shaoqing Ren", "affiliations": ["Microsoft Research"], "paperCount": 25, "citationCount": 220000, "hIndex": 18},
    "a031": {"authorId": "a031", "name": "Jian Sun", "affiliations": ["MEGVII Technology"], "paperCount": 100, "citationCount": 280000, "hIndex": 55},
    "a032": {"authorId": "a032", "name": "Olga Russakovsky", "affiliations": ["Princeton University"], "paperCount": 50, "citationCount": 60000, "hIndex": 25},
    "a033": {"authorId": "a033", "name": "Jia Deng", "affiliations": ["Princeton University"], "paperCount": 60, "citationCount": 70000, "hIndex": 30},
    "a034": {"authorId": "a034", "name": "Hao Su", "affiliations": ["UCSD"], "paperCount": 80, "citationCount": 65000, "hIndex": 35},
    "a035": {"authorId": "a035", "name": "Jonathan Ho", "affiliations": ["Google Brain"], "paperCount": 25, "citationCount": 20000, "hIndex": 15},
    "a036": {"authorId": "a036", "name": "Ajay Jain", "affiliations": ["UC Berkeley"], "paperCount": 15, "citationCount": 15000, "hIndex": 10},
    "a037": {"authorId": "a037", "name": "Pieter Abbeel", "affiliations": ["UC Berkeley", "Covariant"], "paperCount": 250, "citationCount": 100000, "hIndex": 70},
    "a038": {"authorId": "a038", "name": "Robin Rombach", "affiliations": ["Ludwig Maximilian University"], "paperCount": 15, "citationCount": 12000, "hIndex": 10},
    "a039": {"authorId": "a039", "name": "Andreas Blattmann", "affiliations": ["Ludwig Maximilian University"], "paperCount": 10, "citationCount": 10000, "hIndex": 7},
    "a040": {"authorId": "a040", "name": "Dominik Lorenz", "affiliations": ["Ludwig Maximilian University"], "paperCount": 8, "citationCount": 9500, "hIndex": 5},
    "a041": {"authorId": "a041", "name": "Chitwan Saharia", "affiliations": ["Google Brain"], "paperCount": 20, "citationCount": 8000, "hIndex": 12},
    "a042": {"authorId": "a042", "name": "Antoine Bordes", "affiliations": ["Meta AI"], "paperCount": 60, "citationCount": 30000, "hIndex": 30},
    "a043": {"authorId": "a043", "name": "Nicolas Usunier", "affiliations": ["Meta AI"], "paperCount": 40, "citationCount": 15000, "hIndex": 22},
    "a044": {"authorId": "a044", "name": "Jason Weston", "affiliations": ["Meta AI"], "paperCount": 150, "citationCount": 80000, "hIndex": 60},
    "a045": {"authorId": "a045", "name": "Zhiqing Sun", "affiliations": ["Carnegie Mellon University"], "paperCount": 20, "citationCount": 6000, "hIndex": 12},
    "a046": {"authorId": "a046", "name": "Zhi-Hong Deng", "affiliations": ["Peking University"], "paperCount": 100, "citationCount": 8000, "hIndex": 25},
    "a047": {"authorId": "a047", "name": "Theo Trouillon", "affiliations": ["CNRS"], "paperCount": 12, "citationCount": 4000, "hIndex": 8},
    "a048": {"authorId": "a048", "name": "Johannes Welbl", "affiliations": ["UCL"], "paperCount": 15, "citationCount": 4500, "hIndex": 10},
    "a049": {"authorId": "a049", "name": "Volodymyr Mnih", "affiliations": ["DeepMind"], "paperCount": 30, "citationCount": 50000, "hIndex": 20},
    "a050": {"authorId": "a050", "name": "Koray Kavukcuoglu", "affiliations": ["DeepMind"], "paperCount": 50, "citationCount": 60000, "hIndex": 30},
    "a051": {"authorId": "a051", "name": "David Silver", "affiliations": ["DeepMind", "UCL"], "paperCount": 80, "citationCount": 90000, "hIndex": 45},
    "a052": {"authorId": "a052", "name": "Aaron van den Oord", "affiliations": ["DeepMind"], "paperCount": 40, "citationCount": 40000, "hIndex": 25},
    "a053": {"authorId": "a053", "name": "Nal Kalchbrenner", "affiliations": ["DeepMind"], "paperCount": 25, "citationCount": 20000, "hIndex": 18},
    "a054": {"authorId": "a054", "name": "John Schulman", "affiliations": ["OpenAI"], "paperCount": 25, "citationCount": 45000, "hIndex": 18},
    "a055": {"authorId": "a055", "name": "Filip Wolski", "affiliations": ["OpenAI"], "paperCount": 8, "citationCount": 15000, "hIndex": 6},
    "a056": {"authorId": "a056", "name": "Ting Chen", "affiliations": ["Google Brain"], "paperCount": 30, "citationCount": 25000, "hIndex": 18},
    "a057": {"authorId": "a057", "name": "Simon Kornblith", "affiliations": ["Google Brain"], "paperCount": 25, "citationCount": 20000, "hIndex": 15},
    "a058": {"authorId": "a058", "name": "Geoffrey Hinton", "affiliations": ["University of Toronto", "Google Brain"], "paperCount": 350, "citationCount": 800000, "hIndex": 170},
    "a059": {"authorId": "a059", "name": "Haoqi Fan", "affiliations": ["Meta AI"], "paperCount": 20, "citationCount": 18000, "hIndex": 14},
    "a060": {"authorId": "a060", "name": "H. Brendan McMahan", "affiliations": ["Google"], "paperCount": 50, "citationCount": 35000, "hIndex": 25},
    "a061": {"authorId": "a061", "name": "Eider Moore", "affiliations": ["Google"], "paperCount": 10, "citationCount": 20000, "hIndex": 8},
    "a062": {"authorId": "a062", "name": "Peter Kairouz", "affiliations": ["Google"], "paperCount": 60, "citationCount": 15000, "hIndex": 22},
    "a063": {"authorId": "a063", "name": "Barret Zoph", "affiliations": ["Google Brain"], "paperCount": 20, "citationCount": 20000, "hIndex": 14},
    "a064": {"authorId": "a064", "name": "Quoc Le", "affiliations": ["Google Brain"], "paperCount": 100, "citationCount": 120000, "hIndex": 55},
    "a065": {"authorId": "a065", "name": "Hanxiao Liu", "affiliations": ["Google Brain"], "paperCount": 25, "citationCount": 12000, "hIndex": 15},
    "a066": {"authorId": "a066", "name": "Karen Simonyan", "affiliations": ["DeepMind"], "paperCount": 30, "citationCount": 100000, "hIndex": 20},
    "a067": {"authorId": "a067", "name": "Joseph Redmon", "affiliations": ["University of Washington"], "paperCount": 8, "citationCount": 50000, "hIndex": 7},
    "a068": {"authorId": "a068", "name": "Santosh Divvala", "affiliations": ["Allen Institute for AI"], "paperCount": 30, "citationCount": 25000, "hIndex": 18},
    "a069": {"authorId": "a069", "name": "Ali Farhadi", "affiliations": ["University of Washington", "Allen Institute for AI"], "paperCount": 100, "citationCount": 80000, "hIndex": 50},
    "a070": {"authorId": "a070", "name": "Ross Girshick", "affiliations": ["Meta AI"], "paperCount": 50, "citationCount": 150000, "hIndex": 35},
    "a071": {"authorId": "a071", "name": "Jason Yosinski", "affiliations": ["Uber AI Labs"], "paperCount": 15, "citationCount": 18000, "hIndex": 10},
    "a072": {"authorId": "a072", "name": "Jeff Clune", "affiliations": ["OpenAI", "University of Wyoming"], "paperCount": 80, "citationCount": 30000, "hIndex": 35},
    "a073": {"authorId": "a073", "name": "Jared Kaplan", "affiliations": ["Johns Hopkins University", "Anthropic"], "paperCount": 30, "citationCount": 10000, "hIndex": 15},
    "a074": {"authorId": "a074", "name": "Sam McCandlish", "affiliations": ["Anthropic"], "paperCount": 12, "citationCount": 6000, "hIndex": 8},
    "a075": {"authorId": "a075", "name": "Nitish Srivastava", "affiliations": ["University of Toronto"], "paperCount": 10, "citationCount": 45000, "hIndex": 8},
    "a076": {"authorId": "a076", "name": "Diederik Kingma", "affiliations": ["Google Brain"], "paperCount": 25, "citationCount": 180000, "hIndex": 18},
    "a077": {"authorId": "a077", "name": "Jimmy Ba", "affiliations": ["University of Toronto"], "paperCount": 40, "citationCount": 150000, "hIndex": 20},
    "a078": {"authorId": "a078", "name": "Sergey Ioffe", "affiliations": ["Google"], "paperCount": 20, "citationCount": 55000, "hIndex": 15},
    "a079": {"authorId": "a079", "name": "Christian Szegedy", "affiliations": ["Google Brain"], "paperCount": 30, "citationCount": 100000, "hIndex": 20},
    "a080": {"authorId": "a080", "name": "Yuntao Bai", "affiliations": ["Anthropic"], "paperCount": 10, "citationCount": 3000, "hIndex": 6},
    "a081": {"authorId": "a081", "name": "Saurav Kadavath", "affiliations": ["Anthropic"], "paperCount": 5, "citationCount": 2500, "hIndex": 4},
}


def generate_sample_data(output_dir="data/raw"):
    """Generate sample data files for testing."""
    os.makedirs(output_dir, exist_ok=True)

    # Papers dict: paperId -> paper data
    papers_dict = {}
    for paper in PAPERS:
        papers_dict[paper["paperId"]] = paper

    # Save papers
    papers_path = os.path.join(output_dir, "s2_papers.json")
    with open(papers_path, "w") as f:
        json.dump(papers_dict, f, indent=2)
    print(f"Generated {len(papers_dict)} sample papers -> {papers_path}")

    # Save authors
    authors_path = os.path.join(output_dir, "s2_authors.json")
    with open(authors_path, "w") as f:
        json.dump(AUTHOR_DETAILS, f, indent=2)
    print(f"Generated {len(AUTHOR_DETAILS)} sample authors -> {authors_path}")

    return papers_path, authors_path


if __name__ == "__main__":
    generate_sample_data()
