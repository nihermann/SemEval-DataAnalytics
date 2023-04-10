import numpy as np

import torch
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator

from typing import List, Tuple
from torch.utils.data import DataLoader

from dataset import load_data

# region Flair

def cos_sim(s1, s2):
    s1 = s1[0].embedding.unsqueeze(0)
    s2 = s2[0].embedding.unsqueeze(0)
    return torch.cosine_similarity(s1, s2)


def evaluate(sentences, emb):
    sims = []
    for s1, s2, score in sentences:
        s1, s2 = emb.embed(s1), emb.embed(s2)
        sim = cos_sim(s1, s2)
        sims += sim
    return sims


def flair() -> None:
    import pandas as pd

    from flair.embeddings import SentenceTransformerDocumentEmbeddings, BertEmbeddings
    from flair.data import Sentence

    df = pd.read_feather("data/preprocessed.feather")
    df.score /= 5

    embedder = SentenceTransformerDocumentEmbeddings("bert-base-nli-mean-tokens")

    sims = evaluate(
        zip(df.sen1.apply(Sentence), df.sen2.apply(Sentence), df.score),
        embedder
    )
    print(np.corrcoef(np.array(sims), df.score))

# endregion


def kaggle(model_name: str = "stsb-mpnet-base-v2", num_epochs: int = 20) -> None:
    train, validation, test = load_data()

    train_loader = DataLoader(train, batch_size=16, shuffle=True)

    valid_evaluator = CECorrelationEvaluator.from_input_examples(
        validation, name="corr-valid"
    )

    # download pretrained bert model with num_labels=1 which outputs a continuous score between
    # 0 and 1 indicating the similarity between the two input sentences.
    model = CrossEncoder(model_name, num_labels=1)
    save_path = f"models/{model_name}"

    model.fit(
        train_dataloader=train_loader,
        evaluator=valid_evaluator,
        epochs=num_epochs, save_best_model=True,
        warmup_steps=int(len(train_loader) * num_epochs * .1),
        output_path=save_path
    )
    model = CrossEncoder(save_path)
    test_evaluator = CECorrelationEvaluator.from_input_examples(test, name="corr-test")
    test_evaluator(model, save_path)


if __name__ == '__main__':
    import sys
    kaggle(sys.argv[1])

    # flair()



