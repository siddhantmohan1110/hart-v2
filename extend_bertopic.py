import argparse
import json
from pathlib import Path

import torch
from bertopic import BERTopic

from hart.clustering.algos.bert_topic import load_qwen


def _read_prompts(path: Path) -> list[str]:
    """Load prompts from a JSON list or newline-delimited text file."""
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(item) for item in data]
    except json.JSONDecodeError:
        pass
    return [line.strip() for line in text.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Load a saved BERTopic model and extend it with new prompts.")
    parser.add_argument("--model-path", required=True, help="Path to the saved BERTopic model directory/file.")
    parser.add_argument("--text-model-path", required=True, help="Path or HF id for the Qwen text model.")
    parser.add_argument("--prompts-file", required=True, help="Path to prompts (JSON list or newline-delimited text).")
    parser.add_argument("--output-model-path", default=None, help="Optional path to save the updated model (defaults to overwrite --model-path).")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for Qwen embedding.")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers for embedding.")
    parser.add_argument("--max-token-length", type=int, default=10, help="Max token length used during embedding.")
    args = parser.parse_args()

    prompts_path = Path(args.prompts_file)
    prompts = _read_prompts(prompts_path)
    if not prompts:
        raise ValueError(f"No prompts loaded from {prompts_path}")

    print(f"Loaded {len(prompts)} prompts from {prompts_path}")

    embeddings = load_qwen(
        prompts,
        args.text_model_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_token_length=args.max_token_length,
    )
    np_embed = embeddings.cpu().numpy()
    del embeddings
    torch.cuda.empty_cache()

    topic_model = BERTopic.load(args.model_path)
    print(f"Loaded BERTopic model from {args.model_path}")

    topics, probs = topic_model.fit_transform(prompts, np_embed)
    print(f"Extended model with {len(prompts)} prompts; topics found: {len(set(topics)) - 1} (excluding outliers)")

    output_path = args.output_model_path or args.model_path
    topic_model.save(output_path)
    print(f"Saved updated BERTopic model to: {output_path}")


if __name__ == "__main__":
    main()
