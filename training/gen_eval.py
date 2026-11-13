"""Evaluation metrics for GLOSS-MT-2: scores both raw model generation and post-substitution translation."""

import math
import random
from typing import Callable, Dict, List, Optional
import sacrebleu


def sample_examples(examples: List[Dict], n: int, seed: int = 42) -> List[Dict]:
    if n <= 0 or n >= len(examples):
        return list(examples)
    idx = sorted(random.Random(seed).sample(range(len(examples)), n))
    return [examples[i] for i in idx]


def text_metrics(hyps: List[str], refs: List[str]) -> Dict[str, float]:
    return {
        "bleu": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
        "chrf2": round(sacrebleu.corpus_chrf(hyps, [refs], beta=2).score, 2),
    }


def composite(metrics: Dict[str, float], loss: float) -> Dict[str, float]:
    loss_score = round(100.0 * math.exp(-loss), 2)
    return {
        "loss_score": loss_score,
        "composite": round((metrics["bleu"] + metrics["chrf2"] + loss_score) / 3, 3),
    }


def with_gen_metrics(trainer_cls):
    """Subclass Trainer to report BLEU, chrF2, and composite checkpoint selection."""
    class GenEvalTrainer(trainer_cls):
        gen_examples: List[Dict] = []
        gen_fn: Optional[Callable] = None

        def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None,
                            metric_key_prefix="eval"):
            out = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys,
                                          metric_key_prefix)
            if self.gen_examples and self.gen_fn is not None:
                was_training = self.model.training
                self.model.eval()
                hyps_sub, hyps_raw = self.gen_fn(self.model, self.gen_examples)
                if was_training:
                    self.model.train()

                refs = [e["gold"] for e in self.gen_examples]
                m_sub = text_metrics(hyps_sub, refs)
                m_raw = text_metrics(hyps_raw, refs)

                comp = composite(m_sub, out.metrics[f"{metric_key_prefix}_loss"])

                out.metrics[f"{metric_key_prefix}_bleu"] = m_sub["bleu"]
                out.metrics[f"{metric_key_prefix}_chrf2"] = m_sub["chrf2"]
                out.metrics[f"{metric_key_prefix}_raw_bleu"] = m_raw["bleu"]
                out.metrics[f"{metric_key_prefix}_raw_chrf2"] = m_raw["chrf2"]
                out.metrics[f"{metric_key_prefix}_composite"] = comp["composite"]

                print(
                    f"\n[gen_eval] Substituted (Final) -> BLEU: {m_sub['bleu']}, chrF2: {m_sub['chrf2']} | "
                    f"Raw (Preserved) -> BLEU: {m_raw['bleu']}, chrF2: {m_raw['chrf2']} | "
                    f"Composite: {comp['composite']}",
                    flush=True,
                )
            return out

    return GenEvalTrainer
