import logging

import numpy as np
import torch
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from torchmetrics.text.rouge import ROUGEScore

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class Evaluator:
    """
    Evaluator class to generate evaluation metrics for sentiment analysis and
    response generation tasks.
    """

    def evaluate(
        self,
        all_sentiment_outputs: list,
        all_sentiment_labels: list,
        generated_responses: list,
        true_responses: list,
    ) -> dict:
        """
        Perform overall evaluation including sentiment and response.

        Args:
            all_sentiment_outputs (list): The predicted labels for sentiment.
            all_sentiment_labels (list): The ground truth labels for sentiment.
            generated_responses (list): The generated text responses by model.
            true_responses (list): The ground truth text responses.

        Returns:
            dict: A dictionary containing combined metrics for sentiment and
                response.
        """
        pred_labels = torch.tensor(
            [output.argmax().item() for output in all_sentiment_outputs]
        )
        true_labels = torch.tensor(
            [label.item() for label in all_sentiment_labels]
        )
        sentiment_results = self.evaluate_sentiment(true_labels, pred_labels)
        response_results = self.evaluate_response(
            true_responses, generated_responses
        )
        results = {**sentiment_results, **response_results}
        logging.info(f"Results: {results}")

        return results

    def evaluate_sentiment(
        self, pred_labels: np.ndarray, true_labels: np.ndarray
    ) -> dict:
        """
        Evaluate sentiment labels using various metrics.

        Args:
            pred_labels (list): The predicted labels for sentiment.
            true_labels (list): The ground truth labels for sentiment.

        Returns:
            dict: A dictionary containing sentiment metrics like Accuracy,
                F1 Score, and Confusion Matrix.
        """
        accuracy_metric = MulticlassAccuracy(num_classes=3)
        f1_metric = MulticlassF1Score(num_classes=3, average="macro")
        accuracy = accuracy_metric(pred_labels, true_labels)
        f1 = f1_metric(pred_labels, true_labels)

        return {"Accuracy": accuracy.item(), "F1 Score": f1.item()}

    def evaluate_response(
        self, true_responses: list, generated_responses: list
    ) -> dict:
        """
        Evaluate generated text responses using METEOR and ROUGE.

        Parameters:
            true_responses (list[list[str]]): The ground truth text responses.
            generated_responses (list[list[str]]): The generated text responses
                by the model.

        Returns:
            dict: A dictionary containing response metrics like METEOR and
                ROUGE.
        """
        # Initialize ROUGE Scorer
        rouge = ROUGEScore()
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

        for true_resp, generated_resp in zip(
            true_responses, generated_responses
        ):
            for tr, gr in zip(true_resp, generated_resp):
                # ROUGE
                rouge_scores = rouge(tr, gr)
                rouge1_scores.append(rouge_scores["rouge1_fmeasure"])
                rouge2_scores.append(rouge_scores["rouge2_fmeasure"])
                rougeL_scores.append(rouge_scores["rougeL_fmeasure"])

        # Average Scores
        avg_rouge1_score = (
            sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
        )
        avg_rouge2_score = (
            sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
        )
        avg_rougeL_score = (
            sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0
        )

        return {
            "ROUGE-1": avg_rouge1_score.item(),
            "ROUGE-2": avg_rouge2_score.item(),
            "ROUGE-L": avg_rougeL_score.item(),
        }
