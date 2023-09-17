import logging
import os
import random
import sys
from datetime import datetime

import hydra
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../")

from evaluate import Evaluator

from data_processing.dataloaders import DataPreparation
from model.multitask_model import MultitaskModelWithT5
from utils.enums.enums import DatasetType

logging.warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)


class Trainer:
    """
    Trainer class to perform model training, validation and evaluation.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize a Trainer object and initialise database.

        Args:
            cfg (DictConfig): Hydra configuration.

        Returns:
            None
        """
        self.cfg = cfg
        self.evaluator = Evaluator()
        self._initialize_dataloader()
        self._setup_training_configurations()
        self._setup_model_configurations()
        self._configure_loss_functions()
        self._configure_optimizer()
        self._configure_scheduler()

    def _configure_loss_functions(self) -> None:
        """
        Configure loss functions for sentiment classification and
        response generation. Assign weights for sentiment and response loss.
        """
        # Loss functions
        self.encoder_criterion = torch.nn.CrossEntropyLoss()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.modelling.model.encoder_decoder
        )
        self.decoder_criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.eos_token_id
        )

        # Respective loss weightage
        self.sentiment_weightage, self.response_weightage = (
            self.cfg.modelling.loss_weights.sentiment_loss_weightage,
            self.cfg.modelling.loss_weights.response_loss_weightage,
        )

        # Set up variable to store best validation loss
        self.best_validation_loss = float("inf")

        logging.info("Configured Loss Functions")

    def _configure_optimizer(self) -> None:
        """
        Configure the optimizer for model parameter updates. Initializes an
        AdamW optimizer with the specified learning rate.
        """
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.modelling.optimizer.learning_rate,
        )
        logging.info("Configured Optimizer")

    def _configure_scheduler(self) -> None:
        """
        Configure the learning rate scheduler. Initializes a ReduceLROnPlateau
        scheduler to adjust learning rates during training. Monitors the
        validation loss and reduces learning rate on plateau.
        """
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.cfg.modelling.scheduler.factor,
            patience=self.cfg.modelling.scheduler.patience,
            verbose=True,
        )
        logging.info("Configured Learning Rate Scheduler")

    def _compute_batch_loss(
        self,
        encoder_output: torch.Tensor,
        decoder_output: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the total loss for a batch of data:
        - Calculates the encoder and decoder losses using provided model
        outputs and ground truth labels.

        Args:
            encoder_output (torch.Tensor): Model output from the encoder.
            decoder_output (torch.Tensor): Model output from the decoder.
            decoder_input_ids (torch.Tensor): Input IDs for the decoder.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Total loss for the batch.
        """
        encoder_loss = self.encoder_criterion(encoder_output, labels)
        decoder_loss = self.decoder_criterion(
            decoder_output.logits.view(-1, decoder_output.logits.size(-1)),
            decoder_input_ids.view(-1),
        )
        total_loss = (
            self.sentiment_weightage * encoder_loss
            + self.response_weightage * decoder_loss
        )

        return total_loss

    def _initialize_dataloader(self):
        """
        Initialize a DataPreparation instance with the given configuration and
        retrieves dataloaders for training, validation, and testing.
        """
        dataloader = DataPreparation(self.cfg)
        (
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
        ) = dataloader.get_dataloaders(DatasetType)

        logging.info("Initialized DataLoaders")

    def _setup_model_configurations(self) -> None:
        """
        Set up model configurations and initializes a MultitaskModel with
        specified encoder, decoder, and sentiment dimensions.
        """
        sentiment_classes = self.cfg.modelling.model.sentiment_classes
        self.model = MultitaskModelWithT5(
            model_name=self.cfg.modelling.model.encoder_decoder,
            sentiment_dim=sentiment_classes,
        ).to(self.device)
        logging.info("Set up model configurations")

    def _setup_training_configurations(self) -> None:
        """
        Set up general training configurations:
        - Training device
        - Early stopping parameters
        - Gradient Clipping value
        - TensorBoard Writer
        """
        # Set up device
        self.device = self.cfg.modelling.general.device

        # Early stopping
        self.max_epoch = self.cfg.modelling.general.epochs
        self.patience = self.cfg.modelling.general.patience
        self.patience_counter = 0

        # Gradient Clipping
        self.gradient_clipping = (
            self.cfg.modelling.general.gradient_clipping_threshold
        )

        # TensorBoard Writer
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.writer = SummaryWriter(f"runs/experiment_{current_time}")
        self.cfg.modelling.general.checkpoint_path = os.path.join(
            self.cfg.modelling.general.checkpoint_path,
            f"runs/experiment_{current_time}",
        )

        # Set seed
        seed = self.cfg.modelling.general.random_seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        set_seed(seed)

        logging.info("Set up training configurations")

    def train(self):
        """
        Trains the model over a given number of epochs using the training
        dataloader.

        Steps involved:
        1. Sets the model to train mode.
        2. Iterates through each batch of the training data loader.
        3. Computes the loss and performs backpropagation.
        4. Evaluates the training data after each epoch.
        5. Logs metrics to TensorBoard.
        6. Validates the model.
        7. Checks for early stopping criteria.
        """
        epoch_counter = 0

        for _ in range(self.max_epoch):
            self.model.train()
            total_train_loss = 0

            # Arrays for storing model outputs and labels for evaluation
            all_sentiment_outputs, all_sentiment_labels = [], []
            all_generated_texts, all_response_labels = [], []

            for batch in tqdm(self.train_dataloader):
                (
                    input_ids,
                    attention_mask,
                    labels,
                    decoder_input_ids,
                    decoder_attention_mask,
                ) = [x.to(self.device) for x in batch]
                encoder_output, decoder_output = self.model(
                    input_ids,
                    attention_mask,
                    decoder_input_ids,
                    decoder_attention_mask,
                )

                # Compute loss
                total_loss = self._compute_batch_loss(
                    encoder_output, decoder_output, decoder_input_ids, labels
                )
                total_loss.backward()
                total_train_loss += total_loss.item()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.modelling.general.gradient_clipping_threshold,
                )

                # Perform backpropagation
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Generate text
                with torch.no_grad():
                    generated_ids = self.model.t5.generate(
                        input_ids,
                        max_length=40,
                        repetition_penalty=1.5,
                        no_repeat_ngram_size=2,
                        num_beams=5,
                        temperature=0.7,
                        eos_token_id=self.model.t5.config.eos_token_id,
                        early_stopping=True,
                    )

                    generated_text = self.tokenizer.decode(
                        generated_ids[0], skip_special_tokens=True
                    )

                # Store model outputs and labels
                all_sentiment_outputs.extend(encoder_output)
                all_sentiment_labels.extend(labels)
                all_generated_texts.append(generated_text)

                eos_index = (
                    decoder_input_ids[0] == self.tokenizer.eos_token_id
                ).nonzero(as_tuple=True)[0]
                truncated_input_ids = decoder_input_ids[0][: eos_index[0] + 1]
                all_response_labels.append(
                    self.tokenizer.decode(
                        truncated_input_ids, skip_special_tokens=True
                    )
                )

            # Compute average training loss
            avg_train_loss = total_train_loss / len(self.train_dataloader)
            logging.info(f"Average Training Loss: {avg_train_loss}")

            # Evaluation
            train_evaluation = self.evaluator.evaluate(
                all_sentiment_outputs,
                all_sentiment_labels,
                all_generated_texts,
                all_response_labels,
            )
            random_index = random.randint(0, len(all_generated_texts) - 1)
            logging.info(f"Randomly selected sample at index {random_index}:")
            logging.info(f"Generated Text: {all_generated_texts[random_index]}")
            logging.info(f"Response Label: {all_response_labels[random_index]}")

            # Log metrics to TensorBoard
            for metric, value in train_evaluation.items():
                self.writer.add_scalar(
                    f"Training/{metric}", value, epoch_counter
                )

            # Perform validation
            validation_evaluation, avg_validation_loss = self.validate(
                epoch_counter
            )
            self.writer.add_scalars(
                "Loss",
                {"Train": avg_train_loss, "Validation": avg_validation_loss},
                epoch_counter,
            )

            # Complete epoch
            epoch_counter += 1

            # Check for early stopping
            early_stopping_criteria = self._check_for_early_stopping(
                epoch_counter, avg_validation_loss
            )
            if early_stopping_criteria:
                self._perform_early_stopping(epoch_counter)
                return

    def validate(self, current_epoch: int):
        """
        Validates the model using the validation data loader.

        Steps involved:
        1. Sets the model to evaluation mode.
        2. Iterates through each batch of the validation data loader.
        3. Computes the loss for each batch.
        4. Evaluates the model on the entire validation set.
        5. Logs metrics to TensorBoard.

        Returns:
            tuple: A tuple containing:
                validation_evaluation (dict): The evaluation metrics for the
                    validation set.
                avg_validation_loss (float): The average validation loss.
        """
        self.model.eval()
        total_validation_loss = 0

        # Arrays for storing model outputs and labels for evaluation
        all_sentiment_outputs, all_sentiment_labels = [], []
        all_generated_texts, all_response_labels = [], []

        for batch in tqdm(self.val_dataloader):
            (
                input_ids,
                attention_mask,
                labels,
                decoder_input_ids,
                decoder_attention_mask,
            ) = [x.to(self.device) for x in batch]
            with torch.no_grad():
                encoder_output, decoder_output = self.model(
                    input_ids,
                    attention_mask,
                    decoder_input_ids,
                    decoder_attention_mask,
                )
                generated_ids = self.model.t5.generate(
                    input_ids,
                    max_length=40,
                    repetition_penalty=1.5,
                    no_repeat_ngram_size=2,
                    num_beams=5,
                    temperature=0.7,
                    eos_token_id=self.model.t5.config.eos_token_id,
                    early_stopping=True,
                )

                generated_text = self.tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                )

            # Compute loss
            total_loss = self._compute_batch_loss(
                encoder_output, decoder_output, decoder_input_ids, labels
            )
            total_validation_loss += total_loss.item()

            # Store model outputs and labels
            all_sentiment_outputs += encoder_output
            all_sentiment_labels += labels
            all_generated_texts += generated_text
            all_response_labels += self.tokenizer.decode(decoder_input_ids[0])

        # Compute average validation loss
        avg_validation_loss = total_validation_loss / len(self.val_dataloader)
        logging.info(f"Average Validation Loss: {avg_validation_loss}")

        # Learning Rate Scheduler
        self.scheduler.step(avg_validation_loss)
        logging.info("Performing scheduler step")

        # Evaluation
        validation_evaluation = self.evaluator.evaluate(
            all_sentiment_outputs,
            all_sentiment_labels,
            all_generated_texts,
            all_response_labels,
        )
        random_index = random.randint(0, len(all_generated_texts) - 1)
        logging.info(f"Randomly selected sample at index {random_index}:")
        logging.info(f"Generated Text: {all_generated_texts[random_index]}")
        logging.info(f"Response Label: {all_response_labels[random_index]}")

        for metric, value in validation_evaluation.items():
            self.writer.add_scalar(f"Validation/{metric}", value, current_epoch)

        return validation_evaluation, avg_validation_loss

    def test(self):
        """
        Tests the model using the test data loader.

        Steps involved:
        1. Sets the model to evaluation mode.
        2. Iterates through each batch of the test data loader.
        3. Stores the model outputs and labels.
        4. Evaluates the model on the entire test set.

        Returns:
            dict: The evaluation metrics for the test set.
        """
        self.model.eval()

        # Arrays for storing model outputs and labels for evaluation
        all_sentiment_outputs, all_sentiment_labels = [], []
        all_generated_texts, all_response_labels = [], []

        for batch in tqdm(self.test_dataloader):
            (
                input_ids,
                attention_mask,
                labels,
                decoder_input_ids,
                decoder_attention_mask,
            ) = [x.to(self.device) for x in batch]
            with torch.no_grad():
                encoder_output, decoder_output = self.model(
                    input_ids,
                    attention_mask,
                    decoder_input_ids,
                    decoder_attention_mask,
                )
                generated_ids = self.model.t5.generate(
                    input_ids,
                    max_length=40,
                    no_repeat_ngram_size=2,
                    num_beams=5,
                    temperature=0.7,
                    eos_token_id=self.model.t5.config.eos_token_id,
                    early_stopping=True,
                )

                generated_text = self.tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                )

            # Store model outputs and labels
            all_sentiment_outputs.extend(encoder_output.detach().cpu().numpy())
            all_sentiment_labels.extend(labels.detach().cpu().numpy())
            all_generated_texts.extend(generated_text)
            all_response_labels.extend(
                self.tokenizer.decode(decoder_input_ids[0])
            )

        # Evaluation
        test_evaluation = self.evaluator.evaluate(
            all_sentiment_outputs,
            all_sentiment_labels,
            all_generated_texts,
            all_response_labels,
        )
        random_index = random.randint(0, len(all_generated_texts) - 1)
        logging.info(f"Randomly selected sample at index {random_index}:")
        logging.info(f"Generated Text: {all_generated_texts[random_index]}")
        logging.info(f"Response Label: {all_response_labels[random_index]}")

        return test_evaluation

    def _check_for_early_stopping(
        self, current_epoch: int, current_validation_loss: torch.Tensor
    ) -> bool:
        """
        Checks conditions for early stopping based on validation loss and
        maximum epochs.

        Parameters:
            current_epoch (int): The current epoch number during training.
            current_validation_loss (torch.Tensor): The validation loss for the
                current epoch.

        Returns:
            bool: True if early stopping conditions are met, False otherwise.
        """
        if current_validation_loss < self.best_validation_loss:
            self.patience_counter = 0
            self.best_validation_loss = current_validation_loss
            self.model.t5.save_pretrained(
                self.cfg.modelling.general.checkpoint_path
            )
            logging.info("New best model saved")
        else:
            self.patience_counter += 1
            logging.info(f"Patience counter: {self.patience_counter}")

        if self.patience_counter >= self.patience:
            logging.info("Early stopping triggered due to patience")
            return True

        elif current_epoch >= self.max_epoch - 1:
            logging.info("Early stopping triggered due to reaching max epoch")
            return True

        return False

    def _perform_early_stopping(self, current_epoch: int):
        """
        Performs the following actions:
        - Perform evaluation on test set
        - Save best model
        """
        test_evaluation = self.test()
        for metric, value in test_evaluation.items():
            self.writer.add_scalar(f"Testing/{metric}", value, current_epoch)
        self.model.t5.save_pretrained(
            self.cfg.modelling.general.checkpoint_path
        )
        self.writer.close()

        logging.info("Early stopping performed. Best model saved.")
        return test_evaluation


@hydra.main(config_path="../../conf/base", config_name="pipelines.yaml")
def run_standalone(cfg: DictConfig) -> None:
    """
    Initialize Hydra configuration and run standalone Trainer class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        None
    """
    logging.info(run(cfg))


def run(cfg: DictConfig) -> str:
    """
    Pass in Hydra configuration and run Trainer class.

    Args:
        cfg (DictConfig): Hydra configuration.

    Returns:
        str : Completion of Model Training.
    """
    trainer = Trainer(cfg)
    trainer.train()

    return "Completed Model Training"


if __name__ == "__main__":
    run_standalone()
