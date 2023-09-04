import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
)


class MultitaskModel(nn.Module):
    def __init__(self, encoder, decoder, sentiment_dim):
        super(MultitaskModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(encoder)
        decoder_config = AutoConfig.from_pretrained(
            decoder, add_cross_attention=True, is_decoder=True
        )
        self.decoder = AutoModelForCausalLM.from_pretrained(
            decoder, config=decoder_config
        )
        self.sentiment_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.encoder.config.hidden_size, sentiment_dim),
        )
        self.layer_norm = nn.LayerNorm(
            self.encoder.config.hidden_size, eps=1e-12
        )
        self.sentiment_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.encoder.config.hidden_size, 128),
            nn.LayerNorm(128, eps=1e-12),
            nn.ReLU(),
            nn.Linear(128, sentiment_dim),
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
    ):
        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)
        encoder_hidden_states = self.layer_norm(
            encoder_outputs.last_hidden_state
        )

        # Sentiment Classification
        sentiment_output = self.sentiment_head(encoder_hidden_states[:, 0, :])

        # Response Generation
        if decoder_input_ids is not None:
            decoder_outputs = self.decoder(
                decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=decoder_attention_mask,
            )
            return sentiment_output, encoder_hidden_states, decoder_outputs

        return sentiment_output, encoder_hidden_states, None


class MultitaskModelWithT5(nn.Module):
    def __init__(self, model_name, sentiment_dim):
        super(MultitaskModelWithT5, self).__init__()

        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)

        self.sentiment_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.t5.config.d_model, 128),
            nn.LayerNorm(128, eps=1e-12),
            nn.ReLU(),
            nn.Linear(128, sentiment_dim),
        )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
    ):
        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )

        last_hidden_states = outputs.encoder_last_hidden_state

        # Sentiment classification
        sentiment_output = self.sentiment_head(last_hidden_states[:, 0, :])

        return sentiment_output, outputs
