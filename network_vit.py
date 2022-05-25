from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm
import wandb

import torch
from torch import nn
from torch.nn import L1Loss
from transformers import ViTPreTrainedModel, ViTModel, AdamW, get_scheduler
from transformers.modeling_outputs import SequenceClassifierOutput


class ViTForRegression(ViTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.vit = ViTModel(config, add_pooling_layer=False)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(
            self,
            pixel_values=None,
            head_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            loss_fct = L1Loss()
            loss = loss_fct(logits.view(-1), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ViTTrainer:

    def __init__(self, pretrained_model_name, save_model_name, wandb_logging=False):
        self.save_model_name = save_model_name
        datetime_prefix = datetime.today().strftime('%Y%m%d%H%M%S')
        self.save_dir = Path("models") / f"{datetime_prefix}_{self.save_model_name}"
        self.save_dir.mkdir(parents=True, exist_ok=False)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = ViTForRegression.from_pretrained(pretrained_model_name)
        self.model.to(self.device)

        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)

        self.wandb_logging = wandb_logging

    def train(self, trainloader, validloader, num_epochs=100, patience=10):
        if self.wandb_logging:
            wandb.init(project="lanefollowing-ut-vahi")

        num_training_steps = num_epochs * len(trainloader)
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        best_valid_loss = float('inf')
        epochs_of_no_improve = 0

        for epoch in range(num_epochs):

            train_loss = self.train_epoch(trainloader)
            valid_loss, _ = self.evaluate(validloader)

            if self.wandb_logging:
                wandb.log({"epoch": epoch+1, "train_loss": train_loss, "val_los": valid_loss})

            print("Saving model.")
            self.model.save_pretrained(self.save_dir / "last.pt")

            if valid_loss < best_valid_loss:
                print("Saving best model.")
                best_valid_loss = valid_loss
                self.model.save_pretrained(self.save_dir / "best.pt")
                epochs_of_no_improve = 0
            else:
                epochs_of_no_improve += 1

            if self.wandb_logging:
                wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_los": valid_loss})

            if epochs_of_no_improve == patience:
                print(f'Early stopping, on epoch: {epoch + 1}.')
                break

    def evaluate(self, iterator):
        epoch_loss = 0.0
        self.model.eval()

        predictions = []
        with torch.no_grad():
            progress_bar = tqdm(total=len(iterator), smoothing=0)
            for i, batch in enumerate(iterator):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(pixel_values=batch["image"], labels=batch["steering_angle"])
                epoch_loss += outputs.loss.item()

                logits = outputs.logits
                predictions.extend(logits.cpu().squeeze(1))

                progress_bar.update(1)
                progress_bar.set_description(f'valid loss: {(epoch_loss / (i + 1)):.4f}')

        return epoch_loss / len(iterator), predictions

    def train_epoch(self, loader):
        epoch_loss = 0.0
        self.model.train()

        progress_bar = tqdm(total=len(loader), smoothing=0)
        for i, batch in enumerate(loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.model(pixel_values=batch["image"], labels=batch["steering_angle"])
            loss = outputs.loss
            loss.backward()

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            epoch_loss += loss.item()

            progress_bar.update(1)
            progress_bar.set_description(f'train loss: {(epoch_loss / (i + 1)):.4f}')

        return epoch_loss / len(loader)

