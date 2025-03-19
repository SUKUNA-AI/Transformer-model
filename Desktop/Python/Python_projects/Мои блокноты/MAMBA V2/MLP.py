import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class GatedMLP(pl.LightningModule):
    def __init__(
            self,
            in_features: int,
            hidden_features: int = None,
            out_features: int = None,
            activation=F.silu,
            bias: bool = False,
            multiple_of: int = 128,
            learning_rate: float = 1e-3,  # Добавляем параметр для оптимизатора
            device=None,
            dtype=None,
    ):
        """
        Gated MLP слой с использованием PyTorch Lightning.

        Args:
            in_features: Размер входных признаков.
            hidden_features: Размер скрытого слоя (если None, вычисляется как 8 * in_features / 3).
            out_features: Размер выходных признаков (если None, равен in_features).
            activation: Функция активации (по умолчанию F.silu).
            bias: Использовать ли смещение в линейных слоях (по умолчанию False).
            multiple_of: Округление hidden_features до кратного значения (по умолчанию 128).
            learning_rate: Скорость обучения для оптимизатора.
            device: Устройство (CPU/GPU).
            dtype: Тип данных (например, torch.float32).
        """
        super().__init__()
        self.save_hyperparameters()  # Сохраняем гиперпараметры для логирования

        factory_kwargs = {"device": device, "dtype": dtype}

        # Устанавливаем выходной размер
        out_features = out_features if out_features is not None else in_features

        # Устанавливаем размер скрытого слоя
        hidden_features = (
            hidden_features if hidden_features is not None else int(8 * in_features / 3)
        )
        hidden_features = (hidden_features + multiple_of - 1) // multiple_of * multiple_of

        # Линейные слои
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через GatedMLP.

        Args:
            x: Входной тензор формы (batch, ..., in_features).

        Returns:
            Выходной тензор формы (batch, ..., out_features).
        """
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)  # Разделяем на две части по последней размерности
        y = y * self.activation(gate)  # Применяем гейтинг
        y = self.fc2(y)
        return y

    def configure_optimizers(self):
        """
        Настройка оптимизатора для PyTorch Lightning.

        Returns:
            Оптимизатор (torch.optim.Adam).
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hyperparameters["learning_rate"])
        return optimizer
