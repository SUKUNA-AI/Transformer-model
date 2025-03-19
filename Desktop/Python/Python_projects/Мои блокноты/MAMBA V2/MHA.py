import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

try:
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None

try:
    from flash_attn.layers.rotary import RotaryEmbedding
except ImportError:
    RotaryEmbedding = None

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None


def _update_kv_cache(kv, inference_params, layer_idx):
    # Извлекаем размеры последних двух измерений тензора kv: количество голов (num_heads) и размер головы (head_dim)
    num_heads, head_dim = kv.shape[-2:]

    # Проверяем, что указанный layer_idx присутствует в словаре памяти ключей и значений inference_params
    assert layer_idx in inference_params.key_value_memory_dict

    # Получаем кэш ключей и значений (kv_cache) и второе значение (conv_state, здесь не используется) для данного слоя
    kv_cache, _ = inference_params.key_value_memory_dict[layer_idx]

    # Определяем начало и конец батча для обновления кэша на основе смещения и размера текущего батча
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]

    # Определяем начало и конец последовательности на основе смещения длины последовательности
    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + kv.shape[1]

    # Убеждаемся, что конечный индекс батча не превышает размер кэша по оси батча
    assert batch_end <= kv_cache.shape[0]

    # Убеждаемся, что конечный индекс последовательности не превышает размер кэша по оси последовательности
    assert sequence_end <= kv_cache.shape[1]

    # Проверяем, что кэш не является None (должен быть инициализирован)
    assert kv_cache is not None

    # Обновляем соответствующую часть кэша новыми значениями из kv
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv

    # Возвращаем обновленный фрагмент кэша, обрезанный до текущей длины последовательности
    return kv_cache[batch_start:batch_end, :sequence_end, ...]


class MHA(pl.LightningModule):
    def __init__(
            self,
            embed_dim,              # Размер входного/выходного эмбеддинга
            num_heads,              # Количество голов внимания для запросов (query)
            num_heads_kv=None,      # Количество голов для ключей и значений (если None, равно num_heads)
            head_dim=None,          # Размер одной головы (если None, вычисляется как embed_dim / num_heads)
            mlp_dim=0,              # Размер скрытого слоя в MLP (если 0, MLP не используется)
            qkv_proj_bias=True,     # Использовать ли смещение (bias) в проекции QKV
            out_proj_bias=True,     # Использовать ли смещение в выходной проекции
            softmax_scale=None,     # Масштаб для softmax (если None, используется стандартный)
            causal=False,           # Причинное внимание (True для маскирования будущих токенов)
            layer_idx=None,         # Индекс слоя (нужен для инференса с кэшем)
            d_conv=0,               # Размер ядра свертки (если 0, свертка не используется)
            rotary_emb_dim=0,       # Размер ротационных эмбеддингов (если 0, не используются)
            rotary_emb_base=10000.0,# База для ротационных эмбеддингов
            rotary_emb_interleaved=False,  # Использовать ли чередующийся формат ротационных эмбеддингов
            device=None,            # Устройство (CPU/GPU)
            dtype=None,             # Тип данных (например, float32)
    ):
        # Вызываем конструктор родительского класса LightningModule
        super().__init__()

        # Сохраняем гиперпараметры для использования в будущем (например, при загрузке модели)
        self.save_hyperparameters()

        # Сохраняем ключевые параметры как атрибуты класса
        self.embed_dim = embed_dim
        self.layer_idx = layer_idx
        self.d_conv = d_conv
        self.rotary_emb_dim = rotary_emb_dim
        self.softmax_scale = softmax_scale
        self.causal = causal

        # Устанавливаем количество голов для запросов и ключей/значений
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads

        # Проверяем, что num_heads делится на num_heads_kv без остатка (для группового внимания)
        assert (
                self.num_heads % self.num_heads_kv == 0
        ), "num_heads must be divisible by num_heads_kv"

        # Если head_dim не задан, вычисляем его как embed_dim / num_heads
        if head_dim is None:
            assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = head_dim if head_dim is not None else self.embed_dim // num_heads

        # Округляем mlp_dim до ближайшего числа, кратного 256 (оптимизация памяти/вычислений)
        self.mlp_dim = math.ceil(mlp_dim / 256) * 256

        # Вычисляем размер входной проекции: запросы + ключи + значения
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)

        # Вычисляем размер выходного внимания: только запросы
        out_dim = self.head_dim * self.num_heads

        # Если используются ротационные эмбеддинги, инициализируем их
        if self.rotary_emb_dim > 0:
            assert RotaryEmbedding is not None, "rotary requires flash_attn to be installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        # Создаем входную проекцию: из embed_dim в qkv_dim + mlp_dim
        self.in_proj = nn.Linear(embed_dim, qkv_dim + self.mlp_dim, bias=qkv_proj_bias, device=device, dtype=dtype)

        # Если используется свертка, создаем 1D причинную сверточную сеть
        if self.d_conv > 0:
            self.conv1d = nn.Conv1d(
                qkv_dim, qkv_dim,               # Входной и выходной размер каналов
                kernel_size=self.d_conv,        # Размер ядра свертки
                padding=self.d_conv - 1,        # Дополнение для сохранения длины последовательности
                groups=qkv_dim,                 # Групповая свертка (каждый канал независим)
                device=device, dtype=dtype
            )

        # Создаем выходную проекцию: из out_dim + половины mlp_dim в embed_dim
        self.out_proj = nn.Linear(out_dim + self.mlp_dim // 2, embed_dim, bias=out_proj_bias, device=device, dtype=dtype)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        # Определяем тип данных: используем тип весов выходной проекции, если dtype не задан
        dtype = self.out_proj.weight.dtype if dtype is None else dtype

        # Определяем устройство на основе весов выходной проекции
        device = self.out_proj.weight.device

        # Если используется свертка, создаем тензор состояния свертки
        if self.d_conv > 0:
            conv_state = torch.zeros(
                batch_size, self.conv1d.weight.shape[0], self.d_conv,  # Форма: (батч, каналы, размер ядра)
                device=device, dtype=dtype
            )
        else:
            conv_state = None

        # Создаем пустой кэш ключей и значений с заданной формой
        kv_cache = torch.empty(
            batch_size, max_seqlen, 2, self.num_heads_kv, self.head_dim,  # Форма: (батч, длина, 2, головы_kv, размер головы)
            dtype=dtype, device=device,
        )

        # Возвращаем кортеж из кэша ключей/значений и состояния свертки
        return kv_cache, conv_state

    def _update_kv_cache(self, kv, inference_params):
        # Проверяем, что layer_idx задан (необходим для инференса с кэшем)
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"

        # Вызываем внешнюю функцию для обновления кэша
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def _apply_rotary_update_kvcache_attention(self, q, kv, inference_params):
        # Проверяем, что inference_params задан и это не первый шаг инференса
        assert inference_params is not None and inference_params.seqlen_offset > 0

        # Если используются ротационные эмбеддинги, обновляем кэш косинусов и синусов
        if self.rotary_emb_dim > 0:
            self.rotary_emb._update_cos_sin_cache(
                inference_params.max_seqlen, device=q.device, dtype=q.dtype
            )
            rotary_cos, rotary_sin = self.rotary_emb._cos_cached, self.rotary_emb._sin_cached
        else:
            rotary_cos, rotary_sin = None, None

        # Получаем размер батча из формы запросов
        batch = q.shape[0]

        # Извлекаем кэш ключей и значений для текущего слоя
        kv_cache, _ = inference_params.key_value_memory_dict[self.layer_idx]
        kv_cache = kv_cache[:batch]  # Обрезаем до текущего размера батча

        # Определяем длины последовательностей для кэша
        cache_seqlens = (
            inference_params.lengths_per_sample[:batch]
            if inference_params.lengths_per_sample is not None
            else inference_params.seqlen_offset
        )

        # Проверяем наличие оптимизированной функции внимания с кэшем
        assert flash_attn_with_kvcache is not None, "flash_attn must be installed"

        # Выполняем внимание с использованием кэша и ротационных эмбеддингов
        context = flash_attn_with_kvcache(
            q,                          # Запросы
            kv_cache[:, :, 0],         # Ключи из кэша
            kv_cache[:, :, 1],         # Значения из кэша
            kv[:, :, 0],              # Новые ключи
            kv[:, :, 1],              # Новые значения
            rotary_cos=rotary_cos,     # Косинусы для ротации
            rotary_sin=rotary_sin,     # Синусы для ротации
            cache_seqlens=cache_seqlens,  # Длины последовательностей
            softmax_scale=self.softmax_scale,  # Масштаб для softmax
            causal=self.causal,        # Причинное внимание
            rotary_interleaved=self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False,  # Формат ротации
        )
        return context

    def _update_kvcache_attention(self, q, kv, inference_params):
        # Если это первый шаг или оптимизированная функция недоступна
        if (
                inference_params.seqlen_offset == 0
                or flash_attn_with_kvcache is None
        ):
            # Обновляем кэш ключей и значений
            kv = self._update_kv_cache(kv, inference_params)

            # Разделяем ключи и значения из обновленного кэша
            k, v = kv.unbind(dim=-3)

            # Повторяем ключи и значения для соответствия количеству голов запросов
            k = torch.repeat_interleave(k, dim=2, repeats=self.num_heads // self.num_heads_kv)
            v = torch.repeat_interleave(v, dim=2, repeats=self.num_heads // self.num_heads_kv)

            # Выполняем стандартное масштабированное точечное произведение внимания
            return F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),  # Транспонируем для внимания
                is_causal=self.causal, scale=self.softmax_scale          # Параметры внимания
            ).transpose(1, 2)  # Возвращаем исходную форму
        else:
            # Получаем размер батча
            batch = q.shape[0]

            # Извлекаем кэш для текущего слоя
            kv_cache, _ = inference_params.key_value_memory_dict[self.layer_idx]
            kv_cache = kv_cache[:batch]

            # Определяем длины последовательностей
            cache_seqlens = (
                inference_params.lengths_per_sample[:batch]
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )

            # Выполняем оптимизированное внимание с кэшем
            return flash_attn_with_kvcache(
                q, kv_cache[:, :, 0], kv_cache[:, :, 1], kv[:, :, 0], kv[:, :, 1],  # Входные данные
                cache_seqlens=cache_seqlens, softmax_scale=self.softmax_scale, causal=self.causal,  # Параметры
            )

    def forward(self, x, inference_params=None):
        # Если переданы параметры инференса и кэш для слоя не инициализирован, выделяем память
        if inference_params is not None and self.layer_idx not in inference_params.key_value_memory_dict:
            inference_params.key_value_memory_dict[self.layer_idx] = self.allocate_inference_cache(
                x.shape[0], inference_params.max_seqlen, dtype=x.dtype
            )

        # Определяем смещение длины последовательности
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )

        # Определяем максимальную длину для ротационных эмбеддингов
        rotary_max_seqlen = inference_params.max_seqlen if inference_params is not None else None

        # Применяем входную проекцию для получения QKV и MLP-компоненты
        qkv = self.in_proj(x)

        # Если используется MLP, разделяем QKV и MLP-компоненту
        if self.mlp_dim > 0:
            qkv, x_mlp = qkv.split([qkv.shape[-1] - self.mlp_dim, self.mlp_dim], dim=-1)
            x_mlp_up, x_mlp_gate = x_mlp.chunk(2, dim=-1)  # Делим на две части
            x_mlp = x_mlp_up * F.silu(x_mlp_gate)          # Применяем SiLU-активацию

        # Если используется свертка
        if self.d_conv > 0:
            if inference_params is None or inference_params.seqlen_offset == 0:
                if causal_conv1d_fn is None:
                    # Транспонируем для свертки: (batch, seq, dim) -> (batch, dim, seq)
                    qkv = qkv.transpose(1, 2)
                    qkv = self.conv1d(qkv)[..., :-(self.d_conv - 1)]  # Применяем свертку и обрезаем padding
                    qkv = qkv.transpose(1, 2).contiguous()            # Возвращаем исходную форму
                else:
                    # Используем оптимизированную причинную свертку
                    qkv = causal_conv1d_fn(
                        qkv.transpose(1, 2),
                        self.conv1d.weight.squeeze(1),
                        self.conv1d.bias
                    ).transpose(1, 2)
                if inference_params is not None:
                    _, conv_state = inference_params.key_value_memory_dict[self.layer_idx]
                    qkv_t = qkv.transpose(1, 2)  # Транспонируем для обновления состояния
                    conv_state.copy_(F.pad(qkv_t, (self.d_conv - qkv_t.shape[-1], 0)))  # Обновляем состояние
            else:
                _, conv_state = inference_params.key_value_memory_dict[self.layer_idx]
                assert qkv.shape[1] == 1, "Only support decoding with 1 token at a time for now"
                qkv = qkv.squeeze(1)  # Убираем размер последовательности (1)
                if causal_conv1d_update is None:
                    # Сдвигаем состояние и добавляем новое значение
                    conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
                    conv_state[:, :, -1] = qkv
                    qkv = torch.sum(conv_state * self.conv1d.weight.squeeze(1), dim=-1)  # Свертка вручную
                    if self.conv1d.bias is not None:
                        qkv = qkv + self.conv1d.bias
                else:
                    # Используем оптимизированное обновление свертки
                    qkv = causal_conv1d_update(
                        qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias
                    )
                qkv = qkv.unsqueeze(1)  # Восстанавливаем размер последовательности

        # Разделяем запросы (q) и ключи/значения (kv)
        q, kv = qkv.split([self.num_heads * self.head_dim, self.num_heads_kv * 2 * self.head_dim], dim=-1)

        # Преобразуем форму запросов: (..., num_heads * head_dim) -> (..., num_heads, head_dim)
        q = q.view(*q.shape[:-1], self.num_heads, self.head_dim)

        # Преобразуем форму ключей/значений: (..., 2 * num_heads_kv * head_dim) -> (..., 2, num_heads_kv, head_dim)
        kv = kv.view(*kv.shape[:-1], 2, self.num_heads_kv, self.head_dim)

        # Выбираем метод внимания в зависимости от условий
        if (
                inference_params is None
                or inference_params.seqlen_offset == 0
                or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
        ):
            # Применяем ротационные эмбеддинги, если они включены
            if self.rotary_emb_dim > 0:
                q, kv = self.rotary_emb(
                    q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                )
            if inference_params is None:
                # Разделяем ключи и значения
                k, v = kv.unbind(dim=-3)
                k = torch.repeat_interleave(k, dim=2, repeats=self.num_heads // self.num_heads_kv)
                v = torch.repeat_interleave(v, dim=2, repeats=self.num_heads // self.num_heads_kv)
                # Выполняем стандартное внимание
                context = F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                    is_causal=self.causal, scale=self.softmax_scale
                ).transpose(1, 2)
            else:
                # Обновляем кэш и выполняем внимание
                context = self._update_kvcache_attention(q, kv, inference_params)
        else:
            # Используем оптимизированное внимание с ротацией и кэшем
            context = self._apply_rotary_update_kvcache_attention(q, kv, inference_params)

        # Преобразуем контекст обратно в плоский вид: (..., num_heads, head_dim) -> (..., num_heads * head_dim)
        context = context.view(*context.shape[:-2], -1)

        # Если используется MLP, объединяем контекст с MLP-компонентой
        if self.mlp_dim > 0:
            context = torch.cat([context, x_mlp], dim=-1)

        # Применяем выходную проекцию
        out = self.out_proj(context)
        return out

    def step(self, hidden_states, kv_cache, conv_state):
        """
        Пошаговый проход модели MHA для обработки одного токена (используется в инференсе).

        Аргументы:
            hidden_states: Входной тензор формы (batch, 1, embed_dim) — один токен на батч.
            kv_cache: Кэш ключей и значений формы (batch, seqlen, 2, num_heads_kv, head_dim).
            conv_state: Состояние свертки формы (batch, qkv_dim, d_conv) для причинной свертки (если d_conv > 0).

        Возвращает:
            Кортеж (out, kv_cache, conv_state):
                - out: Выходной тензор формы (batch, 1, embed_dim).
                - kv_cache: Обновленный кэш ключей и значений.
                - conv_state: Обновленное состояние свертки (или None, если d_conv == 0).
        """
        # Проверяем, что вход содержит ровно один токен по оси последовательности
        assert hidden_states.shape[1] == 1, "Метод step поддерживает только один токен за раз"

        # Извлекаем размер батча и тип данных
        batch = hidden_states.shape[0]
        dtype = hidden_states.dtype

        # Создаем временный объект inference_params для пошагового инференса
        inference_params = type('InferenceParams', (), {
            'key_value_memory_dict': {self.layer_idx: (kv_cache, conv_state)},  # Словарь с кэшем и состоянием
            'seqlen_offset': kv_cache.shape[1],                                 # Текущая длина последовательности
            'max_seqlen': kv_cache.shape[1] + 1,                                # Максимальная длина с новым токеном
            'batch_size_offset': 0,                                             # Смещение батча (не используется)
            'lengths_per_sample': None                                          # Длины последовательностей (не используются)
        })()

        # Применяем входную проекцию, убирая размер последовательности (1)
        qkv = self.in_proj(hidden_states.squeeze(1))  # (batch, qkv_dim + mlp_dim)

        # Разделяем на QKV и MLP-компоненту, если MLP включен
        if self.mlp_dim > 0:
            qkv, x_mlp = qkv.split([qkv.shape[-1] - self.mlp_dim, self.mlp_dim], dim=-1)
            x_mlp_up, x_mlp_gate = x_mlp.chunk(2, dim=-1)  # Делим MLP на две части
            x_mlp = x_mlp_up * F.silu(x_mlp_gate)          # Применяем SiLU-активацию

        # Применяем свертку, если она включена
        if self.d_conv > 0:
            if causal_conv1d_update is None:
                # Сдвигаем состояние свертки и добавляем новое значение
                conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
                conv_state[:, :, -1] = qkv
                # Вычисляем свертку вручную
                qkv = torch.sum(conv_state * self.conv1d.weight.squeeze(1), dim=-1)
                if self.conv1d.bias is not None:
                    qkv = qkv + self.conv1d.bias
            else:
                # Используем оптимизированное обновление свертки
                qkv = causal_conv1d_update(
                    qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias
                )
            qkv = qkv.unsqueeze(1)  # Восстанавливаем размер последовательности: (batch, 1, qkv_dim)
        else:
            qkv = qkv.unsqueeze(1)  # Добавляем размер последовательности: (batch, 1, qkv_dim)

        # Разделяем запросы и ключи/значения
        q, kv = qkv.split([self.num_heads * self.head_dim, self.num_heads_kv * 2 * self.head_dim], dim=-1)

        # Преобразуем форму запросов: (batch, 1, num_heads * head_dim) -> (batch, 1, num_heads, head_dim)
        q = q.view(batch, 1, self.num_heads, self.head_dim)

        # Преобразуем форму ключей/значений: (batch, 1, 2 * num_heads_kv * head_dim) -> (batch, 1, 2, num_heads_kv, head_dim)
        kv = kv.view(batch, 1, 2, self.num_heads_kv, self.head_dim)

        # Применяем ротационные эмбеддинги, если они включены
        if self.rotary_emb_dim > 0:
            self.rotary_emb._update_cos_sin_cache(inference_params.max_seqlen, device=q.device, dtype=q.dtype)
            rotary_cos, rotary_sin = self.rotary_emb._cos_cached, self.rotary_emb._sin_cached
            q, kv = self.rotary_emb(q, kv, seqlen_offset=inference_params.seqlen_offset)

        # Обновляем кэш, добавляя новые ключи и значения
        kv_cache_new = inference_params.key_value_memory_dict[self.layer_idx][0]
        kv_cache_new = torch.cat([kv_cache, kv], dim=1)  # Конкатенируем по оси последовательности
        inference_params.key_value_memory_dict[self.layer_idx] = (kv_cache_new, conv_state)

        # Выполняем внимание
        if flash_attn_with_kvcache is not None:
            # Используем оптимизированное внимание с кэшем
            context = flash_attn_with_kvcache(
                q,                          # Запросы
                kv_cache_new[:, :, 0],     # Ключи из кэша
                kv_cache_new[:, :, 1],     # Значения из кэша
                kv[:, :, 0],              # Новые ключи
                kv[:, :, 1],              # Новые значения
                cache_seqlens=inference_params.seqlen_offset,  # Длина кэша
                softmax_scale=self.softmax_scale,              # Масштаб softmax
                causal=self.causal,                            # Причинное внимание
                rotary_cos=rotary_cos if self.rotary_emb_dim > 0 else None,  # Косинусы ротации
                rotary_sin=rotary_sin if self.rotary_emb_dim > 0 else None,  # Синусы ротации
                rotary_interleaved=self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False,  # Формат ротации
            )
        else:
            # Используем стандартное внимание
            k, v = kv_cache_new.unbind(dim=-3)  # Разделяем ключи и значения
            k = torch.repeat_interleave(k, dim=2, repeats=self.num_heads // self.num_heads_kv)  # Повторяем для голов
            v = torch.repeat_interleave(v, dim=2, repeats=self.num_heads // self.num_heads_kv)
            context = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),  # Транспонируем для внимания
                is_causal=self.causal, scale=self.softmax_scale           # Параметры внимания
            ).transpose(1, 2)  # Возвращаем исходную форму

        # Преобразуем контекст: (batch, 1, num_heads, head_dim) -> (batch, 1, num_heads * head_dim)
        context = context.view(batch, 1, -1)

        # Добавляем MLP-компоненту, если она есть
        if self.mlp_dim > 0:
            context = torch.cat([context, x_mlp.unsqueeze(1)], dim=-1)

        # Применяем выходную проекцию
        out = self.out_proj(context)  # (batch, 1, embed_dim)

        # Возвращаем выход и обновленные состояния
        return out, kv_cache_new, conv_state

