# Eldorado Fighter

Controle de personagens de **Street Fighter II** usando visão computacional:
uma câmera observa os movimentos do jogador, um modelo de Machine Learning
treinado com vídeos próprios classifica o movimento executado, e o
resultado é traduzido em teclas enviadas ao sistema operacional — **sem
alterar o código-fonte do jogo**.

```
Jogador na câmera → keypoints (YOLOv8-Pose) → classificação (TensorFlow) → tecla simulada → jogo
```

## Visão geral do pipeline

```
┌──────────┐    ┌───────────────┐    ┌───────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐
│  Câmera  │ -> │ OpenCV capture│ -> │ YOLOv8-Pose   │ -> │ Normalização  │ -> │ Buffer de      │ -> │ Modelo TensorFlow │
│ (webcam) │    │ (frame BGR)   │    │ (17 keypoints)│    │ (invariante a │    │ sequência      │    │ (classificação de│
└──────────┘    └───────────────┘    └───────────────┘    │ posição/escala)│    │ (janela de N   │    │ movimento)        │
                                                            └───────────────┘    │  frames)       │    └─────────┬─────────┘
                                                                                 └────────────────┘              │
                                                                                                                  v
                                                            ┌──────────────┐    ┌───────────────┐    ┌──────────────────┐
                                                            │     Jogo     │ <- │ Backend de    │ <- │   ActionMapper    │
                                                            │  (SF2, não   │    │ input (teclado│    │ (regras hold/tap, │
                                                            │  modificado) │    │  emulado)     │    │  cooldown)        │
                                                            └──────────────┘    └───────────────┘    └──────────────────┘
```

Cada seta é uma interface (ver [Padrões de projeto](#padrões-de-projeto)),
não uma dependência concreta. Isso é o que permite, no futuro, trocar
YOLOv8-Pose por outro extrator de pose, trocar o modelo TensorFlow por outra
arquitetura, ou trocar a emulação de teclado por um gamepad virtual — sem
reescrever o pipeline.

## Movimentos suportados (v1)

| Movimento | Categoria | Comportamento |
|---|---|---|
| `idle` | postura | neutro, nenhuma tecla |
| `walk_left` / `walk_right` | direção | tecla mantida pressionada enquanto detectado |
| `crouch` | postura | tecla mantida pressionada enquanto detectado |
| `jump` | ataque | tap único |
| `punch_light` / `punch_medium` / `punch_heavy` | ataque | tap único |
| `kick_light` / `kick_medium` / `kick_heavy` | ataque | tap único |

A lista completa (incluindo as teclas mapeadas) vive em
[`config/movements.yaml`](config/movements.yaml) — é a fonte única de
verdade dos movimentos. **Adicionar um movimento novo no futuro não exige
alterar código**, apenas uma nova entrada nesse arquivo + amostras de
dataset + retreino.

## Padrões de projeto

O projeto nasce pequeno (poucos movimentos, um backend de pose, um backend
de input), mas a estrutura já prevê o crescimento (mais movimentos,
combos, múltiplos jogadores, outro jogo). Por isso cada estágio do
pipeline é isolado por uma interface:

- **Strategy** — `FrameSource`, `PoseEstimator`, `MovementClassifier` e
  `InputBackend` são interfaces abstratas. As implementações atuais
  (`WebcamSource`/`VideoFileSource`, `YoloPoseEstimator`,
  `TensorFlowMovementClassifier`, `KeyboardInputBackend`) podem ser trocadas
  por outras sem alterar quem as usa.
- **Registry** — `MovementRegistry` carrega os movimentos de
  `config/movements.yaml` em tempo de execução. Nenhum módulo tem nomes de
  movimento "hardcoded"; todos consultam o registry.
- **Observer / Event Bus** — `EventBus` desacopla a detecção de movimento
  (pipeline) de quem reage a ela (`ActionMapper` hoje; um logger de debug
  ou métricas de acurácia em produção, no futuro, sem o pipeline saber que
  existem).
- **Pipe-and-Filter** — `RealtimePipeline` encadeia captura → pose →
  normalização → buffer → classificação → evento, cada etapa testável
  isoladamente.
- **Factory** — `training/model_factory.py` constrói a arquitetura de
  rede (`lstm` ou `dense`) a partir de uma string de config, permitindo
  comparar arquiteturas sem duplicar o script de treino.
- **Dependency Injection via config** — `config/settings.yaml` define qual
  implementação cada estágio usa; os `scripts/*.py` são as únicas fábricas
  concretas que leem a config e montam os objetos.

## Estrutura do projeto

```
Eldorado_fighter/
├── README.md
├── PROJECT_LOG.md              # histórico de etapas executadas
├── requirements.txt
├── pyproject.toml
├── config/
│   ├── settings.yaml           # config geral (qual backend usar em cada etapa)
│   └── movements.yaml          # catálogo de movimentos (fonte única de verdade)
├── src/eldorado_fighter/
│   ├── capture/                # FrameSource: webcam / arquivo de vídeo
│   ├── pose/                   # PoseEstimator: YOLOv8-Pose
│   ├── features/               # normalização de landmarks + buffer de sequência
│   ├── classification/         # MovementRegistry + MovementClassifier (TensorFlow)
│   ├── actions/                # InputBackend (teclado) + ActionMapper
│   ├── pipeline/                # EventBus + RealtimePipeline (orquestração)
│   ├── dataset/                 # gravação e schema das amostras de treino
│   ├── training/                 # data loader, factory de modelos, treino
│   └── utils/                    # config e logging
├── scripts/
│   ├── record_dataset.py        # grava amostras rotuladas (câmera -> dataset)
│   ├── extract_keypoints.py     # reextrai keypoints dos vídeos brutos
│   ├── train_model.py           # treina o classificador
│   └── run_live.py              # roda o controle em tempo real
├── data/
│   ├── raw_videos/<movimento>/  # vídeos brutos gravados (não versionados)
│   ├── keypoints/<movimento>/   # keypoints extraídos (.npz, não versionados)
│   └── models/<timestamp>/      # modelos treinados + metadata.json
└── tests/
```

## Setup

Testado em macOS. Requer Python 3.10+ (o `python3` padrão do sistema pode
ser mais antigo — use `python3.12` ou a versão equivalente do Homebrew).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # torna o pacote eldorado_fighter importável pelos scripts
```

Notas específicas do macOS:
- **Permissão de Acessibilidade**: a emulação de teclado (`pynput`) só
  funciona se o terminal (ou o interpretador Python) tiver permissão em
  *Ajustes do Sistema > Privacidade e Segurança > Acessibilidade*.
- **Permissão de Câmera**: o terminal também precisa de permissão em
  *Privacidade e Segurança > Câmera*.
- Em Apple Silicon, o pacote `tensorflow` do PyPI já roda nativamente em
  arm64 (não é necessário `tensorflow-macos` para TensorFlow >= 2.13).

## Como usar

### 1. Gravar o dataset

Para cada movimento, grave várias repetições (varie ângulo, roupa,
iluminação e velocidade para o modelo generalizar melhor):

```bash
python scripts/record_dataset.py --movement punch_light --samples 40 --clip-seconds 2
python scripts/record_dataset.py --movement walk_left --samples 30
# repita para cada movimento em config/movements.yaml
```

Isso salva o vídeo bruto em `data/raw_videos/<movimento>/` e os keypoints
já extraídos em `data/keypoints/<movimento>/`.

### 2. (Opcional) Reextrair keypoints

Se trocar/ajustar o `PoseEstimator`, reprocesse os vídeos sem regravar:

```bash
python scripts/extract_keypoints.py
```

### 3. Treinar o modelo

```bash
python scripts/train_model.py
```

O modelo treinado é salvo em `data/models/<timestamp>/model.keras` e
copiado para `data/models/latest/`, que é o caminho lido por
`config/settings.yaml` (`classification.model_path`).

### 4. Rodar o controle em tempo real

Abra o emulador/jogo de Street Fighter II, deixe a janela em foco e rode:

```bash
python scripts/run_live.py
```

## Roadmap

- [x] v1: parado, andar (esquerda/direita), agachar, pular, socos e chutes
      (fraco/médio/forte)
- [ ] Combos e golpes especiais (sequências de movimentos, não só poses)
- [ ] Backend de input alternativo (gamepad virtual) para emuladores que
      não aceitam teclado
- [ ] Suporte a dois jogadores simultâneos (câmera dividida ou duas câmeras)
- [ ] Modo de calibração (ajuste de confidence_threshold e cooldown por jogador)
- [ ] Métricas de acurácia em tempo real (overlay de debug)

Cada etapa efetivamente executada é registrada em
[`PROJECT_LOG.md`](PROJECT_LOG.md).
