# HAR Fighter

Controle de personagens de **Street Fighter II** usando visão computacional:
uma câmera observa os movimentos do jogador, dois modelos de Machine
Learning treinados com vídeos próprios classificam, **simultaneamente**, a
movimentação lateral e a ação do jogador, e o resultado é traduzido em
teclas enviadas ao sistema operacional — **sem alterar o código-fonte do
jogo**.

```
Jogador na câmera → keypoints (YOLOv8-Pose) → classificação (2 eixos, TensorFlow) → teclas simuladas → jogo
```

## Visão geral do pipeline

O sistema reconhece **dois eixos independentes e simultâneos** a partir da
mesma sequência de pose — por isso é possível, por exemplo, andar para a
esquerda e socar ao mesmo tempo:

- **movement** — `idle` / `walk_left` / `walk_right`
- **action** — `idle` / `crouch` / `jump` / `punch_light|medium|heavy` /
  `kick_light|medium|heavy`

```
┌──────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐  ┌────────────────┐
│  Câmera  │->│ OpenCV capture│->│ YOLOv8-Pose   │->│ Normalização  │->│ Buffer de       │
│ (webcam) │  │ (frame BGR)   │  │ (17 keypoints)│  │ (invariante a │  │ sequência       │
└──────────┘  └───────────────┘  └───────────────┘  │ posição/escala)│  │ (janela de N   │
                                                      └───────────────┘  │  frames, única) │
                                                                          └────────┬────────┘
                                   ┌──────────────────────────────────────────────┤
                                   v                                              v
                      ┌──────────────────────┐                     ┌──────────────────────┐
                      │ Modelo "movement"     │                     │ Modelo "action"       │
                      │ (TensorFlow)          │                     │ (TensorFlow)          │
                      └──────────┬────────────┘                     └──────────┬────────────┘
                                  v                                              v
                      ┌──────────────────────────────────────────────────────────────────┐
                      │                       ActionMapper                                 │
                      │  (regras hold/tap e cooldown POR EIXO — duas teclas podem ficar    │
                      │   seguradas ao mesmo tempo, uma de cada eixo)                       │
                      └───────────────────────────────┬──────────────────────────────────┘
                                                        v
                                           ┌──────────────────────┐    ┌──────────────┐
                                           │ Backend de input      │ -> │     Jogo     │
                                           │ (teclado emulado)     │    │ (SF2, não    │
                                           └──────────────────────┘    │  modificado) │
                                                                        └──────────────┘
```

Cada seta é uma interface (ver [Padrões de projeto](#padrões-de-projeto)),
não uma dependência concreta. Isso é o que permite, no futuro, trocar
YOLOv8-Pose por outro extrator de pose, trocar o modelo TensorFlow por outra
arquitetura, ou trocar a emulação de teclado por um gamepad virtual — sem
reescrever o pipeline. A captura, a pose e o buffer de sequência são
**únicos e compartilhados** entre os dois eixos — só a classificação final
duplica (o YOLO-pose, que é a parte cara, roda uma vez por frame).

## Classes suportadas (v2 — dois eixos)

| Eixo | Classe | Comportamento |
|---|---|---|
| `movement` | `idle` | neutro, nenhuma tecla |
| `movement` | `walk_left` / `walk_right` | tecla mantida pressionada enquanto detectado |
| `action` | `idle` | neutro, nenhuma tecla |
| `action` | `crouch` | tecla mantida pressionada enquanto detectado |
| `action` | `jump` | tap único |
| `action` | `punch_light` / `punch_medium` / `punch_heavy` | tap único |
| `action` | `kick_light` / `kick_medium` / `kick_heavy` | tap único |

A lista completa (incluindo as teclas mapeadas) vive em
[`config/movements.yaml`](config/movements.yaml), organizada em
`axes.movement` e `axes.action` — é a fonte única de verdade das classes.
**Adicionar uma classe nova no futuro não exige alterar código**, apenas
uma nova entrada no eixo certo desse arquivo + amostras de dataset +
retreino.

Cada amostra do dataset carrega um label por eixo (ver
[`dataset/schema.py`](src/har_fighter/dataset/schema.py)). Isso
significa que o dataset gravado hoje (cada clipe variando só um eixo, o
outro fica `idle`) já treina os dois modelos — mas o reconhecimento de
**combos** (ex.: andar socando) só generaliza bem se houver amostras
gravadas com os dois eixos variando juntos (`record_dataset.py --movement
walk_left --action punch_light`). Esse é o principal custo da abordagem de
dois eixos/dois modelos: o dataset de combos cresce combinatoriamente.

## Padrões de projeto

O projeto nasce pequeno (poucos movimentos, um backend de pose, um backend
de input), mas a estrutura já prevê o crescimento (mais movimentos,
combos, múltiplos jogadores, outro jogo). Por isso cada estágio do
pipeline é isolado por uma interface:

- **Strategy** — `FrameSource`, `PoseEstimator`, `MovementClassifier` e
  `InputBackend` são interfaces abstratas. As implementações atuais
  (`WebcamSource`/`VideoFileSource`, `YoloPoseEstimator`,
  `TensorFlowMovementClassifier`, `KeyboardInputBackend`) podem ser trocadas
  por outras sem alterar quem as usa. `MovementClassifier` é instanciado
  duas vezes (uma por eixo) com registries e modelos diferentes.
- **Registry** — `MovementRegistry.from_config(axis)` carrega as classes de
  um eixo de `config/movements.yaml` em tempo de execução. Nenhum módulo
  tem nomes de classe "hardcoded"; todos consultam o registry do eixo
  certo.
- **Observer / Event Bus** — `EventBus` desacopla a detecção de movimento
  (pipeline) de quem reage a ela (`ActionMapper` hoje; um logger de debug
  ou métricas de acurácia em produção, no futuro, sem o pipeline saber que
  existem).
- **Pipe-and-Filter** — `RealtimePipeline` encadeia captura → pose →
  normalização → buffer → classificação (dois eixos) → evento, cada etapa
  testável isoladamente.
- **Factory** — `training/model_factory.py` constrói a arquitetura de
  rede (`lstm`, `dense`, `fnet`, `mlp_mixer` ou `transformer`) a partir de
  uma string de config, permitindo comparar arquiteturas sem duplicar o
  script de treino; `train_model.py --axis` chama a fábrica uma vez por
  eixo.
- **Dependency Injection via config** — `config/settings.yaml` define qual
  implementação cada estágio usa; os `scripts/*.py` são as únicas fábricas
  concretas que leem a config e montam os objetos.

## Estrutura do projeto

```
HAR_fighter/
├── README.md
├── PROJECT_LOG.md              # histórico de etapas executadas
├── requirements.txt
├── pyproject.toml
├── config/
│   ├── settings.yaml           # config geral (qual backend usar em cada etapa)
│   └── movements.yaml          # catálogo de classes por eixo (fonte única de verdade)
├── src/har_fighter/
│   ├── capture/                # FrameSource: webcam / arquivo de vídeo
│   ├── pose/                   # PoseEstimator: YOLOv8-Pose
│   ├── features/               # normalização de landmarks + buffer de sequência
│   ├── classification/         # MovementRegistry + MovementClassifier (TensorFlow)
│   ├── actions/                # InputBackend (teclado) + ActionMapper (por eixo)
│   ├── pipeline/                # EventBus + RealtimePipeline (orquestração, 2 eixos)
│   ├── dataset/                 # gravação e schema (2 labels) das amostras de treino
│   ├── training/                 # data loader, factory de modelos, treino (por eixo)
│   └── utils/                    # config e logging
├── scripts/
│   ├── record_dataset.py        # grava amostras rotuladas (câmera -> dataset, --movement/--action)
│   ├── extract_keypoints.py     # reextrai keypoints dos vídeos brutos
│   ├── train_model.py           # treina o(s) classificador(es) (--axis movement|action)
│   ├── migrate_legacy_dataset.py # one-off: migra data/obsoleto/v1 para o schema de 2 labels
│   └── run_live.py              # roda o controle em tempo real (2 classificadores)
├── data/
│   ├── raw_videos/<movement>__<action>/  # vídeos brutos gravados (não versionados)
│   ├── keypoints/<movement>__<action>/   # keypoints extraídos (.npz, não versionados)
│   ├── models/<axis>/<timestamp>/        # modelos treinados + metadata.json, por eixo
│   └── obsoleto/v1/                      # dataset da v1 (schema de 1 label), arquivado
└── tests/
```

## Setup

Testado em macOS. Requer Python 3.10+ (o `python3` padrão do sistema pode
ser mais antigo — use `python3.12` ou a versão equivalente do Homebrew).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # torna o pacote har_fighter importável pelos scripts
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

### 0. (Já feito uma vez) Migrar o dataset legado

O dataset da v1 (um label só por clipe) está arquivado em
`data/obsoleto/v1/`. Ele já foi convertido para o schema de dois labels com:

```bash
python scripts/migrate_legacy_dataset.py
```

Não precisa rodar de novo — é um script one-off, mantido só para
referência/reprodutibilidade.

### 1. Gravar o dataset

Para cada classe, grave várias repetições (varie ângulo, roupa, iluminação
e velocidade para o modelo generalizar melhor). `--movement` e `--action`
são independentes — grave um eixo só (o outro fica `idle`) ou um combo dos
dois:

```bash
python scripts/record_dataset.py --action punch_light --samples 40 --clip-seconds 2
python scripts/record_dataset.py --movement walk_left --samples 30
python scripts/record_dataset.py --movement walk_left --action punch_light --samples 20  # combo
# repita para cada classe/combo em config/movements.yaml
```

Isso salva o vídeo bruto em `data/raw_videos/<movement>__<action>/` e os
keypoints já extraídos em `data/keypoints/<movement>__<action>/`.

### 2. (Opcional) Reextrair keypoints

Se trocar/ajustar o `PoseEstimator`, reprocesse os vídeos sem regravar:

```bash
python scripts/extract_keypoints.py
```

### 3. Treinar os modelos

```bash
python scripts/train_model.py              # treina os dois eixos
python scripts/train_model.py --axis movement
python scripts/train_model.py --axis action
```

Cada modelo treinado é salvo em `data/models/<axis>/<timestamp>/model.keras`
e copiado para `data/models/<axis>/latest/`, caminhos lidos por
`config/settings.yaml` (`classification.movement.model_path` /
`classification.action.model_path`).

### 4. Rodar o controle em tempo real

Abra o emulador/jogo de Street Fighter II, deixe a janela em foco e rode:

```bash
python scripts/run_live.py
```

Os dois classificadores rodam a cada frame; a janela de preview mostra uma
linha por eixo (movimentação e ação).

## Roadmap

- [x] v1: parado, andar (esquerda/direita), agachar, pular, socos e chutes
      (fraco/médio/forte) — um único classificador, uma classe por vez
- [x] v2: dois eixos simultâneos (movimentação + ação) com classificadores
      independentes, permitindo andar e atacar ao mesmo tempo
- [ ] Gravar dataset de combos (movimento + ação variando juntos) — hoje o
      dataset migrado só cobre cada eixo isoladamente
- [ ] Combos e golpes especiais (sequências de movimentos, não só poses)
- [ ] Backend de input alternativo (gamepad virtual) para emuladores que
      não aceitam teclado
- [ ] Suporte a dois jogadores simultâneos (câmera dividida ou duas câmeras)
- [ ] Modo de calibração (ajuste de confidence_threshold e cooldown por jogador)
- [ ] Métricas de acurácia em tempo real (overlay de debug)

Cada etapa efetivamente executada é registrada em
[`PROJECT_LOG.md`](PROJECT_LOG.md).

## Licença

Este projeto é licenciado sob **CC BY-NC 4.0** (Creative Commons
Attribution-NonCommercial 4.0 International) — uso comercial proibido e
créditos ao autor original (matteo167) são obrigatórios em qualquer
cópia, adaptação ou redistribuição. Veja o arquivo [`LICENSE`](LICENSE)
para o texto completo.
