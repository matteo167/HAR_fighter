# Log do projeto — Eldorado Fighter

Registro cronológico de cada etapa executada. Para a visão de arquitetura e
instruções de uso, ver [`README.md`](README.md).

---

## 2026-11-03 — Etapa 0: Setup inicial do projeto

**O que foi feito:**
- Criada a estrutura completa de diretórios do projeto (`src/eldorado_fighter`,
  `config/`, `scripts/`, `data/`, `tests/`).
- Definido o catálogo inicial de movimentos em `config/movements.yaml`:
  `idle`, `walk_left`, `walk_right`, `crouch`, `jump`, `punch_light`,
  `punch_medium`, `punch_heavy`, `kick_light`, `kick_medium`, `kick_heavy`.
- Implementadas as interfaces (Strategy) e primeiras implementações de
  cada estágio do pipeline:
  - `capture`: `FrameSource` (interface) + `WebcamSource` + `VideoFileSource`.
  - `pose`: `PoseEstimator` (interface) + `MediaPipePoseEstimator`.
  - `features`: `normalize_landmarks` (normalização por centro do quadril e
    escala do tronco) + `SequenceBuffer` (janela deslizante).
  - `classification`: `MovementRegistry` (carregado do YAML),
    `MovementClassifier` (interface) + `TensorFlowMovementClassifier`.
  - `actions`: `InputBackend` (interface) + `KeyboardInputBackend` (pynput)
    + `ActionMapper` (regras de hold vs. tap e cooldown).
  - `pipeline`: `EventBus` (Observer) + `RealtimePipeline` (orquestração
    Pipe-and-Filter).
  - `dataset`: schema das amostras (`.npz` com keypoints brutos + vídeo
    bruto) + `DatasetRecorder`.
  - `training`: `dataset_loader` (gera janelas de sequência a partir dos
    clipes gravados), `model_factory` (arquiteturas `lstm`/`dense`),
    `train.py` (treino + avaliação + versionamento do modelo por timestamp).
- Criados os scripts de linha de comando: `record_dataset.py`,
  `extract_keypoints.py`, `train_model.py`, `run_live.py`.
- Criados `requirements.txt`, `pyproject.toml` (empacotamento `src/`) e
  `.gitignore` (dados grandes/gerados fora do controle de versão).
- Escrito o `README.md` com visão geral, diagrama de arquitetura, padrões
  de projeto adotados, estrutura de diretórios e instruções de uso.

**Por quê:**
- O pipeline foi desenhado em estágios isolados por interface (Strategy)
  desde o início porque o próprio pedido do projeto é incremental: hoje são
  7 categorias de movimento com um único backend de pose/classificação/input,
  mas a lista de movimentos, a arquitetura do modelo e até o backend de
  input devem evoluir. Isolar por interface evita que essas trocas futuras
  exijam reescrever o pipeline.
- Vídeo bruto e keypoints extraídos são persistidos separadamente
  (`data/raw_videos` vs. `data/keypoints`) para permitir reprocessar o
  dataset com um `PoseEstimator` melhor no futuro sem precisar regravar.

**Status:** Estrutura do projeto pronta. Ainda **não há dataset gravado nem
modelo treinado** — são os próximos passos manuais (gravação depende do
usuário executar os movimentos na câmera).

**Próximos passos:**
1. Gravar amostras de cada movimento com `scripts/record_dataset.py`.
2. Treinar o primeiro modelo com `scripts/train_model.py`.
3. Validar o controle em tempo real com `scripts/run_live.py` contra um
   emulador de Street Fighter II e calibrar `confidence_threshold` e
   `cooldown_seconds` em `config/settings.yaml`.

---

## 2026-11-05 — Etapa 1: Dois eixos simultâneos (movimentação + ação)

**O que foi feito:**
- `data/keypoints/` e `data/raw_videos/` (280 clipes gravados na v1) foram
  movidos para `data/obsoleto/v1/`, preservando o histórico git dos
  keypoints via `git mv`.
- `config/movements.yaml` passou de uma lista única de movimentos para
  dois catálogos independentes sob `axes`: `movement` (`idle`, `walk_left`,
  `walk_right`) e `action` (`idle`, `crouch`, `jump`, `punch_*`, `kick_*`).
  O campo `category` (não lido por nenhum código) foi removido.
- `MovementRegistry.from_config(axis)`, `dataset/schema.KeypointSample`
  (agora com `movement_label` + `action_label`), `DatasetRecorder.
  record_clip`, `training/dataset_loader.load_dataset(axis=...)` e
  `training/train.run_training(axis=...)` foram adaptados para o schema de
  dois labels — cada clipe gravado é identificado pela combinação
  `<movement>__<action>` e pode ser reaproveitado no treino dos dois eixos.
- `RealtimePipeline` passou a receber um `MovementClassifier` por eixo
  (`dict[str, MovementClassifier]`) e publica `movement_detected` /
  `action_detected` separadamente a cada frame, sobre a MESMA sequência de
  pose — captura, pose e buffer continuam únicos.
- `ActionMapper` passou a indexar tecla segurada e cooldown de tap por
  eixo (`on_label_detected(axis, prediction)`), permitindo duas teclas
  seguradas ao mesmo tempo (uma por eixo) sem colisão entre as classes
  `idle` dos dois eixos.
- `scripts/run_live.py`, `train_model.py` (`--axis`), `record_dataset.py`
  (`--movement`/`--action` independentes) e `extract_keypoints.py` foram
  atualizados para o novo esquema.
- Criado `scripts/migrate_legacy_dataset.py` (one-off) e executado: migrou
  os 280 clipes de `data/obsoleto/v1/` para o novo schema de dois labels
  (o eixo não gravado em cada clipe antigo foi marcado `idle`), deixando o
  dataset existente imediatamente utilizável pelos dois classificadores.
- `README.md` e `config/settings.yaml` atualizados para refletir os dois
  eixos (`classification.movement.*` / `classification.action.*`,
  `data/models/<axis>/...`).

**Por quê:**
- O jogador quer andar e atacar ao mesmo tempo (ex.: avançar socando), o
  que é impossível com um único classificador softmax sobre todas as
  classes — movimento e ação precisam ser sinais independentes.
- Dois modelos independentes (em vez de um multi-head) foram escolhidos
  por simplicidade de implementação/depuração, aceitando 2x inferência de
  classificação — custo pequeno comparado ao YOLO-pose, que não duplica.
- Migrar o dataset da v1 em vez de descartá-lo evita perder as 280
  gravações já feitas; o risco conhecido é que elas só cobrem cada eixo
  isoladamente — reconhecer combos vai exigir gravar amostras com os dois
  eixos variando juntos (ver Roadmap no README).

**Status:** Refatoração completa e verificada (treino dos dois eixos
ponta-a-ponta contra o dataset migrado). **Não testado**: fluxo completo de
webcam + teclado real (`scripts/run_live.py` interativo) — requer validação
manual do usuário.

**Próximos passos:**
1. Validar `scripts/run_live.py` contra o emulador com os dois eixos.
2. Gravar amostras de combo (`--movement` + `--action` juntos) para o
   modelo de ação generalizar quando o jogador está se movendo.
3. Recalibrar `confidence_threshold`/`cooldown_seconds` por eixo, se
   necessário.
