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
