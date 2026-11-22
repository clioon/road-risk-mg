# Road Risk MG

## Contexto

Projeto de pesquisa para prever a ocorrência semanal de acidentes em segmentos de rodovias federais de Minas Gerais, usando dados da PRF e a malha do OpenStreetMap. O fluxo inclui tratamento dos dados, construção do grafo, treinamento, avaliação e geração de tabelas, gráficos e relatórios.

São comparados Random Forest, XGBoost, MLP, GraphSAGE e GATv2, com treino em 2019–2023, validação em 2024 e testes em 2025 e janeiro–julho de 2026. O método, os resultados e as limitações estão no [relatório final oficial](docs/trab_final.tex).

Os resultados dependem de cobertura temporal assumida, não comprovada, e de uma malha estática de 2018.

## Estrutura dos arquivos

```text
codigo/
  src/road_accidents/     Dados, atributos, grafos, modelos, avaliação e relatórios
  configs/               Configurações de fontes, dataset, modelos e experimentos
  tests/                 Testes e fixtures sintéticas
  dvc.yaml               Pipeline: preparação → treinamento → relatório
  pyproject.toml         Dependências e comando road-risk
  uv.lock                Versões das dependências do experimento
docs/
  trab_final.tex         Relatório final oficial em LaTeX
  imagens/               Imagens do relatório
resultados/              Tabelas, gráficos e recibos do experimento histórico
reproducao/
  confirmatory.tar.gz    Pacote histórico com código, dados e modelos treinados
```

O pacote histórico usa Git LFS. Ao clonar o repositório, tenha o Git LFS instalado para obter o arquivo completo.

## Execução

### Instalar o ambiente

Requisitos: **uv** e **Python 3.12.10**. Execute os comandos do programa dentro de `codigo/`. A instalação das dependências requer acesso à rede.

```sh
cd codigo
uv sync --locked --group dev --system-certs
uv run --locked road-risk --help
```

### Executar testes e o fluxo sintético

```sh
uv run --locked pytest -q -m "not external"
uv run --locked road-risk --result-format json pipeline smoke --config configs/experiments/smoke.yaml
```

O smoke percorre preparação, treinamento, avaliação, relatório e reprodução com dados sintéticos, sem baixar PRF ou OSM. Não exige extrair o pacote histórico e não produz conclusões científicas. As saídas ficam em `codigo/artifacts/smoke/`.

### Treinar e avaliar com dados reais

Antes de executar, restaure os snapshots históricos, seus manifestos e o limite de MG nos caminhos esperados por [dvc.yaml](codigo/dvc.yaml) e pelas [configurações](codigo/configs/). Esses dados estão no pacote de reprodução; a extração sozinha não os coloca nos caminhos de trabalho. Não substitua os snapshots por fontes atuais.

Com as entradas restauradas, execute dentro de `codigo/`:

```sh
uv run --locked dvc repro
```

O DVC prepara os dados, treina os modelos e gera avaliações e relatórios. As saídas ficam em `codigo/data/processed/` e `codigo/artifacts/`. Os comandos individuais estão disponíveis em `road-risk --help`.

### Reproduzir os resultados históricos sem treinar

Na raiz do repositório, extraia o pacote e execute com o ambiente instalado:

```sh
python -m tarfile -e reproducao/confirmatory.tar.gz reproducao/extraido
cd codigo
uv run --locked road-risk --result-format json reproduce --mode full --manifest ../reproducao/extraido/confirmatory/package-manifest.json
```

Esse comando reconstrói os dados e repete as inferências com o código e os pesos arquivados. A saída fica em `codigo/artifacts/reproductions/`. Exige o ambiente exato de `runtime-environment.json`; diferenças em relação ao Windows original podem bloquear a execução com `environment_mismatch`. Não edite o manifesto para contornar essa verificação.
