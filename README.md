# Road Risk MG

A documentação científica oficial é [docs/artigo.tex](docs/artigo.tex).
O [PDF](docs/artigo.pdf) é a versão compilada desse mesmo texto.
Este README serve somente para instalação, navegação e reprodução.

## Organização

| Pasta | Conteúdo |
| --- | --- |
| `codigo/` | Pacote Python original, testes, fixtures, configurações, lock e pipeline DVC. |
| `docs/` | Um único documento LaTeX principal, atualizado com os resultados existentes. |
| `resultados/` | Tabelas, figuras e recibos originais que sustentam o documento. |
| `reproducao/` | Pacote científico original comprimido, sem alteração dos arquivos internos. |
| `.github/workflows/` | CI na raiz do novo repositório, executando dentro de `codigo/`. |

Foram excluídos ambientes virtuais, caches, bytecode, documentos duplicados,
apresentações, relatórios intermediários e ferramentas de planejamento.
O pacote de reprodução mantém seus arquivos históricos de planejamento:
eles fazem parte do inventário congelado e não podem ser removidos sem invalidá-lo.

O código Python, os testes, as configurações e o lock foram copiados sem edições.
Somente o LaTeX, este guia e a configuração externa do novo repositório foram
adaptados. Não houve novos treinamentos nem alterações nos resultados.

## Copiar para um novo repositório

Copie **todo o conteúdo desta pasta**, incluindo os arquivos ocultos.
Não copie o ambiente do projeto antigo e não copie sua pasta `.git`.

O arquivo `reproducao/confirmatory.tar.gz` é maior que o limite normal do GitHub.
Instale Git LFS e execute, na cópia e antes de adicionar os arquivos:

```sh
git init
git lfs install
git add .
```

A regra LFS já está em [.gitattributes](.gitattributes).
Crie o commit e configure o remoto somente quando desejar publicar.
Não há licença de redistribuição adicionada: mantenha o novo repositório privado
até revisar as permissões dos dados e do material acadêmico.

## Ambiente e execução

Os comandos do programa devem ser executados **dentro de `codigo/`**.
São necessários `uv` e Python 3.12.10; o lock preserva as versões do experimento.
Os índices de pacotes originais foram mantidos, sem desabilitar TLS.

```sh
cd codigo
uv sync --locked --group dev --system-certs
uv lock --check --offline
uv run --locked road-risk --help
uv run --locked pytest -q -m "not external"
uv run --locked road-risk --result-format json pipeline smoke --config configs/experiments/smoke.yaml
```

O smoke usa fixtures sintéticas e não baixa PRF ou OSM. Seus resultados não são
conclusões científicas. Instalação de dependências requer rede; os testes bloqueiam
sockets por padrão.

Não é necessário extrair o pacote histórico para executar testes ou smoke.
Os dados reais e pesos históricos estão somente no pacote, evitando duplicação.
Os seletores das fontes em `configs/sources.yaml` referenciam o inventário histórico;
comandos reais de preparação exigem restaurar essas fontes e o limite de MG antes
de usá-los. Não baixe fontes atuais para substituir silenciosamente os snapshots.

## Compilar a documentação oficial

O texto usa a classe padrão `article`, sem depender do template SBC ausente.
O conteúdo e as referências foram preservados, com atualização dos resultados;
não se afirma conformidade visual com o template SBC.

Com uma instalação LaTeX que contenha os pacotes usados:

```sh
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error artigo.tex
```

Alternativa com Tectonic:

```sh
cd docs
tectonic artigo.tex
```

Tectonic pode baixar os pacotes TeX na primeira compilação; a compilação ocorre
localmente. As tabelas e figuras referenciam somente arquivos desta pasta.
A revisão bibliográfica herdada não foi revalidada externamente nesta organização.

## Reproduzir o experimento histórico

Execute a extração na raiz do novo repositório:

```sh
python -m tarfile -e reproducao/confirmatory.tar.gz reproducao/extraido
```

Depois, com o ambiente instalado:

```sh
cd codigo
uv run --locked road-risk --result-format json reproduce --mode full --manifest ../reproducao/extraido/confirmatory/package-manifest.json
```

Esse comando usa o código arquivado no pacote, reconstrói os dados e repete as
inferências sem treinar. A saída fica em `codigo/artifacts/reproductions/`.
Ele exige o Python e o conjunto exato de distribuições
registrados em `runtime-environment.json` dentro do pacote. A instalação Linux
pode diferir do ambiente Windows original; nesse caso o programa bloqueia com
`environment_mismatch`. Não contorne essa verificação editando o manifesto.

O [recibo original](resultados/verificacao/reproduction.json), de 23/09/2026,
registra diferença máxima de 0,0 ponto percentual, sem treinamento, no mesmo
ambiente instalado. Ele não comprova uma nova reprodução nesta pasta ou no Linux.
O LaTeX atualizado não foi inserido retroativamente no pacote congelado.

## Verificação desta organização

- 100 arquivos do projeto copiados byte a byte, sem alteração de código ou configuração.
- 47 testes offline passaram, incluindo integração do smoke e preparação DVC.
- Lint, formatação, tipos e construção do pacote wheel passaram.
- PDF compilado sem avisos; tabelas preditivas, sazonais, ablações e custos conferidos com os CSVs.
- Os 733 componentes do arquivo comprimido correspondem aos hashes originais.

Os 67 ponteiros Git LFS encontrados no pacote local foram substituídos pelos
conteúdos originais recuperados, somente na cópia temporária usada para criar
este arquivo. Nenhum ponteiro foi tratado como dado válido.
A reprodução completa foi tentada no Linux e bloqueou corretamente por diferença
de ambiente: o original contém `pywin32` e `waitress`, enquanto a instalação Linux
contém `gunicorn` e `nvidia-nccl-cu13`. O recibo histórico não foi sobrescrito.

## Limites que continuam válidos

- Cobertura PRF de 2019–2025 e janeiro–julho de 2026 foi assumida, não comprovada.
- Sem prova de publicação anterior, históricos de acidentes permaneceram ausentes.
- A malha é estática, de 2018, e não há denominador de fluxo de tráfego.
- A comparação usou cinco sementes, um candidato por variante e orçamento limitado.
- XGBoost teve maior AP nos dois testes; não se demonstrou superioridade universal.
- A política de dependências preservada é exclusiva de pesquisa privada, não de produção.
