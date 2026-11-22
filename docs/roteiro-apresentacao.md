# Roteiro de apresentação do código-fonte — Road Risk MG

**Duração: 5min30s.** O [vídeo sincronizado (MP4, sem áudio)](road-risk-mg-codigo-fonte.mp4) troca de trecho a cada 11 segundos, no início de cada bloco abaixo. Leiam apenas os parágrafos de fala. Comecem na marca inicial; se terminarem cedo, façam uma pausa até o próximo corte.

Na edição, mantenham o vídeo em velocidade normal e sobreponham a câmera na área vazia do canto inferior direito. Cada pessoa grava 1min50s de fala.

## João — 00:00–01:50

### 00:00–00:11 — Ponto de entrada

**Na tela:** [`pyproject.toml`, linhas 38–39](../codigo/pyproject.toml#L38-L39) — O comando aponta para a função `main`.

Olá! Somos João, Giovanni e Lion. O comando road-risk começa aqui: o arquivo aponta para a função principal da interface.

### 00:11–00:22 — Grupos de comandos

**Na tela:** [`cli.py`, linhas 38–46](../codigo/src/road_accidents/cli.py#L38-L46) — O Typer registra os quatro grupos.

No Typer, registramos quatro grupos: fontes, conjunto de dados, experimentos e pipeline. Cada grupo reúne comandos de uma etapa.

### 00:22–00:33 — Preparação pelo DVC

**Na tela:** [`dvc.yaml`, linhas 3–10](../codigo/dvc.yaml#L3-L10) — A etapa `prepare` recebe a configuração e os manifestos.

O DVC encadeia as etapas. Na primeira, o módulo de dados recebe a configuração e os manifestos de fontes e cobertura.

### 00:33–00:44 — Orquestração dos experimentos

**Na tela:** [`pipeline.py`, linhas 150–166](../codigo/src/road_accidents/pipeline.py#L150-L166) — A segunda etapa congela o experimento; a terceira avalia o pacote.

A etapa de experimento usa a base preparada para treinar e congelar o pacote. A etapa de relatório avalia esse pacote.

### 00:44–00:55 — Montagem da base

**Na tela:** [`data/pipeline.py`, linhas 140–149](../codigo/src/road_accidents/data/pipeline.py#L140-L149) — Rede, segmentos, grafo e calendário são preparados antes das tabelas.

Esta função carrega a malha, cria os segmentos e monta a vizinhança. Depois, usa a cobertura para construir o calendário.

### 00:55–01:06 — Alinhamento das tabelas

**Na tela:** [`data/pipeline.py`, linhas 175–182](../codigo/src/road_accidents/data/pipeline.py#L175-L182) — Atributos e rótulos precisam ter as mesmas chaves, na mesma ordem.

Atributos e rótulos saem de funções separadas. Se as chaves de segmento e semana não estiverem na mesma ordem, a preparação para.

### 01:06–01:17 — Curadoria da PRF

**Na tela:** [`prf.py`, linhas 380–396](../codigo/src/road_accidents/data/prf.py#L380-L396) — A curadoria anota duplicatas e ocorrências fora do recorte.

Na curadoria da PRF, a função normaliza o identificador e marca dois tipos de registro inválido: duplicatas e ocorrências de outros estados.

### 01:17–01:28 — Validação das coordenadas

**Na tela:** [`prf.py`, linhas 409–420](../codigo/src/road_accidents/data/prf.py#L409-L420) — Coordenadas inválidas são sinalizadas; pontos válidos passam pelo teste do polígono.

A função marca coordenadas ausentes, zeradas ou fora da faixa. Quando consegue criar um ponto, verifica se ele fica em Minas Gerais.

### 01:28–01:39 — Leitura da malha histórica

**Na tela:** [`osm.py`, linhas 416–425](../codigo/src/road_accidents/data/osm.py#L416-L425) — O carregador confere o hash antes de identificar as vias federais.

Na leitura do OpenStreetMap, a função confere o hash do arquivo, valida a extração histórica e identifica as referências das rodovias federais.

### 01:39–01:50 — Identidade de cada trecho

**Na tela:** [`segments.py`, linhas 72–76](../codigo/src/road_accidents/graph/segments.py#L72-L76) — A chave usa as pontas ordenadas e a geometria normalizada.

Só entram trechos inteiros no estado. A chave combina pontas ordenadas e geometria normalizada, unificando os dois sentidos quando a geometria é igual.

## Giovanni — 01:50–03:40

### 01:50–02:01 — Vizinhança no grafo

**Na tela:** [`adjacency.py`, linhas 23–36](../codigo/src/road_accidents/graph/adjacency.py#L23-L36) — Os pares de vizinhos vêm de junções compartilhadas.

Cada trecho é um nó. O código forma pares em junções compartilhadas: cruzar no mapa, sem uma junção, não cria ligação.

### 02:01–02:12 — Acidente e trecho mais próximo

**Na tela:** [`matching.py`, linhas 32–49](../codigo/src/road_accidents/graph/matching.py#L32-L49) — O candidato mais próximo só é aceito dentro do limite de distância.

A função ordena os candidatos por distância e identificador. Só aceita o trecho mais próximo se estiver a até quinhentos metros.

### 02:12–02:23 — Semanas elegíveis

**Na tela:** [`weekly.py`, linhas 37–47](../codigo/src/road_accidents/features/weekly.py#L37-L47) — O calendário exclui semanas incompletas ou fora da cobertura declarada.

O calendário avança de sete em sete dias. Uma semana fica fora se cruza partições ou não tem cobertura declarada completa.

### 02:23–02:34 — Rótulo por trecho e semana

**Na tela:** [`weekly.py`, linhas 119–131](../codigo/src/road_accidents/features/weekly.py#L119-L131) — O SQL cruza segmentos com semanas e converte a contagem em rótulo binário.

O SQL combina cada segmento com cada semana elegível. O rótulo vale um se houve acidente registrado e zero caso contrário.

### 02:34–02:45 — Atributos da via e do calendário

**Na tela:** [`temporal.py`, linhas 123–138](../codigo/src/road_accidents/features/temporal.py#L123-L138) — A função junta os atributos fixos aos campos que variam por semana.

A função pega comprimento, tipo de via e faixas. Para cada semana, junta os campos do calendário e tenta recuperar o histórico.

### 02:45–02:56 — Histórico sem informação futura

**Na tela:** [`temporal.py`, linhas 69–77](../codigo/src/road_accidents/features/temporal.py#L69-L77) — Sem fonte disponível antes da previsão, a contagem fica ausente.

Sem comprovação de cobertura e publicação prévia, a contagem fica ausente, não zero. Assim, o código não usa informação futura.

### 02:56–03:07 — Ajuste apenas no treino

**Na tela:** [`preprocessing.py`, linhas 27–38](../codigo/src/road_accidents/features/preprocessing.py#L27-L38) — O pré-processador recusa ajuste fora do período de treino.

O pré-processador só aceita datas de treino no ajuste. Assim, validação e teste não influenciam as estatísticas usadas na transformação.

### 03:07–03:18 — Mediana e padronização

**Na tela:** [`preprocessing.py`, linhas 44–57](../codigo/src/road_accidents/features/preprocessing.py#L44-L57) — As colunas numéricas passam pelo imputador e pelo padronizador.

Nas colunas numéricas, a transformação preenche ausências com a mediana, sinaliza os dados faltantes e padroniza a escala de cada coluna.

### 03:18–03:29 — Ordem dos nós por semana

**Na tela:** [`graph/dataset.py`, linhas 44–59](../codigo/src/road_accidents/graph/dataset.py#L44-L59) — Toda semana é alinhada à mesma lista de nós.

Para as redes de grafos, agrupamos as linhas por semana. Cada grupo precisa ter todos os segmentos; o código os ordena.

### 03:29–03:40 — Tensor de arestas

**Na tela:** [`graph/dataset.py`, linhas 68–85](../codigo/src/road_accidents/graph/dataset.py#L68-L85) — A variante sem topologia recebe um tensor vazio, mas mantém os atributos.

O tensor guarda as ligações nos dois sentidos. Na variante sem topologia, ele fica vazio; os atributos dos nós são mantidos.

## Lion — 03:40–05:30

### 03:40–03:51 — Modelos tabulares

**Na tela:** [`baselines.py`, linhas 33–51](../codigo/src/road_accidents/models/baselines.py#L33-L51) — O módulo encaminha a matriz para Random Forest, XGBoost ou MLP.

O caminho tabular recebe uma matriz. A função usa Random Forest ou XGBoost ou encaminha essa matriz para a rede MLP.

### 03:51–04:02 — Convolução no grafo

**Na tela:** [`graphsage.py`, linhas 19–25](../codigo/src/road_accidents/models/graphsage.py#L19-L25) — As duas camadas recebem os atributos e as conexões.

No GraphSAGE, as duas camadas recebem atributos e arestas. Elas agregam informações dos vizinhos e devolvem um escore bruto por segmento.

### 04:02–04:13 — Limiar na validação

**Na tela:** [`training.py`, linhas 265–278](../codigo/src/road_accidents/experiments/training.py#L265-L278) — Os escores de validação determinam o ponto de corte.

A rotina ajusta o modelo no treino e calcula escores na validação. O ponto de corte é escolhido ali, nunca no teste.

### 04:13–04:24 — Congelamento do experimento

**Na tela:** [`freeze.py`, linhas 131–147](../codigo/src/road_accidents/experiments/freeze.py#L131-L147) — O manifesto liga as execuções escolhidas aos hashes dos artefatos.

O manifesto registra as execuções escolhidas e os hashes dos arquivos. Assim, dá para detectar arquivos alterados depois do congelamento.

### 04:24–04:35 — Inferência no teste

**Na tela:** [`runner.py`, linhas 164–179](../codigo/src/road_accidents/evaluation/runner.py#L164-L179) — A avaliação usa o modelo e o limiar já salvos.

No teste, o avaliador chama o modelo salvo, junta os rótulos pelas chaves e reaplica o ponto de corte congelado.

### 04:35–04:46 — Cálculo das métricas

**Na tela:** [`metrics.py`, linhas 57–69](../codigo/src/road_accidents/evaluation/metrics.py#L57-L69) — A precisão média usa escores; o F1 usa decisões binárias.

A precisão média usa os escores. Após o corte, verdadeiros positivos, falsos positivos e falsos negativos entram no cálculo do F1.

### 04:46–04:57 — Escrita das tabelas

**Na tela:** [`tables.py`, linhas 340–351](../codigo/src/road_accidents/reporting/tables.py#L340-L351) — Métricas, execuções, custos, populações e ablações vão para arquivos distintos.

O relatório separa as métricas agregadas das execuções individuais. Também grava tabelas com as populações, as ablações e os custos.

### 04:57–05:08 — Inventário do pacote

**Na tela:** [`packaging.py`, linhas 257–266](../codigo/src/road_accidents/reporting/packaging.py#L257-L266) — Cada componente tem caminho, tamanho e hash registrados.

No empacotamento, cada entrada registra caminho, tamanho e hash. Assim, podemos conferir se os arquivos recebidos correspondem ao inventário.

### 05:08–05:19 — Reconstrução do conjunto

**Na tela:** [`reproduce.py`, linhas 322–339](../codigo/src/road_accidents/reporting/reproduce.py#L322-L339) — A reprodução refaz o conjunto e confere a versão reconstruída.

Na reprodução, as fontes arquivadas voltam para a preparação. O programa valida a base reconstruída e compara a assinatura dos dados de desenvolvimento.

### 05:19–05:30 — Repetição da inferência

**Na tela:** [`reproduce.py`, linhas 349–361](../codigo/src/road_accidents/reporting/reproduce.py#L349-L361) — O mesmo pacote volta à avaliação e as métricas são comparadas.

Por fim, repetimos a inferência sem treinar de novo e comparamos as métricas. Esse é o caminho pelo código. Obrigado!
