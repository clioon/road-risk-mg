# Roteiro de apresentação do código-fonte — Road Risk MG

**Duração: 5min30s.** O [vídeo-guia de gravação](road-risk-mg-codigo-fonte.mp4) mostra o relógio e a próxima troca de assunto. Cada bloco abaixo dura **22 segundos**. Leiam só o parágrafo depois de **Na tela**. Comecem na marca inicial e, se terminarem cedo, façam uma pausa até o próximo corte.

Na edição, troquem o guia pelo [vídeo limpo para entrega](road-risk-mg-apresentacao-final.mp4) sem mudar o início nem a velocidade. Os dois MP4 são silenciosos e têm os mesmos cortes. Sobreponham a câmera no canto inferior direito. Cada pessoa tem 1min50s.

## João — 00:00–01:50

### 00:00–00:22 — Mapa do repositório

**Na tela:** [Raiz do repositório](../) — As pastas separam o programa, a documentação, os resultados e o pacote de reprodução.

Olá! Somos João, Giovanni e Lion. O Road Risk MG estima em quais trechos das rodovias federais de Minas Gerais pode haver um acidente registrado em uma semana. Antes de abrir as funções, vamos ver onde fica cada parte do programa.

### 00:22–00:44 — Dentro da pasta de código

**Na tela:** [`codigo/`](../codigo/) — Código-fonte, configurações, testes, scripts de apoio e ordem de execução.

Na pasta código, a parte chamada src guarda o programa. As configurações definem fontes e parâmetros, e os testes conferem o comportamento das etapas. Também há um script de apoio para checar dependências. O arquivo do DVC encadeia preparação, treino e relatório.

### 00:44–01:06 — Dados, mapa e atributos

**Na tela:** [`src/road_accidents/`](../codigo/src/road_accidents/) — `data/`, `graph/` e `features/` transformam as fontes em tabelas por semana.

No código-fonte, a pasta de dados prepara os registros da Polícia Rodoviária Federal e a malha do OpenStreetMap. A de grafos organiza os trechos e suas ligações. A de atributos monta as tabelas semanais: o que o modelo recebe e a resposta que precisa prever.

### 01:06–01:28 — Do treino ao relatório

**Na tela:** [`src/road_accidents/`](../codigo/src/road_accidents/) — `models/`, `experiments/`, `evaluation/` e `reporting/` cobrem o restante do fluxo.

A pasta de modelos reúne as implementações que fazem as previsões. A de experimentos coordena o treino e fixa as versões escolhidas. A avaliação aplica esses modelos aos dados de teste e calcula as métricas. A pasta de relatórios transforma essas saídas em tabelas e gráficos.

### 01:28–01:50 — Onde consultar as saídas

**Na tela:** [Raiz do repositório](../) — `docs/`, `resultados/` e `reproducao/` separam documentação, saídas e material arquivado.

A pasta docs reúne a documentação. Resultados guarda as tabelas e os gráficos já gerados, e reprodução contém o pacote para repetir a avaliação com o material arquivado. Com esse mapa em mente, o Giovanni vai mostrar como um registro de acidente vira uma linha na tabela.

## Giovanni — 01:50–03:40

### 01:50–02:12 — Como os trechos se conectam

**Na tela:** [`adjacency.py`, linhas 23–36](../codigo/src/road_accidents/graph/adjacency.py#L23-L36) — Só trechos com uma junção em comum viram vizinhos.

No nosso grafo, cada trecho de rodovia é um nó. Esta função agrupa os trechos que compartilham uma junção e forma pares de vizinhos. Por exemplo, um viaduto que passa por cima de outra via, sem ligação entre elas, não cria um par.

### 02:12–02:34 — Do ponto do acidente ao trecho

**Na tela:** [`matching.py`, linhas 32–49](../codigo/src/road_accidents/graph/matching.py#L32-L49) — O trecho mais próximo só é aceito até 500 metros.

O registro do acidente traz uma localização, mas o modelo trabalha com trechos. Aqui, o código escolhe o trecho mais próximo e só aceita a associação se a distância for de até quinhentos metros. Acima desse limite, o ponto fica sem trecho associado.

### 02:34–02:56 — Uma linha por trecho e semana

**Na tela:** [`weekly.py`, linhas 119–130](../codigo/src/road_accidents/features/weekly.py#L119-L130) — A contagem vira um rótulo binário: houve ou não registro.

Para cada semana incluída, esta consulta cria uma linha por trecho. Se dois acidentes foram registrados, a contagem é dois, mas o rótulo é um. Sem registro, o rótulo é zero. Esse zero não prova que não houve acidente: depende da cobertura assumida.

### 02:56–03:18 — O que o modelo pode saber

**Na tela:** [`temporal.py`, linhas 69–77](../codigo/src/road_accidents/features/temporal.py#L69-L77) — A contagem histórica fica ausente se a fonte não comprovar cobertura e publicação prévia.

Comprimento da via e semana do ano já são conhecidos. O histórico de acidentes é diferente: precisa ter cobertura comprovada e estar disponível antes da previsão. Se não há essa comprovação, o código deixa o campo como dado ausente, em vez de preencher com zero.

### 03:18–03:40 — Ajuste apenas com dados de treino

**Na tela:** [`training.py`, linhas 213–225](../codigo/src/road_accidents/experiments/training.py#L213-L225) — O `fit` recebe o treino; a validação só passa pelo `transform`.

Antes de treinar, precisamos preparar as colunas. Repare que a função de ajuste recebe só os dados de treino. A validação reaproveita a transformação sem recalcular valores como a mediana. Com as colunas preparadas, o Lion vai mostrar como os modelos usam esses dados.

## Lion — 03:40–05:30

### 03:40–04:02 — Modelos que recebem a tabela

**Na tela:** [`baselines.py`, linhas 33–51](../codigo/src/road_accidents/models/baselines.py#L33-L51) — Random Forest, XGBoost e MLP recebem atributos sem a lista de vizinhos.

No caminho tabular, cada linha descreve um trecho em uma semana. A função escolhe qual modelo recebe essa tabela: Random Forest, XGBoost ou MLP. Nenhum deles recebe a lista de vizinhos. Essa lista entra no outro caminho, o dos modelos de grafo.

### 04:02–04:24 — Como os vizinhos entram no modelo

**Na tela:** [`graphsage.py`, linhas 19–25](../codigo/src/road_accidents/models/graphsage.py#L19-L25) — As camadas combinam os atributos do trecho com informações da vizinhança.

Neste exemplo, o GraphSAGE recebe duas entradas: as características dos trechos e a lista de conexões. As camadas combinam a informação de cada trecho com a dos vizinhos. No fim, sai um valor por trecho, que depois é convertido em uma pontuação.

### 04:24–04:46 — Da pontuação à decisão

**Na tela:** [`metrics.py`, linhas 39–49](../codigo/src/road_accidents/evaluation/metrics.py#L39-L49) — A função escolhe o ponto de corte com o maior F1.

A previsão começa como uma pontuação. Para decidir quais trechos sinalizar, a função testa vários pontos de corte na validação. Ela escolhe o corte com maior F1, uma medida que combina a proporção de alertas corretos com a proporção de casos encontrados.

### 04:46–05:08 — Teste sem mudar o modelo

**Na tela:** [`runner.py`, linhas 164–179](../codigo/src/road_accidents/evaluation/runner.py#L164-L179) — O avaliador reaproveita o modelo e o corte já escolhidos.

O teste usa o modelo salvo. Aqui, o programa alinha as previsões e as respostas esperadas pelo trecho e pela semana. Depois compara cada pontuação com o corte que veio da validação. O código não recalcula esse corte nem treina novamente no teste.

### 05:08–05:30 — Como repetir a avaliação

**Na tela:** [`reproduce.py`, linhas 349–361](../codigo/src/road_accidents/reporting/reproduce.py#L349-L361) — A reprodução compara uma nova avaliação com as métricas arquivadas.

Por fim, a reprodução refaz a base com as fontes arquivadas. Neste trecho, o programa usa os modelos salvos para repetir a avaliação e comparar as métricas com as originais, sem treinar de novo. Assim fechamos o caminho dos dados até a avaliação. Obrigado!
